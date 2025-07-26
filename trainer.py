import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader as PyGLoader
from models import STx_encoder, STx_discriminator, STx_ARGVA
from utils import cal_auc_ap_acc
from torch.nn.utils import clip_grad_value_
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
import numpy as np
from scripts.CustomFunc import onehot_to_label, sqrtm
from utils import compute_block_matrix, compare_block_matrices
import copy
import os
from torch_geometric.data import Data

class MultiDataTrainer:
    def __init__(self, model, sp_graph_lists, sc_graph_lists, sp_adj_lists=None, 
                 device="cuda", pretrain_epochs=50, patience=50, batch_size=1,
                 save_dir='checkpoints'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.graphs_list = sp_graph_lists  # List of lists of graphs
        self.sc_graphs_list = sc_graph_lists
        self.adj_list = sp_adj_lists
        
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.pretrain_epochs = pretrain_epochs
        self.patience = patience
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics tracking for each dataset (仍然保留)
        self.best_metrics = {i: {'auc': 0, 'epoch': 0} for i in range(len(self.graphs_list))}
        self.best_model_states = {i: None for i in range(len(self.graphs_list))}
        
        # 用于整体AUC均值early stopping/保存
        self.best_auc_mean = 0
        self.best_auc_mean_epoch = 0
        self.best_model_state = None

        # Debug information
        print("=== Data Structure Debug Info ===")
        print(f"Number of datasets: {len(self.graphs_list)}")
        # for i, graphs in enumerate(self.graphs_list):
            # print(f"\nDataset {i}:")
            # print(f"Type: {type(graphs)}")
            # print(f"Length: {len(graphs)}")
            # if len(graphs) > 0:
            #     print(f"First item type: {type(graphs[0])}")
            #     if isinstance(graphs[0], tuple):
            #         print(f"Tuple length: {len(graphs[0])}")
            #         for j, item in enumerate(graphs[0]):
            #             print(f"Tuple item {j} type: {type(item)}")
    
    def load_graph_data(self, data_item):
        """Helper function to load and process graph data"""
        if isinstance(data_item, Data):
            return data_item
        elif isinstance(data_item, tuple):
            for item in data_item:
                if isinstance(item, Data):
                    return item
            return data_item[0]
        elif isinstance(data_item, str):
            print(f"Warning: Received string data: {data_item}")
            raise ValueError(f"Received string data but don't know how to load it: {data_item}")
        else:
            return data_item
    
    def pretrain(self):
        l2_lambda = 0.001
        no_improve_count = 0
        print("Start Pretrain...")

        for epoch in range(1, self.pretrain_epochs + 1):
            total_running_loss = 0.0
            metrics_per_dataset = []

            # Training phase
            self.model.train()
            for dataset_idx, graphs in enumerate(self.graphs_list):
                running_loss = 0.0
                for data in graphs:
                    data = data.to(self.device)
                    s = data.x.shape[0]

                    z, _ = self.model.encode(None, data.x, None, None, None, None)
                    w = self.model.__u__
                    cl = w.cpu().softmax(dim=-1)

                    # Discriminator step
                    for _ in range(5):
                        self.model.discriminator.train()
                        disc_optimizer = Adam(self.model.discriminator.parameters(), lr=1e-4)
                        disc_optimizer.zero_grad()
                        disc_loss = self.model.discriminator_loss(z, None, None)
                        disc_loss.backward(retain_graph=True)
                        disc_optimizer.step()

                    # Loss computation
                    loss = 0
                    loss = loss + self.model.reg_loss(z, None, None)
                    loss = loss + 10 * self.model.recon_loss(z, data.pos_edge, data.train_edge_index, data.test_edge_index, None)
                    loss = loss + self.model.cluster_loss(None, data.y)
                    loss = loss + (1 / data.num_nodes) * self.model.kl_loss()

                    # L2 regularization
                    l2_regularization = torch.tensor(0., requires_grad=True, device=self.device)
                    for name, param in self.model.encoder.named_parameters():
                        if 'bias' not in name:
                            l2_regularization = l2_regularization + torch.norm(param, p=2)
                    loss = loss + l2_lambda * l2_regularization

                    loss.backward()
                    clip_grad_value_(self.model.encoder.parameters(), 0.1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss += loss.item()

                total_running_loss += running_loss

                # Validation for each dataset
                with torch.no_grad():
                    self.model.eval()
                    val_data = graphs[0].to(self.device)
                    z, _ = self.model.encode(None, val_data.x, None, None, None, None)
                    w = self.model.__u__
                    cl = w.cpu().softmax(dim=-1)
                    ari1 = adjusted_rand_score(onehot_to_label(cl.t()).values.reshape(-1), val_data.y.cpu())
                    acc1 = accuracy_score(val_data.y.cpu(), onehot_to_label(cl.t()).values.reshape(-1))
                    nmi1 = normalized_mutual_info_score(val_data.y.cpu(), onehot_to_label(cl.t()).values.reshape(-1))
                    adj = self.model.decoder.forward_all(z)
                    auroc1, ap_score1, acc_score1 = cal_auc_ap_acc(1, val_data.test_edge_index, z, adj, val_data.neg_edge)
                    auc = np.mean(auroc1)
                    ap = np.mean(ap_score1)
                    acc = np.mean(acc_score1)
                    metrics = {
                        'dataset_idx': dataset_idx,
                        'loss': running_loss,
                        'ari': ari1,
                        'acc': acc1,
                        'nmi': nmi1,
                        'auc': auc,
                        'ap': ap,
                        'acc1': acc
                    }
                    metrics_per_dataset.append(metrics)

                    print(f"[Pretrain] Epoch {epoch:03d} | Dataset {dataset_idx} | "
                          f"Loss {running_loss:.4f} | ARI {ari1:.3f} | ACC {acc1:.3f} | "
                          f"NMI {nmi1:.3f} | AUC {auc:.3f} | AP {ap:.3f} | ACC1 {acc:.3f}")

            # === 计算所有dataset的AUC均值，并用其early stopping和模型保存 ===
            aucs = [m['auc'] for m in metrics_per_dataset]
            mean_auc = float(np.mean(aucs))

            if mean_auc > self.best_auc_mean:
                self.best_auc_mean = mean_auc
                self.best_auc_mean_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_state,
                           os.path.join(self.save_dir, "best_model_auc_mean.pkl"))
                print(f"The best mean AUROC is {self.best_auc_mean:.4f}, Epoch {epoch}）")
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(f"Epoch {epoch}")
            print(f"Average Loss across all datasets: {total_running_loss / len(self.graphs_list):.4f}")
            print(f"Mean AUC across all datasets: {mean_auc:.4f} | Best Mean: {self.best_auc_mean:.4f} (Epoch {self.best_auc_mean_epoch})")

            # for metrics in metrics_per_dataset:
            #     idx = metrics['dataset_idx']
            #     print(f"Dataset {idx} - Best AUC so far: {self.best_metrics[idx]['auc']:.4f} "
            #           f"(Epoch {self.best_metrics[idx]['epoch']})")
            
            if no_improve_count > self.patience:
                print(f" Early stopping at epoch {epoch} (patience={self.patience})")
                break

        print("\n=== Training Complete ===")
        print(f"Best Mean AUC: {self.best_auc_mean:.4f} at Epoch {self.best_auc_mean_epoch}")
        return self.best_model_state   # 注意，最终只返回均值最优这一份模型权重