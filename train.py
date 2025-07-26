import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader as PyGLoader
from models import STx_encoder, STx_discriminator, STx_ARGVA, Alignment_Model
from utils import cal_auc_ap_acc
from torch.nn.utils import clip_grad_value_
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
import numpy as np
from scripts.CustomFunc import onehot_to_label, sqrtm
from utils import compute_block_matrix, compare_block_matrices
import copy
import random



# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子为42
set_seed(42)

class Trainer:
    def __init__(self, model, sp_graph_list,F_inputs,sc_graph_list, sp_adj_list=None, device="cuda", pretrain_epochs=50):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.graphs = sp_graph_list
        self.sc_graphs = sc_graph_list
        self.adj = sp_adj_list
        self.F_inputs = F_inputs
        self.loader = PyGLoader(self.graphs, batch_size=1, shuffle=False)
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.pretrain_epochs = pretrain_epochs

    def pretrain(self):
        l2_lambda = 0.001
        print("Start Pretrain...")
        for epoch in range(1, self.pretrain_epochs + 1):
            running_loss = 0.0
            for data in self.loader:
                self.model.train()
                data = data.to(self.device)
                s = data.x.shape[0]

                z, _ = self.model.encode(None, data.x, None, None, None, None)
                w = self.model.__u__
                cl = w.cpu().softmax(dim=-1)

                # --- Discriminator step ---
                for _ in range(5):
                    self.model.discriminator.train()
                    disc_optimizer = Adam(self.model.discriminator.parameters(), lr=1e-4)
                    disc_optimizer.zero_grad()
                    disc_loss = self.model.discriminator_loss(z, None, None)
                    disc_loss.backward(retain_graph=True)
                    disc_optimizer.step()

                # --- Loss computation ---
                loss = 0
                loss = loss + self.model.reg_loss(z, None, None)
                loss = loss + 10 * self.model.recon_loss(z, data.pos_edge, data.train_edge_index, data.test_edge_index, None)
                loss = loss + self.model.cluster_loss(None, data.y)
                loss = loss + (1 / data.num_nodes) * self.model.kl_loss()

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

            # --- Validation log ---
            with torch.no_grad():
                self.model.eval()
                data = self.graphs[0].to(self.device)
                z, _ = self.model.encode(None, data.x, None, None, None, None)
                w = self.model.__u__
                cl = w.cpu().softmax(dim=-1)
                ari1 = adjusted_rand_score(onehot_to_label(cl.t()).values.reshape(-1), data.y.cpu())
                acc1 = accuracy_score(data.y.cpu(), onehot_to_label(cl.t()).values.reshape(-1))
                nmi1 = normalized_mutual_info_score(data.y.cpu(), onehot_to_label(cl.t()).values.reshape(-1))
                adj = self.model.decoder.forward_all(z)
                auroc1, ap_score1, acc_score1 = cal_auc_ap_acc(1, data.test_edge_index, z, adj, data.neg_edge)
                auc = np.mean(auroc1)
                ap = np.mean(ap_score1)
                acc = np.mean(acc_score1)

                print(f"[Pretrain] Epoch {epoch:03d} | Loss {running_loss:.4f} | ARI {ari1:.3f} | ACC {acc1:.3f} | NMI {nmi1:.3f} | AUC {auc:.3f} | AP {ap:.3f} | ACC1 {acc:.3f}")
    def compute_block_alignment_loss(self, adj1, adj2, labels1, labels2, threshold=0.6):
        block1 = compute_block_matrix(adj1, labels1, threshold=threshold, is_prediction=True)
        block2 = compute_block_matrix(adj2, labels2, threshold=threshold, is_prediction=True)

        # Align common clusters
        common_clusters = sorted(set(block1.columns).intersection(block2.columns))
        block1 = block1.reindex(index=common_clusters, columns=common_clusters)
        block2 = block2.reindex(index=common_clusters, columns=common_clusters)

        # Extract upper triangles
        triu_indices = np.triu_indices(len(common_clusters), k=1)
        vec1 = torch.tensor(block1.values[triu_indices], dtype=torch.float32)
        vec2 = torch.tensor(block2.values[triu_indices], dtype=torch.float32)

        # Compute MSE Loss
        loss = F.mse_loss(vec1, vec2)

        return loss

    def finetune(self, epochs=50, patience=10, lr=1e-4):
        print("\n Start Fine-tuning Alignment between spatial and scRNA")
        
        # 确保encoder有alignment功能
        if not hasattr(self.model.encoder, 'activate_sc_alignment'):
            raise ValueError("Encoder doesn't support alignment. Please use an encoder with activate_sc_alignment parameter.")
        
        # 激活alignment并创建alignment_model
        self.model.encoder.activate_sc_alignment = True
        if self.model.encoder.alignment_model is None:
            print("Creating new alignment model...")
            # 使用固定的维度
            self.model.encoder.alignment_model = Alignment_Model(
                input_dim= self.F_inputs,  # 保持原始维度
                output_dim= self.F_inputs,  # 保持原始维度
                hidden_dim=4000   # 中间层维度
            ).to(self.device)
        
        # 设置模型为训练模式
        self.model.train()
        
        # 冻结除了alignment_model之外的所有参数
        for name, param in self.model.named_parameters():
            if "alignment_model" in name:
                param.requires_grad = True
                # print(f"Training parameter: {name}")
            else:
                param.requires_grad = False
        
        # 获取需要训练的参数
        alignment_model_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(alignment_model_params) == 0:
            raise ValueError("No parameters to train! Check if alignment_model is properly initialized.")
        
        # print(f"Number of trainable parameters: {sum(p.numel() for p in alignment_model_params)}")
        
        # 使用AdamW优化器
        finetune_optimizer = torch.optim.AdamW(alignment_model_params, lr=lr, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            finetune_optimizer, 
            T_0=10,  # 每10个epoch重启一次
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=1e-6  # 最小学习率
        )

        best_corr = -float('inf')
        best_epoch = 0
        best_model_state = None
        
        # 用于计算移动平均的损失
        loss_history = []
        window_size = 5

        num_datasets = len(self.graphs)
        if num_datasets == 0:
            raise ValueError("No spatial datasets provided for fine-tuning.")
        if len(self.sc_graphs) != num_datasets:
            print(f"Warning: Number of spatial datasets ({num_datasets}) and scRNA-seq datasets ({len(self.sc_graphs)}) differ. Using min_len for pairing.")
            num_datasets = min(num_datasets, len(self.sc_graphs))


        for epoch in range(1, epochs + 1):
            self.model.train()
            finetune_optimizer.zero_grad()
            
            epoch_total_loss = 0.0
            epoch_m_alignment_loss = 0.0

            for i in range(num_datasets):
                spatial_data = self.graphs[i].to(self.device)
                sc_data = self.sc_graphs[i].to(self.device)

                # 空间数据编码 - 不需要梯度，作为对齐目标
                with torch.no_grad():
                    z_spatial, _ = self.model.encode(None, spatial_data.x, None, None, None, None, is_single_cell=False)
                    m_spatial = self.model.__m__.clone().detach()
                
                # 单细胞数据编码 - 需要梯度，通过alignment_model
                z_sc, _ = self.model.encode(None, sc_data.x, None, None, None, None, is_single_cell=True)
                m_sc = self.model.__m__
                
                # 1. 特征对齐损失
                m_alignment_loss = F.mse_loss(m_sc, m_spatial)
                
                # 总损失 (目前只有特征对齐损失)
                total_loss_single_dataset = m_alignment_loss
                
                epoch_total_loss += total_loss_single_dataset.item()
                epoch_m_alignment_loss += m_alignment_loss.item()

                # 累积梯度，每个数据集的损失权重为 1/num_datasets
                (total_loss_single_dataset / num_datasets).backward()

            # 记录损失 (平均到每个数据集)
            current_epoch_avg_loss = epoch_total_loss / num_datasets
            loss_history.append(current_epoch_avg_loss)
            if len(loss_history) > window_size:
                loss_history.pop(0)
            
            # 计算移动平均损失
            avg_loss = sum(loss_history) / len(loss_history)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(alignment_model_params, max_norm=1.0)
            
            # 更新参数
            finetune_optimizer.step()
            
            # 更新学习率
            scheduler.step()

            # 评估
            if epoch % 5 == 0 or epoch == 1:
                with torch.no_grad():
                    self.model.eval()
                    
                    all_block_corrs = []
                    all_aucs = []
                    
                    avg_eval_m_alignment_loss = 0.0

                    for i in range(num_datasets):
                        spatial_data_eval = self.graphs[i].to(self.device)
                        sc_data_eval = self.sc_graphs[i].to(self.device)
                        adj_spatial = self.adj[i].to(self.device)
                        z_spatial_eval, _ = self.model.encode(None, spatial_data_eval.x, None, None, None, None, is_single_cell=False)
                        m_spatial_eval = self.model.__m__.clone().detach()
                        z_sc_eval, _ = self.model.encode(None, sc_data_eval.x, None, None, None, None, is_single_cell=True)
                        m_sc_eval = self.model.__m__

                        eval_m_alignment_loss = F.mse_loss(m_sc_eval, m_spatial_eval)
                        avg_eval_m_alignment_loss += eval_m_alignment_loss.item()
                        
                        z_spatial = self.model.decoder.forward_all(z_spatial_eval)
                        adj_sc = self.model.decoder.forward_all(z_sc_eval) # Not used if structure loss is removed but kept for consistency
                        
                        block_spatial = compute_block_matrix(adj_spatial, spatial_data_eval.y, threshold=0.6, is_prediction=True)
                        # For block_sc, we'd typically use sc_data_eval.y if available and meaningful for block comparison
                        # Assuming sc_data_eval.y has cluster labels for scRNA-seq data that can be compared to spatial
                        if hasattr(sc_data_eval, 'y') and sc_data_eval.y is not None:
                             block_sc = compute_block_matrix(adj_sc, sc_data_eval.y, threshold=0.6, is_prediction=True)
                             block_corr, _ = compare_block_matrices(block_spatial, block_sc)
                             all_block_corrs.append(block_corr)
                        else:
                             # Handle cases where sc_data might not have 'y' or it's not comparable for block_corr
                             # Or simply skip block_corr if sc_data.y is not suitable
                             pass


                        # 计算AUC
                        try:
                            auroc, ap_score, acc_score = cal_auc_ap_acc(1, spatial_data_eval.test_edge_index, z_spatial_eval, z_spatial, spatial_data_eval.neg_edge)
                            auc = np.mean(auroc)
                            all_aucs.append(auc)
                        except Exception as e:
                            # print(f"Warning: Could not compute AUC for dataset {i}: {e}")
                            all_aucs.append(-1) # Or handle as NaN, or skip

                    avg_block_corr = np.mean(all_block_corrs) if all_block_corrs else -1.0 # Default if no block_corrs
                    avg_auc = np.mean(all_aucs) if all_aucs else -1.0 # Default if no AUCs

                    print(f"[Finetune] Epoch {epoch:03d} | "
                          f"Avg Loss: {avg_loss:.6f} | "
                          f"Avg Align Loss (Train): {(epoch_m_alignment_loss / num_datasets):.6f} | "
                          f"Avg Align Loss (Eval): {(avg_eval_m_alignment_loss / num_datasets):.6f} | "
                          f"Avg Corr: {avg_block_corr:.4f} | "
                          f"Avg AUC: {avg_auc:.4f}")

                    if avg_block_corr > best_corr:
                        best_corr = avg_block_corr
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        print(f" New best model state saved at epoch {epoch} with Avg Corr: {best_corr:.4f}")
                    elif epoch - best_epoch > patience:
                        print(f" Early stopping at epoch {epoch} (best epoch was {best_epoch} with Avg Corr: {best_corr:.4f})")
                        break
        
        # if best_model_state:
        #     torch.save(best_model_state,'/shareN/data8/SwapTmp/fy/Spatial/ARGA-ARVGA/GBM/Verify/model/finetuned_best_model.pkl')
        #     self.model.load_state_dict(best_model_state) # Load best model before finishing
        #     print(f"Fine-tuning Finished. Loaded best model from epoch {best_epoch} with Avg Corr: {best_corr:.4f}")
        # else:
        #     print(f"Fine-tuning Finished. No best model state was saved (possibly due to no improvement or errors).")


        return best_model_state


if __name__ == "__main__":
    raise NotImplementedError("Use this module by importing Trainer and passing graph lists externally.")
