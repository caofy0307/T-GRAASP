import os
import copy
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

from tgraasp.models import Alignment_Model
from tgraasp.utils import cal_auc_ap_acc, compute_block_matrix, compare_block_matrices,compute_neighborhood_enrichment
from tgraasp.CustomFunc import onehot_to_label, sqrtm

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, name: str = "tgraasp") -> logging.Logger:
    """Create a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{name}_{id(log_dir)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        # console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # file
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------
# class TGraaspTrainer:
import os
import copy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.data import Data
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
from scipy.stats import pearsonr




class TGraaspTrainer:
    def __init__(
        self,
        model,
        sp_graph_list: Union[List[Data], List[List[Data]]],
        F_inputs: int,
        sc_graph_list: Optional[Union[List[Data], List[List[Data]]]] = None,
        sp_adj_list: Optional[List[torch.Tensor]] = None,
        *,
        device: str = "cuda",
        pretrain_epochs: int = 50,
        patience: int = 50,
        batch_size: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        log_dir: Union[str, Path] = "logs",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        disc_steps: int = 5,
        l2_lambda: float = 1e-3,
        seed: int = 42,
    ):
        set_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.graphs = self._flatten(sp_graph_list)
        self.sc_graphs = self._flatten(sc_graph_list) if sc_graph_list is not None else None
        self.adj_list = sp_adj_list
        self.F_inputs = F_inputs

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.pretrain_epochs = pretrain_epochs
        self.patience = patience
        self.batch_size = batch_size

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.disc_steps = disc_steps
        self.l2_lambda = l2_lambda

        self.best_auc_mean = 0.0
        self.best_auc_mean_epoch = 0
        self.best_model_state = None

        print("=== TGraaspTrainer: Data Debug ===")
        print(f"Spatial datasets: {len(self.graphs)}")
        if self.sc_graphs is not None:
            print(f"scRNA datasets:  {len(self.sc_graphs)}")
        if self.adj_list is not None:
            print(f"Spatial adjs:    {len(self.adj_list)}")

        self.loader = PyGLoader(self.graphs, batch_size=self.batch_size, shuffle=False)

    @staticmethod
    def _flatten(x):
        if x is None:
            return None
        if len(x) == 0:
            return []
        if isinstance(x[0], list):
            out = []
            for chunk in x:
                out.extend(chunk)
            return out
        return x

    # -----------------------------------------------------------------
    # Pretraining
    # -----------------------------------------------------------------
    def pretrain(self) -> Optional[dict]:
        print("Start Pretrain...")
        no_improve_count = 0

        for epoch in range(1, self.pretrain_epochs + 1):
            total_running_loss = 0.0

            # ---- train ----
            self.model.train()
            for p in self.model.discriminator.parameters():
                p.requires_grad = True

            for batch in self.loader:
                datas = batch if isinstance(batch, list) else [batch]
                for data in datas:
                    data = data.to(self.device)

                    z, _ = self.model.encode(None, data.x, None, None, None, None)
                    w = self.model.__u__
                    cl = w.detach().cpu().softmax(dim=-1)

                    # discriminator steps
                    for _ in range(self.disc_steps):
                        self.model.discriminator.train()
                        disc_optim = Adam(self.model.discriminator.parameters(), lr=1e-4)
                        disc_optim.zero_grad()
                        disc_loss = self.model.discriminator_loss(z, None, None)
                        disc_loss.backward(retain_graph=True)
                        disc_optim.step()

                    # loss assembly
                    loss = 0
                    loss = loss + self.model.reg_loss(z, None, None)
                    loss = loss + 10 * self.model.recon_loss(
                        z, data.pos_edge, data.train_edge_index, data.test_edge_index, None
                    )
                    loss = loss + self.model.cluster_loss(None, data.y)
                    loss = loss + (1 / data.num_nodes) * self.model.kl_loss()

                    # L2 on encoder (except biases)
                    l2_reg = torch.tensor(0., requires_grad=True, device=self.device)
                    for name, p in self.model.encoder.named_parameters():
                        if 'bias' not in name:
                            l2_reg = l2_reg + torch.norm(p, p=2)
                    loss = loss + self.l2_lambda * l2_reg

                    loss.backward()
                    clip_grad_value_(self.model.encoder.parameters(), 0.1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_running_loss += float(loss.item())

            # ---- eval ----
            with torch.no_grad():
                self.model.eval()
                all_aucs, all_logs = [], []
                for idx, data in enumerate(self.graphs):
                    data = data.to(self.device)
                    z, _ = self.model.encode(None, data.x, None, None, None, None)
                    w = self.model.__u__
                    cl = w.detach().cpu().softmax(dim=-1)

                    pred = onehot_to_label(cl.t()).values.reshape(-1)
                    y_true = data.y.detach().cpu().numpy()
                    ari = adjusted_rand_score(pred, y_true)
                    acc = accuracy_score(y_true, pred)
                    nmi = normalized_mutual_info_score(y_true, pred)

                    adj_pred = self.model.decoder.forward_all(z)
                    auroc, ap_score, acc_score = cal_auc_ap_acc(
                        1, data.test_edge_index, z, adj_pred, data.neg_edge
                    )
                    auc = float(np.mean(auroc))
                    ap = float(np.mean(ap_score))
                    acc_lp = float(np.mean(acc_score))

                    all_aucs.append(auc)
                    all_logs.append((idx, ari, acc, nmi, auc, ap, acc_lp))

                mean_auc = float(np.mean(all_aucs)) if all_aucs else 0.0
                for (idx, ari, acc, nmi, auc, ap, acc_lp) in all_logs:
                    print(
                        f"[Pretrain] Epoch {epoch:03d} | Dataset {idx:02d} | "
                        f"ARI {ari:.3f} | ACC {acc:.3f} | NMI {nmi:.3f} | "
                        f"AUC {auc:.3f} | AP {ap:.3f} | ACC1 {acc_lp:.3f}"
                    )

                if mean_auc > self.best_auc_mean:
                    self.best_auc_mean = mean_auc
                    self.best_auc_mean_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_model_state, self.save_dir / "best_model_auc_mean.pkl")
                    print(f"✔ New best mean AUROC {self.best_auc_mean:.4f} at epoch {epoch}")
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                avg_loss = total_running_loss / max(1, len(self.graphs))
                print(
                    f"Epoch {epoch:03d} | Avg Loss {avg_loss:.4f} | "
                    f"Mean AUC {mean_auc:.4f} | Best {self.best_auc_mean:.4f} "
                    f"(Epoch {self.best_auc_mean_epoch})"
                )

                if no_improve_count > self.patience:
                    print(f"⏹ Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        print("\n=== Pretraining Finished ===")
        print(f"Best Mean AUC: {self.best_auc_mean:.4f} at Epoch {self.best_auc_mean_epoch}")
        return self.best_model_state

    # -----------------------------------------------------------------
    # Finetune helpers
    # -----------------------------------------------------------------
    def _precompute_spatial_targets(self):
        spatial_targets = []
        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.graphs)):
                data = self.graphs[i].to(self.device)
                z_spatial, _ = self.model.encode(
                    None, data.x, None, None, None, None, is_single_cell=False
                )
                adj_cluster = self.adj_list[i]
                spatial_targets.append({
                    'z': z_spatial.clone().detach(),
                    'm': self.model.__m__.clone().detach(),
                    'adj_cluster': adj_cluster.clone().detach(),
                    'data': data,
                })
                print(
                    f"    Spatial[{i}]: Z shape={z_spatial.shape}, "
                    f"adj shape={adj_cluster.shape}, "
                    f"M shape={self.model.__m__.shape}"
                )
        return spatial_targets

    def _compute_cluster_centroids(self, z, y, num_classes):
        one_hot = F.one_hot(y, num_classes).float()
        counts = one_hot.sum(0).clamp(min=1).unsqueeze(1)
        return (one_hot.T @ z) / counts

    def _normalized_sigmoid_adj(self, centroids):
        c_norm = F.normalize(centroids, dim=1)
        return torch.sigmoid(c_norm @ c_norm.T)

    # -----------------------------------------------------------------
    # Finetune
    # -----------------------------------------------------------------
    def finetune(
        self,
        epochs: int = 200,
        patience: int = 30,
        lr: float = 1e-4,
        lambda_recon: float = 1.0,
        lambda_mse: float = 1.0,
        n_perms: int = 500,
        eval_spatial_idx=None,
    ):
        logger = setup_logger(self.log_dir, name="finetune")

        logger.info("=" * 60)
        logger.info("  Fine-tuning [Auto-Balanced]")
        logger.info("=" * 60)

        # ---- activate alignment ----
        self.model.encoder.activate_sc_alignment = True
        if self.model.encoder.alignment_model is None:
            self.model.encoder.alignment_model = Alignment_Model(
                input_dim=self.F_inputs, output_dim=self.F_inputs, hidden_dim=4000
            ).to(self.device)

        # ---- precompute spatial enrichment GT ----
        logger.info(f"  Pre-computing Spatial Enrichment GT (n_perms={n_perms})...")
        spatial_enrich_gt_list = []
        for i in range(len(self.graphs)):
            en_gt = compute_neighborhood_enrichment(
                self.adj_list[i], self.graphs[i].y, n_perms=n_perms, device=str(self.device)
            )
            spatial_enrich_gt_list.append(en_gt)
            logger.info(f"    SP[{i}] enrichment shape: {en_gt.shape}")

        # ---- precompute spatial targets ----
        logger.info("  Pre-computing spatial targets...")
        spatial_targets = self._precompute_spatial_targets()

        # ---- precompute spatial cluster centroids ----
        sp_cluster_centroids = []
        for i in range(len(self.graphs)):
            data_i = self.graphs[i].to(self.device)
            z_sp_i = spatial_targets[i]['z']
            y_sp_i = data_i.y.long()
            num_classes = int(y_sp_i.max().item()) + 1
            centroids = self._compute_cluster_centroids(z_sp_i, y_sp_i, num_classes)
            sp_cluster_centroids.append(centroids)
            logger.info(f"    SP[{i}] centroids: {centroids.shape}")

        # ---- check __m__ dimension ----
        z_dim = sp_cluster_centroids[0].shape[1]
        m_dim = spatial_targets[0]['m'].shape[1]
        use_m_for_mse = (m_dim == z_dim)
        logger.info(f"  __m__ dim={m_dim}, z dim={z_dim}, use_m_for_mse={use_m_for_mse}")

        # ---- freeze everything except alignment_model ----
        for name, param in self.model.named_parameters():
            param.requires_grad = ("alignment_model" in name)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"  Trainable params: {sum(p.numel() for p in trainable_params)}")

        finetune_optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            finetune_optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        num_spatial = len(self.graphs)
        sc_data = self.sc_graphs[0].to(self.device)
        y_sc = sc_data.y.long()
        num_classes_sc = int(y_sc.max().item()) + 1

        if eval_spatial_idx is not None:
            eval_indices = [eval_spatial_idx] if isinstance(eval_spatial_idx, int) else eval_spatial_idx
        else:
            eval_indices = list(range(num_spatial))

        # ---- warmup: compute auto-scale ----
        logger.info("  Computing auto-scale...")
        self.model.train()
        with torch.no_grad():
            z_w, _ = self.model.encode(None, sc_data.x, None, None, None, None, is_single_cell=True)
            sc_cen_w = self._compute_cluster_centroids(z_w, y_sc, num_classes_sc)

            raw_recon = 0.0
            for i in range(num_spatial):
                sp_cen = sp_cluster_centroids[i]
                K = min(sc_cen_w.shape[0], sp_cen.shape[0])
                adj_sc_w = self._normalized_sigmoid_adj(sc_cen_w[:K])
                adj_sp_w = self._normalized_sigmoid_adj(sp_cen[:K])
                raw_recon += F.mse_loss(adj_sc_w, adj_sp_w).item()
            raw_recon /= num_spatial

            raw_mse = 0.0
            for i in range(num_spatial):
                if use_m_for_mse:
                    src, tgt = self.model.__m__, spatial_targets[i]['m']
                else:
                    src, tgt = sc_cen_w, sp_cluster_centroids[i]
                K_m = min(src.shape[0], tgt.shape[0])
                raw_mse += F.mse_loss(src[:K_m], tgt[:K_m]).item()
            raw_mse /= num_spatial

        if raw_recon > 1e-8 and raw_mse > 1e-8:
            recon_scale = 1.0 / raw_recon
            mse_scale = 1.0 / raw_mse
        else:
            recon_scale = 1.0
            mse_scale = 1.0

        logger.info(f"  [SCALE] raw_recon={raw_recon:.6f}, raw_mse={raw_mse:.6f}")
        logger.info(f"  [SCALE] recon_scale={recon_scale:.2e}, mse_scale={mse_scale:.2e}")
        logger.info(
            f"  [SCALE] w_recon={lambda_recon * recon_scale:.4f}, "
            f"w_mse={lambda_mse * mse_scale:.4f}"
        )
        logger.info("=" * 60)

        best_corr = -float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # ---- train step ----
            self.model.train()
            finetune_optimizer.zero_grad()

            z_sc, _ = self.model.encode(None, sc_data.x, None, None, None, None, is_single_cell=True)
            sc_centroids = self._compute_cluster_centroids(z_sc, y_sc, num_classes_sc)

            # Loss 1: recon (normalized sigmoid adjacency matching)
            total_recon = torch.tensor(0.0, device=self.device)
            for i in range(num_spatial):
                sp_cen = sp_cluster_centroids[i]
                K = min(sc_centroids.shape[0], sp_cen.shape[0])
                adj_sc = self._normalized_sigmoid_adj(sc_centroids[:K])
                adj_sp = self._normalized_sigmoid_adj(sp_cen[:K]).detach()
                total_recon += F.mse_loss(adj_sc, adj_sp)
            avg_recon = total_recon / num_spatial

            # Loss 2: MSE (centroid or __m__ alignment)
            total_mse = torch.tensor(0.0, device=self.device)
            for i in range(num_spatial):
                if use_m_for_mse:
                    src, tgt = self.model.__m__, spatial_targets[i]['m']
                else:
                    src, tgt = sc_centroids, sp_cluster_centroids[i]
                K_m = min(src.shape[0], tgt.shape[0])
                total_mse += F.mse_loss(src[:K_m], tgt[:K_m].detach())
            avg_mse = total_mse / num_spatial

            total_loss = (
                lambda_recon * recon_scale * avg_recon
                + lambda_mse * mse_scale * avg_mse
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            finetune_optimizer.step()

            # ---- eval step ----
            with torch.no_grad():
                self.model.eval()
                z_sc_eval, _ = self.model.encode(
                    None, sc_data.x, None, None, None, None, is_single_cell=True
                )
                adj_sc_eval = self.model.decoder.forward_all(z_sc_eval)
                en_pred = compute_neighborhood_enrichment(
                    adj_sc_eval, sc_data.y, n_perms=n_perms, device=str(self.device)
                )

                all_pearsons = []
                all_aucs = []

                for idx in eval_indices:
                    en_gt = spatial_enrich_gt_list[idx]
                    common_clusters = sorted(
                        set(en_gt.columns).intersection(set(en_pred.columns))
                    )
                    if len(common_clusters) >= 2:
                        mat_true = en_gt.loc[common_clusters, common_clusters].values
                        mat_pred = en_pred.loc[common_clusters, common_clusters].values
                        triu_idx = np.triu_indices(len(common_clusters), k=1)
                        vec_true = mat_true[triu_idx]
                        vec_pred = mat_pred[triu_idx]
                        p_corr, _ = pearsonr(vec_true, vec_pred)
                        if not np.isnan(p_corr):
                            all_pearsons.append(p_corr)

                    try:
                        sp_eval = self.graphs[idx].to(self.device)
                        z_sp_eval, _ = self.model.encode(
                            None, sp_eval.x, None, None, None, None, is_single_cell=False
                        )
                        adj_sp_pred = self.model.decoder.forward_all(z_sp_eval)
                        auroc, _, _ = cal_auc_ap_acc(
                            1, sp_eval.test_edge_index, z_sp_eval, adj_sp_pred, sp_eval.neg_edge
                        )
                        all_aucs.append(np.mean(auroc))
                    except Exception:
                        all_aucs.append(-1)

                avg_corr = np.mean(all_pearsons) if all_pearsons else float('nan')
                avg_auc = np.mean(all_aucs) if all_aucs else float('nan')

                s_r = lambda_recon * recon_scale * avg_recon.item()
                s_m = lambda_mse * mse_scale * avg_mse.item()
                logger.info(
                    f"[FT] Ep {epoch:03d} | "
                    f"Loss {total_loss.item():.4f} (R:{s_r:.4f} M:{s_m:.4f}) | "
                    f"Corr {avg_corr:.4f} | AUC {avg_auc:.4f} | "
                    f"LR {finetune_optimizer.param_groups[0]['lr']:.2e}"
                )

                if not np.isnan(avg_corr):
                    scheduler.step(avg_corr)

                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    save_path = self.save_dir / "best_finetune_model.pkl"
                    torch.save(best_model_state, save_path)
                    logger.info(f"  -> Best! Corr={best_corr:.4f} @ {save_path}")

                if epoch - best_epoch > patience:
                    logger.info(f"  -> Early stop ep {epoch} (best {best_epoch})")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        final_path = self.save_dir / "final_finetune_model.pkl"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"\nDone. Best ep {best_epoch}, Corr {best_corr:.4f}")
        return best_model_state


# -----------------------------------------------------------------
if __name__ == "__main__":
    raise NotImplementedError("Import TGraaspTrainer and use it in your training script.")
