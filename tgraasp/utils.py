"""
TGRAASP Utilities Module
========================
Utility functions for spatial transcriptomics data processing, evaluation, and analysis.

Main Functions:
- Data loading: SGEDataset, load_ppi_connectivity
- Data manipulation: sparse2tuple, torch_delete, edge_sampling, train_test_split
- Evaluation metrics: cal_auc_ap_acc, compute_block_matrix, compare_block_matrices
- Helpers: create_mask
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, remove_self_loops

# Constants
EPS = 1e-15  # Small epsilon for numerical stability

__all__ = [
    "sparse2tuple", "torch_delete", "edge_sampling", "train_test_split",
    "SGEDataset", "data_counts", "data_counts1", "SGEDataset1", 
    "create_mask", "cal_auc_ap_acc",
    "compute_block_matrix", "compare_block_matrices", "load_ppi_connectivity"
]


def set_seed(seed: int = 42) -> None:
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, name: str = "train"):
    """
    Simple file + console logger.
    Returns a logging.Logger instance.
    """
    import logging
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{name}_{id(log_dir)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def sparse2tuple(sparse_mx):
    """
    Convert a scipy sparse matrix to tuple representation.
    
    Args:
        sparse_mx: Scipy sparse matrix
        
    Returns:
        Tuple of (coordinates, values, shape)
    """
    from scipy import sparse as sp

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = coo_matrix(sparse_mx)
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def torch_delete(tensor: torch.Tensor, index: torch.Tensor, dim: int = 0):
    """
    Delete elements from tensor along specified dimension (CUDA-safe).
    
    Args:
        tensor: Input tensor
        index: Indices to delete
        dim: Dimension along which to delete (default: 0)
        
    Returns:
        Tensor with specified elements removed
    """
    device = tensor.device
    mask = torch.ones(tensor.size(dim), dtype=torch.bool, device=device)
    mask[index] = False
    index_keep = torch.nonzero(mask, as_tuple=False).squeeze()
    return torch.index_select(tensor, dim, index_keep)


def edge_sampling(pos_edge: torch.Tensor, train_edge_index: torch.Tensor, test_edge_index: torch.Tensor):
    """Generate negative samples matching the cardinality of provided positive edges."""
    edge_false = negative_sampling(pos_edge.T)
    N = edge_false.size(1)
    random_indices = torch.randperm(N, device=edge_false.device)
    n_train = train_edge_index.size(1)
    n_test = test_edge_index.size(1)
    idx_train = random_indices[:n_train]
    idx_test = random_indices[n_train:n_train + n_test]
    train_edge_false = edge_false[:, idx_train]
    test_edge_false = edge_false[:, idx_test]
    return train_edge_false, test_edge_false


def train_test_split(adj_values: torch.Tensor, test_ratio: float):
    """Create train / test positive edges + corresponding negative samples."""
    import scipy.sparse as sp
    device = adj_values.device
    edges_single = torch.as_tensor(sparse2tuple(sp.triu(adj_values.cpu()))[0], dtype=torch.long, device=device)
    # Shuffle
    if test_ratio > 1:
        test_ratio = test_ratio / edges_single.shape[0]
    num_test = int(np.floor(edges_single.shape[0] * test_ratio))
    perm = torch.randperm(edges_single.shape[0], device=device)
    test_idx = perm[:num_test]
    test_edges = edges_single[test_idx].T
    train_edges = torch_delete(edges_single, test_idx, dim=0).T
    train_false, test_false = edge_sampling(edges_single, train_edges, test_edges)
    return edges_single, train_edges, test_edges, train_false, test_false




def SGEDataset(sample_ids, raw_dir, test_ratio=0.1):
    """Load spatial gene expression dataset and build PyG Data objects."""
    data_list, adj_list = [], []
    for sid in sample_ids:
        print(f"Loading sample {sid} ")
        nodes = pd.read_csv(f"{raw_dir}/{sid}_hvg_counts.txt", sep=' ', header=0,index_col=0)
        x = torch.tensor(nodes.values, dtype=torch.float).T  # [N, F]
        node_label = pd.read_csv(f"{raw_dir}/{sid}_truth.txt",  sep=' ', header=0,index_col=0)
        y = torch.tensor(node_label['seurat_clusters'].values, dtype=torch.float32)
        valid = ~torch.isnan(y)
        x, y = x[valid], y[valid].long()
        pos_df = pd.read_csv(f"{raw_dir}/{sid}_position.txt", sep=" ", header=0,index_col=0)
        coord = torch.tensor(pos_df.iloc[:, [0, 1]].values, dtype=torch.float32)
        dst = torch.tensor(pairwise_distances(coord, metric='euclidean'))
        cut = torch.sort(dst[:, 100])[0][6] + 5
        adj = (dst < cut).int()
        pos_edge, tr_edges, te_edges, tr_false, te_false = train_test_split(adj, test_ratio)
        data = Data(x=x, y=y, train_edge_index=tr_edges, test_edge_index=te_edges,
                    pos_edge=pos_edge, neg_edge=te_false,
                    train_edge_weight=torch.ones(tr_edges.size(1)),
                    test_edge_weight=torch.ones(te_edges.size(1)),
                    batch=torch.zeros(x.size(0), dtype=torch.long))
        data_list.append(data)
        adj_list.append(adj)
    return data_list, adj_list


def SGEDataset1(id: str, raw_dir: str):
    data_list = []
    for i in id:
        print(f'sample ID: {i}')
        nodes = pandas.read_csv(f'{raw_dir}/{i}_hvg_data.txt', sep=' ', header=0,index_col=0)
        x = torch.tensor(nodes.values, dtype=torch.float).transpose(0, 1)
        node_id = nodes.columns.values
        node_label = pandas.read_csv(f'{raw_dir}/{i}_meta.txt', sep=' ', header=0,index_col=0)
        y = torch.tensor(node_label['integrated_snn_res.0.4'], dtype=torch.float32)
        j = torch.where(~torch.isnan(y))
        y = y[j].long()
        g = Data(x=x[j],y=y)
        data_list.append(g)
        # adj_list.append(adj)
    return data_list
        


def load_ppi_connectivity(ppi_path: str, num_nodes: int) -> torch.Tensor:
    """
    读取 PPI.connect1.txt：必须至少包含两�?0-based 基因索引�?    会额外追加自环，确保卷积层可以取到自身特征�?    返回 edge_index: shape (2, E)
    """
    df = pd.read_csv(ppi_path, sep=' ', header=None)      
    edge = df.iloc[:, 1:3].astype(int).values
    edge_index = torch.tensor(edge.T, dtype=torch.long)
    self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    return torch.cat([edge_index, self_loops], dim=1)


def data_counts(sample_ids, raw_dir, test_ratio=0.1):
    """Load spatial gene expression dataset and build PyG Data objects."""
    data_list, adj_list = [], []
    for sid in sample_ids:
        print(f"Loading sample {sid} ")
        nodes = pd.read_csv(f"{raw_dir}/{sid}_sp_counts.txt", sep=" ", index_col=0)
        x = torch.tensor(nodes.values, dtype=torch.float).T  # [N, F]
        node_label = pd.read_csv(f"{raw_dir}/{sid}_meta_sp.txt", sep=" ", index_col=0)
        y = torch.tensor(node_label['integrated_snn_res.0.4'].values, dtype=torch.float32)
        valid = ~torch.isnan(y)
        x, y = x[valid], y[valid].long()
        pos_df = pd.read_csv(f"{raw_dir}/{sid}_position_sp.txt", sep=" ", index_col=0)
        coord = torch.tensor(pos_df.iloc[:, [0, 1]].values, dtype=torch.float32)
        dst = torch.tensor(pairwise_distances(coord, metric='euclidean'))
        cut = torch.sort(dst[:, 100])[0][8]
        adj = (dst < cut).int()
        pos_edge, tr_edges, te_edges, tr_false, te_false = train_test_split(adj, test_ratio)
        data = Data(x=x, y=y, train_edge_index=tr_edges, test_edge_index=te_edges,
                    pos_edge=pos_edge, neg_edge=te_false,
                    train_edge_weight=torch.ones(tr_edges.size(1)),
                    test_edge_weight=torch.ones(te_edges.size(1)),
                    batch=torch.zeros(x.size(0), dtype=torch.long))
        data_list.append(data)
        adj_list.append(adj)
    return data_list, adj_list

def data_counts1(sample_ids, raw_dir):
    """Simple dataset without edges (expression + labels)."""
    data_list = []
    for sid in sample_ids:
        nodes = pd.read_csv(f"{raw_dir}/{sid}_sc_counts.txt", sep=" ", index_col=0)
        x = torch.tensor(nodes.values, dtype=torch.float).T
        node_label = pd.read_csv(f"{raw_dir}/{sid}_meta.txt", sep=" ", index_col=0)
        y = torch.tensor(node_label['integrated_snn_res.0.4'].values, dtype=torch.float32)
        valid = ~torch.isnan(y)
        data_list.append(Data(x=x[valid], y=y[valid].long()))
    return data_list

def create_mask(num_nodes: int, train_ratio: float = 0.5):
    return torch.rand(num_nodes) < train_ratio


def cal_auc_ap_acc(sampling_n, edge_index,z,adj,neg_edge_index):
    auroc = []
    ap_score = []
    acc_score = []
    preds = []
    pos = []
    preds = []
    for src, dst in zip(edge_index[0], edge_index[1]):
        preds.append(adj[src, dst].item())
    preds = torch.tensor(preds)
    if neg_edge_index is None:
        for i in range(sampling_n):
            neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=z.size(0))
            neg_edge_index, _ = remove_self_loops(neg_edge_index)
    preds_neg = []
    for src1, dst1 in zip(neg_edge_index[0], neg_edge_index[1]):
        preds_neg.append(adj[src1, dst1].item())
    preds_neg = torch.tensor(preds_neg)
    preds_all = torch.cat([preds, preds_neg],dim=0)
    labels_all = torch.cat([torch.ones(len(preds)), torch.zeros(len(preds_neg))])
    # # labels_all = torch.cat([torch.ones(len(preds_neg)), torch.zeros(len(preds))])
    labels_all = labels_all.detach().cpu()
    preds_all = preds_all.detach().cpu()
    roc_score = roc_auc_score(labels_all, preds_all)
    auroc.append(float(roc_score))
    ap = average_precision_score(labels_all, preds_all)
    ap_score.append(ap)
    acc = accuracy_score(labels_all, np.round(preds_all))
    acc_score.append(acc)
    return auroc,ap_score,acc_score

# ---- Block matrix utilities ----

def compute_neighborhood_enrichment(adj, labels, n_perms=1000, seed=42, device="cuda"):
    """
    计算空间邻域富集度 (Neighborhood Enrichment) 的高性能版本。
    支持 GPU 加速和置换检验。
    """
    if isinstance(adj, torch.Tensor):
        W = adj.detach().to(device, dtype=torch.float32)
    else:
        W = torch.tensor(adj, device=device, dtype=torch.float32)
    
    W = (W + W.T) / 2
    W.fill_diagonal_(0)
    
    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy().flatten()
    else:
        y = np.array(labels).flatten()

    P_df = pd.get_dummies(y)
    cluster_names = P_df.columns
    P = torch.tensor(P_df.values, device=device, dtype=torch.float32) # (N, C)
    
    N, C = P.shape
    observed = P.T @ (W @ P) 

    if n_perms == 0:
        k = observed.sum(dim=1)
        total = k.sum()
        expected = torch.outer(k, k) / (total + 1e-12)
        z_score = (observed - expected) / (torch.sqrt(expected) + 1e-12)
        return pd.DataFrame(z_score.cpu().numpy(), index=cluster_names, columns=cluster_names)

    torch.manual_seed(seed)
    perm_vals = torch.empty((n_perms, C, C), device=device)
    
    for i in range(n_perms):
        idx = torch.randperm(N, device=device)
        P_shuffled = P[idx, :]
        perm_vals[i] = P_shuffled.T @ (W @ P_shuffled)
        
    mean_perm = perm_vals.mean(dim=0)
    std_perm = perm_vals.std(dim=0)
    z_scores = (observed - mean_perm) / (std_perm + 1e-12)
    
    return pd.DataFrame(z_scores.cpu().numpy(), index=cluster_names, columns=cluster_names)



def compute_block_matrix(adj, labels, threshold=0.6, is_prediction=True):
    """Compute normalized block‐connectivity matrix for clusters."""
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    sign_adj = (adj > threshold).astype(int) if is_prediction else np.sign(adj)
    P = pd.get_dummies(labels)
    n = P.values.sum(axis=0)
    P_norm2 = np.outer(n, n) + 1e-10
    A = (P.values.T @ sign_adj @ P.values) / P_norm2
    diag = np.diag(A) + 1e-10
    result = A / diag[:, None]
    clusters = P.columns
    return pd.DataFrame(result, index=clusters, columns=clusters)


def compare_block_matrices(m1: pd.DataFrame, m2: pd.DataFrame):
    common = sorted(set(m1.columns) & set(m2.columns))
    m1_common = m1.loc[common, common]
    m2_common = m2.loc[common, common]
    
    # 对称化处理：取上三角与下三角的平均   
    m1_sym = (m1_common + m1_common.T) / 2
    m2_sym = (m2_common + m2_common.T) / 2
    
    # 提取上三角（排除对角线）
    idx = np.triu_indices(len(common), k=1)
    from scipy.stats import pearsonr
    corr, pval = pearsonr(m1_sym.values[idx], m2_sym.values[idx])
    return corr, pval



