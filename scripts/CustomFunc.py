from csv import QUOTE_MINIMAL
from token import DOUBLESLASHEQUAL
from tokenize import Double
from typing import List, Optional, Tuple, Union
import pandas
#import faiss
import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch import Tensor, cdist
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_batch, to_dense_adj, dense_to_sparse, to_undirected
from torch.linalg import matrix_power, cholesky
from scripts.sqrtm import sqrtm

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    # fact = 1.0 / (m.size(1) - 1) # if the covariance matrix shoud be scaled
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return m.matmul(mt).squeeze()

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask
    row, col = edge_index
    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p
    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]
    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()
    return edge_index, edge_mask

def matrix_regulize(matrix: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eig(matrix)
    vals = torch.view_as_real(vals)[:, 0]
    i = vals < 0. 
    vals[i] = 1E-5 
    vals = torch.stack((vals, torch.zeros(len(vals)))).t()
    vals = torch.view_as_complex(vals.contiguous())
    new_matrix = torch.matmul(vecs, torch.matmul(torch.diag(vals), torch.inverse(vecs)))
    return torch.view_as_real(new_matrix)[:, :, 0]    

def onehot_to_label(assign_mat: torch.Tensor) -> torch.Tensor:
    cl, n = assign_mat.size()
    out = torch.zeros(n, dtype=torch.int64)
    _, out = torch.max(assign_mat, dim=0)
    return(pandas.DataFrame(out))

def Dykstra_projection(P_: torch.Tensor, tau: torch.float32, max_iter: int=1000, k_max: int=1000) -> torch.Tensor:
    v1, v2 = P_.size()
    P0 = torch.exp(P_/tau)
    Q_ = torch.ones(v1, v2, dtype=torch.float32)
    Q0 = Q_
    Ksi = (P0 * Q_)
    L = Ksi.sum(1)
    L[L > k_max] = k_max
    L[L < 1.] = 1.
    P = torch.diag(L / Ksi.sum(1)).mm(Ksi)
    Q = (Q_ * P0) / P
    for i in range(1, max_iter + 1):
        if i == 1:
            Ksi = (P * Q0)
            L = Ksi.sum(1)
            L[L > k_max] = k_max
            L[L < 1.] = 1.
            P_=P
            P = torch.diag(L / Ksi.sum(1)).mm(Ksi)
            Q_=Q
            Q = (Q0 * P_) / P
        elif i - i // 2 == 0:
            Ksi = (P * Q_)
            L = Ksi.sum(0)
            P_=P
            P = Ksi.mm(torch.diag(1. / Ksi.sum(0)))
            Q_=Q
            Q = (Q_ * P_) / P
        else:
            Ksi = (P * Q_)
            L = Ksi.sum(1)
            L[L > k_max] = k_max
            L[L < 1.] = 1.
            P_= P
            P = torch.diag(L / Ksi.sum(1)).mm(Ksi)
            Q_=Q
            Q = (Q_ * P_) / P
    return P / P.sum(0)

def graph_cov(D: torch.Tensor, knn: int, normalization: str=None, Weights: torch.Tensor=None) -> torch.Tensor:
    N = D.size(0)
    A = torch.zeros(N, N, dtype=torch.float32) 
    for i in range(N):
        _, k = torch.sort(D[i, :], descending=False)
        j = k[:knn]
        A[i, j] = 1.0
        A[j, i] = 1.0
    if Weights is None:
        W = A
    else:
        W = A * Weights
    Deg = torch.diag(W.sum(dim=0))
    L = Deg - W
    return A, torch.linalg.inv(L + 1E-5 * torch.eye(N))

def FrobeniusNorm(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Return the Forbenius norm of the input matrix A (a_ij)
    ||A||Fr = (\Sigma\Sigma a_ij^2)^1/2
    """
    return torch.sqrt(torch.sum(matrix**2))

def NNeighbors(x: torch.Tensor, batch: torch.Tensor, min_dist: torch.float32=None) -> torch.Tensor: 
    X, _ = to_dense_batch(x, batch)
    d = cdist(X, X)
    if min_dist is not None:
        adj = (d <= min_dist).float()
    edge_index, _ = dense_to_sparse(adj)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index

def SGEGrid(n: torch.Tensor) -> torch.Tensor:
    m = torch.sqrt(n).int() + 1
    x = torch.arange(m) / m
    ans = torch.zeros(n, 2)
    c = 0
    for i in range(m):
        for j in range(m):
            if c>=n: 
                break
            ans[c, 0] = x[j]
            ans[c, 1] = x[i]
            c += 1
    return ans

def faiss_graph(x: torch.Tensor, k: int, batch: torch.Tensor = None, loop: bool = False,
              flow: str = 'source_to_target', force_undirected: bool = True) -> torch.Tensor:
    if batch is not None:
        batch_size = int(batch.max()) + 1
        N_i = scatter_add(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    else:
        N_i = x.size(0)
    edge_index = torch.zeros(2, 0, dtype=torch.int64)
    x, _ = to_dense_batch(x, batch)
    x_np = x.cpu().detach().numpy() 
    for b in range(batch_size):
        x_i = x_np[b][range(N_i[b]), :]
        d = x_i.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(x_i)
        if loop:
            _, nn = index.search(x_i, k)
        else:
            _, nn = index.search(x_i, k + 1)
            nn = nn[:, range(1, k + 1)]
        src = torch.tensor(range(N_i[b])).repeat_interleave(k)
        tgt = torch.tensor(nn, dtype=torch.int64).reshape(-1)
        if b > 0:
            src += N_i[b - 1]
            tgt += N_i[b - 1]    
        edge_index = torch.cat((edge_index, torch.stack((src, tgt))), dim=1) 
    if force_undirected:
        edge_index = to_undirected(edge_index)
    return edge_index

def scale(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.mean(dim)
    sd = x.std(dim, unbiased=False)
    x -= m
    return x
