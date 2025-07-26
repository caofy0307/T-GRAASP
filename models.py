# Python built-in
from math import e
from typing import Optional
# Third-party
import numpy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, LeakyReLU, SELU, Softmax, Parameter, Module, CrossEntropyLoss
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

# PyTorch Geometric
from torch_geometric.nn import (
    MessagePassing,
    knn_graph,
    DMoNPooling,
    GINConv,
    EdgePooling,
    GraphNorm,
    MessageNorm,
    GAE,
)
from torch_geometric.utils import (
    erdos_renyi_graph,
    unbatch,
    to_dense_adj,
    to_dense_batch,
    dense_to_sparse,
    remove_self_loops,
    add_remaining_self_loops,
    negative_sampling,
)
from torch_geometric.nn.dense.mincut_pool import _rank3_trace
from torch_geometric.nn.inits import reset

# Custom modules
from scripts.SGWTConv import SGWTConv
from scripts.CustomFunc import scale, dropout_edge
from scripts.sqrtm import sqrtm
from scripts.CustomFunc import onehot_to_label, sqrtm
from utils import *
# External local module
from sparselinear import SparseLinear
EPS = 1e-15
MAX_LOGSTD = 10
MAX = 10.
MIN = -10.
class FixedSparseEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, connect):
        super().__init__(aggr='max') #  "Maximum" aggregation.
        self.out_dim = out_channels
        self.lin1 = SparseLinear(2 * in_channels, in_channels, connectivity=connect)
        self.lin2 = Linear(in_channels, out_channels)
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    def message(self, x_i, x_j, edge_weight):
        out = torch.cat([x_i, x_j], dim=1)
        # print(out.shape)
        out = F.relu(self.lin1(out)) + x_i
        out = self.lin2(out)
        # print(out)
        # print(out.shape)
        n_h = edge_weight.size(1)
        s_h = self.out_dim // n_h
        out = torch.matmul(out.view(-1, s_h, n_h), edge_weight)
        # print(out.shape)
        # print(out.view(-1, n_h * s_h).shape)
        return out.view(-1, n_h * s_h)



class FullEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') #  "Mean" aggregation.
        self.out_dim = out_channels
        self.lin1 = Linear(2 * in_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    def message(self, x_i, x_j, edge_weight):
        out = torch.cat([x_i, x_j], dim=1)
        out = F.relu(self.lin1(out)) + x_i
        out = self.lin2(out)
        n_h = edge_weight.size(1)
        s_h = self.out_dim // n_h
        out = torch.matmul(out.view(-1, s_h, n_h), edge_weight)
        return out.view(-1, n_h * s_h)





class PPIEdgeConv(FixedSparseEdgeConv):
    def __init__(self, in_channels, out_channels, connect, pi=None, dropout=None, k=None, n_heads=2):
        super().__init__(in_channels, out_channels, connect)
        self.pi = pi
        self.dropout = dropout
        self.k = k
        self.n_heads = n_heads
        self.head_size = out_channels // n_heads
        self.d_k = out_channels
        self.qnet = Sequential(
                        SparseLinear(2 * in_channels, in_channels, connectivity=connect),
                        Linear(in_channels, out_channels),
                    )
        self.knet = Sequential(
                        SparseLinear(2 * in_channels, in_channels, connectivity=connect),
                        Linear(in_channels, out_channels),
                    )
    def reshape(self, e):
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e
    def forward(self, x, edge_index, batch=None):
        if self.k is None:
            n = x.size(0)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            for v in unbatch(torch.arange(n, device=x.device), batch):
                e = erdos_renyi_graph(n, self.pi, directed=False)
                e = e.to(x.device)
                edge_index = torch.cat([edge_index, v[e]], dim=1)
        if self.pi is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = dropout_edge(edge_index, self.dropout, force_undirected=True)
        x_src, x_dst = x[edge_index]
        Q = self.qnet(torch.cat([x_src, x_dst], dim=1))
        K = self.knet(torch.cat([x_src, x_dst], dim=1))
        Q = self.reshape(Q)
        K = self.reshape(K)
        edge_weight = torch.matmul(Q, K.permute(0, 2, 1)) / torch.tensor(self.d_k).sqrt()
        return super().forward(x, edge_index, edge_weight)



class DynamicEdgeConv(FullEdgeConv):
    def __init__(self, in_channels, out_channels, k=4, dropout=0.0, n_heads=2):
        super().__init__(in_channels, out_channels)
        self.k = k
        self.dropout = dropout
        self.n_heads = n_heads
        self.head_size = out_channels // n_heads
        self.d_k = out_channels
        self.qnet = Linear(2 * in_channels, out_channels)
        self.knet = Linear(2 * in_channels, out_channels)
    def reshape(self, e):
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e
    def forward(self, x, edge_index, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, self.dropout, force_undirected=True)
        x_src, x_dst = x[edge_index]
        Q = self.qnet(torch.cat([x_src, x_dst], dim=1))
        K = self.knet(torch.cat([x_src, x_dst], dim=1))
        Q = self.reshape(Q)
        K = self.reshape(K)
        edge_weight = torch.matmul(Q, K.permute(0, 2, 1)) / torch.tensor(self.d_k).sqrt()
        return super().forward(x, edge_index, edge_weight)



class STx_encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, m, l, connect, pi, n_heads, K, 
                 activate_sc_alignment: bool = False):
        super().__init__()
        torch.manual_seed(12345)
        self.n_heads = n_heads
        self.conv0 = SGWTConv(in_channels, hidden_channels, K,normalization="rw")
        self.norm1 = GraphNorm(hidden_channels)
        self.conv1 = SGWTConv(in_channels + hidden_channels, hidden_channels, K,normalization="rw")
        self.fc1 = Sequential(
                            Linear(hidden_channels, m * 16),
                            SELU(),
                            Linear(m * 16, m * 4),
                            SELU(),
                            Linear(m * 4, m)
                            )
        
        self.activate_sc_alignment = activate_sc_alignment
        if self.activate_sc_alignment:
            self.alignment_model = Alignment_Model(in_channels, in_channels)
        else:
            self.alignment_model = None

        self.conv2 = PPIEdgeConv(in_channels, hidden_channels, connect, pi=pi, k=None, dropout=0.0, n_heads=self.n_heads)
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels, k=3, dropout=0.1, n_heads=self.n_heads)
        self.fc2 = Sequential(
                            Linear(hidden_channels, hidden_channels // 2),
                            SELU(),
                            Linear(hidden_channels // 2, hidden_channels // 4),
                            SELU(),
                            Linear(hidden_channels // 4, out_channels)
                            )
        self.fc3 = Sequential(
                            Linear(hidden_channels, hidden_channels * 2),
                            SELU(),
                            Linear(hidden_channels * 2, hidden_channels * 4),
                            SELU(),
                            Linear(hidden_channels * 4, out_channels * l)
                            )
    def forward(self, x, edge_index, edge_weight, batch, mask: Optional[torch.Tensor] = None,
                  is_single_cell_input: bool = False):
        initial_x = x # Keep original x for reference if needed for S calculation base
        edge_index = knn_graph(initial_x, 3, batch, loop=False) if edge_index is None else edge_index
        u = self.conv0(initial_x, edge_index, edge_weight, batch)
        u = self.norm1(u, batch)
        u = self.fc1(u)
        S = u.softmax(dim=-1) # S shape: [batch_size, num_nodes_in_batch, m_clusters]
        
        S_for_aggregation = S.unsqueeze(0) if S.dim() == 2 else S # Ensure S is [batch_size, num_nodes, m]
        
        # x_features_to_pool shape: [batch_size, num_nodes_in_batch, D_in_features_model]
        x_features_to_pool, batch_info_dense = to_dense_batch(initial_x, batch) 
        batch_size = x_features_to_pool.size(0)
        num_original_nodes = x_features_to_pool.size(1) # Max nodes in batch
        model_in_channels = x_features_to_pool.size(2) # Should be self.conv0.in_channels

        if mask is not None:
            # Ensure mask is compatible with x_features_to_pool and S_for_aggregation
            mask_dense = mask.view(batch_size, num_original_nodes, 1).to(x_features_to_pool.dtype)
            x_features_to_pool = x_features_to_pool * mask_dense
            S_for_aggregation = S_for_aggregation * mask_dense # Mask S as well

        # x_aggregated_profiles shape: [batch_size, m_clusters, D_in_features_model]
        x_aggregated_profiles = torch.matmul(S_for_aggregation.transpose(1, 2), x_features_to_pool)
        
        # Normalize aggregated profiles
        # sum_S_per_cluster shape: [batch_size, m_clusters, 1]
        sum_S_per_cluster = torch.sum(S_for_aggregation, dim=1, keepdim=True).transpose(1, 2).clamp(min=EPS)
        x_agg_norm = x_aggregated_profiles / sum_S_per_cluster
        
        # Reshape for GCN input: [total_cluster_instances_in_batch, D_in_features_model]
        # D_in_features_model should be STx_encoder's in_channels.
        m_clusters = S_for_aggregation.size(2)
        mu_features_for_gcn = x_agg_norm.reshape(batch_size * m_clusters, model_in_channels)

        # Conditionally apply alignment model
        if self.alignment_model is not None and is_single_cell_input:
            # 对齐模型应用于单细胞数据
            aligned_mu_features = self.alignment_model(mu_features_for_gcn)
        else:
            aligned_mu_features = mu_features_for_gcn # Use original if not SC or no alignment model / alignment not active

        # Batch tensor for the GCN layers operating on cluster instances
        batch_for_gcn = torch.arange(batch_size, device=initial_x.device).repeat_interleave(m_clusters)
        
        x_conv2_out = self.conv2(aligned_mu_features, None, batch_for_gcn) # EdgeIndex set to None for KNN
        y_conv3_out = self.conv3(x_conv2_out, None, batch_for_gcn) # EdgeIndex set to None for KNN
        
        z_final_conv = x_conv2_out + y_conv3_out
        
        mu_to_return = aligned_mu_features # Shape [bs*m, D_aligned_features (which is in_channels)]

        return u, mu_to_return, self.fc2(z_final_conv), self.fc3(z_final_conv), edge_index, batch_for_gcn



class STx_discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(54321)
        self.conv = GINConv(Sequential(
                                        Linear(in_channels, hidden_channels),
                                        SELU(),
                                        Linear(hidden_channels, hidden_channels)
                            ))
        self.fc = Sequential(
                            Linear(hidden_channels, hidden_channels // 3),
                            SELU(),
                            Linear(hidden_channels // 3, out_channels)
                            )
    def forward(self, z, edge_index, batch):
        edge_index = knn_graph(z, 4, batch, loop=False) if edge_index is None else edge_index
        z = self.conv(z, edge_index)
        z = self.fc(z)
        return z





class STx_ARGA(GAE):
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Optional[Module] = None,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.criterion = CrossEntropyLoss()
    # def recon_loss(self, z: torch.Tensor,
    #             pos_edge_index: torch.Tensor,
    #             pos_edge_weight: torch.Tensor,
    #             # pos_edge_weight: Optional[torch.Tensor] = None,
    #             neg_edge_index: torch.Tensor) -> torch.Tensor:
    #             # neg_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
    #     if pos_edge_weight is None:
    #         pos_edge_weight = torch.ones_like(pos_pred, dtype=torch.float32)
    #     pos_loss = ((pos_edge_weight - pos_pred) ** 2).mean()
    #     if neg_edge_index is None:
    #         neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    #     neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
    #     neg_loss = (neg_pred ** 2).mean()
    #     return pos_loss + neg_loss
    # def recon_loss(self, z: torch.Tensor,
    #             pos_edge_index: torch.Tensor,
    #             neg_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     pos_loss = -torch.log( self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
    #     if neg_edge_index is None:
    #         neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    #         neg_edge_index, _ = remove_self_loops(neg_edge_index)
    #     neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
    #     return pos_loss + neg_loss
    def recon_loss(self, z: torch.Tensor,
                pos_edge:torch.Tensor,
                pos_edge_index: torch.Tensor,
                pos_edge_index1: torch.Tensor,
                neg_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos_loss = -torch.log( self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index,_ = edge_sampling(pos_edge, pos_edge_index,pos_edge_index1)
            # neg_edge_index, _ = remove_self_loops(neg_edge_index)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss
    def reset_parameters(self):
        super().reset_parameters()
        reset(self.discriminator)
    def reg_loss(self, z: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(self.discriminator(z, edge_index, batch))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss
    def discriminator_loss(self, z: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(self.discriminator(torch.randn_like(z), edge_index, batch))
        fake = torch.sigmoid(self.discriminator(z.detach(), edge_index, batch))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss
    




class STx_ARGVA(STx_ARGA):
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Optional[Module] = None,
        l: Optional[torch.Tensor] = 1,
    ):
        super().__init__(encoder, discriminator, decoder)
        self.l = l
    def reparameterize(self, s: torch.Tensor, mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(s, mu)
        if self.training:
            C = torch.Tensor(onehot_to_label(s.t().cpu()).values).int().view(-1)
            z = torch.cat([
                    torch.matmul(torch.randn(1, self.l, device=mu.device), A[C[i]].view(self.l, -1))
                        for i in range(s.size(0))
                    ]) + z
        return z
    def encode(self, s: torch.Tensor,
                    x: torch.Tensor,
                    edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
                    batch: torch.Tensor,
                    mask: torch.Tensor,
                    is_single_cell: bool = False) -> torch.Tensor:
        self.__u__, self.__m__, self.__v__, self.__A__, self.__e__, self.__b__ = self.encoder(x, edge_index, edge_weight, batch, mask, is_single_cell_input=is_single_cell)
        self.__A__ = self.__A__.clamp(min=MIN, max=MAX)
        self.__logstd__ =  torch.cat([
                        torch.matmul(self.__A__[i].view(self.l, -1).t(), self.__A__[i].view(self.l, -1)).diag().view(1, -1)
                            for i in range(self.__m__.size(0))
                        ]).log()
        S = self.__u__.softmax(dim=-1)
        self.__adj__ = to_dense_adj(self.__e__)
        self.__adj1__ = torch.matmul(torch.matmul(S.t(), self.__adj__), S)
        s = S if s is None else s
        z = self.reparameterize(s, self.__v__, self.__A__)
        return z, self.__b__
    
    def cluster_loss(self, u: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        u = self.__u__ if u is None else u
        k = u.size(1)
        s = u.softmax(dim=-1)
        degrees = torch.einsum('ijk->ik', self.__adj__).t()
        m = torch.einsum('ij->', degrees)
        ca = torch.matmul(s.t(), degrees)
        cb = torch.matmul(degrees.t(), s)
        normalizer = torch.matmul(ca, cb) / 2 / m
        decompose = self.__adj1__ - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = torch.mean(spectral_loss)
        # Orthogonal loss
        ss = torch.matmul(s.t(), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)
        # Cluster loss:
        cluster_loss = torch.norm(torch.einsum('ij->i', ss)) / self.__adj__.size(1) * torch.norm(i_s) - 1
        out = cluster_loss + ortho_loss + spectral_loss if y is None else cluster_loss + ortho_loss + spectral_loss + F.cross_entropy(u,y) #mse_loss(s, y)#F.cross_entropy(u, y)        \n",
        return out
    def kl_loss(
        self,
        mu: Optional[torch.Tensor] = None,
        logstd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = self.__v__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
class Alignment_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
            
        # 增加dropout和batch normalization
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # 修改这里，因为fc2的输出维度是hidden_dim
        
        # 使用更深的网络结�?        
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 残差连接
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # 应用batch normalization
        x = self.bn1(x)
        
        # 主路�?        
        h = self.fc1(x)
        h = self.bn2(h)
        h = self.fc2(h)
        h = self.bn3(h)
        h = self.fc3(h)
        
        # 残差连接
        res = self.residual(x)
        
        # 合并主路径和残差
        return h + res
        