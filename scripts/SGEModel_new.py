from math import e
import numpy
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Dropout, ReLU, LeakyReLU, Softmax, Parameter, MultiheadAttention
from sparselinear import SparseLinear
from torch_geometric.nn import MessagePassing, knn_graph, DMoNPooling, GINConv, EdgePooling, GraphNorm, MessageNorm
from torch_geometric.utils import erdos_renyi_graph, unbatch, to_dense_adj, to_dense_batch, dense_to_sparse, remove_self_loops, add_remaining_self_loops
from scripts.sqrtm import sqrtm
from scripts.CustomFunc import scale, dropout_edge
from scripts.SGWTConv import SGWTConv


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, inputs):
        return self.module(inputs) + inputs
    
class MHA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.mha = MultiheadAttention(out_channels, n_heads) 
        self.qkv = Sequential(
                            Linear(in_channels, hidden_channels, bias=False),
                            ReLU(),
                            Linear(hidden_channels, out_channels * 3, bias=False),
                            )   
    def forward(self, inputs):
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)
        return self.mha(q, k, v)[0]  

'''
class ResNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.module = Sequential(
                    Linear(in_channels, hidden_channels),
                    LeakyReLU(0.1),
                    Linear(hidden_channels, in_channels)
                ) 
    def forward(self, inputs):
        return self.module(inputs) + inputs
'''

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
        out = F.relu(self.lin1(out)) + x_i
        out = self.lin2(out)
        n_h = edge_weight.size(1)   
        s_h = self.out_dim // n_h   
        out = torch.matmul(out.view(-1, s_h, n_h), edge_weight) 
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

"""
'''This is the original SGENet3'''

class SGENet3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, m, n, connect, k1=16, k2=6):
        super().__init__()
        torch.manual_seed(12345)
        self.pool1 = DMoNPooling([in_channels, hidden_channels], m)
        self.conv1 = PPIEdgeConv(in_channels, hidden_channels, connect, pi=None, k=k1)  
        self.conv2 = DynamicEdgeConv(n, n, k=k2, dropout=0.25)  
        self.norm2 = GraphNorm(n)
        self.conv3 = DynamicEdgeConv(n, out_channels, k=k2, dropout=0.25)  
        #self.norm3 = GraphNorm(out_channels)
        self.mlp = Sequential(
                        ResNet(hidden_channels, out_channels),
                        ResNet(hidden_channels, out_channels),
                        #Dropout(p=0.2),
                        Linear(hidden_channels, out_channels),
                        #Dropout(p=0.2),
                        Linear(out_channels, n),
                        Softmax(dim=1)
                    ) 
    def forward(self, x, edge_index, batch):
        adj = to_dense_adj(edge_index, batch)
        x, _ = to_dense_batch(x, batch)
        S, x, adj, _, _, _ = self.pool1(x, adj)
        x = x.transpose(1, 2) / torch.sum(S, dim=(1), keepdim=True)
        S = S.view(-1, S.size(2)).t()
        batch = torch.tensor(range(x.size(0)))
        batch = batch.repeat_interleave(x.size(2))
        x = x.view(-1, x.size(2)).t()
        edge_index, _ = dense_to_sparse(adj)   
        x, _ = self.conv1(x, edge_index, batch) 
        x = self.mlp(x)         
        y, _ = self.conv2(x, edge_index, batch)
        y = self.norm2(y)
        y, _ = self.conv3(y, edge_index, batch)
        return x, y, S
"""

class SGENet3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, m, n, connect, pi=0.5, k1=6, k2=6, n_heads=3):
        super().__init__()
        torch.manual_seed(12345)
        self.pool1 = DMoNPooling([in_channels, hidden_channels], m)
        self.conv1 = PPIEdgeConv(in_channels, hidden_channels * 2, connect, pi=pi, k=None, dropout=0.2, n_heads=n_heads)  
        self.conv2 = DynamicEdgeConv(hidden_channels * 2, hidden_channels, k=k1, dropout=0.25)  
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = DynamicEdgeConv(hidden_channels, out_channels, k=k2, dropout=0.25)  
        self.norm3 = GraphNorm(out_channels)
        self.conv4 = GINConv(Sequential(
                                Linear(out_channels, hidden_channels),
                                Linear(hidden_channels, n)
                            )) 
    def forward(self, x, edge_index, batch):
        adj = to_dense_adj(edge_index, batch)
        x0, _ = to_dense_batch(x, batch)
        S, x0, adj, _, _, _ = self.pool1(x0, adj)
        x0 = x0.transpose(1, 2) / torch.sum(S, dim=1, keepdim=True)
        S = S.view(-1, S.size(2)).t()
        batch = torch.tensor(range(x0.size(0)))
        batch = batch.repeat_interleave(x0.size(2))
        x0 = x0.view(-1, x0.size(2)).t()
        edge_index, _ = dense_to_sparse(adj)  
        y = x0 
        y = self.conv1(y, _, batch)    
        y, _ = self.conv2(y, _, batch)
        y = self.norm2(y)
        y, edge_index = self.conv3(y, _, batch)
        y = self.norm3(y)    
        z = y        
        z = self.conv4(z, edge_index)
        z = z.softmax(dim=1)
        return z, y, S

class SGENet4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 J, K,
                 m, n, connect, 
                 pi=0.5, k1=6, k2=6, n_heads=3):
        super().__init__()
        torch.manual_seed(12345)
        self.pool1 = DMoNPooling([in_channels, hidden_channels], m)
        self.conv0 = SGWTConv(in_channels, J, K)        
        self.conv1 = PPIEdgeConv(in_channels, hidden_channels, connect, pi=pi, k=None, dropout=0.2, n_heads=n_heads)  
        self.conv2 = DynamicEdgeConv(hidden_channels, hidden_channels, k=k1, dropout=0.25)  
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = DynamicEdgeConv(hidden_channels, out_channels, k=k2, dropout=0.25)  
        self.norm3 = GraphNorm(out_channels)
        self.conv4 = GINConv(Sequential(
                                Linear(out_channels, hidden_channels),
                                Linear(hidden_channels, n)
                            )) 
    def forward(self, x, edge_index, batch):
        adj = to_dense_adj(edge_index, batch)
        x, _ = to_dense_batch(x, batch)
        S, x, adj, _, _, _ = self.pool1(x, adj)
        x = x.transpose(1, 2) / torch.sum(S, dim=1, keepdim=True)
        S = S.view(-1, S.size(2)).t()
        batch = torch.tensor(range(x.size(0)))
        batch = batch.repeat_interleave(x.size(2))
        x = x.view(-1, x.size(2)).t()
        edge_index, _ = dense_to_sparse(adj)   
        x = self.conv0(x, edge_index)
        x = x.relu()            
        x = self.conv1(x, _, batch)  
        x, _ = self.conv2(x, _, batch)
        x = self.norm2(x)
        x, edge_index = self.conv3(x, _, batch)
        x = self.norm3(x)    
        y = x        
        y = self.conv4(y, edge_index)
        y = y.softmax(dim=1)
        return y, x, S

class GraphWasserstein1Loss_V(torch.nn.Module):
    def __init__(self, weight=None, size_average=False, normalize=False):
        super(GraphWasserstein1Loss_V, self).__init__() 
        self.normalization = normalize
    def forward(self, inputs, targets, pmat):
        inputs = inputs.detach()
        Z = scale(inputs, dim=0)
        A = torch.matmul(Z, Z.t())
        A = A.relu()
        D = torch.diag(A.sum(dim=1))      
        if self.normalization:
            D_ = D ** -.5
            L = torch.eye(inputs.size(0)) - D_.mm(A).mm(D_)
        else:            
            L = D - A
        k = pmat.size(0)
        L_adj = pmat.mm(L).mm(pmat.t()) + 1E-5 * torch.eye(k)
        numpy.savetxt('A.txt', A.detach().numpy(), fmt='%1.3f')
        numpy.savetxt('L_adj.txt', L_adj.detach().numpy(), fmt='%1.3f')       
        # S1 = torch.linalg.inv(L_adj)  
        S1 = torch.linalg.inv(torch.eye(pmat.size(0))  + .5 * L_adj) 
        # S1 = torch.linalg.matrix_exp(-0.05 / 2 * L_adj)
        S0 = targets.to(dtype=torch.float32)
        S0_sqrt = sqrtm(S0)
        S_ = sqrtm(S0_sqrt.mm(S1).mm(S0_sqrt))
        return S1.trace() + S0.trace() - 2.0 * S_.trace()
    
# class GraphWasserstein1Loss_V(torch.nn.Module):
#     def __init__(self, weight=None, size_average=False, normalize=False):
#         super(GraphWasserstein1Loss_V, self).__init__() 
#         self.normalization = normalize
#     def forward(self, inputs, targets, pmat):
#         Z = scale(inputs, dim=0)
#         A = torch.matmul(Z, Z.t())
#         A = A.relu()
#         D = torch.diag(A.sum(dim=1))      
#         if self.normalization:
#             D_ = D ** -.5
#             L = torch.eye(inputs.size(0)) - D_.mm(A).mm(D_)
#         else:            
#             L = D - A
#         k = pmat.size(0)
#         L_adj = pmat.mm(L).mm(pmat.t()) + 1E-5 * torch.eye(k)
#         numpy.savetxt('A.txt', A.detach().numpy(), fmt='%1.3f')
#         numpy.savetxt('L_adj.txt', L_adj.detach().numpy(), fmt='%1.3f')
#         S1 = torch.linalg.inv(L_adj)  
#         S0 = targets.to(dtype=torch.float32)
#         S0_sqrt = sqrtm(S0)
#         S_ = sqrtm(S0_sqrt.mm(S1).mm(S0_sqrt))
#         return S1.trace() + S0.trace() - 2.0 * S_.trace()

class GraphWasserstein1Loss_E(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GraphWasserstein1Loss_E, self).__init__() 
        self.normalization = size_average
    def forward(self, inputs, targets, edge_index, pmat):
        V = inputs.size(0)
        edge_full, _ = add_remaining_self_loops(edge_index)
        E = edge_full.size(1)
        W = torch.eye(V)
        for k in range(E):
            i, j = edge_full[:, k]
            d = (inputs[i] - inputs[j]).pow(2).sum()
            w = torch.exp(-1.0 * d)
            W[i, j] = w
            W[j, i] = w
        D = torch.diag(W.sum(dim=1))
        L = D - W
        k = pmat.size(0)
        L_adj = pmat.mm(L).mm(pmat.t()) + 1E-5 * torch.eye(k)
        S1 = torch.linalg.inv(L_adj)  
        S0 = targets.to(dtype=torch.float32)
        S0_sqrt = sqrtm(S0)
        S_ = sqrtm(S0_sqrt.mm(S1).mm(S0_sqrt))
        return S1.trace() + S0.trace() - 2.0 * S_.trace()

class GraphKLLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, max_dist=0.5):
        super(GraphKLLoss, self).__init__()
        self.d_max = max_dist 
    def forward(self, inputs, targets, edge_index, pmat):         
        V = inputs.size(0)
        E = edge_index.size(1)
        W = to_dense_adj(edge_index)[0] + torch.eye(V)
        for k in range(E):
            i, j = edge_index[:, k]
            d = (inputs[i] - inputs[j]).pow(2).sum()
            if d <= self.d_max:
                w = torch.exp(-1.0 * d)
                W[i, j] = w
                W[j, i] = w
            else:
                W[i, j] = 1E-8
                W[j, i] = 1E-8
        D = torch.diag(W.sum(dim=1))
        L = D - W
        k = pmat.size(0)
        L_adj = pmat.mm(L).mm(pmat.t()) + 1E-5 * torch.eye(k)    
        #torch.save(L_adj, './L_adj.pt') 
        S0 = targets.to(dtype=torch.float32)
        S1 = torch.linalg.inv(L_adj)      
        S_ = L_adj.mm(S0)
        return torch.det(S1).log() - torch.det(S0).log() - k + S_.trace()

class OrthogonalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(OrthogonalLoss, self).__init__()
    def forward(self, inputs, targets):
        S1 = inputs.mm(inputs.t()).type_as(targets)
        return torch.norm(S1 / torch.norm(S1) - targets / torch.norm(targets))         

class myKLLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(myKLLoss, self).__init__() 
    def forward(self, inputs, targets, K_sample):         
        n = inputs.size()[1]
        k, _ = targets.unique(dim=0, return_counts=True)   
        mu = torch.zeros(len(k), n)
        Sc = torch.zeros(len(k),n, n)
        for i in k:
            X = inputs[targets == i]
            mu[i] = X.mean(0)
            Sc[i] = cov(X)
        D = torch.zeros(len(k), len(k), dtype=float)
        for i in k:
            for j in k:
                S1 = Sc[i]
                S2 = Sc[j]
                S_ = (S1 + S2) / 2
                #Bhattacharyya distance
                D[i, j] = (mu[i] - mu[j]).view(1,n).mm(torch.inverse(S_)).mm((mu[i] - mu[j]).view(n,1)) /8 + .5 * torch.log(torch.det(S_) / torch.sqrt(torch.det(S1) * torch.det(S2)))
        K = torch.exp(-D.t() * D / 0.01) #Gaussian kernel 
        #K = 1 - D.t() * D #Epanechnikov kernel
        #K = torch.sqrt(D.t() * D / 0.01 + 0.0)
        K_ = K_sample[0, 0] * (0.5 * K + 0.5 * torch.diag(torch.ones(len(k))))
        #print(torch.det(K))
        p = MultivariateNormal(torch.zeros(len(k), dtype=float), K_sample.clone().detach())
        q = MultivariateNormal(torch.zeros(len(k), dtype=float), K_)
        return kl_divergence(q, p)

