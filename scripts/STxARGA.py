from math import e
from re import L
import numpy
import torch
import os.path as osp
from typing import List
#import torch.nn.functional as F
import pandas
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, SELU
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, ChebConv, GINConv, GraphNorm, GAE
from sklearn.metrics.cluster import adjusted_rand_score
from scripts.CustomFunc import onehot_to_label, sqrtm
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from scripts.SGEDataset import SGEDataset
from scripts.SGWTConv import SGWTConv
from scripts.SGEModel import ResNet, PPIEdgeConv, DynamicEdgeConv

EPS = 1e-15
sample_id = list(range(151507, 151511)) + list(range(151669, 151677))
SGE_train = SGEDataset(root='/home/ubuntu/sda1/qiyuan/fengyang/', name='SGE2', id=sample_id)
SGE_test = SGEDataset(root='/home/ubuntu/sda1/qiyuan/fengyang/', name='SGE2', id=sample_id)
train_loader = DataLoader(SGE_train[0, 2, 4, 6, 8, 10], batch_size=1, shuffle=True)
test_loader = DataLoader(SGE_train[1, 3, 5, 7, 9, 11], batch_size=1, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
Nv, N = SGE_train[0].x.size()    
ppi = pandas.read_csv('/home/ubuntu/sda1/qiyuan/fengyang/SGE2/PPI.connect.txt', sep=' ', header=0)
con = ppi.values[:, range(1,3)].transpose()
con = torch.tensor(con.astype(int), dtype=torch.long)
emd = pandas.read_csv('/home/ubuntu/sda1/qiyuan/fengyang/emd_8_samples.txt',sep = '\t', header=0)
emd = torch.tensor(emd.values)[:, [6, 0, 1, 2, 3, 4, 5]]
emd = emd[[6, 0, 1, 2, 3, 4, 5]]
#K_empiric = 1E4 * torch.exp(-.5 * (emd.t() * emd))
K_empiric = torch.exp(-5E2 * torch.log(emd / 4))
K_sparse = dense_to_sparse(K_empiric)

class STx_encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, m, connect, pi, n_heads, K):
        super().__init__()
        torch.manual_seed(12345)
        self.n_heads = n_heads
        self.conv0 = SGWTConv(in_channels, hidden_channels, K)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv1 = SGWTConv(hidden_channels, m * 4, K)
        self.fc1 = Sequential(
                            Linear(m * 4, m * 2),
                            SELU(),
                            Linear(m * 2, m),
                            SELU(),
                            Linear(m, m)
                            )
        self.conv2 = PPIEdgeConv(in_channels, hidden_channels, connect, pi=pi, k=None, dropout=0.0, n_heads=self.n_heads)   
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels, k=3, dropout=0.1, n_heads=self.n_heads)
        self.fc2 = Sequential(
                            Linear(hidden_channels, hidden_channels // 2),  
                            SELU(),
                            Linear(hidden_channels // 2, hidden_channels // 4),
                            SELU(),                 
                            Linear(hidden_channels // 4, out_channels)
                            )        
    def forward(self, x, edge_weight, batch):
        edge_index = knn_graph(x, 3, batch, loop=False)
        u = self.conv0(x, edge_index, edge_weight, batch)
        u = self.norm1(u, batch)
        edge_index = knn_graph(u, 3, batch, loop=False)
        u = self.conv1(u, edge_index, edge_weight, batch)
        u = self.fc1(u)
        S = u.softmax(dim=-1)
        S = S.unsqueeze(0) if S.dim() == 2 else S
        x, _ = to_dense_batch(x, batch)  
        x = torch.matmul(S.transpose(1, 2), x)
        x = x.transpose(1, 2) / torch.sum(S, dim=1, keepdim=True)
        batch = torch.tensor(range(S.size(0)))
        batch = batch.repeat_interleave(S.size(2))        
        x = x.view(-1, x.size(2)).t()
        S = S.view(-1, S.size(2))
        x = self.conv2(x, _, batch)
        y = self.conv3(x, _, batch)
        z = x + y
        v = self.fc2(z)
        return u, v

class STx_discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(54321)
        #self.conv = DynamicEdgeConv(in_channels, hidden_channels, k=4, dropout=0.1, n_heads=4) 
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
        if edge_index is None:
            edge_index =  knn_graph(z, 4, batch, loop=False)
        z = self.conv(z, edge_index)
        z = self.fc(z)
        return z
    
class STx_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj    

class STx_ARGA(GAE):
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Module,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.criterion = CrossEntropyLoss()
        reset(self.discriminator)
    def recon_loss(self, z: torch.Tensor, K: torch.Tensor, p = 1) -> torch.Tensor:
        _, k = K.size()
        A = self.decoder(z)
        D = torch.diag(A.sum(dim=1)) 
        L = D - A  
        L_reg = L + torch.eye(k, device=z.device)
        S1 = torch.linalg.inv(L_reg) 
        S0 = K.to(dtype=torch.float32) 
        S0_sqrt = sqrtm(S0)
        S_ = sqrtm(S0_sqrt.mm(S1).mm(S0_sqrt))
        W_p = S1.trace() ** p + S0.trace() ** p - 2.0 * S_.trace() ** p
        W_p = W_p ** (1 / p)
        return W_p.log()
    def cluster_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(z, y)
    def reset_parameters(self):
        super().reset_parameters()
        reset(self.discriminator)
    def reg_loss(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(self.discriminator(z, None, batch))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss
    def discriminator_loss(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        real = torch.sigmoid(self.discriminator(torch.randn_like(z), None, batch))
        fake = torch.sigmoid(self.discriminator(z.detach(), None, batch))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss

encoder = STx_encoder(in_channels=N, 
             hidden_channels=64, 
             out_channels=16, 
             m=7, K=3,
             connect=con, pi=0.75, n_heads=8)
discriminator = STx_discriminator(in_channels=16, hidden_channels=8, out_channels=1)
decoder = STx_decoder()
model = STx_ARGA(encoder, discriminator, decoder)
model = model.to(device)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1E-4)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1E-5)
#data = next(iter(train_loader))
#z = model.encode(data.x, None, data.batch)

def train(loader):
    model.train()
    running_loss = 0
    for data in loader: # Iterate in batches over the training dataset.
        data = data.to(device)    
        w, z = model.encode(data.x, None, data.batch)
        for _ in range(5):
            discriminator.train()
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(torch.matmul(w, z), data.batch) 
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step() 
        loss = 0
        loss = loss + model.reg_loss(torch.matmul(w, z), data.batch) 
        loss = loss + model.recon_loss(z, K_empiric.to(device))
        loss = loss + model.cluster_loss(w, data.y)
        loss.backward()
        encoder_optimizer.step()  
        encoder_optimizer.zero_grad()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def test(loader):
    model.eval()
    loss_reg = 0
    loss_clust = 0
    loss_recon = 0
    for data in loader: # Iterate in batches over the training/test dataset.
        data = data.to(device)
        w, z = model.encode(data.x, None, data.batch)
        cl = w.cpu().softmax(dim=-1)
        ari = adjusted_rand_score(onehot_to_label(cl.t()).values.reshape(-1), data.y.cpu())
        loss_reg = loss_reg + model.reg_loss(torch.matmul(w, z), data.batch)  
        loss_clust = loss_clust + model.cluster_loss(w, data.y)   
        loss_recon = loss_recon + model.recon_loss(z, K_empiric.to(device))
    return loss_reg / len(loader), loss_clust / len(loader), loss_recon / len(loader), ari # Derive ratio of correct predictions.

f = open(r'arga0728.csv', 'w')
f.writelines(['train loss,', 'test reg loss,', 'test class loss,', 'test recon loss,', 'test ari\n'])
f.close()
#f = open(r'arga0728.csv', 'a')
for epoch in range(1, 5001):
    best_loss = 2**25 
    loss = train(train_loader)
    loss_reg, loss_clust, loss_recon, ari = test(test_loader)
    f.write( ','.join(map(str, torch.tensor([loss, loss_reg, loss_clust, loss_recon, ari]).numpy())))
    f.write('\n')
    print((f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test loss reg: {loss_reg:.3f}, '
           f'Test loss class: {loss_clust:.3f}, Test loss recon: {loss_recon:.3f}, ' 
           f'Test ARI: {ari:.3f}'))
#f.close()

import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
import time
def train(): 
    best_loss = 2**25 
    since = time.time()
    f = open(r'/home/ubuntu/sda1/qiyuan/fengyang/arga0807.csv', 'w')
    f.writelines(['train loss,', 'test reg loss,', 'test class loss,', 'test recon loss,', 'test ari\n'])    
    for epoch in range(1, 5001):
        model.eval()
        loss_reg = 0
        loss_clust = 0
        loss_recon = 0
        model.train()
        running_loss = 0
        for data in train_loader: # Iterate in batches over the training dataset.
            data = data.to(device)    
            w, z = model.encode(data.x, None, data.batch)
            for _ in range(5):
                discriminator.train()
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(torch.matmul(w, z), data.batch) 
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimizer.step() 
            loss = 0
            loss = loss + model.reg_loss(torch.matmul(w, z), data.batch) 
            loss = loss + model.recon_loss(z, K_empiric.to(device))
            loss = loss + model.cluster_loss(w, data.y)
            loss.backward()
            encoder_optimizer.step()  
            encoder_optimizer.zero_grad()
            running_loss += loss.item()
            loss =  running_loss / len(train_loader)
        loss_reg, loss_clust, loss_recon, ari = test(test_loader)
        f.write( ','.join(map(str, torch.tensor([loss, loss_reg, loss_clust, loss_recon, ari]).numpy())))
        f.write('\n')
        print((f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test loss reg: {loss_reg:.3f}, '
                f'Test loss class: {loss_clust:.3f}, Test loss recon: {loss_recon:.3f}, ' 
                f'Test ARI: {ari:.3f}'))
        if epoch == 1:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        if(loss_recon < best_loss):
            print(f"Now best model is epoch{epoch}")
            # best_loss = loss3
            best_loss = loss_recon
            best_model = model.state_dict()
            torch.save(best_model,os.path.join(r'/home/ubuntu/sda1/qiyuan/fengyang',rf'best_model_test0807.pkl'))
    f.close()




'''

##################################################################################
Old codes


   
class STx_decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(53124)
        self.lin0 = Linear(in_channels, in_channels // 2)
        self.lin1 = Linear(in_channels // 2, in_channels // 4)
        self.lin2 = Linear(in_channels // 4, in_channels // 8)
        self.lin3 = Linear(in_channels // 8, out_channels)
        self.lin4 = Linear(out_channels, in_channels // 8)
        self.lin5 = Linear(in_channels // 8, in_channels // 4)
        self.lin6 = Linear(in_channels // 4, in_channels // 2)
        self.lin7 = Linear(in_channels // 2, in_channels)
        self.fc = Sequential(
                            Linear(in_channels, hidden_channels),
                            SELU(),
                            Linear(hidden_channels, out_channels)
                            )
    def forward(self, z):
        z0 = self.lin0(z)
        z1 = self.lin1(z0)
        z2 = self.lin2(z1)
        z3 = self.lin3(z2)
        z4 = self.lin4(z3)
        z4 = z4 + z2
        z5 = self.lin5(z4)
        z5 = z5 + z1
        z6 = self.lin6(z5)
        z6 = z6 + z0
        z7 = self.lin7(z6)
        z7 = z7 + z
        z7 = F.selu(z7)
        z7 = self.fc(z7)
        return torch.matmul(z7.t(), z7)

   
class STx_decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(53124)
        self.fc = Sequential(
                            Linear(in_channels, hidden_channels),
                            SELU(),
                            Linear(hidden_channels, out_channels),
                            ResNet(Linear(out_channels, out_channels)),
                            ResNet(Linear(out_channels, out_channels)),
                            ResNet(Linear(out_channels, out_channels))                       
                            )
    def forward(self, z):
        z = self.fc(z)
        return torch.matmul(z.t(), z)
        
'''