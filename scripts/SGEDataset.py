import pandas
import torch
import os.path as osp
from typing import List
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import dense_to_sparse, remove_self_loops, to_undirected
from sklearn.metrics import pairwise_distances

#generate dataset from whole-genome HiC data
#SGE2 = SGEDataset(root='/public/home/qiyuan/Projects/Spatial/', name='SGE2', id=sample_id)
class SGEDataset(InMemoryDataset):
    url = ''
    def __init__(self, root: str, name: str, id: List[str], transform=None, pre_transform=None):
        self.name = name
        self.id = id
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self) -> List[str]:         
        files = ['hvg_counts.txt', 'truth.txt']    
        return [f'{i}_{f}' for i in self.id for f in files]  
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def process(self):
        data_list = []
        for i in self.id:
            print(f'sample ID: {i}')
            nodes = pandas.read_csv(f'{self.raw_dir}/{i}_hvg_counts.txt', sep=' ', header=0)
            x = torch.tensor(nodes.values, dtype=torch.float).transpose(0, 1)
            node_id = nodes.columns.values
            node_label = pandas.read_csv(f'{self.raw_dir}/{i}_truth.txt', sep=' ', header=0) 
            node_barcode = node_label['barcode'].values
            if not all(node_id == node_barcode):
                try:
                    raise Exception('Node IDs do not match!')
                except Exception as error:
                    print(error) 
            node_celltype = node_label['layer'].values.tolist()
            y = torch.tensor(node_celltype, dtype=torch.float32)
            j = torch.where(~torch.isnan(y))    
            edge_index = torch.Tensor(2, 0) #empty edges    
            pos = pandas.read_csv(f'{self.raw_dir}/{i}_position.txt', sep=' ', header=0) 
            # coord = torch.tensor(pos.iloc[:,-2:].values, dtype=torch.float32) 
            coord = torch.tensor(pos.iloc[:,[4, 5]].values, dtype=torch.float32)   
            dst = torch.tensor(pairwise_distances(coord[j], metric='euclidean'))
            adj = (dst < 150.).int()
            edge_index, _ = dense_to_sparse(adj)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_weight = torch.ones_like(edge_index[0])
            g = Data(x=x[j], y=y[j].long(), edge_index=edge_index, edge_weight=edge_weight)
            data_list = data_list + [g]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
