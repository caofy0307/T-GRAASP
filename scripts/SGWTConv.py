from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, get_laplacian, to_dense_adj,to_undirected

class SGWTConv(MessagePassing):
    r"""The SGWT spectral graph convolutional operator

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        #width: List[float],
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        #[self.x1, self.x2] = width
        self.J = out_channels - 1
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        #super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)


    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        lambda_min: OptTensor = 1.e-4,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None
            
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if  edge_weight.min() > lambda_min:
            lambda_min = edge_weight.min()
        if not isinstance(lambda_min, Tensor):
            lambda_min = torch.tensor(lambda_min, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_min > 0 

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
            lambda_min = lambda_min[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]  #remove self-loop
        edge_weight[loop_mask] -= 1
        
        return edge_index, edge_weight, lambda_max, lambda_min

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        edge_index, norm, lambda_max, _ = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        if batch is not None:
            L = to_dense_adj(edge_index, batch, edge_attr=norm)
        else:
            L = to_dense_adj(edge_index, edge_attr=norm)

        n = x.size(0)
        Tx_0 = torch.eye(n, device=edge_index.device)
        Tx_1 = 2. * L.view(n, n) / lambda_max - torch.eye(n, device=edge_index.device)
        Wx = self.lins[0](torch.matmul(Tx_0, x))
        if len(self.lins) > 1:
            Wx = Wx + self.lins[1](torch.matmul(Tx_1, x))
        for lin in self.lins[2:]:
            Tx_2 = 2. * torch.matmul(2. * L.view(n, n) / lambda_max - torch.eye(n, device=edge_index.device), Tx_1) - Tx_0
            Wx = Wx + lin(torch.matmul(Tx_2, x))
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = Wx

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
