"""
TGRAASP Models Module
=====================
Deep learning model components for spatial transcriptomics data analysis.

Main Components:
- FixedSparseEdgeConv, FullEdgeConv: Graph convolutional layers
- PPIEdgeConv, DynamicEdgeConv: Dynamic graph convolutional layers
- STx_encoder: Spatial transcriptomics encoder
- STx_discriminator: Discriminator network
- STx_ARGA, STx_ARGVA: Adversarially regularized graph autoencoder models
- Alignment_Model: Model for cross-modality alignment
"""

# Python built-in
from typing import Optional

# Third-party
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, SELU, Parameter, Module, CrossEntropyLoss

# PyTorch Geometric
from torch_geometric.nn import (
    MessagePassing,
    knn_graph,
    GINConv,
    GraphNorm,
    GAE,
)
from torch_geometric.utils import (
    erdos_renyi_graph,
    unbatch,
    to_dense_adj,
    to_dense_batch,
    remove_self_loops,
    negative_sampling,
)
from torch_geometric.nn.dense.mincut_pool import _rank3_trace
from torch_geometric.nn.inits import reset

# Local imports
try:
    from .SGWTConv import SGWTConv
    from .CustomFunc import dropout_edge, onehot_to_label
    from .utils import edge_sampling
except ImportError:
    from SGWTConv import SGWTConv
    from CustomFunc import dropout_edge, onehot_to_label
    from utils import edge_sampling

# External dependency
try:
    from sparselinear import SparseLinear
except ImportError:
    raise ImportError("Please install sparselinear: pip install sparselinear")
# Constants
EPS = 1e-15  # Small epsilon for numerical stability
MAX_LOGSTD = 10  # Maximum log standard deviation
MAX = 10.  # Maximum value for clamping
MIN = -10.  # Minimum value for clamping


class FixedSparseEdgeConv(MessagePassing):
    """
    Fixed Sparse Edge Convolution Layer.
    
    Uses sparse linear transformations and multi-head edge attention weights
    for message passing on graphs.
    
    Args:
        in_channels (int): Number of input features per node
        out_channels (int): Number of output features per node
        connect: Connectivity pattern for sparse linear layer
    """
    def __init__(self, in_channels, out_channels, connect):
        super().__init__(aggr='max')  # Maximum aggregation
        self.out_dim = out_channels
        self.lin1 = SparseLinear(2 * in_channels, in_channels, connectivity=connect)
        self.lin2 = Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        """Forward pass through the layer."""
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_i, x_j, edge_weight):
        """
        Construct messages from source nodes to target nodes.
        
        Args:
            x_i: Target node features
            x_j: Source node features
            edge_weight: Edge attention weights (shape: [num_edges, num_heads])
            
        Returns:
            Messages weighted by edge attention
        """
        # Concatenate source and target features
        out = torch.cat([x_i, x_j], dim=1)
        # Apply transformations with residual connection
        out = F.relu(self.lin1(out)) + x_i
        out = self.lin2(out)
        # Apply multi-head attention weights
        n_h = edge_weight.size(1)  # Number of heads
        s_h = self.out_dim // n_h  # Size per head
        out = torch.matmul(out.view(-1, s_h, n_h), edge_weight)
        return out.view(-1, n_h * s_h)



class FullEdgeConv(MessagePassing):
    """
    Full Edge Convolution Layer.
    
    Similar to FixedSparseEdgeConv but uses dense linear layers and mean aggregation.
    
    Args:
        in_channels (int): Number of input features per node
        out_channels (int): Number of output features per node
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # Mean aggregation
        self.out_dim = out_channels
        self.lin1 = Linear(2 * in_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        """Forward pass through the layer."""
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_i, x_j, edge_weight):
        """
        Construct messages with multi-head attention.
        
        Args:
            x_i: Target node features
            x_j: Source node features
            edge_weight: Edge attention weights
            
        Returns:
            Weighted messages
        """
        out = torch.cat([x_i, x_j], dim=1)
        out = F.relu(self.lin1(out)) + x_i
        out = self.lin2(out)
        n_h = edge_weight.size(1)
        s_h = self.out_dim // n_h
        out = torch.matmul(out.view(-1, s_h, n_h), edge_weight)
        return out.view(-1, n_h * s_h)





class PPIEdgeConv(FixedSparseEdgeConv):
    """
    Protein-Protein Interaction inspired Edge Convolution.
    
    Constructs dynamic graph edges using either Erdos-Renyi random graphs or k-NN,
    then applies multi-head attention mechanism for edge weights.
    
    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        connect: Connectivity pattern for sparse layers
        pi (float, optional): Probability for Erdos-Renyi graph construction
        dropout (float, optional): Edge dropout probability
        k (int, optional): Number of nearest neighbors for k-NN graph
        n_heads (int): Number of attention heads (default: 2)
    """
    def __init__(self, in_channels, out_channels, connect, pi=None, dropout=None, k=None, n_heads=2):
        super().__init__(in_channels, out_channels, connect)
        self.pi = pi
        self.dropout = dropout
        self.k = k
        self.n_heads = n_heads
        self.head_size = out_channels // n_heads
        self.d_k = out_channels
        
        # Query and Key networks for attention
        self.qnet = Sequential(
            SparseLinear(2 * in_channels, in_channels, connectivity=connect),
            Linear(in_channels, out_channels),
        )
        self.knet = Sequential(
            SparseLinear(2 * in_channels, in_channels, connectivity=connect),
            Linear(in_channels, out_channels),
        )
        
    def reshape(self, e):
        """Reshape tensor for multi-head attention."""
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with dynamic graph construction.
        
        Args:
            x: Node features
            edge_index: Edge indices (can be None for dynamic construction)
            batch: Batch assignment vector
            
        Returns:
            Updated node features
        """
        # Construct edges using Erdos-Renyi if pi is provided
        if self.k is None:
            n = x.size(0)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            for v in unbatch(torch.arange(n, device=x.device), batch):
                e = erdos_renyi_graph(n, self.pi, directed=False)
                e = e.to(x.device)
                edge_index = torch.cat([edge_index, v[e]], dim=1)
        # Construct edges using k-NN if k is provided
        if self.pi is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            
        # Apply edge dropout
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = dropout_edge(edge_index, self.dropout, force_undirected=True)
        
        # Compute attention weights
        x_src, x_dst = x[edge_index]
        Q = self.qnet(torch.cat([x_src, x_dst], dim=1))
        K = self.knet(torch.cat([x_src, x_dst], dim=1))
        Q = self.reshape(Q)
        K = self.reshape(K)
        edge_weight = torch.matmul(Q, K.permute(0, 2, 1)) / torch.tensor(self.d_k).sqrt()
        
        return super().forward(x, edge_index, edge_weight)



class DynamicEdgeConv(FullEdgeConv):
    """
    Dynamic Edge Convolution with k-NN graph construction.
    
    Dynamically constructs graph using k-nearest neighbors and applies
    multi-head attention for edge weights.
    
    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        k (int): Number of nearest neighbors (default: 4)
        dropout (float): Edge dropout probability (default: 0.0)
        n_heads (int): Number of attention heads (default: 2)
    """
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
        """Reshape tensor for multi-head attention."""
        new_shape = e.size()[:-1] + (self.n_heads, self.head_size)
        e = e.view(*new_shape)
        return e
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with dynamic k-NN graph construction.
        
        Args:
            x: Node features
            edge_index: Edge indices (ignored, reconstructed using k-NN)
            batch: Batch assignment vector
            
        Returns:
            Updated node features
        """
        # Dynamically construct k-NN graph
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, self.dropout, force_undirected=True)
        
        # Compute attention weights
        x_src, x_dst = x[edge_index]
        Q = self.qnet(torch.cat([x_src, x_dst], dim=1))
        K = self.knet(torch.cat([x_src, x_dst], dim=1))
        Q = self.reshape(Q)
        K = self.reshape(K)
        edge_weight = torch.matmul(Q, K.permute(0, 2, 1)) / torch.tensor(self.d_k).sqrt()
        
        return super().forward(x, edge_index, edge_weight)



class STx_encoder(torch.nn.Module):
    """
    Spatial Transcriptomics Encoder.
    
    Encodes spatial transcriptomics data using spectral graph wavelets,
    performs soft clustering, and generates latent representations.
    
    Args:
        in_channels (int): Number of input gene features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output latent dimensions
        m (int): Number of clusters for soft assignment
        l (int): Latent dimension multiplier for covariance
        connect: PPI connectivity pattern
        pi (float): Probability for random graph construction
        n_heads (int): Number of attention heads
        K (int): Order of spectral graph wavelet filter
        activate_sc_alignment (bool): Whether to activate single-cell alignment model
    """
    def __init__(self, in_channels, hidden_channels, out_channels, m, l, connect, pi, n_heads, K, 
                 activate_sc_alignment: bool = False):
        super().__init__()
        torch.manual_seed(12345)
        self.n_heads = n_heads
        
        # Spectral graph wavelet convolutions for spatial structure
        self.conv0 = SGWTConv(in_channels, hidden_channels, K, normalization="rw")
        self.norm1 = GraphNorm(hidden_channels)
        self.conv1 = SGWTConv(in_channels + hidden_channels, hidden_channels, K, normalization="rw")
        
        # Clustering network - produces soft assignments
        self.fc1 = Sequential(
            Linear(hidden_channels, m * 16),
            SELU(),
            Linear(m * 16, m * 4),
            SELU(),
            Linear(m * 4, m)
        )
        
        # Optional alignment model for cross-modality integration
        self.activate_sc_alignment = activate_sc_alignment
        if self.activate_sc_alignment:
            self.alignment_model = Alignment_Model(in_channels, in_channels)
        else:
            self.alignment_model = None

        # PPI-based and dynamic graph convolutions for feature learning
        self.conv2 = PPIEdgeConv(in_channels, hidden_channels, connect, pi=pi, k=None, 
                                 dropout=0.0, n_heads=self.n_heads)
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels, k=3, dropout=0.1, 
                                     n_heads=self.n_heads)
        
        # Output networks for mean and covariance
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
        """
        Forward pass through the encoder.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights
            batch: Batch assignment vector
            mask: Optional mask for nodes
            is_single_cell_input: Whether input is single-cell data (applies alignment if True)
            
        Returns:
            Tuple containing:
                - u: Cluster assignment logits
                - mu: Cluster centroids (aligned if single-cell)
                - v: Mean of latent distribution
                - A: Covariance components of latent distribution
                - edge_index: Updated edge indices
                - batch_for_gcn: Batch assignments for cluster-level graph
        """
        initial_x = x  # Keep original features
        
        # Construct k-NN graph if not provided
        edge_index = knn_graph(initial_x, 3, batch, loop=False) if edge_index is None else edge_index
        
        # Spectral wavelet convolution for clustering
        u = self.conv0(initial_x, edge_index, edge_weight, batch)
        u = self.norm1(u, batch)
        u = self.fc1(u)
        S = u.softmax(dim=-1)  # Soft cluster assignments [num_nodes, m_clusters]
        
        # Ensure correct batch dimension
        S_for_aggregation = S.unsqueeze(0) if S.dim() == 2 else S
        
        # Convert to dense batch format for pooling
        x_features_to_pool, batch_info_dense = to_dense_batch(initial_x, batch) 
        batch_size = x_features_to_pool.size(0)
        num_original_nodes = x_features_to_pool.size(1)
        model_in_channels = x_features_to_pool.size(2)

        # Apply mask if provided
        if mask is not None:
            mask_dense = mask.view(batch_size, num_original_nodes, 1).to(x_features_to_pool.dtype)
            x_features_to_pool = x_features_to_pool * mask_dense
            S_for_aggregation = S_for_aggregation * mask_dense

        # Aggregate features by cluster assignments (soft pooling)
        x_aggregated_profiles = torch.matmul(S_for_aggregation.transpose(1, 2), x_features_to_pool)
        
        # Normalize by cluster sizes
        sum_S_per_cluster = torch.sum(S_for_aggregation, dim=1, keepdim=True).transpose(1, 2).clamp(min=EPS)
        x_agg_norm = x_aggregated_profiles / sum_S_per_cluster
        
        # Reshape cluster features for GCN processing
        m_clusters = S_for_aggregation.size(2)
        mu_features_for_gcn = x_agg_norm.reshape(batch_size * m_clusters, model_in_channels)

        # Apply alignment model if processing single-cell data
        if self.alignment_model is not None and is_single_cell_input:
            aligned_mu_features = self.alignment_model(mu_features_for_gcn)
        else:
            aligned_mu_features = mu_features_for_gcn

        # Create batch assignments for cluster-level processing
        batch_for_gcn = torch.arange(batch_size, device=initial_x.device).repeat_interleave(m_clusters)
        
        # Process cluster features through graph convolutions
        x_conv2_out = self.conv2(aligned_mu_features, None, batch_for_gcn)
        y_conv3_out = self.conv3(x_conv2_out, None, batch_for_gcn)
        
        # Residual connection
        z_final_conv = x_conv2_out + y_conv3_out
        
        mu_to_return = aligned_mu_features

        return u, mu_to_return, self.fc2(z_final_conv), self.fc3(z_final_conv), edge_index, batch_for_gcn



class STx_discriminator(torch.nn.Module):
    """
    Discriminator Network for Adversarial Training.
    
    Uses GIN convolution to distinguish between real and generated latent representations.
    
    Args:
        in_channels (int): Number of input latent dimensions
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output dimensions (typically 1 for binary classification)
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(54321)
        
        # GIN convolution for invariant graph features
        self.conv = GINConv(Sequential(
            Linear(in_channels, hidden_channels),
            SELU(),
            Linear(hidden_channels, hidden_channels)
        ))
        
        # Classification head
        self.fc = Sequential(
            Linear(hidden_channels, hidden_channels // 3),
            SELU(),
            Linear(hidden_channels // 3, out_channels)
        )
        
    def forward(self, z, edge_index, batch):
        """
        Forward pass of discriminator.
        
        Args:
            z: Latent representations [num_nodes, in_channels]
            edge_index: Edge indices (if None, constructs k-NN graph)
            batch: Batch assignment vector
            
        Returns:
            Discrimination logits [num_nodes, out_channels]
        """
        edge_index = knn_graph(z, 4, batch, loop=False) if edge_index is None else edge_index
        z = self.conv(z, edge_index)
        z = self.fc(z)
        return z





class STx_ARGA(GAE):
    """
    Adversarially Regularized Graph Autoencoder (ARGA).
    
    Combines graph autoencoding with adversarial training to learn robust
    latent representations of graph-structured data.
    
    Args:
        encoder: Encoder network
        discriminator: Discriminator network for adversarial regularization
        decoder: Decoder network (optional, uses inner product if None)
    """
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
        """
        Compute graph reconstruction loss.
        
        Args:
            z: Latent node representations
            pos_edge: All positive edges
            pos_edge_index: Training positive edges
            pos_edge_index1: Test positive edges
            neg_edge_index: Negative edges (sampled if None)
            
        Returns:
            Combined positive and negative reconstruction loss
        """
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index, _ = edge_sampling(pos_edge, pos_edge_index, pos_edge_index1)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss
    
    def reset_parameters(self):
        """Reset all model parameters."""
        super().reset_parameters()
        reset(self.discriminator)
    
    def reg_loss(self, z: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial regularization loss for generator.
        
        Args:
            z: Latent representations (generated)
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            Generator loss (fool discriminator)
        """
        real = torch.sigmoid(self.discriminator(z, edge_index, batch))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss
    
    def discriminator_loss(self, z: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            z: Latent representations
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            Discriminator loss (distinguish real vs fake)
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z), edge_index, batch))
        fake = torch.sigmoid(self.discriminator(z.detach(), edge_index, batch))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss
    




class STx_ARGVA(STx_ARGA):
    """
    Adversarially Regularized Variational Graph Autoencoder (ARVGA).
    
    Extends ARGA with variational inference, learning probabilistic latent
    representations with reparameterization trick.
    
    Args:
        encoder: Variational encoder network
        discriminator: Discriminator network
        decoder: Decoder network (optional)
        l: Latent dimension for covariance parameterization
    """
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
        """
        Reparameterization trick for variational inference.
        
        Args:
            s: Soft cluster assignments [num_nodes, num_clusters]
            mu: Cluster means [num_clusters, latent_dim]
            A: Covariance components [num_clusters, latent_dim * l]
            
        Returns:
            Sampled latent representations [num_nodes, latent_dim]
        """
        z = torch.matmul(s, mu)
        if self.training:
            # Sample noise according to cluster assignments
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
        """
        Encode input data into latent space with variational inference.
        
        Args:
            s: Soft cluster assignments (optional, computed if None)
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
            batch: Batch assignment
            mask: Node mask
            is_single_cell: Whether to apply alignment for single-cell data
            
        Returns:
            Tuple of (latent_representations, batch_for_clusters)
        """
        # Encode through encoder network
        self.__u__, self.__m__, self.__v__, self.__A__, self.__e__, self.__b__ = \
            self.encoder(x, edge_index, edge_weight, batch, mask, is_single_cell_input=is_single_cell)
        
        # Clamp covariance components for stability
        self.__A__ = self.__A__.clamp(min=MIN, max=MAX)
        
        # Compute log standard deviation from covariance components
        self.__logstd__ = torch.cat([
            torch.matmul(self.__A__[i].view(self.l, -1).t(), 
                        self.__A__[i].view(self.l, -1)).diag().view(1, -1)
            for i in range(self.__m__.size(0))
        ]).log()
        
        # Soft cluster assignments
        S = self.__u__.softmax(dim=-1)
        
        # Compute adjacency matrices for clustering loss
        self.__adj__ = to_dense_adj(self.__e__)
        self.__adj1__ = torch.matmul(torch.matmul(S.t(), self.__adj__), S)
        
        # Use provided or computed assignments
        s = S if s is None else s
        
        # Reparameterize to sample latent code
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
    """
    Cross-Modality Alignment Model.
    
    This model aligns features from different modalities 
    (e.g., spatial vs single-cell gene expression data)
    using a deep residual network with batch normalization and dropout.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        hidden_dim (int, optional): Dimension of the hidden layers.
            Defaults to input_dim * 2.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
            
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # fc2 outputs hidden_dim, so use hidden_dim here
        
        # First fully connected block
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Second fully connected block
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Third fully connected block
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Residual connection to match input/output dimensions
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Apply batch normalization to the input
        x = self.bn1(x)
        
        # Forward through main path
        h = self.fc1(x)
        h = self.bn2(h)
        h = self.fc2(h)
        h = self.bn3(h)
        h = self.fc3(h)
        
        # Residual connection (skip connection)
        res = self.residual(x)
        
        # Combine main path and residual
        return h + res

        