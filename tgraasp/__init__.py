"""
TGRAASP: Topological Graph-based Representation for Adversarial Analysis of Spatial Proteomics
================================================================================================

A deep learning framework for spatial transcriptomics data analysis using graph neural networks
and adversarial training.

Main Components:
---------------
- models: Core neural network architectures (STx_encoder, STx_ARGVA, etc.)
- utils: Data loading and evaluation utilities
- CustomFunc: Custom mathematical functions
- SGWTConv: Spectral graph wavelet convolution layer
- train, trainer: Training utilities

Example Usage:
-------------
```python
from tgraasp import STx_encoder, STx_discriminator, STx_ARGVA
from tgraasp.utils import SGEDataset, load_ppi_connectivity

# Load data
data_list, adj_list = SGEDataset(sample_ids, raw_dir)

# Create model
encoder = STx_encoder(in_channels=2000, hidden_channels=128, ...)
discriminator = STx_discriminator(in_channels=32, hidden_channels=64, ...)
model = STx_ARGVA(encoder, discriminator)

# Train model
# ... (see examples for full training code)
```

For more information, see the documentation and examples.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main model components
from .models import (
    FixedSparseEdgeConv,
    FullEdgeConv,
    PPIEdgeConv,
    DynamicEdgeConv,
    STx_encoder,
    STx_discriminator,
    STx_ARGA,
    STx_ARGVA,
    Alignment_Model,
)

# Import spectral graph wavelet convolution
from .SGWTConv import SGWTConv

# Import utility functions
from .utils import (
    sparse2tuple,
    torch_delete,
    edge_sampling,
    train_test_split,
    SGEDataset,
    load_ppi_connectivity,
    create_mask,
    cal_auc_ap_acc,
    compute_block_matrix,
    compare_block_matrices,
)

# Import Data loading functions
from .TGraaspData import (
    DatasetScheme,
    TGraaspDataBuilder,
)

# Import custom functions
from .CustomFunc import (
    cov,
    dropout_edge,
    onehot_to_label,
    scale,
    matrix_regulize,
)

__all__ = [
    # Version info
    "__version__",
    # Models
    "FixedSparseEdgeConv",
    "FullEdgeConv",
    "PPIEdgeConv",
    "DynamicEdgeConv",
    "STx_encoder",
    "STx_discriminator",
    "STx_ARGA",
    "STx_ARGVA",
    "Alignment_Model",
    # Layers
    "SGWTConv",
    # Utilities
    "sparse2tuple",
    "torch_delete",
    "edge_sampling",
    "train_test_split",
    "SGEDataset",
    "load_ppi_connectivity",
    "create_mask",
    "cal_auc_ap_acc",
    "compute_block_matrix",
    "compare_block_matrices",
    # Custom functions
    "cov",
    "dropout_edge",
    "onehot_to_label",
    "scale",
    "matrix_regulize",
]

