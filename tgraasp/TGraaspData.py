from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Literal

import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from .utils import train_test_split
# Expected signature of your own split function:
# train_test_split(adj: torch.Tensor, test_ratio: float)
#   -> Tuple[pos_edge, tr_edges, te_edges, tr_false, te_false]
# from your_module import train_test_split


@dataclass
class DatasetScheme:
    """
    Unified file scheme for both basic and graph datasets.
    Set `with_edges=True` to enable graph construction using spatial coordinates.
    """
    # Core files
    counts_tmpl: str
    labels_tmpl: str
    label_col: str = "integrated_snn_res.0.4"

    # Graph-specific (required when with_edges=True)
    positions_tmpl: Optional[str] = None       # e.g., "{sid}_position_sp.txt"

    # Parsing options
    sep: str = r"\s+"
    header: Optional[int] = 0
    index_col: Optional[int] = 0

    # Mode switch
    with_edges: bool = False                   # False -> basic; True -> graph

    # Optional coordinate column indices (0-based) if positions file has extra columns
    coord_cols: Tuple[int, int] = (0, 1)


class TGraaspDataBuilder:
    """
    Unified data builder for T-GRAASP using a single DatasetScheme.
    - When scheme.with_edges is True: build graph dataset (train/test edges)
    - When scheme.with_edges is False: build basic dataset (no edges)
    - Also supports loading PPI connectivity with self-loops
    """

    def __init__(self, raw_dir: str):
        self.root = Path(raw_dir)

    # -------- utilities --------
    @staticmethod
    def _read_df(path: Path, sep: str, header: Optional[int], index_col: Optional[int]) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, sep=sep, header=header, index_col=index_col)

    @staticmethod
    def _labels_with_mask(series: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tensor(series.values, dtype=torch.float32)
        valid = ~torch.isnan(y)
        return y[valid].long(), valid

    # -------- public API --------
    def load_ppi_connectivity(self, ppi_path: str, num_nodes: int) -> torch.Tensor:
        """
        Load PPI edges and append self-loops.
        Assumes the file has edges in columns 1 and 2 (0-based indices), whitespace-separated.
        Returns edge_index with shape (2, E).
        """
        df = pd.read_csv(ppi_path, sep=r"\s+", header=None)
        edge = df.iloc[:, 1:3].astype(int).values
        edge_index = torch.tensor(edge.T, dtype=torch.long)
        self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
        return torch.cat([edge_index, self_loops], dim=1)

    def build_dataset(
        self,
        sample_ids: List[str],
        scheme: DatasetScheme,
        *,
        # graph-only params
        test_ratio: float = 0.1,
        distance_ref_k: int = 100,
        distance_ref_rank: int = 8,
        distance_margin: float = 0.0,
    ):
        """
        Build dataset according to scheme.with_edges.
        Returns:
          - if with_edges: (data_list, adj_list)
          - else: data_list
        """
        if scheme.with_edges and not scheme.positions_tmpl:
            raise ValueError("positions_tmpl is required when with_edges=True.")

        data_list: List[Data] = []
        adj_list: List[torch.Tensor] = []

        for sid in sample_ids:
            print(f"[build_dataset] Loading sample {sid} (with_edges={scheme.with_edges})")

            counts_fp = self.root / scheme.counts_tmpl.format(sid=sid)
            labels_fp = self.root / scheme.labels_tmpl.format(sid=sid)

            nodes_df = self._read_df(counts_fp, scheme.sep, scheme.header, scheme.index_col)
            x = torch.tensor(nodes_df.values, dtype=torch.float32).T

            label_df = self._read_df(labels_fp, scheme.sep, scheme.header, scheme.index_col)
            if scheme.label_col not in label_df.columns:
                raise KeyError(f"Label column not found in {labels_fp}: {scheme.label_col}")
            y, valid = self._labels_with_mask(label_df[scheme.label_col])

            if not scheme.with_edges:
                data_list.append(Data(x=x[valid], y=y))
                continue

            # graph mode
            pos_fp = self.root / scheme.positions_tmpl.format(sid=sid)
            pos_df = self._read_df(pos_fp, scheme.sep, scheme.header, scheme.index_col)

            # align x and coords by the valid mask
            x = x[valid]
            i0, i1 = scheme.coord_cols
            coord = torch.tensor(pos_df.iloc[:, [i0, i1]].values, dtype=torch.float32)[valid]

            # pairwise distances and adaptive threshold
            dst = torch.tensor(pairwise_distances(coord, metric="euclidean"))
            kth_col = min(max(distance_ref_k, 0), dst.shape[1] - 1)
            rank = min(max(distance_ref_rank, 0), dst.shape[0] - 1)
            cut = torch.sort(dst[:, kth_col])[0][rank].item() + float(distance_margin)

            adj = (dst < cut).int()

            pos_edge, tr_edges, te_edges, tr_false, te_false = train_test_split(adj, test_ratio)

            data = Data(
                x=x, y=y,
                train_edge_index=tr_edges, test_edge_index=te_edges,
                pos_edge=pos_edge, neg_edge=te_false,
                train_edge_weight=torch.ones(tr_edges.size(1)),
                test_edge_weight=torch.ones(te_edges.size(1)),
                batch=torch.zeros(x.size(0), dtype=torch.long),
            )
            data_list.append(data)
            adj_list.append(adj)

        return (data_list, adj_list) if scheme.with_edges else data_list
