from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------
EdgeBuilder = Callable[..., torch.Tensor]
EdgeSplitter = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def normalize_barcodes(
    index: pd.Index,
    prefix_to_strip: Optional[str] = None,
    strip_numeric_suffix: bool = True,
) -> pd.Index:
    """Normalize cell/spot barcodes for robust cross-file alignment.

    Parameters
    ----------
    index:
        Input pandas index.
    prefix_to_strip:
        Optional prefix to remove, e.g. ``"SC_"`` or ``"SP_"``.
    strip_numeric_suffix:
        If True, strip trailing patterns like ``_1`` / ``_1_2``.
    """
    index = index.astype(str)
    if prefix_to_strip:
        index = index.str.replace(rf"^{prefix_to_strip}", "", regex=True)
    if strip_numeric_suffix:
        index = index.str.replace(r"_[0-9_]+$", "", regex=True)
    return index



def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path



def align_by_index(left: pd.DataFrame, right: pd.DataFrame, sample_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly align two dataframes by index intersection while preserving order."""
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValueError(f"[{sample_id}] Expression matrix and metadata have zero overlapping barcodes.")

    if len(common) != len(left) or len(common) != len(right):
        logger.warning(
            "[%s] Auto-align by index intersection: expr=%d, meta=%d, aligned=%d",
            sample_id, len(left), len(right), len(common)
        )

    return left.loc[common], right.loc[common]



def validate_required_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}")


# ---------------------------------------------------------------------
# Cell-type mapping
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CellTypeMapping:
    """Shared cell-type vocabulary for SC and SP datasets."""

    celltypes: List[str]
    celltype_to_idx: Dict[str, int] = field(init=False)
    idx_to_celltype: Dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        unique_celltypes = list(dict.fromkeys(self.celltypes))
        object.__setattr__(self, "celltypes", unique_celltypes)
        object.__setattr__(self, "celltype_to_idx", {ct: i for i, ct in enumerate(unique_celltypes)})
        object.__setattr__(self, "idx_to_celltype", {i: ct for i, ct in enumerate(unique_celltypes)})

    @property
    def num_classes(self) -> int:
        return len(self.celltypes)

    def encode_series(self, labels: pd.Series, context: str = "labels") -> torch.Tensor:
        unknown = sorted(set(labels.dropna().unique()) - set(self.celltype_to_idx.keys()))
        if unknown:
            raise ValueError(
                f"[{context}] Found unknown cell types not present in shared mapping: {unknown}"
            )
        return torch.tensor(labels.map(self.celltype_to_idx).values, dtype=torch.long)

    @classmethod
    def from_obs_csv(
        cls,
        obs_csv_path: str | Path,
        start_col: str = "B_cell",
        end_col: str = "cDC1_CLEC9A",
    ) -> "CellTypeMapping":
        path = ensure_exists(Path(obs_csv_path))
        df = pd.read_csv(path, header=0, index_col=0, nrows=0)
        cols = df.columns.tolist()
        if start_col not in cols or end_col not in cols:
            raise ValueError(
                f"Cannot infer cell types from {path}. start_col={start_col!r}, end_col={end_col!r}."
            )
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col)
        if start_idx > end_idx:
            raise ValueError("start_col must appear before end_col.")
        return cls(cols[start_idx : end_idx + 1])

    def summary(self) -> str:
        lines = [f"CellTypeMapping(num_classes={self.num_classes})"]
        lines.extend([f"  {idx:>2} : {name}" for idx, name in self.idx_to_celltype.items()])
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------
@dataclass
class BaseDatasetConfig:
    raw_dir: str | Path
    sample_ids: Sequence[str]
    expression_suffix: str
    metadata_suffix: str
    expression_sep: str = " "
    expression_index_col: int = 0
    metadata_index_col: int = 0
    strip_numeric_suffix: bool = True

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)


@dataclass
class SingleCellDatasetConfig(BaseDatasetConfig):
    expression_suffix: str = "_hvg_data.txt"
    metadata_suffix: str = "_meta.csv"
    barcode_prefix: Optional[str] = "SC_"
    celltype_column: str = "celltype"


@dataclass
class SpatialDatasetConfig(BaseDatasetConfig):
    expression_suffix: str = "_sp_hvg_data.txt"
    metadata_suffix: str = "_obs.csv"
    barcode_prefix: Optional[str] = "SP_"
    coord_columns: Tuple[str, str] = ("array_row", "array_col")
    index_column_fallback: Optional[str] = "X"
    test_ratio: float = 0.1
    k_max: int = 50
    target_degree: int = 8


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------
class SingleCellDatasetBuilder:
    def __init__(self, config: SingleCellDatasetConfig, mapping: CellTypeMapping):
        self.config = config
        self.mapping = mapping

    def _read_expression(self, sample_id: str) -> pd.DataFrame:
        path = ensure_exists(self.config.raw_path / f"{sample_id}{self.config.expression_suffix}")
        df = pd.read_csv(
            path,
            sep=self.config.expression_sep,
            header=0,
            index_col=self.config.expression_index_col,
        )
        df.index = normalize_barcodes(
            df.index,
            prefix_to_strip=self.config.barcode_prefix,
            strip_numeric_suffix=self.config.strip_numeric_suffix,
        )
        return df

    def _read_metadata(self, sample_id: str) -> pd.DataFrame:
        path = ensure_exists(self.config.raw_path / f"{sample_id}{self.config.metadata_suffix}")
        df = pd.read_csv(path, header=0, index_col=self.config.metadata_index_col)
        df.index = normalize_barcodes(
            df.index,
            prefix_to_strip=self.config.barcode_prefix,
            strip_numeric_suffix=self.config.strip_numeric_suffix,
        )
        validate_required_columns(df, [self.config.celltype_column], f"SC {sample_id}")
        return df

    def build_sample(self, sample_id: str) -> Data:
        logger.info("Processing SC sample: %s", sample_id)
        expr = self._read_expression(sample_id)
        meta = self._read_metadata(sample_id)
        expr, meta = align_by_index(expr, meta, sample_id)

        x = torch.tensor(expr.values, dtype=torch.float32)
        y = self.mapping.encode_series(meta[self.config.celltype_column], context=f"SC {sample_id}")

        data = Data(x=x, y=y)
        data.sample_id = sample_id
        data.num_classes = self.mapping.num_classes
        return data

    def build(self) -> List[Data]:
        datasets = [self.build_sample(sid) for sid in self.config.sample_ids]
        logger.info("Built %d single-cell datasets.", len(datasets))
        return datasets


class SpatialDatasetBuilder:
    def __init__(
        self,
        config: SpatialDatasetConfig,
        mapping: CellTypeMapping,
        edge_builder: EdgeBuilder,
        edge_splitter: EdgeSplitter,
    ):
        self.config = config
        self.mapping = mapping
        self.edge_builder = edge_builder
        self.edge_splitter = edge_splitter

    def _read_expression(self, sample_id: str) -> pd.DataFrame:
        path = ensure_exists(self.config.raw_path / f"{sample_id}{self.config.expression_suffix}")
        df = pd.read_csv(
            path,
            sep=self.config.expression_sep,
            header=0,
            index_col=self.config.expression_index_col,
        )
        df.index = normalize_barcodes(
            df.index,
            prefix_to_strip=self.config.barcode_prefix,
            strip_numeric_suffix=self.config.strip_numeric_suffix,
        )
        return df

    def _read_metadata(self, sample_id: str) -> pd.DataFrame:
        path = ensure_exists(self.config.raw_path / f"{sample_id}{self.config.metadata_suffix}")
        df = pd.read_csv(path, header=0, index_col=self.config.metadata_index_col)
        if self.config.index_column_fallback and self.config.index_column_fallback in df.columns:
            df = df.set_index(self.config.index_column_fallback)
        validate_required_columns(
            df,
            list(self.mapping.celltypes) + list(self.config.coord_columns),
            f"SP {sample_id}",
        )
        return df

    def build_sample(self, sample_id: str) -> Tuple[Data, torch.Tensor]:
        logger.info("Processing SP sample: %s", sample_id)
        expr = self._read_expression(sample_id)
        meta = self._read_metadata(sample_id)
        expr, meta = align_by_index(expr, meta, sample_id)

        x_all = torch.tensor(expr.values, dtype=torch.float32)
        y_prob_raw = torch.tensor(meta[self.mapping.celltypes].values, dtype=torch.float32)
        coord_all = torch.tensor(meta[list(self.config.coord_columns)].values, dtype=torch.float32)

        row_sums = torch.sum(y_prob_raw, dim=1)
        valid_mask = ~torch.isnan(row_sums)

        x = x_all[valid_mask]
        y_prob = y_prob_raw[valid_mask]
        y = torch.argmax(y_prob, dim=1)
        coord = coord_all[valid_mask]
        n = coord.shape[0]

        logger.info("[%s] Valid spots: %d", sample_id, n)

        edge_index = self.edge_builder(
            coord,
            k_max=self.config.k_max,
            target_degree=self.config.target_degree,
        )

        adj_tensor = to_dense_adj(edge_index, max_num_nodes=n)[0]
        adj_sparse = sp.csr_matrix(adj_tensor.cpu().numpy().astype(np.float32))

        pos_edge, train_edges, test_edges, train_edge_false, test_edge_false = self.edge_splitter(
            adj_sparse,
            test_ratio=self.config.test_ratio,
        )

        train_edge_weight = torch.ones(train_edges.size(1), dtype=torch.float32)
        test_edge_weight = torch.ones(test_edges.size(1), dtype=torch.float32)
        batch = torch.zeros(n, dtype=torch.long)
        adj = adj_tensor.to(torch.int8)

        data = Data(
            x=x,
            y=y,
            y_prob=y_prob,
            coord=coord,
            train_edge_index=train_edges,
            test_edge_index=test_edges,
            pos_edge=pos_edge,
            neg_edge=test_edge_false,
            train_edge_weight=train_edge_weight,
            test_edge_weight=test_edge_weight,
            batch=batch,
        )
        data.sample_id = sample_id
        data.num_classes = self.mapping.num_classes
        return data, adj

    def build(self) -> Tuple[List[Data], List[torch.Tensor]]:
        data_list: List[Data] = []
        adj_list: List[torch.Tensor] = []
        for sid in self.config.sample_ids:
            data, adj = self.build_sample(sid)
            data_list.append(data)
            adj_list.append(adj)
        logger.info("Built %d spatial datasets.", len(data_list))
        return data_list, adj_list


# ---------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------
def infer_celltype_list_from_sp(
    obs_csv_path: str | Path,
    start_col: str = "B_cell",
    end_col: str = "cDC1_CLEC9A",
) -> List[str]:
    mapping = CellTypeMapping.from_obs_csv(obs_csv_path, start_col=start_col, end_col=end_col)
    logger.info("Inferred %d cell types from %s", mapping.num_classes, obs_csv_path)
    return mapping.celltypes



def build_sc_datasets(
    sample_ids: Sequence[str],
    raw_dir: str | Path,
    mapping: CellTypeMapping,
    *,
    expression_suffix: str = "_hvg_data.txt",
    metadata_suffix: str = "_meta.csv",
    celltype_column: str = "celltype",
    barcode_prefix: Optional[str] = "SC_",
) -> List[Data]:
    config = SingleCellDatasetConfig(
        raw_dir=raw_dir,
        sample_ids=sample_ids,
        expression_suffix=expression_suffix,
        metadata_suffix=metadata_suffix,
        celltype_column=celltype_column,
        barcode_prefix=barcode_prefix,
    )
    return SingleCellDatasetBuilder(config=config, mapping=mapping).build()



def build_sp_datasets(
    sample_ids: Sequence[str],
    raw_dir: str | Path,
    mapping: CellTypeMapping,
    *,
    edge_builder: EdgeBuilder,
    edge_splitter: EdgeSplitter,
    test_ratio: float = 0.1,
    expression_suffix: str = "_sp_hvg_data.txt",
    metadata_suffix: str = "_obs.csv",
    barcode_prefix: Optional[str] = "SP_",
    coord_columns: Tuple[str, str] = ("array_row", "array_col"),
    index_column_fallback: Optional[str] = "X",
    k_max: int = 50,
    target_degree: int = 8,
) -> Tuple[List[Data], List[torch.Tensor]]:
    config = SpatialDatasetConfig(
        raw_dir=raw_dir,
        sample_ids=sample_ids,
        expression_suffix=expression_suffix,
        metadata_suffix=metadata_suffix,
        barcode_prefix=barcode_prefix,
        coord_columns=coord_columns,
        index_column_fallback=index_column_fallback,
        test_ratio=test_ratio,
        k_max=k_max,
        target_degree=target_degree,
    )
    return SpatialDatasetBuilder(
        config=config,
        mapping=mapping,
        edge_builder=edge_builder,
        edge_splitter=edge_splitter,
    ).build()


# ---------------------------------------------------------------------
# Recommended package-style usage example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Example:
    # from yourpkg.graph import build_edges_auto_radius
    # from yourpkg.split import train_test_split
    #
    # mapping = CellTypeMapping.from_obs_csv(
    #     "data/ZWJBT_obs.csv",
    #     start_col="B_cell",
    #     end_col="cDC1_CLEC9A",
    # )
    # print(mapping.summary())
    #
    # sc_graphs = build_sc_datasets(
    #     sample_ids=["sample1", "sample2"],
    #     raw_dir="data/sc",
    #     mapping=mapping,
    # )
    #
    # sp_graphs, sp_adjs = build_sp_datasets(
    #     sample_ids=["sampleA", "sampleB"],
    #     raw_dir="data/sp",
    #     mapping=mapping,
    #     edge_builder=build_edges_auto_radius,
    #     edge_splitter=train_test_split,
    #     test_ratio=0.1,
    # )
    #
    # print(len(sc_graphs), len(sp_graphs), len(sp_adjs))
    pass
