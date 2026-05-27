from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class NetworkData:
    name: str
    edges: pd.DataFrame  # columns: src, dst, weight
    G: nx.DiGraph


def _read_edgelist_3col(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst", "weight"])
    if df.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (src, dst, weight) in {path}, got {df.shape[1]}")
    return df


def load_ythan(path: str | Path) -> NetworkData:
    """
    Load Ythan food web.

    Dataset convention (per `dataset_description.md`):
      edge src -> dst means src is consumed by dst (prey -> predator).
    """
    path = Path(path)
    edges = _read_edgelist_3col(path)
    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0).astype(float)

    G = nx.from_pandas_edgelist(
        edges,
        source="src",
        target="dst",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )
    # Ensure isolates (if any) are included
    all_nodes = pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True))
    G.add_nodes_from(int(x) for x in all_nodes)

    return NetworkData(name=path.stem, edges=edges, G=G)


def giant_weakly_connected_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    if G.number_of_nodes() == 0:
        return G.copy()
    comps: Iterable[set[int]] = nx.weakly_connected_components(G)
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

