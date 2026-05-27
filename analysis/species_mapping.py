from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_species_mapping(path: str | Path) -> pd.DataFrame:
    """
    Load node_id -> species table.

    Graph nodes in Ythan.txt are 0..133; species_mapping.csv uses node_id 1..134.
    Alignment: graph node g maps to node_id g for g>=1 (shared indexing).
    Graph node 0 maps to node_id 1 (graph has 0..133, CSV has 1..134; only 0 is offset).
    """
    df = pd.read_csv(path)
    required = {"node_id", "species_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"species_mapping must contain {required}")
    return df


def graph_to_csv_node_id(graph_node: int) -> int:
    """Map graph node index to species_mapping.node_id."""
    g = int(graph_node)
    return 1 if g == 0 else g


def annotate_with_species(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    node_col: str = "node",
) -> pd.DataFrame:
    """Merge species names onto a table with a graph node column (0-based)."""
    m = mapping.copy()
    out = df.copy()
    out["_map_id"] = out[node_col].astype(int).map(graph_to_csv_node_id)
    m = m.rename(
        columns={
            "node_id": "_map_id",
            "species_name": "species",
            "common_name": "common_name",
            "category": "category",
        }
    )
    cols = ["_map_id", "species", "common_name", "category"]
    cols = [c for c in cols if c in m.columns]
    out = out.merge(m[cols], on="_map_id", how="left").drop(columns=["_map_id"])
    return out


def format_node(node: int, mapping: pd.DataFrame | None = None) -> str:
    if mapping is None:
        return str(node)
    row = mapping[mapping["node_id"] == graph_to_csv_node_id(int(node))]
    if row.empty:
        return str(node)
    sp = row.iloc[0]["species_name"]
    cat = row.iloc[0].get("category", "")
    return f"{sp} (node {node}, {cat})" if cat else f"{sp} (node {node})"
