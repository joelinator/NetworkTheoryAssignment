from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd


def directed_reachability_report(G: nx.DiGraph) -> dict:
    """Summarise strong/weak connectivity — relevant for interpreting closeness."""
    n = G.number_of_nodes()
    sccs = list(nx.strongly_connected_components(G))
    wccs = list(nx.weakly_connected_components(G))
    largest_scc = max(sccs, key=len)
    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "is_strongly_connected": nx.is_strongly_connected(G),
        "n_strongly_connected_components": len(sccs),
        "largest_scc_size": len(largest_scc),
        "largest_scc_fraction": len(largest_scc) / n if n else 0.0,
        "n_weakly_connected_components": len(wccs),
        "closeness_caveat": (
            "Food webs are typically NOT strongly connected. Standard directed closeness "
            "uses only reachable pairs and can inflate scores for nodes in small reachable sets. "
            "We report harmonic centrality as a more appropriate reachability measure."
        ),
    }


def save_reachability_report(G: nx.DiGraph, out_path: str | Path) -> dict:
    report = directed_reachability_report(G)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


def basal_breakdown(G: nx.DiGraph, species_annotated: pd.DataFrame) -> pd.DataFrame:
    """Count basal nodes (in-degree 0) by ecological category."""
    basal = species_annotated[species_annotated["in_degree"] == 0]
    if "category" not in basal.columns:
        return pd.DataFrame()
    return (
        basal.groupby("category", dropna=False)
        .agg(n_basal=("node", "count"))
        .reset_index()
        .sort_values("n_basal", ascending=False)
    )
