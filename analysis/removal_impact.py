from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from robustness_analysis import cascade_extinctions


def weak_connectivity_snapshot(G: nx.DiGraph) -> dict:
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    sizes = [len(c) for c in wccs]
    return {
        "n_wcc": len(wccs),
        "largest_wcc": sizes[0] if sizes else 0,
        "second_largest_wcc": sizes[1] if len(sizes) > 1 else 0,
        "n_isolated": sum(1 for s in sizes if s == 1),
        "wcc_sizes_top5": sizes[:5],
    }


def removal_impact_row(
    G: nx.DiGraph,
    node: int,
    species: str | None = None,
    category: str | None = None,
) -> dict:
    base = weak_connectivity_snapshot(G)
    G2 = G.copy()
    if node in G2:
        G2.remove_node(node)
    after = weak_connectivity_snapshot(G2)
    ext = cascade_extinctions(G, int(node))
    return {
        "node": int(node),
        "species": species,
        "category": category,
        "baseline_n_wcc": base["n_wcc"],
        "after_n_wcc": after["n_wcc"],
        "baseline_largest_wcc": base["largest_wcc"],
        "after_largest_wcc": after["largest_wcc"],
        "largest_wcc_drop": base["largest_wcc"] - after["largest_wcc"],
        "cascade_total_extinctions": len(ext),
        "cascade_secondary_extinctions": len(ext) - 1,
        "interpretation_note": _interpret(base, after, len(ext)),
    }


def _interpret(base: dict, after: dict, cascade_n: int) -> str:
    parts = []
    if after["n_wcc"] > base["n_wcc"]:
        parts.append(
            f"removal splits weak components ({base['n_wcc']}→{after['n_wcc']}); "
            f"largest component {base['largest_wcc']}→{after['largest_wcc']}"
        )
    elif after["largest_wcc"] < base["largest_wcc"]:
        parts.append(f"largest weak component shrinks by {base['largest_wcc'] - after['largest_wcc']}")
    else:
        parts.append("web stays weakly connected as one giant component")
    parts.append(f"cascade model: {cascade_n} total extinctions")
    if after["n_wcc"] > base["n_wcc"] and cascade_n <= 2:
        parts.append(
            "topology breaks (betweenness) before directed prey-loss cascades "
            "because predators retain other prey"
        )
    return "; ".join(parts)


def key_node_removal_study(
    G: nx.DiGraph,
    nodes: list[int],
    species_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    lookup = {}
    if species_df is not None:
        for _, r in species_df.iterrows():
            lookup[int(r["node"])] = (r.get("species"), r.get("category"))

    for n in nodes:
        sp, cat = lookup.get(int(n), (None, None))
        rows.append(removal_impact_row(G, int(n), sp, cat))
    return pd.DataFrame(rows)


def save_removal_impact_outputs(
    G: nx.DiGraph,
    out_dir: str | Path,
    prefix: str,
    species_df: pd.DataFrame | None = None,
    focus_nodes: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compare weak connectivity and cascade impact for structurally important nodes.

    Default focus: top betweenness (118), top out-degree parasites (3, 5, 19),
    top PageRank-in producer (132).
    """
    out_dir = Path(out_dir)
    if focus_nodes is None:
        focus_nodes = [118, 3, 5, 19, 132]

    df = key_node_removal_study(G, focus_nodes, species_df)
    df.to_csv(out_dir / f"{prefix}_removal_impact.csv", index=False)

    summary = {
        "baseline": weak_connectivity_snapshot(G),
        "focus_nodes": focus_nodes,
        "rows": df.to_dict(orient="records"),
    }
    (out_dir / f"{prefix}_removal_impact.json").write_text(
        json.dumps(summary, indent=2)
    )
    return df
