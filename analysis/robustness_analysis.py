from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def _basal_nodes(G: nx.DiGraph) -> set[int]:
    """Species with no prey in prey→predator orientation (in-degree 0)."""
    return {int(n) for n in G.nodes() if G.in_degree(n) == 0}


def cascade_extinctions(G: nx.DiGraph, removed: int, basal: set[int] | None = None) -> set[int]:
    """
    Simple secondary-extinction cascade (Dunne-style, prey→predator edges).

    After removing `removed`, repeatedly delete non-basal species with no remaining prey
    (in-degree 0 in the remaining graph).
    """
    if basal is None:
        basal = _basal_nodes(G)

    G2 = G.copy()
    if removed in G2:
        G2.remove_node(removed)

    extinct: set[int] = {removed}
    changed = True
    while changed:
        changed = False
        for n in list(G2.nodes()):
            if int(n) in basal:
                continue
            if G2.in_degree(n) == 0:
                extinct.add(int(n))
                G2.remove_node(n)
                changed = True
    return extinct


def robustness_by_removal(G: nx.DiGraph, top_n: int = 25) -> pd.DataFrame:
    basal = _basal_nodes(G)
    rows = []
    for n in G.nodes():
        ext = cascade_extinctions(G, int(n), basal=basal)
        rows.append(
            {
                "node": int(n),
                "primary_extinctions": 1,
                "total_extinctions": len(ext),
                "secondary_extinctions": len(ext) - 1,
            }
        )
    df = pd.DataFrame(rows).sort_values("total_extinctions", ascending=False)
    return df.head(top_n) if top_n else df


def save_robustness_outputs(
    G: nx.DiGraph,
    out_dir: str | Path,
    prefix: str,
    species_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    df = robustness_by_removal(G, top_n=G.number_of_nodes())
    if species_df is not None:
        df = df.merge(
            species_df[["node", "species", "category"]],
            on="node",
            how="left",
        )
    df.to_csv(out_dir / f"{prefix}_robustness_cascade.csv", index=False)

    top = df.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        (r["species"][:28] + "…") if species_df is not None and pd.notna(r.get("species")) else str(int(r["node"]))
        for _, r in top.iterrows()
    ]
    ax.barh(labels, top["total_extinctions"], color="#C44E52", edgecolor="white")
    ax.set_xlabel("total extinctions after removal (cascade)")
    ax.set_title("Top species by secondary extinction impact")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_robustness_top15.png", dpi=200)
    plt.close(fig)
    return df
