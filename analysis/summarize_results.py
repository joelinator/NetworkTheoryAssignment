from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    cent = pd.read_csv(OUT / "ythan_centralities.csv")
    troph = pd.read_csv(OUT / "ythan_trophic_levels.csv")
    comm = pd.read_csv(OUT / "ythan_communities_membership.csv")
    top = pd.read_csv(OUT / "ythan_top_nodes.csv")

    # trophic table contains degrees too; keep the centrality table's degree columns
    troph = troph.drop(columns=[c for c in ["in_degree", "out_degree"] if c in troph.columns])
    profiles = cent.merge(troph, on="node", how="left").merge(comm, on="node", how="left")

    # Focus on nodes that appear in any top-3 list
    focus_nodes = sorted(top["node"].unique().tolist())
    focus = profiles[profiles["node"].isin(focus_nodes)].copy()

    # Add rank info per measure (rename rank columns to avoid collisions)
    ranks = top.pivot_table(index="node", columns="measure", values="rank", aggfunc="min").reset_index()
    ranks = ranks.rename(columns={c: f"rank_{c}" for c in ranks.columns if c != "node"})
    focus = focus.merge(ranks, on="node", how="left")

    # Build final ordered list
    ordered = [
        "node",
        "community_id",
        "basal",
        "trophic_level",
        "in_degree",
        "out_degree",
        "betweenness",
        "pagerank",
        "rank_in_degree",
        "rank_out_degree",
        "rank_betweenness",
        "rank_pagerank",
    ]
    ordered = [c for c in ordered if c in focus.columns]
    remaining = [c for c in focus.columns if c not in ordered]
    focus = focus[ordered + remaining]

    focus.to_csv(OUT / "ythan_top_node_profiles.csv", index=False)
    print("Wrote outputs/ythan_top_node_profiles.csv")


if __name__ == "__main__":
    main()

