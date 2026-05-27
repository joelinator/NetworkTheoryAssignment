from __future__ import annotations

from pathlib import Path

import pandas as pd

from species_mapping import annotate_with_species, graph_to_csv_node_id, load_species_mapping


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MAPPING = ROOT / "species_mapping.csv"


def main() -> None:
    mapping = load_species_mapping(MAPPING)
    cent = pd.read_csv(OUT / "ythan_centralities.csv")
    troph = pd.read_csv(OUT / "ythan_trophic_levels.csv")
    comm = pd.read_csv(OUT / "ythan_communities_louvain_membership.csv")
    top = pd.read_csv(OUT / "ythan_top_nodes.csv")

    profiles = annotate_with_species(
        cent.merge(troph, on="node", how="left", suffixes=("", "_troph"))
        .merge(comm, on="node", how="left"),
        mapping,
    )

    focus_nodes = sorted(top["node"].unique().tolist())
    focus = profiles[profiles["node"].isin(focus_nodes)].copy()
    ranks = top.pivot_table(index="node", columns="measure", values="rank", aggfunc="min").reset_index()
    ranks = ranks.rename(columns={c: f"rank_{c}" for c in ranks.columns if c != "node"})
    focus = focus.merge(ranks, on="node", how="left")

    focus.to_csv(OUT / "ythan_top_node_profiles.csv", index=False)

    # Human-readable top nodes with species for slides
    rows = []
    for measure, grp in top.groupby("measure"):
        g = grp.copy()
        g["map_id"] = g["node"].astype(int).map(graph_to_csv_node_id)
        g = g.merge(mapping, left_on="map_id", right_on="node_id", how="left")
        for _, r in g.iterrows():
            rows.append(
                {
                    "measure": measure,
                    "rank": int(r["rank"]),
                    "node": int(r["node"]),
                    "species": r.get("species_name", ""),
                    "category": r.get("category", ""),
                    "value": float(r["value"]),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "ythan_top_nodes_with_species.csv", index=False)
    print("Wrote outputs/ythan_top_node_profiles.csv")
    print("Wrote outputs/ythan_top_nodes_with_species.csv")


if __name__ == "__main__":
    main()
