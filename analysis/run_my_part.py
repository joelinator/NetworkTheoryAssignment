from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from centrality_analysis import centrality_summary_text, compute_centralities, save_centrality_outputs
from community_embeddings import save_all_embedding_plots, save_kmeans_on_embedding_plots
from extra_tool_analysis import save_extra_tool_outputs
from graph_diagnostics import basal_breakdown, save_reachability_report
from load_network import giant_weakly_connected_subgraph, load_ythan
from motif_analysis import save_motif_outputs
from robustness_analysis import save_robustness_outputs
from species_mapping import annotate_with_species, load_species_mapping
from summarize_results import main as summarize_main
from visualize_network import (
    compute_community_clustered_layout,
    compute_layout,
    plot_network_by_centrality,
    plot_network_by_communities,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Ythan.txt"
MAPPING = ROOT / "data" / "species_mapping.csv"
OUT = ROOT / "outputs"


def write_discussion_summary(
    mapping: pd.DataFrame,
    cent,
    comm,
    troph,
    reach: dict,
    basal_df: pd.DataFrame,
    robust_top: pd.DataFrame,
) -> None:
    top_species = pd.read_csv(OUT / "ythan_top_nodes_with_species.csv")
    lines = [
        "YTHAN FOOD WEB — DISCUSSION SUMMARY (addresses critics + species names)",
        "=" * 72,
        "",
        "LITERATURE: Huxham, Begg & Raffaelli (1996, Oikos) — parasite-rich Ythan estuary web;",
        "parasites increase complexity and disrupt intervality vs cascade-model webs (Cohen et al.).",
        "",
        f"Network: n={reach['n_nodes']}, m={reach['n_edges']}. "
        f"Strongly connected? {reach['is_strongly_connected']}. "
        f"Largest SCC: {reach['largest_scc_size']} nodes ({reach['largest_scc_fraction']:.1%}).",
        reach["closeness_caveat"],
        "",
        "TOP SPECIES BY CENTRALITY (prey → predator):",
    ]
    for measure in top_species["measure"].unique():
        lines.append(f"  [{measure}]")
        sub = top_species[top_species["measure"] == measure].sort_values("rank")
        for _, r in sub.iterrows():
            lines.append(
                f"    {int(r['rank'])}. {r['species']} ({r['category']}) — node {int(r['node'])}"
            )
    lines.extend(
        [
            "",
            "KEY ECOLOGICAL READINGS:",
            "  • Generalist feeders: Littorina saxatilis (118), Macoma balthica (122), Nematoda (124) — high in-degree.",
            "  • Parasite transmission hubs: Cercariae lebouri (3), Cryptocotyle jejuna (5), Parvatrema affine (19) — high out-degree.",
            "  • Connector: Littorina saxatilis (118) — highest betweenness; cascade removal impact lower than key parasites.",
            "  • Phaeophyta / brown algae (132): highest PageRank-in — primary producer channelling",
            "    many trophic paths in this parasite-rich web (Huxham et al.); not a vertebrate apex predator.",
            "  • Intervality: parasite-heavy Ythan webs are known to violate simple cascade/interval models.",
            "",
            f"COMMUNITIES (Louvain): k={len(comm.communities)}, Q={comm.modularity:.3f}.",
            "  Parasite-heavy Ythan web shows modular compartments (bird/mollusc/benthic channels).",
            "  Girvan–Newman first split: Q≈0, degenerate — documented, not used.",
            "",
            f"BASAL NODES (in-degree=0): n={len(troph.basal_nodes)} ({100*len(troph.basal_nodes)/reach['n_nodes']:.0f}% of web).",
            "  High count reflects many parasites without resolved prey links + detritus/POM categories (Huxham et al.).",
        ]
    )
    if len(basal_df):
        lines.append("  Basal by category:")
        for _, r in basal_df.iterrows():
            lines.append(f"    - {r['category']}: {int(r['n_basal'])}")
    lines.extend(
        [
            "",
            "EXTRA TOOLS:",
            "  • Directed motif census vs degree-preserving null (see ythan_motifs_vs_null.csv).",
            "  • Cascade robustness: species whose removal triggers most secondary extinctions.",
            "",
            "TOP ROBUSTNESS (cascade removals):",
        ]
    )
    for _, r in robust_top.head(8).iterrows():
        sp = r.get("species", r["node"])
        lines.append(f"  • {sp}: {int(r['total_extinctions'])} total extinctions")
    lines.append("")
    (OUT / "ythan_discussion_summary.txt").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_species_mapping(MAPPING)

    nd = load_ythan(DATA)
    G = nd.G
    G_wcc = giant_weakly_connected_subgraph(G)

    meta = {
        "dataset": nd.name,
        "path": str(DATA),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "species_mapping": str(MAPPING),
        "node_id_convention": "graph node g -> species node_id g (g>=1); graph 0 -> node_id 1",
        "note": "Edges are prey->predator per dataset_description.md",
    }
    (OUT / "ythan_meta.json").write_text(json.dumps(meta, indent=2))

    reach = save_reachability_report(G_wcc, OUT / "ythan_reachability_report.json")

    cent = compute_centralities(G_wcc)
    save_centrality_outputs(cent, OUT, prefix="ythan")
    cent_annot = annotate_with_species(cent.centralities.reset_index(), mapping)
    cent_annot.to_csv(OUT / "ythan_centralities_with_species.csv", index=False)
    (OUT / "ythan_centrality_summary.txt").write_text(centrality_summary_text(cent))

    comm, troph = save_extra_tool_outputs(G_wcc, OUT, prefix="ythan")
    troph_annot = annotate_with_species(troph.node_table, mapping)
    troph_annot.to_csv(OUT / "ythan_trophic_levels_with_species.csv", index=False)
    basal_df = basal_breakdown(G_wcc, troph_annot)
    basal_df.to_csv(OUT / "ythan_basal_by_category.csv", index=False)

    save_motif_outputs(G_wcc, OUT, prefix="ythan", seed=7)
    robust = save_robustness_outputs(G_wcc, OUT, prefix="ythan", species_df=cent_annot)

    extra_meta = {
        "best_community_method": comm.method,
        "best_modularity_Q": comm.modularity,
        "n_communities": len(comm.communities),
        "basal_nodes_count": len(troph.basal_nodes),
        "basal_fraction": len(troph.basal_nodes) / G_wcc.number_of_nodes(),
        "largest_scc_fraction": reach["largest_scc_fraction"],
    }
    (OUT / "ythan_extra_tool_meta.json").write_text(json.dumps(extra_meta, indent=2))

    pos = compute_layout(G_wcc, seed=7)
    pos_comm = compute_community_clustered_layout(G_wcc, comm.membership, seed=7)
    cdf = cent.centralities.reset_index().rename(columns={"index": "node"})

    centrality_cols = [
        ("in_degree", "viridis"),
        ("out_degree", "magma"),
        ("betweenness", "plasma"),
        ("pagerank_in", "cividis"),
        ("pagerank_out", "cividis"),
    ]

    for col, cmap in centrality_cols:
        values = {int(r["node"]): float(r[col]) for _, r in cdf.iterrows()}
        plot_network_by_centrality(
            G_wcc, pos, values, OUT / f"ythan_network_{col}.png",
            title=f"Ythan — {col}", cmap=cmap, uniform_node_size=False,
        )
        plot_network_by_centrality(
            G_wcc, pos, values, OUT / f"ythan_network_{col}_fixedsize.png",
            title=f"Ythan — {col}", cmap=cmap, uniform_node_size=True,
        )

    plot_network_by_communities(
        G_wcc, pos, comm.membership,
        OUT / f"ythan_network_communities_{comm.method}.png",
        title=f"Ythan — communities ({comm.method}), global layout",
    )
    plot_network_by_communities(
        G_wcc, pos_comm, comm.membership,
        OUT / f"ythan_network_communities_{comm.method}_clustered.png",
        title=f"Ythan — communities ({comm.method}), clustered layout",
    )

    plotted = save_all_embedding_plots(G_wcc, comm.membership, OUT, prefix="ythan", seed=7)
    (OUT / "ythan_embedding_methods.json").write_text(json.dumps({"plotted_methods": plotted}, indent=2))

    kmeans_summary = save_kmeans_on_embedding_plots(
        G_wcc, OUT, prefix="ythan", seed=7, embed_dim_cluster=15, k_min=2, k_max=15
    )

    summarize_main()
    write_discussion_summary(mapping, cent, comm, troph, reach, basal_df, robust)

    print("Done.")
    print(f"Wrote outputs to: {OUT}")
    print(kmeans_summary.to_string(index=False))


if __name__ == "__main__":
    main()
