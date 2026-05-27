from __future__ import annotations

import json
from pathlib import Path

from centrality_analysis import centrality_summary_text, compute_centralities, save_centrality_outputs
from community_embeddings import save_all_embedding_plots
from extra_tool_analysis import save_extra_tool_outputs
from load_network import giant_weakly_connected_subgraph, load_ythan
from visualize_network import (
    compute_community_clustered_layout,
    compute_layout,
    plot_network_by_centrality,
    plot_network_by_communities,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Ythan.txt"
OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    nd = load_ythan(DATA)
    G = nd.G
    G_wcc = giant_weakly_connected_subgraph(G)

    meta = {
        "dataset": nd.name,
        "path": str(DATA),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_nodes_giant_wcc": int(G_wcc.number_of_nodes()),
        "n_edges_giant_wcc": int(G_wcc.number_of_edges()),
        "note": "Edges are prey->predator per dataset_description.md",
    }
    (OUT / "ythan_meta.json").write_text(json.dumps(meta, indent=2))

    cent = compute_centralities(G_wcc)
    save_centrality_outputs(cent, OUT, prefix="ythan")
    (OUT / "ythan_centrality_summary.txt").write_text(centrality_summary_text(cent))

    comm, troph = save_extra_tool_outputs(G_wcc, OUT, prefix="ythan")
    extra_meta = {
        "best_community_method": comm.method,
        "best_modularity_Q": comm.modularity,
        "n_communities": len(comm.communities),
        "basal_nodes_count": len(troph.basal_nodes),
    }
    (OUT / "ythan_extra_tool_meta.json").write_text(json.dumps(extra_meta, indent=2))

    pos = compute_layout(G_wcc, seed=7)
    pos_comm = compute_community_clustered_layout(G_wcc, comm.membership, seed=7)
    cdf = cent.centralities.reset_index().rename(columns={"index": "node"})

    centrality_cols = [
        ("in_degree", "viridis"),
        ("out_degree", "magma"),
        ("betweenness", "plasma"),
        ("closeness_in", "viridis"),
        ("closeness_out", "magma"),
        ("pagerank_in", "cividis"),
        ("pagerank_out", "cividis"),
    ]

    for col, cmap in centrality_cols:
        values = {int(r["node"]): float(r[col]) for _, r in cdf.iterrows()}
        plot_network_by_centrality(
            G_wcc,
            pos,
            values,
            out_path=OUT / f"ythan_network_{col}.png",
            title=f"Ythan food web — {col}",
            cmap=cmap,
            uniform_node_size=False,
        )
        plot_network_by_centrality(
            G_wcc,
            pos,
            values,
            out_path=OUT / f"ythan_network_{col}_fixedsize.png",
            title=f"Ythan food web — {col}",
            cmap=cmap,
            uniform_node_size=True,
        )

    plot_network_by_communities(
        G_wcc,
        pos,
        membership=comm.membership,
        out_path=OUT / f"ythan_network_communities_{comm.method}.png",
        title=f"Ythan food web — communities ({comm.method}), global layout",
    )
    plot_network_by_communities(
        G_wcc,
        pos_comm,
        membership=comm.membership,
        out_path=OUT / f"ythan_network_communities_{comm.method}_clustered.png",
        title=f"Ythan food web — communities ({comm.method}), clustered layout",
    )

    plotted = save_all_embedding_plots(G_wcc, comm.membership, OUT, prefix="ythan", seed=7)
    (OUT / "ythan_embedding_methods.json").write_text(json.dumps({"plotted_methods": plotted}, indent=2))

    print("Done.")
    print(f"Wrote outputs to: {OUT}")
    print(f"Embedding methods plotted: {plotted}")


if __name__ == "__main__":
    main()
