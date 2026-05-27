from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CentralityResult:
    centralities: pd.DataFrame
    top_nodes: dict[str, list[int]]
    correlations: pd.DataFrame


def compute_centralities(G: nx.DiGraph) -> CentralityResult:
    """
    Compute >=3 centrality measures, including a spectral measure.

    Notes for directed food webs:
    - `in_degree` ~ "number of prey items" (generality / predators' diet breadth) given prey->predator direction.
    - `out_degree` ~ "number of predators consuming this species" (vulnerability).
    - For *directed* measures, we compute both "in" and "out" variants where meaningful:
      - in/out closeness (reachability-to vs reachability-from)
      - PageRank on G vs PageRank on reversed G (a spectral analogue of “in vs out importance”)
    - `betweenness` is computed on the unweighted directed structure (common default).
    """
    nodes = list(G.nodes())

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    betw = nx.betweenness_centrality(G, normalized=True, weight=None)

    # Directed closeness (caveat: food webs are rarely strongly connected; scores use
    # reachable pairs only and can be inflated — see reachability report in outputs).
    closeness_out = nx.closeness_centrality(G)
    closeness_in = nx.closeness_centrality(G.reverse(copy=False))

    # Harmonic centrality: uses sum of inverse distances and is more stable when many
    # directed paths are missing (recommended supplement for food webs).
    harmonic_out = nx.harmonic_centrality(G)
    harmonic_in = nx.harmonic_centrality(G.reverse(copy=False))

    # Spectral measures: PageRank on G and on reversed G.
    # (With prey->predator orientation, PR(G) tends to emphasize high-level predators/sinks;
    # PR(G^R) tends to emphasize influential sources / “prey-side importance”.)
    pr_in = nx.pagerank(G, alpha=0.85, weight="weight")
    pr_out = nx.pagerank(G.reverse(copy=False), alpha=0.85, weight="weight")

    df = pd.DataFrame(
        {
            "node": nodes,
            "in_degree": [in_deg[n] for n in nodes],
            "out_degree": [out_deg[n] for n in nodes],
            "betweenness": [betw[n] for n in nodes],
            "closeness_in": [closeness_in[n] for n in nodes],
            "closeness_out": [closeness_out[n] for n in nodes],
            "harmonic_in": [harmonic_in[n] for n in nodes],
            "harmonic_out": [harmonic_out[n] for n in nodes],
            "pagerank_in": [pr_in[n] for n in nodes],
            "pagerank_out": [pr_out[n] for n in nodes],
        }
    ).set_index("node")

    # Rank / top-k
    top_nodes: dict[str, list[int]] = {}
    for col in [
        "in_degree",
        "out_degree",
        "betweenness",
        "closeness_in",
        "closeness_out",
        "harmonic_in",
        "harmonic_out",
        "pagerank_in",
        "pagerank_out",
    ]:
        top_nodes[col] = (
            df[col]
            .sort_values(ascending=False)
            .head(3)
            .index.astype(int)
            .tolist()
        )

    corr = df[
        [
            "in_degree",
            "out_degree",
            "betweenness",
            "closeness_in",
            "closeness_out",
            "harmonic_in",
            "harmonic_out",
            "pagerank_in",
            "pagerank_out",
        ]
    ].corr(method="spearman")
    return CentralityResult(centralities=df, top_nodes=top_nodes, correlations=corr)


def save_centrality_outputs(
    result: CentralityResult,
    out_dir: str | Path,
    prefix: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_centralities.csv"
    result.centralities.reset_index().to_csv(csv_path, index=False)

    top_rows = []
    for measure, nodes in result.top_nodes.items():
        for rank, node in enumerate(nodes, start=1):
            top_rows.append(
                {
                    "measure": measure,
                    "rank": rank,
                    "node": node,
                    "value": float(result.centralities.loc[node, measure]),
                }
            )
    pd.DataFrame(top_rows).to_csv(out_dir / f"{prefix}_top_nodes.csv", index=False)

    result.correlations.to_csv(out_dir / f"{prefix}_centrality_spearman_corr.csv")

    # Plot: top-10 per measure bar charts
    for col in [
        "in_degree",
        "out_degree",
        "betweenness",
        "closeness_in",
        "closeness_out",
        "harmonic_in",
        "harmonic_out",
        "pagerank_in",
        "pagerank_out",
    ]:
        s = result.centralities[col].sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([str(i) for i in s.index], s.values)
        ax.set_title(f"Top 10 nodes by {col}")
        ax.set_xlabel("node id")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_top10_{col}.png", dpi=200)
        plt.close(fig)

    # Plot: correlation heatmap
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(result.correlations.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(result.correlations.shape[1]), result.correlations.columns, rotation=45, ha="right")
    ax.set_yticks(range(result.correlations.shape[0]), result.correlations.index)
    for i in range(result.correlations.shape[0]):
        for j in range(result.correlations.shape[1]):
            ax.text(j, i, f"{result.correlations.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Spearman correlations between centralities")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_centrality_corr_heatmap.png", dpi=200)
    plt.close(fig)


def centrality_summary_text(result: CentralityResult) -> str:
    lines: list[str] = []
    lines.append("Top nodes (IDs) by measure:")
    for m, nodes in result.top_nodes.items():
        lines.append(f"- {m}: {nodes}")
    lines.append("")
    lines.append("Spearman correlations:")
    lines.append(result.correlations.round(3).to_string())
    return "\n".join(lines)

