from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommunityResult:
    method: str
    communities: list[set[int]]
    modularity: float
    membership: pd.DataFrame  # node, community_id


def _to_undirected(G: nx.DiGraph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from((u, v) for u, v in G.edges())
    return UG


def _membership_from_communities(communities: list[set[int]]) -> pd.DataFrame:
    membership_rows = []
    for cid, comm in enumerate(communities):
        for node in comm:
            membership_rows.append({"node": int(node), "community_id": int(cid)})
    return pd.DataFrame(membership_rows).sort_values(["community_id", "node"])


def detect_communities_undirected(G: nx.DiGraph, method: str) -> CommunityResult:
    """
    Community detection as an "extra tool".

    We use an undirected projection of the food web (common in exploratory analysis),
    then apply one of several NetworkX community algorithms.

    Supported methods:
    - 'louvain' (modularity optimization; good baseline)
    - 'greedy_modularity' (fast modularity heuristic)
    - 'label_propagation' (diffusion-like; no objective)
    - 'fluid' (asynchronous fluid communities; requires connected graph)
    - 'girvan_newman' (edge betweenness divisive; returns hierarchical splits)
    """
    UG = _to_undirected(G)

    method = method.strip().lower()
    if method == "greedy_modularity":
        communities = list(nx.algorithms.community.greedy_modularity_communities(UG))
    elif method == "label_propagation":
        communities = list(nx.algorithms.community.asyn_lpa_communities(UG, seed=7))
    elif method == "fluid":
        # Choose k by heuristic: ~sqrt(n) but clamp to [2, 12] for readability.
        if not nx.is_connected(UG):
            # FluidC requires connected; fall back to greedy.
            communities = list(nx.algorithms.community.greedy_modularity_communities(UG))
            method = "fluid(fallback_greedy_disconnected)"
        else:
            k = int(max(2, min(12, round(np.sqrt(UG.number_of_nodes())))))
            communities = list(nx.algorithms.community.asyn_fluidc(UG, k=k, seed=7))
            method = f"fluid(k={k})"
    elif method == "girvan_newman":
        # Take the first split (2 communities) for a simple, explainable result.
        gen = nx.algorithms.community.girvan_newman(UG)
        first = next(gen)
        communities = [set(c) for c in first]
    elif method == "louvain":
        # NetworkX 3.x includes Louvain. Keep robust with fallback.
        try:
            communities = list(nx.algorithms.community.louvain_communities(UG, seed=7))
        except Exception:
            communities = list(nx.algorithms.community.greedy_modularity_communities(UG))
            method = "louvain(fallback_greedy_unavailable)"
    else:
        raise ValueError(f"Unknown community method: {method}")

    modularity = float(nx.algorithms.community.modularity(UG, communities))
    membership = _membership_from_communities(communities)
    return CommunityResult(method=method, communities=communities, modularity=modularity, membership=membership)


@dataclass(frozen=True)
class TrophicResult:
    node_table: pd.DataFrame  # node, in_degree, out_degree, basal, trophic_level
    basal_nodes: list[int]


def estimate_trophic_levels(G: nx.DiGraph) -> TrophicResult:
    """
    Simple trophic-level estimate for a prey->predator food web.

    Basal nodes: in_degree == 0 (no prey; primary producers/detritus-like).
    Trophic level TL is estimated via:
        TL_i = 1 + mean(TL_preys(i))
    solved iteratively (Gauss-Seidel style). This is a simplified variant commonly used
    for descriptive purposes when detailed diet fractions are unavailable.
    """
    nodes = list(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    basal = {n: (in_deg[n] == 0) for n in nodes}
    basal_nodes = [int(n) for n in nodes if basal[n]]

    # Initialize TL: basal=1, others=2
    TL = {n: (1.0 if basal[n] else 2.0) for n in nodes}

    # Prey of predator i are its predecessors (edges prey->predator)
    max_iter = 5000
    tol = 1e-10
    for _ in range(max_iter):
        max_delta = 0.0
        for i in nodes:
            if basal[i]:
                continue
            preys = list(G.predecessors(i))
            if not preys:
                # Should be basal, but keep safe:
                new = 1.0
            else:
                new = 1.0 + float(np.mean([TL[p] for p in preys]))
            delta = abs(new - TL[i])
            if delta > max_delta:
                max_delta = delta
            TL[i] = new
        if max_delta < tol:
            break

    df = pd.DataFrame(
        {
            "node": [int(n) for n in nodes],
            "in_degree": [int(in_deg[n]) for n in nodes],
            "out_degree": [int(out_deg[n]) for n in nodes],
            "basal": [bool(basal[n]) for n in nodes],
            "trophic_level": [float(TL[n]) for n in nodes],
        }
    ).sort_values("node")
    return TrophicResult(node_table=df, basal_nodes=basal_nodes)


def save_extra_tool_outputs(
    G: nx.DiGraph,
    out_dir: str | Path,
    prefix: str,
) -> tuple[CommunityResult, TrophicResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multiple community algorithms for comparison
    methods = ["louvain", "greedy_modularity", "label_propagation", "fluid", "girvan_newman"]
    comm_results: list[CommunityResult] = []
    for m in methods:
        comm_results.append(detect_communities_undirected(G, method=m))

    # Save memberships + size plots per method
    summary_rows = []
    for cr in comm_results:
        safe_method = cr.method.replace("/", "_").replace(" ", "_")
        cr.membership.to_csv(out_dir / f"{prefix}_communities_{safe_method}_membership.csv", index=False)
        sizes = sorted([len(c) for c in cr.communities], reverse=True)
        pd.DataFrame({"community_id": list(range(len(cr.communities))), "size": [len(c) for c in cr.communities]}).to_csv(
            out_dir / f"{prefix}_communities_{safe_method}_sizes.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(range(1, len(sizes) + 1), sizes, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Community sizes — {cr.method} (Q={cr.modularity:.3f})")
        ax.set_xlabel("community rank (by size)")
        ax.set_ylabel("size (#nodes)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_community_sizes_{safe_method}.png", dpi=200)
        plt.close(fig)

        note = ""
        if "girvan_newman" in cr.method:
            note = (
                "First GN split is degenerate (one giant block + isolates); "
                "Q≈0 — not used for interpretation. Prefer Louvain/greedy."
            )
        summary_rows.append(
            {
                "method": cr.method,
                "n_communities": len(cr.communities),
                "modularity_Q": cr.modularity,
                "largest_community_size": max(sizes) if sizes else 0,
                "note": note,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("modularity_Q", ascending=False)
    summary.to_csv(out_dir / f"{prefix}_communities_methods_summary.csv", index=False)

    # For reporting: exclude Girvan–Newman from "best method" comparison
    comm_results_compare = [c for c in comm_results if "girvan" not in c.method.lower()]

    troph = estimate_trophic_levels(G)
    troph.node_table.to_csv(out_dir / f"{prefix}_trophic_levels.csv", index=False)

    # Plot trophic level histogram
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(troph.node_table["trophic_level"], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_title("Trophic level distribution (simple estimate)")
    ax.set_xlabel("trophic level")
    ax.set_ylabel("#nodes")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_trophic_level_hist.png", dpi=200)
    plt.close(fig)

    # Return the best-modularity partition (excluding degenerate GN first split)
    best = max(comm_results_compare, key=lambda r: r.modularity)
    return best, troph

