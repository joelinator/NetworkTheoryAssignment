from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _undirected_projection(G: nx.DiGraph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from((u, v) for u, v in G.edges())
    return UG


def compute_layout(G: nx.DiGraph, seed: int = 7) -> dict[int, np.ndarray]:
    """Global spring layout on undirected projection (reproducible)."""
    UG = _undirected_projection(G)
    pos = nx.spring_layout(UG, seed=seed, k=None, iterations=400)
    return {int(k): np.asarray(v) for k, v in pos.items()}


def compute_community_clustered_layout(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    seed: int = 7,
    community_scale: float = 1.2,
    cluster_spacing: float = 4.0,
) -> dict[int, np.ndarray]:
    """
    Layout that places each community in its own region.

    Strategy:
    1) Run a local spring layout inside each community subgraph.
    2) Place community centers on a circle and translate local layouts.

    Intuition: nodes in the same community appear close; different communities are separated.
    """
    UG = _undirected_projection(G)
    mem = membership.set_index("node")["community_id"].to_dict()

    communities: dict[int, list[int]] = {}
    for n in UG.nodes():
        cid = int(mem.get(int(n), -1))
        communities.setdefault(cid, []).append(int(n))

    comm_ids = sorted(communities.keys())
    k = max(len(comm_ids), 1)
    centers = {}
    for i, cid in enumerate(comm_ids):
        angle = 2 * np.pi * i / k
        centers[cid] = cluster_spacing * np.array([np.cos(angle), np.sin(angle)])

    pos: dict[int, np.ndarray] = {}
    for cid, nodes in communities.items():
        sub = UG.subgraph(nodes)
        if sub.number_of_nodes() == 1:
            local = {nodes[0]: np.array([0.0, 0.0])}
        else:
            local = nx.spring_layout(sub, seed=seed + cid, k=0.9, iterations=250)
            local = {int(n): np.asarray(p) for n, p in local.items()}
        local_arr = np.vstack(list(local.values()))
        local_arr = (local_arr - local_arr.mean(axis=0)) * community_scale
        center = centers.get(cid, np.zeros(2))
        for j, n in enumerate(local.keys()):
            pos[n] = local_arr[j] + center

    return pos


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def _draw_directed_edges(G: nx.DiGraph, pos: dict[int, np.ndarray], ax: plt.Axes) -> None:
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=7,
        width=0.55,
        alpha=0.09,
        edge_color="#000000",
        connectionstyle="arc3,rad=0.04",
    )


def plot_network_by_centrality(
    G: nx.DiGraph,
    pos: dict[int, np.ndarray],
    values: dict[int, float],
    out_path: str | Path,
    title: str,
    cmap: str = "viridis",
    uniform_node_size: bool = False,
    node_size: float = 220,
) -> None:
    """
    Network plot with centrality encoded by color (colormap gradient).

    - uniform_node_size=False: size also reflects centrality (default)
    - uniform_node_size=True: fixed node size, color only
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    v = np.array([float(values.get(int(n), 0.0)) for n in nodes], dtype=float)

    if uniform_node_size:
        sizes = node_size
    else:
        lo, hi = np.quantile(v, 0.02), np.quantile(v, 0.98)
        v_clip = np.clip(v, lo, hi) if hi > lo else v
        v_norm = (v_clip - v_clip.min()) / (v_clip.max() - v_clip.min() + 1e-12)
        sizes = 60 + 900 * (v_norm**1.6)

    fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    _draw_directed_edges(G, pos, ax)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=v,
        cmap=plt.get_cmap(cmap),
        linewidths=0.6,
        edgecolors="#1f1f1f",
        alpha=0.96,
    )

    suffix = " (fixed size, color = centrality)" if uniform_node_size else " (size + color = centrality)"
    ax.set_title(title + suffix, fontsize=14, pad=14)

    sm = mpl.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.Normalize(vmin=float(v.min()), vmax=float(v.max())),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_network_by_communities(
    G: nx.DiGraph,
    pos: dict[int, np.ndarray],
    membership: pd.DataFrame,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    mem = membership.set_index("node")["community_id"].to_dict()
    comm_ids = sorted(set(int(mem.get(int(n), -1)) for n in nodes))
    n_comm = len([c for c in comm_ids if c >= 0])

    cmap = plt.get_cmap("tab20")
    colors = []
    for n in nodes:
        cid = int(mem.get(int(n), -1))
        colors.append("#BDBDBD" if cid < 0 else cmap(cid % 20))

    fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    _draw_directed_edges(G, pos, ax)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=220,
        node_color=colors,
        linewidths=0.7,
        edgecolors="#1f1f1f",
        alpha=0.96,
    )

    ax.set_title(f"{title} (k={n_comm})", fontsize=15, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
