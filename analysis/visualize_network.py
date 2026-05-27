from __future__ import annotations

from dataclasses import dataclass
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
    """
    Layout strategy (beauty + stability):
    - Use an undirected projection for layout stability.
    - Use spring layout with fixed seed for reproducibility.
    """
    UG = _undirected_projection(G)
    pos = nx.spring_layout(UG, seed=seed, k=None, iterations=400)
    # Ensure int keys for serialization / merging
    return {int(k): np.asarray(v) for k, v in pos.items()}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def _quantile_clip(x: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> np.ndarray:
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if hi <= lo:
        return x
    return np.clip(x, lo, hi)


def plot_network_by_centrality(
    G: nx.DiGraph,
    pos: dict[int, np.ndarray],
    values: dict[int, float],
    out_path: str | Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    """
    Beautiful network plot:
    - node size encodes centrality (log-scaled via quantile clipping)
    - node color encodes centrality (continuous colormap)
    - edges are faint (structure context without clutter)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    v = np.array([float(values.get(int(n), 0.0)) for n in nodes], dtype=float)
    v_clip = _quantile_clip(v)
    # size mapping (avoid giant hubs dominating)
    v_norm = (v_clip - v_clip.min()) / (v_clip.max() - v_clip.min() + 1e-12)
    sizes = 60 + 900 * (v_norm**1.6)

    fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
    ax = fig.add_subplot(111)
    _style_axes(ax)

    # Edges (directed). Keep arrows subtle to avoid clutter at m=593.
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

    # Nodes
    nodes_artist = nx.draw_networkx_nodes(
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

    ax.set_title(title, fontsize=15, pad=14)

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=mpl.colors.Normalize(vmin=float(v.min()), vmax=float(v.max())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_network_by_communities(
    G: nx.DiGraph,
    pos: dict[int, np.ndarray],
    membership: pd.DataFrame,  # columns: node, community_id
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    mem = membership.set_index("node")["community_id"].to_dict()
    comm_ids = sorted(set(int(mem.get(int(n), -1)) for n in nodes))
    n_comm = len([c for c in comm_ids if c >= 0])

    # Discrete palette with many distinct colors
    cmap = plt.get_cmap("tab20")
    colors = []
    for n in nodes:
        cid = int(mem.get(int(n), -1))
        if cid < 0:
            colors.append("#BDBDBD")
        else:
            colors.append(cmap(cid % 20))

    fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
    ax = fig.add_subplot(111)
    _style_axes(ax)

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

