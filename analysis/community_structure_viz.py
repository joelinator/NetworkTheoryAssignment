from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib.path import Path as MplPath


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def _community_palette(n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(max(n, 1))]


def plot_community_adjacency_matrix(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    out_path: str | Path,
    title: str = "Adjacency matrix (nodes sorted by community)",
    directed: bool = True,
) -> pd.DataFrame:
    """
    Matrix plot / adjacency matrix with nodes sorted by community.

    Visual goal: dense diagonal blocks for within-community edges.
    Returns a small table with the node ordering and community ids.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mem = membership.set_index("node")["community_id"].to_dict()
    nodes = sorted([int(n) for n in G.nodes()], key=lambda n: (int(mem.get(n, -1)), n))
    comm_ids = [int(mem.get(n, -1)) for n in nodes]

    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)

    if directed:
        for u, v in G.edges():
            iu = idx.get(int(u))
            iv = idx.get(int(v))
            if iu is not None and iv is not None:
                A[iu, iv] = 1
    else:
        for u, v in G.to_undirected().edges():
            iu = idx.get(int(u))
            iv = idx.get(int(v))
            if iu is not None and iv is not None:
                A[iu, iv] = 1
                A[iv, iu] = 1

    fig = plt.figure(figsize=(10.2, 9.2), facecolor="white")
    ax = fig.add_subplot(111)
    ax.imshow(A, cmap="Greys", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("node (sorted)")
    ax.set_ylabel("node (sorted)")

    # Community boundaries (thin colored separators + block rectangles)
    boundaries = []
    start = 0
    for i in range(1, len(nodes) + 1):
        if i == len(nodes) or comm_ids[i] != comm_ids[start]:
            boundaries.append((comm_ids[start], start, i))
            start = i

    palette = _community_palette(len(boundaries))
    for j, (cid, a, b) in enumerate(boundaries):
        ax.axhline(b - 0.5, color="#BBBBBB", linewidth=0.7, alpha=0.7)
        ax.axvline(b - 0.5, color="#BBBBBB", linewidth=0.7, alpha=0.7)
        rect = mpatches.Rectangle(
            (a - 0.5, a - 0.5),
            (b - a),
            (b - a),
            fill=False,
            linewidth=2.0,
            edgecolor=palette[j],
            alpha=0.9,
        )
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return pd.DataFrame({"node": nodes, "community_id": comm_ids, "order": list(range(len(nodes)))})


@dataclass(frozen=True)
class ChordInputs:
    comm_sizes: dict[int, int]
    comm_order: list[int]
    flow: np.ndarray  # k x k (directed) or symmetric


def _community_flow_matrix(G: nx.DiGraph, membership: pd.DataFrame) -> ChordInputs:
    mem = membership.set_index("node")["community_id"].to_dict()
    comm_ids = sorted({int(c) for c in membership["community_id"].unique().tolist() if int(c) >= 0})
    cid_to_i = {cid: i for i, cid in enumerate(comm_ids)}

    sizes: dict[int, int] = {cid: 0 for cid in comm_ids}
    for n in G.nodes():
        cid = int(mem.get(int(n), -1))
        if cid in sizes:
            sizes[cid] += 1

    k = len(comm_ids)
    M = np.zeros((k, k), dtype=float)
    for u, v in G.edges():
        cu = int(mem.get(int(u), -1))
        cv = int(mem.get(int(v), -1))
        if cu in cid_to_i and cv in cid_to_i:
            M[cid_to_i[cu], cid_to_i[cv]] += 1.0

    return ChordInputs(comm_sizes=sizes, comm_order=comm_ids, flow=M)


def plot_community_chord_diagram(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    out_path: str | Path,
    title: str = "Chord diagram (community-to-community edges)",
    min_fraction: float = 0.02,
    symmetric: bool = True,
) -> pd.DataFrame:
    """
    Chord diagram at the *community* level (no node clutter).

    - Communities are arcs around a circle (arc length ∝ community size).
    - Ribbons connect communities; thickness ∝ number of edges between them.
    - `symmetric=True` aggregates i→j and j→i into one ribbon (cleaner for talks).

    Returns a table of inter-community edge totals.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ci = _community_flow_matrix(G, membership)
    comm_ids = ci.comm_order
    sizes = ci.comm_sizes
    F = ci.flow.copy()
    if symmetric:
        F = F + F.T
        np.fill_diagonal(F, 0.0)

    k = len(comm_ids)
    if k == 0:
        raise ValueError("No communities found in membership table.")

    # Totals and thresholding
    total = float(F.sum())
    if total <= 0:
        total = 1.0
    threshold = min_fraction * total

    palette = _community_palette(k)

    # Arc allocation by community size
    size_arr = np.array([sizes.get(cid, 1) for cid in comm_ids], dtype=float)
    size_arr = np.clip(size_arr, 1.0, None)
    frac = size_arr / size_arr.sum()

    # Geometry
    R = 1.0
    arc_pad = 0.015 * 2 * np.pi
    angles = []
    start = 0.0
    for f in frac:
        span = float(f) * 2 * np.pi
        a0 = start + arc_pad
        a1 = start + span - arc_pad
        angles.append((a0, a1))
        start += span

    def pol(theta: float, r: float) -> np.ndarray:
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    # Build a simple ribbon as a cubic Bezier from arc midpoint to arc midpoint.
    def ribbon(ax: plt.Axes, a: float, b: float, w: float, color, alpha=0.50) -> None:
        p0 = pol(a, R * 0.98)
        p3 = pol(b, R * 0.98)
        c1 = pol(a, R * 0.25)
        c2 = pol(b, R * 0.25)
        verts = [p0, c1, c2, p3]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=w,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(patch)

    fig = plt.figure(figsize=(10.5, 10.5), facecolor="white")
    ax = fig.add_subplot(111)
    _style_axes(ax)

    # Draw community arcs and labels
    for i, cid in enumerate(comm_ids):
        a0, a1 = angles[i]
        arc = mpatches.Arc(
            (0, 0),
            2 * R,
            2 * R,
            angle=0,
            theta1=np.degrees(a0),
            theta2=np.degrees(a1),
            linewidth=10.0,
            color=palette[i],
            alpha=0.95,
            capstyle="round",
        )
        ax.add_patch(arc)

        mid = 0.5 * (a0 + a1)
        label_pos = pol(mid, R * 1.10)
        ax.text(
            label_pos[0],
            label_pos[1],
            f"C{cid}\n(n={sizes.get(cid, 0)})",
            ha="center",
            va="center",
            fontsize=10,
            color="#1f1f1f",
        )

    # Draw ribbons between community midpoints
    mids = [0.5 * (a0 + a1) for (a0, a1) in angles]
    max_w = float(F.max()) if float(F.max()) > 0 else 1.0
    for i in range(k):
        for j in range(i + 1, k):
            w = float(F[i, j])
            if w < threshold:
                continue
            # line width: emphasize differences but keep readable
            lw = 1.5 + 10.0 * (w / max_w) ** 0.75
            col = mcolors.to_rgba(palette[i], alpha=0.55)
            ribbon(ax, mids[i], mids[j], lw, col, alpha=0.55)

    ax.set_title(title, fontsize=15, pad=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    # Return flow table (by community id)
    rows = []
    for i, ci_id in enumerate(comm_ids):
        for j, cj_id in enumerate(comm_ids):
            if ci_id == cj_id:
                continue
            rows.append({"from_community": ci_id, "to_community": cj_id, "edges": float(ci.flow[i, j])})
    return pd.DataFrame(rows).sort_values("edges", ascending=False)

