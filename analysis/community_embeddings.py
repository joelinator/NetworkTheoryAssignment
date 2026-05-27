from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, TSNE


def _undirected_projection(G: nx.DiGraph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from((u, v) for u, v in G.edges())
    return UG


def _sorted_nodes(UG: nx.Graph) -> list[int]:
    return [int(n) for n in sorted(UG.nodes())]


def _adjacency_matrix(UG: nx.Graph, nodes: list[int]) -> np.ndarray:
    return nx.to_numpy_array(UG, nodelist=nodes, dtype=float)


def _shortest_path_distance_matrix(UG: nx.Graph, nodes: list[int]) -> np.ndarray:
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    D = np.full((n, n), fill_value=float(n), dtype=float)
    np.fill_diagonal(D, 0.0)
    spl = dict(nx.all_pairs_shortest_path_length(UG))
    for u in nodes:
        for v, d in spl[u].items():
            if v in idx:
                D[idx[u], idx[v]] = float(d)
    return D


def embed_laplacian_eigen(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """
    Spectral embedding via the graph Laplacian (eigenvectors of normalized Laplacian).

    Intuition: coordinates place nodes so that linked nodes are close — communities
    often form clusters in this space.
    """
    nodes = _sorted_nodes(UG)
    A = _adjacency_matrix(UG, nodes)
    model = SpectralEmbedding(
        n_components=dim,
        affinity="precomputed",
        random_state=seed,
        n_jobs=1,
    )
    X = model.fit_transform(A)
    return nodes, X


def embed_adjacency_eigen(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """
    Use leading eigenvectors of the (undirected) adjacency matrix as coordinates.

    Math: A v = λ v; node i coordinate k is v_k(i).
  """
    nodes = _sorted_nodes(UG)
    A = _adjacency_matrix(UG, nodes)
    # Symmetric adjacency → real eigenvectors
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]
    # Skip the trivial top mode if nearly constant; take next `dim` vectors
    X = evecs[:, order[1 : dim + 1]]
    if X.shape[1] < dim:
        pad = np.zeros((len(nodes), dim - X.shape[1]))
        X = np.hstack([X, pad])
    return nodes, X


def embed_pca_adjacency(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """
    PCA on adjacency rows (each node's neighborhood profile).
    """
    nodes = _sorted_nodes(UG)
    A = _adjacency_matrix(UG, nodes)
    X = PCA(n_components=dim, random_state=seed).fit_transform(A)
    return nodes, X


def embed_mds(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """
    Classical MDS on shortest-path distances.
    Intuition: preserve graph distances in low-dimensional space.
    """
    nodes = _sorted_nodes(UG)
    D = _shortest_path_distance_matrix(UG, nodes)
    X = MDS(n_components=dim, dissimilarity="precomputed", random_state=seed, normalized_stress="auto").fit_transform(D)
    return nodes, X


def embed_isomap(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    nodes = _sorted_nodes(UG)
    D = _shortest_path_distance_matrix(UG, nodes)
    X = Isomap(n_components=dim, metric="precomputed").fit_transform(D)
    return nodes, X


def embed_tsne(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    """
    t-SNE on PCA-preprocessed adjacency rows (stable for n≈134).
    """
    nodes = _sorted_nodes(UG)
    A = _adjacency_matrix(UG, nodes)
    pre = PCA(n_components=min(10, len(nodes) - 1), random_state=seed).fit_transform(A)
    X = TSNE(
        n_components=dim,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, len(nodes) - 1),
    ).fit_transform(pre)
    return nodes, X


def embed_umap(UG: nx.Graph, dim: int, seed: int = 7) -> tuple[list[int], np.ndarray]:
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is not installed") from e

    nodes = _sorted_nodes(UG)
    A = _adjacency_matrix(UG, nodes)
    X = umap.UMAP(n_components=dim, random_state=seed, metric="cosine").fit_transform(A)
    return nodes, X


EMBEDDING_METHODS: dict[str, Callable[..., tuple[list[int], np.ndarray]]] = {
    "laplacian_eigen": embed_laplacian_eigen,
    "adjacency_eigen": embed_adjacency_eigen,
    "pca_adjacency": embed_pca_adjacency,
    "mds": embed_mds,
    "isomap": embed_isomap,
    "tsne": embed_tsne,
    "umap": embed_umap,
}


def _community_colors(membership: pd.DataFrame, nodes: list[int]) -> list:
    mem = membership.set_index("node")["community_id"].to_dict()
    cmap = plt.get_cmap("tab20")
    colors = []
    for n in nodes:
        cid = int(mem.get(int(n), -1))
        colors.append("#BDBDBD" if cid < 0 else cmap(cid % 20))
    return colors


def _draw_edges_2d(ax, G: nx.DiGraph, nodes: list[int], coords: np.ndarray, idx: dict[int, int], alpha: float = 0.08) -> None:
    for u, v in G.edges():
        if int(u) not in idx or int(v) not in idx:
            continue
        iu, iv = idx[int(u)], idx[int(v)]
        ax.plot(
            [coords[iu, 0], coords[iv, 0]],
            [coords[iu, 1], coords[iv, 1]],
            color="#333333",
            alpha=alpha,
            linewidth=0.35,
            zorder=1,
        )


def _draw_edges_3d(ax, G: nx.DiGraph, coords: np.ndarray, idx: dict[int, int], alpha: float = 0.06) -> None:
    for u, v in G.edges():
        if int(u) not in idx or int(v) not in idx:
            continue
        iu, iv = idx[int(u)], idx[int(v)]
        ax.plot(
            [coords[iu, 0], coords[iv, 0]],
            [coords[iu, 1], coords[iv, 1]],
            [coords[iu, 2], coords[iv, 2]],
            color="#444444",
            alpha=alpha,
            linewidth=0.3,
        )


def plot_embedding_2d(
    G: nx.DiGraph,
    nodes: list[int],
    coords: np.ndarray,
    membership: pd.DataFrame,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = {n: i for i, n in enumerate(nodes)}
    colors = _community_colors(membership, nodes)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    _draw_edges_2d(ax, G, nodes, coords, idx)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=55, edgecolors="#1f1f1f", linewidths=0.5, zorder=3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_embedding_3d(
    G: nx.DiGraph,
    nodes: list[int],
    coords: np.ndarray,
    membership: pd.DataFrame,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = {n: i for i, n in enumerate(nodes)}
    colors = _community_colors(membership, nodes)

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _draw_edges_3d(ax, G, coords, idx)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=45, edgecolors="#1f1f1f", linewidths=0.4, depthshade=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_all_embedding_plots(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    out_dir: str | Path,
    prefix: str,
    seed: int = 7,
) -> list[str]:
    """
    Compute and save 2D + 3D embedding plots for every available method.
    Returns list of methods successfully plotted.
    """
    out_dir = Path(out_dir)
    plotted: list[str] = []
    UG = _undirected_projection(G)

    for name, fn in EMBEDDING_METHODS.items():
        try:
            nodes, X2 = fn(UG, 2, seed=seed)
            _, X3 = fn(UG, 3, seed=seed)
        except ImportError:
            continue

        plot_embedding_2d(
            G,
            nodes,
            X2,
            membership,
            out_dir / f"{prefix}_embed2d_{name}.png",
            title=f"Community embedding 2D — {name}",
        )
        plot_embedding_3d(
            G,
            nodes,
            X3,
            membership,
            out_dir / f"{prefix}_embed3d_{name}.png",
            title=f"Community embedding 3D — {name}",
        )
        plotted.append(name)
    return plotted
