from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from sklearn.cluster import KMeans
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
    pre_dim = min(len(nodes) - 1, max(dim, 2))
    pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(A)
    # Barnes-Hut t-SNE only supports n_components <= 3; use exact method for higher dims.
    method = "exact" if dim > 3 else "barnes_hut"
    X = TSNE(
        n_components=dim,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, len(nodes) - 1),
        method=method,
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


# Graph/spectral embeddings — used for k-means clustering (NOT PCA/t-SNE on adjacency).
KMEANS_EMBEDDING_METHODS: dict[str, Callable[..., tuple[list[int], np.ndarray]]] = {
    "laplacian_eigen": embed_laplacian_eigen,
    "adjacency_eigen": embed_adjacency_eigen,
    "mds": embed_mds,
    "isomap": embed_isomap,
}

# All embedders (includes optional viz-only helpers for Louvain-colored plots).
EMBEDDING_METHODS: dict[str, Callable[..., tuple[list[int], np.ndarray]]] = {
    **KMEANS_EMBEDDING_METHODS,
    "pca_adjacency": embed_pca_adjacency,
    "tsne": embed_tsne,
    "umap": embed_umap,
}


def project_embedding_pca(X_hi: np.ndarray, dim: int, seed: int = 7) -> np.ndarray:
    """PCA projection of an embedding matrix — visualization only."""
    dim = min(dim, X_hi.shape[0] - 1, X_hi.shape[1])
    return PCA(n_components=dim, random_state=seed).fit_transform(X_hi)


def project_embedding_tsne_2d(X_hi: np.ndarray, seed: int = 7) -> np.ndarray:
    """t-SNE 2D projection of an embedding matrix — visualization only."""
    perplexity = min(30, X_hi.shape[0] - 1)
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        method="barnes_hut",
    ).fit_transform(X_hi)


def _labels_to_communities(nodes: list[int], labels: np.ndarray) -> list[set[int]]:
    communities: list[set[int]] = []
    for cid in sorted(np.unique(labels)):
        communities.append({nodes[i] for i in range(len(nodes)) if labels[i] == cid})
    return communities


def modularity_for_labels(UG: nx.Graph, nodes: list[int], labels: np.ndarray) -> float:
    communities = _labels_to_communities(nodes, labels)
    return float(nx.algorithms.community.modularity(UG, communities))


def membership_from_labels(nodes: list[int], labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"node": nodes, "community_id": labels.astype(int)})


def select_k_by_modularity(
    UG: nx.Graph,
    nodes: list[int],
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    seed: int = 7,
) -> tuple[int, np.ndarray, float, pd.DataFrame]:
    """
    Run k-means for each k in [k_min, k_max] on embedding X, score partitions by modularity Q,
    return the best k and its labels.
    """
    k_max = min(k_max, len(nodes) - 1)
    if k_max < k_min:
        raise ValueError("Not enough nodes to scan k values.")

    rows: list[dict] = []
    best_q = -np.inf
    best_k = k_min
    best_labels = np.zeros(len(nodes), dtype=int)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        q = modularity_for_labels(UG, nodes, labels)
        rows.append({"k": k, "modularity_Q": q})
        if q > best_q:
            best_q = q
            best_k = k
            best_labels = labels

    return best_k, best_labels, float(best_q), pd.DataFrame(rows)


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


def save_kmeans_on_embedding_plots(
    G: nx.DiGraph,
    out_dir: str | Path,
    prefix: str,
    seed: int = 7,
    embed_dim_cluster: int = 15,
    k_min: int = 2,
    k_max: int = 15,
) -> pd.DataFrame:
    """
    For each graph embedding (Laplacian / adjacency eigen / MDS / Isomap):
      1) embed nodes in `embed_dim_cluster` dimensions
      2) k-means on that embedding; scan k and pick best by modularity Q
      3) visualize with PCA and t-SNE projections to 2D/3D (viz only, not for clustering)
    """
    out_dir = Path(out_dir)
    UG = _undirected_projection(G)
    n = UG.number_of_nodes()
    dim_hi = min(embed_dim_cluster, max(2, n - 1))

    summary_rows: list[dict] = []

    for name, fn in KMEANS_EMBEDDING_METHODS.items():
        nodes, X_hi = fn(UG, dim_hi, seed=seed)

        best_k, labels, best_q, scan = select_k_by_modularity(
            UG, nodes, X_hi, k_min=k_min, k_max=k_max, seed=seed
        )
        scan.to_csv(out_dir / f"{prefix}_kmeans_Q_scan_{name}.csv", index=False)

        membership = membership_from_labels(nodes, labels)
        tag = f"{name}_kmeans_k{best_k}"

        # 2D/3D coords are projections of the embedding used for k-means (not separate embedders).
        X2_pca = project_embedding_pca(X_hi, 2, seed=seed)
        X3_pca = project_embedding_pca(X_hi, 3, seed=seed)
        X2_tsne = project_embedding_tsne_2d(X_hi, seed=seed)

        base_title = f"K-means on {name} ({dim_hi}D) — k={best_k}, Q={best_q:.3f}"

        plot_embedding_2d(
            G,
            nodes,
            X2_pca,
            membership,
            out_dir / f"{prefix}_embed2d_{tag}_vizpca.png",
            title=f"{base_title} | 2D PCA projection",
        )
        plot_embedding_2d(
            G,
            nodes,
            X2_tsne,
            membership,
            out_dir / f"{prefix}_embed2d_{tag}_viztsne.png",
            title=f"{base_title} | 2D t-SNE projection",
        )
        plot_embedding_3d(
            G,
            nodes,
            X3_pca,
            membership,
            out_dir / f"{prefix}_embed3d_{tag}_vizpca.png",
            title=f"{base_title} | 3D PCA projection",
        )

        summary_rows.append(
            {
                "embedding_method": name,
                "embed_dim_for_kmeans": dim_hi,
                "best_k": int(best_k),
                "best_modularity_Q": best_q,
                "viz_2d": "pca+tsne",
                "viz_3d": "pca",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("best_modularity_Q", ascending=False)
    summary.to_csv(out_dir / f"{prefix}_kmeans_embedding_summary.csv", index=False)
    return summary


def save_all_embedding_plots(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    out_dir: str | Path,
    prefix: str,
    seed: int = 7,
    embed_dim: int = 15,
) -> list[str]:
    """
    Louvain-colored plots: build graph embeddings, then PCA/t-SNE for 2D/3D display only.
    """
    out_dir = Path(out_dir)
    plotted: list[str] = []
    UG = _undirected_projection(G)
    dim_hi = min(embed_dim, max(2, UG.number_of_nodes() - 1))

    for name, fn in KMEANS_EMBEDDING_METHODS.items():
        nodes, X_hi = fn(UG, dim_hi, seed=seed)
        X2_pca = project_embedding_pca(X_hi, 2, seed=seed)
        X3_pca = project_embedding_pca(X_hi, 3, seed=seed)
        X2_tsne = project_embedding_tsne_2d(X_hi, seed=seed)

        plot_embedding_2d(
            G,
            nodes,
            X2_pca,
            membership,
            out_dir / f"{prefix}_embed2d_{name}_vizpca.png",
            title=f"{name} ({dim_hi}D) — 2D PCA (Louvain colors)",
        )
        plot_embedding_2d(
            G,
            nodes,
            X2_tsne,
            membership,
            out_dir / f"{prefix}_embed2d_{name}_viztsne.png",
            title=f"{name} ({dim_hi}D) — 2D t-SNE (Louvain colors)",
        )
        plot_embedding_3d(
            G,
            nodes,
            X3_pca,
            membership,
            out_dir / f"{prefix}_embed3d_{name}_vizpca.png",
            title=f"{name} ({dim_hi}D) — 3D PCA (Louvain colors)",
        )
        plotted.append(name)
    return plotted
