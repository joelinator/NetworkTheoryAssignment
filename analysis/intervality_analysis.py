from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd


def competition_graph_from_food_web(G: nx.DiGraph) -> nx.Graph:
    """
    Undirected competition graph: edge between predators that share at least one prey.

    Prey→predator orientation: prey are predecessors of predators.
  """
    prey_sets = {int(n): {int(p) for p in G.predecessors(n)} for n in G.nodes()}
    C = nx.Graph()
    C.add_nodes_from(G.nodes())
    nodes = list(G.nodes())
    for i, j in combinations(nodes, 2):
        if prey_sets[int(i)] & prey_sets[int(j)]:
            C.add_edge(int(i), int(j))
    return C


def _has_asteroidal_triple(G: nx.Graph) -> bool:
    """True if G contains an asteroidal triple (interval graphs are AT-free)."""
    nodes = list(G.nodes())
    nbrs = {v: set(G.neighbors(v)) for v in nodes}

    for a, b, c in combinations(nodes, 3):
        na, nb, nc = nbrs[a], nbrs[b], nbrs[c]
        # Asteroidal: for each vertex in the triple, a path between the other two
        # avoids the closed neighbourhood of that vertex.
        if _pair_avoids_neighbourhood(G, b, c, na):
            if _pair_avoids_neighbourhood(G, a, c, nb):
                if _pair_avoids_neighbourhood(G, a, b, nc):
                    return True
    return False


def _pair_avoids_neighbourhood(
    G: nx.Graph, u: int, v: int, forbidden: set[int]
) -> bool:
    """Is there a u–v path whose internal vertices are not in forbidden ∪ {u, v}?"""
    blocked = set(forbidden) | {u, v}
    H = G.copy()
    H.remove_nodes_from(blocked)
    if u not in H or v not in H:
        return False
    return nx.has_path(H, u, v)


def is_interval_graph(G: nx.Graph) -> bool:
    """
    Interval-graph recognition: chordal and asteroidal-triple-free.

    (Standard characterization; see e.g. Lekkerkerker & Boland 1962.)
    """
    if G.number_of_nodes() == 0:
        return True
    if not nx.is_chordal(G):
        return False
    return not _has_asteroidal_triple(G)


def intervality_report(G: nx.DiGraph) -> dict:
    C = competition_graph_from_food_web(G)
    chordal = nx.is_chordal(C)
    interval = is_interval_graph(C)
    at = _has_asteroidal_triple(C) if chordal else None
    return {
        "competition_n_nodes": int(C.number_of_nodes()),
        "competition_n_edges": int(C.number_of_edges()),
        "competition_density": float(
            2 * C.number_of_edges() / (C.number_of_nodes() * (C.number_of_nodes() - 1))
            if C.number_of_nodes() > 1
            else 0.0
        ),
        "is_chordal": bool(chordal),
        "has_asteroidal_triple": at,
        "is_interval": bool(interval),
        "cascade_interval_hypothesis": (
            "Consistent with cascade / interval model"
            if interval
            else "Violates interval hypothesis (non-interval competition graph)"
        ),
        "interpretation": (
            "Predator diet overlaps cannot be nested on a single niche axis without "
            "contradiction — expected for parasite-rich Ythan webs (Huxham et al. 1996)."
            if not interval
            else "Diet-overlap structure is consistent with an interval / cascade representation."
        ),
    }


def save_intervality_outputs(
    G: nx.DiGraph,
    out_dir: str | Path,
    prefix: str,
) -> dict:
    out_dir = Path(out_dir)
    report = intervality_report(G)
    (out_dir / f"{prefix}_intervality.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([report]).to_csv(out_dir / f"{prefix}_intervality.csv", index=False)
    return report
