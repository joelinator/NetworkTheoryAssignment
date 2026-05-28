#!/usr/bin/env python3
"""Regenerate Ythan_Network_Analysis.ipynb from the current analysis pipeline."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


NOTEBOOK = Path(__file__).resolve().parents[1] / "Ythan_Network_Analysis.ipynb"

cells: list[dict] = []

cells.append(
    md(
        """# Ythan Estuary Food Web — Full Network Analysis

**Assignment (our part):** node centralities + extra tools on the Ythan food web.

**Literature:** Huxham, Begg & Raffaelli (1996, *Oikos*) — parasite-rich estuary web.

**Edge convention:** `prey → predator` (edge *A → B* means *A* is eaten by *B*).

| Measure | Meaning |
|---------|---------|
| **in-degree** | Number of **prey** → **generality** (diet breadth) |
| **out-degree** | Number of **predators** → **vulnerability** |

**Run:** execute all cells top-to-bottom in **conda (base)**. Heavy steps match `analysis/run_my_part.py`.

**Docs:** `methods_and_interpretation_my_part.md`, `presentation_my_part_script.md`
"""
    )
)

cells.append(
    code(
        """# Optional: clone repo if you only have this notebook
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/joelinator/NetworkTheoryAssignment.git"
CLONE_DIR = Path.cwd() / "NetworkTheoryAssignment"

if not (Path.cwd() / "analysis").exists() and not CLONE_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(CLONE_DIR)], check=True)
    os.chdir(CLONE_DIR)
elif CLONE_DIR.exists() and not (Path.cwd() / "analysis").exists():
    os.chdir(CLONE_DIR)
print("Working directory:", Path.cwd().resolve())"""
    )
)

cells.append(
    code(
        """# Setup
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
if not (ROOT / "analysis").exists():
    raise FileNotFoundError("Open notebook from repo root (need analysis/ folder).")

sys.path.insert(0, str(ROOT / "analysis"))

from centrality_analysis import centrality_summary_text, compute_centralities, save_centrality_outputs
from community_embeddings import save_all_embedding_plots, save_kmeans_on_embedding_plots
from extra_tool_analysis import save_extra_tool_outputs
from graph_diagnostics import basal_breakdown, save_reachability_report
from intervality_analysis import save_intervality_outputs
from load_network import giant_weakly_connected_subgraph, load_ythan
from motif_analysis import save_motif_outputs
from random_graph_comparison import save_random_graph_comparison
from removal_impact import save_removal_impact_outputs
from robustness_analysis import save_robustness_outputs
from species_mapping import annotate_with_species, load_species_mapping
from summarize_results import main as summarize_results
from community_structure_viz import plot_community_adjacency_matrix, plot_community_chord_diagram
from community_ecology_analysis import save_community_ecology_outputs
from visualize_network import (
    compute_community_clustered_layout,
    compute_layout,
    plot_network_by_centrality,
    plot_network_by_communities,
)

DATA = ROOT / "data" / "Ythan.txt"
MAPPING = ROOT / "data" / "species_mapping.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

%matplotlib inline
plt.rcParams["figure.dpi"] = 110

def show_img(path, title=None):
    p = Path(path)
    if not p.exists():
        display(Markdown(f"*Missing:* `{p}`"))
        return
    if title:
        display(Markdown(f"### {title}"))
    display(Image(filename=p))

def show_csv(path, n=10):
    p = Path(path)
    if p.exists():
        display(pd.read_csv(p).head(n))
    else:
        display(Markdown(f"*Missing:* `{p}`"))

def gallery(glob_pattern, title=None, max_images=None):
    paths = sorted(OUT.glob(glob_pattern))
    if max_images:
        paths = paths[:max_images]
    if title:
        display(Markdown(f"## {title} ({len(paths)} files)"))
    for p in paths:
        show_img(p)

print("ROOT:", ROOT.resolve())"""
    )
)

cells.append(md("## 1. Load network & species mapping"))

cells.append(
    code(
        """mapping = load_species_mapping(MAPPING)
nd = load_ythan(DATA)
G = nd.G
G_wcc = giant_weakly_connected_subgraph(G)

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Giant WCC: {G_wcc.number_of_nodes()} nodes, {G_wcc.number_of_edges()} edges")
display(nd.edges.head())"""
    )
)

cells.append(md("## 2. Reachability (closeness caveat)"))

cells.append(
    code(
        """reach = save_reachability_report(G_wcc, OUT / "ythan_reachability_report.json")
display(pd.DataFrame([reach]).T.rename(columns={0: "value"}))"""
    )
)

cells.append(md("## 3. Centralities (all measures + species names)"))

cells.append(
    code(
        """cent = compute_centralities(G_wcc)
save_centrality_outputs(cent, OUT, prefix="ythan")

cent_annot = annotate_with_species(cent.centralities.reset_index(), mapping)
cent_annot.to_csv(OUT / "ythan_centralities_with_species.csv", index=False)
(OUT / "ythan_centrality_summary.txt").write_text(centrality_summary_text(cent))

print(centrality_summary_text(cent))
display(cent_annot.sort_values("pagerank_in", ascending=False).head(12))"""
    )
)

cells.append(
    code(
        """summarize_results()
top_species = pd.read_csv(OUT / "ythan_top_nodes_with_species.csv")
display(top_species.sort_values(["measure", "rank"]))

show_img(OUT / "ythan_centrality_corr_heatmap.png", "Centrality correlation heatmap")
corr = pd.read_csv(OUT / "ythan_centrality_spearman_corr.csv", index_col=0)
display(corr.round(2))"""
    )
)

cells.append(
    md(
        """**Two niche axes (MacArthur–Levins):**
- **Generality:** in-degree, PageRank-in, closeness-out, harmonic-out (ρ ≈ 0.98)
- **Vulnerability:** out-degree, PageRank-out, closeness-in, harmonic-in (ρ ≈ 0.91)

*Littorina saxatilis*: in-degree **57** (prey types consumed), out-degree **8** (predators)."""
    )
)

cells.append(md("### 3a. Centrality network maps (fixed node size)"))

cells.append(
    code(
        """pos = compute_layout(G_wcc, seed=7)
cdf = cent.centralities.reset_index().rename(columns={"index": "node"})

for col, cmap in [
    ("in_degree", "viridis"),
    ("out_degree", "magma"),
    ("betweenness", "plasma"),
    ("closeness_in", "viridis"),
    ("closeness_out", "viridis"),
    ("harmonic_in", "viridis"),
    ("harmonic_out", "viridis"),
    ("pagerank_in", "cividis"),
    ("pagerank_out", "cividis"),
]:
    values = {int(r["node"]): float(r[col]) for _, r in cdf.iterrows()}
    out = OUT / f"ythan_network_{col}_fixedsize.png"
    plot_network_by_centrality(
        G_wcc, pos, values, out,
        title=f"Ythan — {col}", cmap=cmap, uniform_node_size=True,
    )
    show_img(out, col)"""
    )
)

cells.append(md("### 3b. Top-10 bar charts per centrality"))

cells.append(code("""gallery("ythan_top10_*.png", "Top-10 centrality bar charts")"""))

cells.append(md("## 4. Community detection"))

cells.append(
    code(
        """comm, troph = save_extra_tool_outputs(G_wcc, OUT, prefix="ythan")
troph_annot = annotate_with_species(troph.node_table, mapping)
troph_annot.to_csv(OUT / "ythan_trophic_levels_with_species.csv", index=False)
basal_df = basal_breakdown(G_wcc, troph_annot)
basal_df.to_csv(OUT / "ythan_basal_by_category.csv", index=False)

display(pd.read_csv(OUT / "ythan_communities_methods_summary.csv").sort_values("modularity_Q", ascending=False))
print(f"Best: {comm.method}, k={len(comm.communities)}, Q={comm.modularity:.3f}")
display(basal_df)"""
    )
)

cells.append(md("## 4a. Louvain community membership with species names (CSV)"))

cells.append(
    code(
        """comm_ecology = save_community_ecology_outputs(
    G_wcc,
    comm.membership,
    mapping,
    annotate_with_species,
    OUT,
    prefix="ythan",
)

show_csv(OUT / "ythan_communities_louvain_with_species.csv", n=15)
display(pd.read_csv(OUT / "ythan_communities_louvain_ecology_summary.csv"))
display(pd.read_csv(OUT / "ythan_communities_louvain_label_counts.csv").head(25))"""
    )
)

cells.append(md("### 4a(i). Community ecological interpretation scaffold"))

cells.append(
    code(
        """p = OUT / "ythan_communities_louvain_interpretation.txt"
if p.exists():
    print(p.read_text())
else:
    print("Missing:", p)"""
    )
)

cells.append(md("## 4b. Community structure visualizations (matrix + chord)"))

cells.append(
    code(
        """# Adjacency matrix (nodes sorted by community) and chord diagram (community interconnections)
plot_community_adjacency_matrix(
    G_wcc,
    comm.membership,
    OUT / "ythan_community_adjacency_matrix.png",
    title="Ythan — adjacency matrix (sorted by Louvain community)",
    directed=True,
)
show_img(OUT / "ythan_community_adjacency_matrix.png")

plot_community_chord_diagram(
    G_wcc,
    comm.membership,
    OUT / "ythan_community_chord.png",
    title="Ythan — chord diagram (community interconnections)",
    min_fraction=0.02,
    symmetric=True,
)
show_img(OUT / "ythan_community_chord.png")"""
    )
)

cells.append(
    code(
        """pos_comm = compute_community_clustered_layout(G_wcc, comm.membership, seed=7)
p_global = OUT / f"ythan_network_communities_{comm.method}.png"
p_clust = OUT / f"ythan_network_communities_{comm.method}_clustered.png"
plot_network_by_communities(G_wcc, pos, comm.membership, p_global,
    title=f"Communities ({comm.method}), global layout")
plot_network_by_communities(G_wcc, pos_comm, comm.membership, p_clust,
    title=f"Communities ({comm.method}), clustered layout")
show_img(p_global)
show_img(p_clust)
gallery("ythan_community_sizes_*.png", "Community size distributions")"""
    )
)

cells.append(md("## 5. Graph embeddings & k-means"))

cells.append(
    code(
        """methods = save_all_embedding_plots(G_wcc, comm.membership, OUT, prefix="ythan", seed=7)
(OUT / "ythan_embedding_methods.json").write_text(json.dumps({"plotted_methods": methods}, indent=2))

kmeans_summary = save_kmeans_on_embedding_plots(
    G_wcc, OUT, prefix="ythan", seed=7, embed_dim_cluster=15, k_min=2, k_max=15,
)
display(kmeans_summary)

best = kmeans_summary.iloc[0]
bm, bk = best["embedding_method"], int(best["best_k"])
show_img(OUT / f"ythan_embed2d_{bm}_kmeans_k{bk}_vizpca.png", f"Best k-means: {bm}, k={bk} (PCA viz)")
show_img(OUT / f"ythan_embed2d_{bm}_kmeans_k{bk}_viztsne.png", "t-SNE viz (display only)")"""
    )
)

cells.append(
    code(
        """# 2D embeddings (PCA projection) for each method
for m in methods:
    show_img(OUT / f"ythan_embed2d_{m}_vizpca.png", f"2D embedding — {m}")"""
    )
)

cells.append(md("## 6. Trophic levels"))

cells.append(
    code(
        """print(f"Basal nodes: {len(troph.basal_nodes)} ({100*len(troph.basal_nodes)/G_wcc.number_of_nodes():.0f}%)")
display(troph_annot.sort_values("trophic_level", ascending=False).head(12))
show_img(OUT / "ythan_trophic_level_hist.png", "Trophic level histogram")"""
    )
)

cells.append(md("## 7. Motif census vs null"))

cells.append(
    code(
        """save_motif_outputs(G_wcc, OUT, prefix="ythan", seed=7)
motifs = pd.read_csv(OUT / "ythan_motifs_vs_null.csv").sort_values("z_score", key=abs, ascending=False)
display(motifs.head(12))
show_img(OUT / "ythan_motifs_bar.png", "Observed vs null motif counts")
show_img(OUT / "ythan_motifs_zscores.png", "Motif z-scores")"""
    )
)

cells.append(md("## 8. Cascade robustness"))

cells.append(
    code(
        """robust = save_robustness_outputs(G_wcc, OUT, prefix="ythan", species_df=cent_annot)
display(robust.head(15))
show_img(OUT / "ythan_robustness_top15.png", "Top cascade impacts")"""
    )
)

cells.append(md("## 8b. Removal impact (betweenness vs cascade)"))

cells.append(
    code(
        """removal = save_removal_impact_outputs(G_wcc, OUT, prefix="ythan", species_df=cent_annot)
display(removal)"""
    )
)

cells.append(md("## 9. Intervality (competition graph)"))

cells.append(
    code(
        """interval = save_intervality_outputs(G_wcc, OUT, prefix="ythan")
display(pd.DataFrame([interval]).T.rename(columns={0: "value"}))"""
    )
)

cells.append(md("## 10. Random graph comparison"))

cells.append(
    code(
        """rg = save_random_graph_comparison(G_wcc, OUT, prefix="ythan", seed=7)
display(rg)"""
    )
)

cells.append(md("## 11. Metadata & discussion summary"))

cells.append(
    code(
        """from run_my_part import write_discussion_summary

extra_meta = {
    "best_community_method": comm.method,
    "best_modularity_Q": comm.modularity,
    "n_communities": len(comm.communities),
    "basal_nodes_count": len(troph.basal_nodes),
    "basal_fraction": len(troph.basal_nodes) / G_wcc.number_of_nodes(),
    "is_interval_competition_graph": interval["is_interval"],
}
(OUT / "ythan_extra_tool_meta.json").write_text(json.dumps(extra_meta, indent=2))
display(pd.DataFrame([extra_meta]))

write_discussion_summary(reach, comm, troph, basal_df, robust, interval, removal)
print((OUT / "ythan_discussion_summary.txt").read_text())"""
    )
)

cells.append(md("## 12. Takeaways"))

cells.append(
    md(
        """1. **Generality ≠ vulnerability** — two centrality axes (MacArthur–Levins).
2. ***Littorina saxatilis*** — 57 prey, top betweenness; WCC splits on removal but few cascade extinctions.
3. **Parasites** — high vulnerability & robustness (*Podocotyle*, *Catatropis*, *Cryptocotyle lingua*).
4. ***Phaeophyta*** — PageRank-in hub (producer), not apex predator.
5. **Louvain** Q ≈ 0.31, 5 communities; **motifs** → hierarchical chains; **non-interval** competition graph.
6. All outputs in `outputs/` — 130+ files including embeddings and community CSVs."""
    )
)

cells.append(
    code(
        """# Optional: list every PNG written to outputs/
pngs = sorted(OUT.glob("*.png"))
print(f"Total PNG files: {len(pngs)}")
for p in pngs:
    print(p.name)"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

NOTEBOOK.write_text(json.dumps(nb, indent=1))
print("Wrote", NOTEBOOK, "with", len(cells), "cells")
