from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# Names for the 16 directed 3-node motifs (node order in census tuple)
MOTIF_NAMES = {
    "003": "empty",
    "012": "single edge",
    "102": "mutual dyad + isolate",
    "021D": "out-chain",
    "021U": "in-chain",
    "021C": "fan-out",
    "111D": "out-chain + mutual",
    "111U": "in-chain + mutual",
    "030T": "transitive triangle",
    "030C": "cyclic triangle",
    "201": "mutual + directed third",
    "120D": "one mutual, one in-edge",
    "120U": "one mutual, one out-edge",
    "120C": "one mutual, one bidirectional",
    "210": "two mutual dyads",
    "300": "3-cycle all mutual",
}


def directed_triad_census(G: nx.DiGraph) -> pd.DataFrame:
    """Count directed 3-node motifs (NetworkX triadic_census)."""
    census = nx.triadic_census(G)
    rows = [{"motif": k, "name": MOTIF_NAMES.get(k, k), "count": int(v)} for k, v in census.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df["fraction"] = df["count"] / df["count"].sum()
    return df


def compare_to_random_null(
    G: nx.DiGraph,
    n_samples: int = 200,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Compare motif counts to directed configuration-model random graphs
    (same in/out degree sequences).
    """
    rng = np.random.default_rng(seed)
    observed = directed_triad_census(G)
    motif_keys = observed["motif"].tolist()

    # directed configuration model via networkx
    in_seq = [d for _, d in G.in_degree()]
    out_seq = [d for _, d in G.out_degree()]

    null_counts = {m: [] for m in motif_keys}
    for _ in range(n_samples):
        try:
            R = nx.directed_configuration_model(in_seq, out_seq, seed=int(rng.integers(1e9)))
            R = nx.DiGraph(R)
            R.remove_edges_from(nx.selfloop_edges(R))
            c = nx.triadic_census(R)
            for m in motif_keys:
                null_counts[m].append(c[m])
        except Exception:
            continue

    rows = []
    for _, row in observed.iterrows():
        m = row["motif"]
        samples = null_counts[m]
        if not samples:
            continue
        mu = float(np.mean(samples))
        sd = float(np.std(samples)) or 1e-12
        z = (row["count"] - mu) / sd
        rows.append(
            {
                "motif": m,
                "name": row["name"],
                "observed": row["count"],
                "null_mean": mu,
                "null_std": sd,
                "z_score": z,
                "fraction": row["fraction"],
            }
        )
    return pd.DataFrame(rows).sort_values("z_score", key=abs, ascending=False)


def save_motif_outputs(G: nx.DiGraph, out_dir: str | Path, prefix: str, seed: int = 7) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = directed_triad_census(G)
    obs.to_csv(out_dir / f"{prefix}_motifs_observed.csv", index=False)

    null = compare_to_random_null(G, n_samples=150, seed=seed)
    null.to_csv(out_dir / f"{prefix}_motifs_vs_null.csv", index=False)

    top = obs.nlargest(8, "count")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["motif"], top["count"], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("count")
    ax.set_title("Directed triad motif census (top 8)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_motifs_bar.png", dpi=200)
    plt.close(fig)

    sig = null.head(8)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#C44E52" if z > 0 else "#55A868" for z in sig["z_score"]]
    ax.barh(sig["motif"], sig["z_score"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("z-score vs degree-preserving null")
    ax.set_title("Motifs over/under-represented (|z| largest)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_motifs_zscores.png", dpi=200)
    plt.close(fig)
