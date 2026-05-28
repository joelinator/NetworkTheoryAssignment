from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommunityEcologySummary:
    membership_species: pd.DataFrame
    community_summary: pd.DataFrame
    label_counts: pd.DataFrame


_BIRD_GENERA = {
    "Larus",
    "Sterna",
    "Somateria",
    "Corvus",
    "Pluvialis",
    "Calidris",
    "Numenius",
    "Haematopus",
    "Arenaria",
    "Tringa",
    "Branta",
    "Anas",
    "Tadorna",
    "Phalacrocorax",
    "Stercorarius",
    "Rissa",
}

_FISH_GENERA = {
    "Pomatoschistus",
    "Ammodytes",
    "Gasterosteus",
    "Myoxocephalus",
    "Agonus",
    "Pleuronectes",
    "Platichthys",
    "Gadus",
    "Merlangius",
    "Sprattus",
    "Clupea",
}


def _genus(species: str | None) -> str | None:
    if not isinstance(species, str) or not species.strip():
        return None
    # Take the first token that starts with a letter (handles "Enteromorpha sp.")
    m = re.match(r"([A-Za-z][A-Za-z-]*)", species.strip())
    return m.group(1) if m else None


def infer_ecological_label(species: str | None, category: str | None, common_name: str | None) -> str:
    """
    Lightweight label inference from species name.

    This is intentionally conservative: it only labels what we can reasonably
    infer from the mapping table (birds/fish/producer/parasite).
    """
    cat = (category or "").strip().lower()
    if cat == "parasite":
        return "parasite"

    sp = (species or "").strip()
    cn = (common_name or "").strip().lower()
    g = _genus(sp)

    if g in _BIRD_GENERA:
        return "bird"
    if g in _FISH_GENERA:
        return "fish"
    if g in {"Phaeophyta", "Enteromorpha"}:
        return "producer"
    if "phytoplankton" in cn or "algae" in cn:
        return "producer"
    return "other_free_living"


def community_membership_with_species(
    membership: pd.DataFrame,
    mapping: pd.DataFrame,
    annotate_fn,
) -> pd.DataFrame:
    """
    Join community membership with species names.

    `annotate_fn` should be `species_mapping.annotate_with_species`.
    """
    df = annotate_fn(membership.copy(), mapping, node_col="node")
    df["label"] = [
        infer_ecological_label(r.get("species"), r.get("category"), r.get("common_name"))
        for _, r in df.iterrows()
    ]
    return df.sort_values(["community_id", "label", "species", "node"])


def summarize_communities(
    G: nx.DiGraph,
    membership_species: pd.DataFrame,
) -> CommunityEcologySummary:
    mem = membership_species.set_index("node")["community_id"].to_dict()
    nodes = sorted(int(n) for n in G.nodes())

    # Basic community stats
    rows = []
    for cid, grp in membership_species.groupby("community_id"):
        comm_nodes = set(int(n) for n in grp["node"].tolist())
        sub = G.subgraph(comm_nodes)
        # directed edge counts internal vs external
        internal = int(sub.number_of_edges())
        outgoing = int(sum(1 for u, v in G.edges() if int(u) in comm_nodes and int(v) not in comm_nodes))
        incoming = int(sum(1 for u, v in G.edges() if int(u) not in comm_nodes and int(v) in comm_nodes))

        labels = grp["label"].value_counts().to_dict()
        top_species = grp["species"].dropna().astype(str).head(8).tolist()

        rows.append(
            {
                "community_id": int(cid),
                "n_nodes": int(len(comm_nodes)),
                "internal_edges": internal,
                "incoming_edges": incoming,
                "outgoing_edges": outgoing,
                "label_bird": int(labels.get("bird", 0)),
                "label_fish": int(labels.get("fish", 0)),
                "label_producer": int(labels.get("producer", 0)),
                "label_parasite": int(labels.get("parasite", 0)),
                "label_other_free_living": int(labels.get("other_free_living", 0)),
                "top_species_examples": "; ".join(top_species),
            }
        )

    community_summary = pd.DataFrame(rows).sort_values("n_nodes", ascending=False)

    label_counts = (
        membership_species.groupby(["community_id", "label"], as_index=False)
        .agg(n=("node", "count"))
        .sort_values(["community_id", "n"], ascending=[True, False])
    )

    return CommunityEcologySummary(
        membership_species=membership_species,
        community_summary=community_summary,
        label_counts=label_counts,
    )


def save_community_ecology_outputs(
    G: nx.DiGraph,
    membership: pd.DataFrame,
    mapping: pd.DataFrame,
    annotate_fn,
    out_dir: str | Path,
    prefix: str,
) -> CommunityEcologySummary:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ms = community_membership_with_species(membership, mapping, annotate_fn)
    ms.to_csv(out_dir / f"{prefix}_communities_louvain_with_species.csv", index=False)

    summary = summarize_communities(G, ms)
    summary.community_summary.to_csv(out_dir / f"{prefix}_communities_louvain_ecology_summary.csv", index=False)
    summary.label_counts.to_csv(out_dir / f"{prefix}_communities_louvain_label_counts.csv", index=False)

    # Human-readable interpretation scaffold (heuristic; for presentation narrative)
    (out_dir / f"{prefix}_communities_louvain_interpretation.txt").write_text(
        _interpretation_text(summary.community_summary),
        encoding="utf-8",
    )
    return summary


def _module_guess(row: pd.Series) -> str:
    n = float(row.get("n_nodes", 0) or 0)
    if n <= 0:
        return "unknown"
    frac_par = float(row.get("label_parasite", 0)) / n
    frac_bird = float(row.get("label_bird", 0)) / n
    frac_fish = float(row.get("label_fish", 0)) / n
    frac_prod = float(row.get("label_producer", 0)) / n

    # If parasite-heavy, interpret as parasite-host subweb
    if frac_par >= 0.35:
        return "parasite–host module"
    if frac_bird >= 0.25:
        return "avian / top-predator module"
    if frac_fish >= 0.20:
        return "pelagic / fish module"
    if frac_prod >= 0.10:
        return "basal producer channel"
    return "benthic / invertebrate module"


def _interpretation_text(comm_summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("LOUVAIN COMMUNITIES — ECOLOGICAL INTERPRETATION (heuristic from species list)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("How to read:")
    lines.append("- 'module guess' is inferred from species names (birds/fish/producers/parasites).")
    lines.append("- Parasites are expected to reduce modularity by linking hosts across trophic levels (Huxham et al.).")
    lines.append("")

    for _, r in comm_summary.sort_values("community_id").iterrows():
        cid = int(r["community_id"])
        n = int(r["n_nodes"])
        module = _module_guess(r)
        lines.append(f"Community C{cid} (n={n}) — {module}")
        lines.append(f"- internal edges: {int(r['internal_edges'])}; incoming: {int(r['incoming_edges'])}; outgoing: {int(r['outgoing_edges'])}")
        lines.append(
            "- composition: "
            f"{int(r.get('label_bird', 0))} birds, "
            f"{int(r.get('label_fish', 0))} fish, "
            f"{int(r.get('label_producer', 0))} producers, "
            f"{int(r.get('label_parasite', 0))} parasites, "
            f"{int(r.get('label_other_free_living', 0))} other free-living"
        )
        ex = str(r.get("top_species_examples", "") or "").strip()
        if ex:
            lines.append(f"- examples: {ex}")
        lines.append("")

    lines.append("Literature connection (what to say):")
    lines.append("- Q≈0.31 is typical for empirical food webs: real compartmentalization plus cross-module energy flow.")
    lines.append("- Parasites act as 'structural glue' by linking multiple hosts across modules, suppressing Q relative to parasite-free webs.")
    lines.append("")
    return "\n".join(lines)

