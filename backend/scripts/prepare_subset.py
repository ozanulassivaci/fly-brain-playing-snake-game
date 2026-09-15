"""
Derives the Phase 0 candidate functional subset (motion pathway + central
complex + descending neurons) from the locally downloaded MaleCNS v1.0
tables and writes two outputs that must stay index-consistent with each
other:

  - frontend/assets/brain-subset.json   (3D points for the brain panel)
  - data/processed/subset.npz           (edge list + cluster labels for
                                          the backend LIF simulation)

Run once from the repo root: python backend/scripts/prepare_subset.py

Reuses the exact filters validated in docs/architecture-plan.md (Phase 0):
motion pathway by type-name prefix, central complex by real ROI synapse
weight (EB/FB/PB/NO), descending neurons by superclass. See that doc for
why each filter looks the way it does.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
FRONTEND_ASSETS = REPO_ROOT / "frontend" / "assets"

MOTION_PATTERN = re.compile(r"^(T4|T5|LC\d|LPLC\d|LT\d)")
CORE_CX_ROI_PATTERN = re.compile(r"^(EB|FB|PB(\(|$)|NO$|NO\()")
MIN_CX_SYNWEIGHT = 5
# FC (fan-shaped body columnar, real goal-direction cell types) instance
# strings look like "FC1E_C4_L" — the "_C{1-9}_" is real fan-shaped-body
# column position, matching the real anatomical column count. Used for
# goal-direction injection (Phase 3.2) — see docs/architecture-plan.md for
# why this replaced Phase 3.1's generic PB-glomerulus ring (wrong anatomical
# target: that ring represents current heading, not goal).
FC_COLUMN_PATTERN = re.compile(r"_C(\d)_")

# EPG ("E-PG", the actual identified compass/heading cell type) instance
# strings look like "EPG(PB08)_R2" — same PB-glomerulus ring Phase 3.1 used,
# but Phase 3.1's mistake was injecting *goal* into this ring (wrong: EPG
# represents current heading, not goal) and reading out via all DN types.
# Phase 3.2 re-adds this ring restricted to EPG specifically (real data
# check: 50 EPG neurons, 25 L / 25 R, glomeruli 1-9, matching this pattern
# for all of them) and uses it only for heading injection, alongside the
# separate FC goal ring above — PFL neurons receive real synapses from both
# (FC->PFL: 1585 edges; EPG->PFL: 247 edges) and are the ones anatomically
# wired to compare them.
HEADING_RING_PATTERN = re.compile(r"\(PB\d+\)_([LR])(\d)")


def load_annotations() -> pd.DataFrame:
    ann = pd.read_feather(DATA_RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    return ann[ann["status"] == "Traced"].copy()


def load_roi_info() -> pd.DataFrame:
    # bodyId/type/superclass/status/roiInfo only — see data/raw/README-ish
    # note in docs/architecture-plan.md: this is a column-projected copy of
    # the full Neuprint_Neurons.feather (the only table with real roiInfo).
    df = pd.read_feather(DATA_RAW / "neurons_roi_subset.feather")
    return df[df["status"] == "Traced"].copy()


def has_core_cx_roi(roi_info_json: str) -> bool:
    if not isinstance(roi_info_json, str) or not roi_info_json:
        return False
    try:
        rois = json.loads(roi_info_json)
    except (json.JSONDecodeError, TypeError):
        return False
    for roi, stats in rois.items():
        if CORE_CX_ROI_PATTERN.match(roi) and stats.get("synweight", 0) >= MIN_CX_SYNWEIGHT:
            return True
    return False


def build_subset() -> pd.DataFrame:
    traced = load_annotations()
    roi_df = load_roi_info()[["bodyId", "roiInfo"]]

    motion_types = {t for t in traced["type"].dropna().unique() if MOTION_PATTERN.match(str(t))}
    motion_mask = traced["type"].isin(motion_types)
    dn_mask = traced["superclass"] == "descending_neuron"

    cx_ids = set(roi_df.loc[roi_df["roiInfo"].apply(has_core_cx_roi), "bodyId"])
    cx_mask = traced["bodyId"].isin(cx_ids)

    def cluster_for(row) -> str | None:
        if row["superclass"] == "descending_neuron":
            return "dn"
        if row["bodyId"] in cx_ids:
            return "cx"
        if row["type"] in motion_types:
            return "motion"
        return None

    subset = traced[motion_mask | cx_mask | dn_mask].copy()
    subset["cluster"] = subset.apply(cluster_for, axis=1)
    subset = subset[subset["somaLocation"].notna()].reset_index(drop=True)
    subset["subset_index"] = subset.index
    return subset


def write_frontend_points(subset: pd.DataFrame) -> None:
    coords = np.array([list(loc) for loc in subset["somaLocation"]], dtype=float)
    coords -= coords.mean(axis=0)
    coords /= np.abs(coords).max()

    points = [
        {"x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4), "c": cluster}
        for (x, y, z), cluster in zip(coords, subset["cluster"])
    ]
    FRONTEND_ASSETS.mkdir(parents=True, exist_ok=True)
    out_path = FRONTEND_ASSETS / "brain-subset.json"
    with open(out_path, "w") as f:
        json.dump({"points": points}, f)
    print(f"wrote {out_path} ({len(points)} points, {out_path.stat().st_size / 1024:.0f} KB)")


def write_backend_edges(subset: pd.DataFrame) -> None:
    body_to_index = dict(zip(subset["bodyId"], subset["subset_index"]))
    weights = pd.read_feather(
        DATA_RAW / "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"
    )
    internal = weights[weights["body_pre"].isin(body_to_index) & weights["body_post"].isin(body_to_index)]

    edge_pre = internal["body_pre"].map(body_to_index).to_numpy(dtype=np.int32)
    edge_post = internal["body_post"].map(body_to_index).to_numpy(dtype=np.int32)
    edge_weight = internal["weight"].to_numpy(dtype=np.float32)
    cluster = subset["cluster"].to_numpy()
    # Phase 3: real per-neuron type (for T4/T5 a/b/c/d direction-tuned
    # sensory targeting) and soma side (for descending-neuron L/R motor
    # readout) — see docs/architecture-plan.md for why these are real,
    # usable biological structure rather than made-up labels.
    neuron_type = subset["type"].fillna("").to_numpy(dtype=str)
    soma_side = subset["somaSide"].fillna("").to_numpy(dtype=str)

    def fc_column_for(row) -> int:
        if not str(row["type"]).startswith("FC"):
            return -1
        m = FC_COLUMN_PATTERN.search(str(row["instance"]))
        return int(m.group(1)) - 1 if m else -1

    def heading_ring_for(row) -> int:
        if not str(row["type"]).startswith("EPG"):
            return -1
        m = HEADING_RING_PATTERN.search(str(row["instance"]))
        if not m:
            return -1
        side, num = m.group(1), int(m.group(2))
        return (num - 1) + (9 if side == "L" else 0)

    fc_column = subset.apply(fc_column_for, axis=1).to_numpy(dtype=np.int32)
    heading_ring = subset.apply(heading_ring_for, axis=1).to_numpy(dtype=np.int32)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "subset.npz"
    np.savez(
        out_path,
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_weight=edge_weight,
        cluster=cluster,
        neuron_type=neuron_type,
        soma_side=soma_side,
        fc_column=fc_column,
        heading_ring=heading_ring,
        n_neurons=len(subset),
    )
    print(f"wrote {out_path} ({len(subset)} neurons, {len(edge_pre)} internal edges)")
    print(f"  FC goal-column neurons (valid fc_column): {int((fc_column >= 0).sum())}")
    print(f"  EPG heading-ring neurons (valid heading_ring): {int((heading_ring >= 0).sum())}")


def main() -> None:
    subset = build_subset()
    print("cluster counts:")
    print(subset["cluster"].value_counts())
    write_frontend_points(subset)
    write_backend_edges(subset)


if __name__ == "__main__":
    main()
