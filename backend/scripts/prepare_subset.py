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

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "subset.npz"
    np.savez(
        out_path,
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_weight=edge_weight,
        cluster=cluster,
        n_neurons=len(subset),
    )
    print(f"wrote {out_path} ({len(subset)} neurons, {len(edge_pre)} internal edges)")


def main() -> None:
    subset = build_subset()
    print("cluster counts:")
    print(subset["cluster"].value_counts())
    write_frontend_points(subset)
    write_backend_edges(subset)


if __name__ == "__main__":
    main()
