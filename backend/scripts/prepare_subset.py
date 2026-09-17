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

# Phase 3.8: the olfactory / mushroom-body pathway. Real flies find food
# primarily by smell, and the whole canonical circuit is present in this
# dataset (checked before adding it): antennal-lobe projection neurons
# (541 with soma positions, 273 L / 273 R) -> Kenyon cells (4050) -> MBON
# (90) -> DNa* steering, that last step at 1145 total weight, the same
# order as LC10's ~970 into the same readout. PAM (314 dopaminergic
# neurons) sits exactly where it does in the real animal: projecting onto
# the Kenyon cells (KCg-m alone at 68,532 weight), i.e. onto the KC->MBON
# synapses whose plasticity is what dopamine actually does. APL is the
# mushroom body's big inhibitory interneuron.
#
# The 2635 ORNs are deliberately *not* included: not one of them has a
# soma position (0/2635), because olfactory receptor neurons sit in the
# antenna rather than the brain volume this reconstruction covers. Odour
# therefore enters at the PN stage — the antennal lobe's output, which is
# the odour representation the rest of the brain actually sees, and the
# right level for a model that cannot claim to reproduce ORN gain control
# anyway.
#
# Worth recording plainly: the olfactory pathway has essentially *no*
# direct route to steering on its own — PN->FC is 1 weight and PN->DNa is
# 2. Smell only reaches behaviour through the mushroom body, which is
# also why odour and dopamine are one feature rather than two.
PN_PATTERN = re.compile(r"^[A-Z]+[0-9a-z]*_[a-z]*PN")
# The lateral horn: olfaction's *innate* output, as opposed to the
# mushroom body's learned one. This is the half that actually drives
# odour-guided behaviour in a naive fly (PN->LH is 378,010 weight, LH->
# descending neurons 22,810), and leaving it out was why the first odour
# attempt had nowhere for the signal to go but the mushroom body, which
# suppressed it.
LH_PATTERN = re.compile(r"^LH")
KC_PATTERN = re.compile(r"^KC")
MBON_PATTERN = re.compile(r"^MBON")
PAM_PATTERN = re.compile(r"^PAM")
APL_PATTERN = re.compile(r"^APL$")
# Phase 3.13: the premotor steering network. Measured against this dataset
# rather than assumed, and it is the largest single omission the project
# has had: the 32 DNa* steering neurons we read the motor decision from
# receive 326,741 total input weight, and everything we inject reaches them
# through populations worth about 2.9% of it —
#
#     LC10 (visual target)  3,863 direct  (1.2%)
#     PFL  (central complex) 2,832        (0.9%)
#     MBON (mushroom body)   1,145        (0.4%)
#     LPLC1 (obstacle)         575        (0.2%)
#     LPLC2                      0
#
# The other 97% comes from LAL (lateral accessory lobe), PS (posterior
# slope), AOTU and VES interneurons. That is exactly where it should come
# from: in insects the LAL *is* the premotor steering centre, where
# descending steering commands are assembled. We had 8 of the LAL's 204
# types and none of PS's 266.
#
# Which means every sensory pathway in this project has been entering
# through a side door. Their real routes to steering:
#
#     LC10  -> DNa* direct 3,863   -> LAL/PS/AOTU 379,073   (98x)
#     LH    -> DNa* direct    66   -> LAL/PS/AOTU  16,672  (253x)
#     PFL   -> DNa* direct 2,832   -> LAL/PS/AOTU  40,010   (14x)
#     MBON  -> DNa* direct 1,145   -> LAL/PS/AOTU  25,164   (22x)
#     LPLC1 -> DNa* direct   575   -> LAL/PS/AOTU   5,126    (9x)
#
# The odour pathway was made to work over a 66-weight connection while its
# real 16,672-weight route was not in the model at all, and obstacle
# avoidance stayed a whisper for the same reason — which is the measured
# explanation for why raising any of these gains only ever traded one kind
# of death for another: they were all competing inside the ~3% of the
# steering readout we could actually reach.
PREMOTOR_PATTERN = re.compile(r"^(LAL|PS|AOTU|VES)\d")
CORE_CX_ROI_PATTERN = re.compile(r"^(EB|FB|PB(\(|$)|NO$|NO\()")
MIN_CX_SYNWEIGHT = 5
# FC (fan-shaped body columnar, real goal-direction cell types) instance
# strings look like "FC1E_C4_L" — the "_C{1-9}_" is real fan-shaped-body
# column position, matching the real anatomical column count, and the
# trailing "_L"/"_R" is a real hemisphere side (all 277 FC neurons in this
# subset have it, 139 R / 138 L). Briefly made this side-aware (an 18-slot
# code duplicating the heading ring's structure) on the theory that a
# side-blind FC code and a side-aware EPG code were mismatched coordinate
# systems for PFL to compare — reverted after checking the real
# connectivity: FC_L and FC_R project to PFL_L/PFL_R almost identically
# (e.g. FC_L->PFL_L 4199 vs FC_L->PFL_R 3859, both sides similar), unlike
# EPG (see below), so side isn't a meaningful axis for FC's positional
# code. Column number only, 9 positions, matching the original design.
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
#
# The PB's 18 glomeruli (9 per side) are the real, well-documented
# "double-wrapped" compass ring in Drosophila literature (Wolff & Rubin,
# Turner-Evans et al.): the same 9 angular positions appear once per
# hemisphere, both halves representing one shared heading (not 18
# independent positions) — checked directly against this dataset before
# relying on it: EPG_L#k projects to EPG_R#k (matching glomerulus number)
# with ~8x the average weight of EPG_L#k to a *different*-numbered EPG_R#k2,
# consistent with paired glomeruli encoding the same angular value rather
# than 18 distinct ones. So heading_ring_for below folds side away too —
# same 9-position space as FC's column code, injecting into both
# hemispheres' matching glomerulus together for one coherent bump — instead
# of the disconnected 18-slot ring used up through Phase 3.3's first draft
# (which is the same mismatched-topology mistake FC's side-aware draft
# made: placing two neurons that jointly encode the *same* angular value
# on opposite sides of an artificial ring, roughly 9 slots apart).
HEADING_RING_PATTERN = re.compile(r"\(PB\d+\)_[LR](\d)")

# Phase 3.3: real per-neuron neurotransmitter predictions exist in this
# dataset (Neuprint_Neurons.feather's consensusNt, from FlyEM's own EM-based
# NT classifier) — checked directly against this exact subset before using
# it: Delta7 (the real, well-known inhibitory ring-attractor-sharpening
# interneuron in the fly compass circuit) is 100% glutamate in this data
# (42/42 neurons), and the central-complex cluster overall is ~32% GABA or
# glutamate (994/3137) — not the "no sign data available" situation
# docs/architecture-plan.md previously assumed. Standard fly-connectome
# convention (matching how FlyWire/hemibrain analyses treat these three
# transmitters): acetylcholine is excitatory, GABA and glutamate are
# inhibitory. Everything else here (dopamine/octopamine/serotonin/unclear/
# missing — a small remainder, see the per-cluster breakdown recorded in
# docs/architecture-plan.md) defaults to excitatory rather than being
# modeled as a separate neuromodulatory channel, which is out of scope for
# this project's simplified LIF.
INHIBITORY_NT = {"gaba", "glutamate"}


def load_annotations() -> pd.DataFrame:
    ann = pd.read_feather(DATA_RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    return ann[ann["status"] == "Traced"].copy()


def load_roi_info() -> pd.DataFrame:
    # bodyId/type/superclass/status/roiInfo only — see data/raw/README-ish
    # note in docs/architecture-plan.md: this is a column-projected copy of
    # the full Neuprint_Neurons.feather (the only table with real roiInfo).
    df = pd.read_feather(DATA_RAW / "neurons_roi_subset.feather")
    return df[df["status"] == "Traced"].copy()


def load_neurotransmitters() -> pd.DataFrame:
    # The full Neuprint_Neurons.feather again (same file load_roi_info's
    # neurons_roi_subset.feather was pre-projected from), a different
    # column pair this time — consensusNt isn't in that smaller projection,
    # so this reads the big file directly, column-projected to keep it fast.
    import pyarrow.feather as feather

    table = feather.read_table(
        DATA_RAW / "Neuprint_Neurons.feather", columns=["bodyId:long", "consensusNt:string"]
    )
    df = table.to_pandas().rename(columns={"bodyId:long": "bodyId", "consensusNt:string": "consensusNt"})
    return df


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
    nt_df = load_neurotransmitters()

    motion_types = {t for t in traced["type"].dropna().unique() if MOTION_PATTERN.match(str(t))}
    motion_mask = traced["type"].isin(motion_types)
    dn_mask = traced["superclass"] == "descending_neuron"

    types = traced["type"].fillna("")
    premotor_mask = types.str.match(PREMOTOR_PATTERN)
    olf_masks = {
        "pn": types.str.match(PN_PATTERN),
        "lh": types.str.match(LH_PATTERN),
        "kc": types.str.match(KC_PATTERN),
        "mbon": types.str.match(MBON_PATTERN),
        "pam": types.str.match(PAM_PATTERN),
        "apl": types.str.match(APL_PATTERN),
    }
    olf_any = olf_masks["pn"]
    for m in olf_masks.values():
        olf_any = olf_any | m

    cx_ids = set(roi_df.loc[roi_df["roiInfo"].apply(has_core_cx_roi), "bodyId"])
    cx_mask = traced["bodyId"].isin(cx_ids)

    olf_cluster_by_id = {}
    for name, mask in olf_masks.items():
        for body_id in traced.loc[mask, "bodyId"]:
            olf_cluster_by_id.setdefault(body_id, name)

    premotor_ids = set(traced.loc[premotor_mask, "bodyId"])

    def cluster_for(row) -> str | None:
        if row["superclass"] == "descending_neuron":
            return "dn"
        if row["bodyId"] in cx_ids:
            return "cx"
        if row["type"] in motion_types:
            return "motion"
        if row["bodyId"] in premotor_ids:
            return "lal"
        return olf_cluster_by_id.get(row["bodyId"])

    subset = traced[motion_mask | cx_mask | dn_mask | olf_any | premotor_mask].copy()
    subset["cluster"] = subset.apply(cluster_for, axis=1)
    subset = subset.merge(nt_df, on="bodyId", how="left")
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
        return int(m.group(1)) - 1 if m else -1

    fc_column = subset.apply(fc_column_for, axis=1).to_numpy(dtype=np.int32)
    heading_ring = subset.apply(heading_ring_for, axis=1).to_numpy(dtype=np.int32)
    nt_sign = np.where(subset["consensusNt"].isin(INHIBITORY_NT), -1.0, 1.0).astype(np.float32)

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
        nt_sign=nt_sign,
        n_neurons=len(subset),
    )
    print(f"wrote {out_path} ({len(subset)} neurons, {len(edge_pre)} internal edges)")
    print(f"  inhibitory (GABA/glutamate) neurons: {int((nt_sign < 0).sum())} / {len(subset)}")
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
