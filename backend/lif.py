"""
Leaky integrate-and-fire simulation of the Phase 0 candidate subset, run on
GPU with PyTorch. Real connectivity and weights (from
data/processed/subset.npz), synthetic dynamics — there is no biologically
calibrated conductance value for "MaleCNS synapse count", so WEIGHT_SCALE
below is an approximate, empirically-tuned knob, not a claimed-accurate
parameter. See docs/architecture-plan.md for the subset's provenance.
"""

import re
from pathlib import Path

import numpy as np
import torch

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

DT_MS = 1.0
LEAK_TAU_MS = 20.0
THRESHOLD = 1.0
RESET = -0.2
REFRACTORY_MS = 3.0
NOISE_STD = 0.1
WEIGHT_SCALE = 2e-3
DRIVE_DECAY_TAU_MS = 300.0

# The real MaleCNS weight table has no excitatory/inhibitory sign (that
# needs neurotransmitter-type data we didn't pull in Phase 0) — every
# connection here is positive. A purely excitatory recurrent network with
# ~50 connections/neuron saturates almost immediately (verified empirically:
# even tiny weight/noise values pushed 15-85% of the whole network to spike
# every 20ms batch). TARGET_RATE + INHIB_GAIN below is a simple population-
# level homeostatic control loop — proportional feedback that damps the
# whole network toward a plausible sparse firing rate — standing in for the
# real inhibitory neurons this simplified subset doesn't model.
#
# Applied *per cluster* (motion/cx/dn separately), not as one global pool:
# a single global average let a small number of CX neurons spike at the
# max refractory-limited rate (measured: ~400k spike-events across CX over
# 4000 steps, dominating the global average) while DN sat almost silent
# (measured: ~2.7e-5 spikes/neuron/step) — globally "on target" overall,
# locally saturated in one cluster and starved in another. Per-cluster
# control keeps each cluster independently near TARGET_RATE.
TARGET_RATE = 0.0003  # target fraction of neurons spiking per 1ms step
# CX needed ~10x more gain than motion/dn to actually reach TARGET_RATE
# (found empirically: at a uniform gain of 40, CX settled ~27x over target
# while motion/dn landed close to it — CX's local recurrent excitation is
# evidently much stronger, so its proportional controller needs a bigger
# correction for the same size of error).
INHIB_GAIN = {"motion": 40.0, "cx": 400.0, "dn": 40.0, "fc": 400.0, "pfl": 400.0, "epg": 400.0, "lc10": 400.0}
ACTIVITY_EMA_TAU_MS = 20.0

# Phase 3: real per-neuron structure used for sensory-in/motor-out, not made
# up. T4/T5 subtypes a/b/c/d are real (Drosophila literature: these four
# subtypes are each tuned to one of the four cardinal motion directions;
# T4 = ON-edge motion, T5 = OFF-edge, same directional tuning, so each
# direction letter combines both).
DIRECTION_TYPE_PATTERN = re.compile(r"^T[45]([abcd])")
SENSORY_SCALE = 0.5
# Phase 3.5: this was 150ms, chosen without measurement back in Phase 3.
# With a strong, fast-changing steering signal (LC10, Phase 3.4) and a
# discrete grid where every turn immediately changes the true bearing, a
# 150ms-lagged readout meant the motor decision still reflected the
# *pre-turn* bearing for a good fraction of a tick, causing a systematic
# overshoot-and-correct oscillation instead of convergence — found by
# directly comparing trajectories at different tau values (a fly that
# orbited at distance 8-12 from the target at 150ms converged to distance
# 0-2 at 20ms, same everything else). Confirmed at the full-gameplay
# level: 15ms took real apple-eating success from 0/24 episodes (150ms) to
# 20/24 (real per-episode measurement, see docs/architecture-plan.md).
MOTOR_EMA_TAU_MS = 15.0

# Phase 3.2: goal direction via the real, published FC -> PFL3 -> DNa02
# steering circuit (Westeinde et al.), checked against this exact dataset's
# connectome-weights table before building this (not assumed): FC->PFL is
# 1585 edges/16296 total weight; PFL->DNa02 specifically is 28 edges at
# 17-51 weight each (strong individual synapses, not a diffuse population
# effect); all 24 PFL3 neurons connect to DNa02_L and/or DNa02_R. This
# replaced Phase 3.1's attempt, which injected into a generic "anything with
# a PB-glomerulus label" ring (wrong anatomical target — that population
# represents current heading, not goal) and read out via all 1308 DN neurons
# regardless of function (diluted by escape/flight/grooming/feeding DNs
# unrelated to steering); measured over multiple independent trials, that
# combination produced a same-sign-as-noise, sometimes wrong-signed shift —
# a real negative result, not a tuning failure.
#
# fc_column (0-8) comes from real fan-shaped-body column labels in FC-type
# instance strings (see prepare_subset.py) — 277 real neurons, column
# number only. Briefly tried a side-aware 18-position version (matching the
# heading ring below) on the theory that FC's side-blind code and EPG's
# side-aware one were mismatched coordinate systems — reverted after
# checking real connectivity: FC_L and FC_R project to PFL_L/PFL_R almost
# identically, so side isn't a meaningful axis for FC (see prepare_subset.py
# for the actual weight comparison). The readout uses all identified DNa*
# (numbered) steering descending neurons (32 neurons, 16 L / 16 R) rather
# than DNa02 alone (2 neurons, too few to read a rate from in this
# uncalibrated LIF) or the full unrelated-DN-diluted 1308-neuron aggregate.
FC_COLUMNS = 9
GOAL_SCALE = 1.0
GOAL_SIGMA = 1.5  # bump width in FC columns
STEERING_DN_TYPE_PATTERN = re.compile(r"^DNa\d+$")

# Phase 3.2b: measured (test_diag_repeat.py, 6 independent trials) that
# injecting only the goal into FC made FC and PFL respond strongly and
# reliably (once the homeostasis split below stopped crushing PFL), but the
# DNa* L/R difference still came out wrong-signed in 5/6 trials — noise, not
# a real steering bias. Root cause: real PFL3 neurons don't relay the goal on
# its own, they compare it against *current heading* (from the EPG compass
# ring) via their real anatomical dendrite geometry — inject the goal alone
# and there is nothing to compare it against, so no reliable lateral signal
# should be expected. EPG->PFL is a real, substantial pathway in this exact
# dataset (checked before adding this: 247 edges, weight 2756 — comparable
# scale to FC->PFL's 1585 edges/16296 weight), so both signals are injected
# in the *same absolute (allocentric) reference frame* — heading and goal
# angle both measured against a fixed world axis, not against each other —
# letting PFL's real synaptic wiring compute the comparison itself, the way
# it does in the actual fly, rather than pre-computing a relative bearing in
# JS and only ever telling the brain "half" of the comparison.
# The PB's 18 glomeruli (9 per side) are a real, well-documented
# "double-wrapped" ring (Wolff & Rubin; Turner-Evans et al.): the same 9
# angular positions appear once per hemisphere, both halves jointly
# representing one shared heading, not 18 independent positions — checked
# directly against this dataset: EPG_L#k projects to EPG_R#k (matching
# glomerulus number) at ~8x the average weight of EPG_L#k to a
# different-numbered EPG_R#k2. So this also folds side away, matching
# FC_COLUMNS' 9-position space, injecting into both hemispheres' matching
# glomerulus together for one coherent bump — not the disconnected 18-slot
# ring briefly used earlier in Phase 3.3 (the same mismatched-topology
# mistake FC's side-aware draft made: placing two neurons that jointly
# encode the *same* angular value ~9 slots apart on an artificial ring).
HEADING_RING_SIZE = 9
HEADING_SCALE = 1.0
HEADING_SIGMA = 1.5  # bump width in ring positions (matches GOAL_SIGMA: same-sized 9-position ring)

# Phase 3.3 follow-up: a real gap between these two (OFF well below ON, a
# genuine hysteresis band) was measured to hold a "left"/"right" decision
# for a long time once triggered — a real sustained bias during injection
# rarely dips all the way back down to a small OFF threshold, so median
# hold length was ~5.3 game ticks and the tail reached ~41 ticks (~6
# seconds) at TURN_OFF_THRESH=0.00003. That's long enough for even
# one-turn-per-tick to spin the snake through several full rotations
# during a single hold, and long enough that a single edge-triggered turn
# leaves it going straight for multiple seconds — the source of both the
# circling bug and the "turns too late, hits a wall" bug reported after
# playing. Measured directly (sweep_off_thresh.py-style test) that
# removing the hysteresis band entirely (OFF == ON) shrinks hold length to
# a median of ~1.6 ticks and a max of ~6.8 ticks — short enough that plain
# once-per-tick turning (frontend/js/snake-game.js) no longer needs a
# game-layer cooldown/edge-trigger workaround to avoid visible spinning.
TURN_ON_THRESH = 0.0001
TURN_OFF_THRESH = 0.0001

# Phase 3.4: direct visual pursuit via LC10, the real, published
# target-pursuit visual projection neuron (Ribeiro et al. 2018 — LC10a
# detects a small salient visual target and drives steering toward it via
# DNp11; a male fly pursuing a female uses exactly this pathway). This is
# the anatomically correct real circuit for "sees something and flies
# straight at it" — unlike central-complex path integration (FC/EPG/PFL,
# above), which is for returning to a remembered location, not real-time
# visual target acquisition. Checked directly against this dataset before
# using it: every LC10 subtype present (a, b, c-1, c-2, d, e; 960 neurons
# total) projects to DNa* steering neurons with a *perfectly* ipsilateral,
# zero-crosstalk pattern — e.g. LC10a_L -> DNa_L is 377 weight / LC10a_L ->
# DNa_R is exactly 0, and the mirror image for LC10a_R — the cleanest,
# least ambiguous real structure found in this whole project, needing no
# delicate goal-vs-heading subtraction the way PFL3 does. DNa10 (the
# single strongest target, 801+424 weight) is already inside the existing
# DNa* steering readout, so no new readout population is needed.
#
# LC10 has no real retinotopic/spatial-position label in this dataset
# (unlike FC's column or EPG's glomerulus), so there is no way to build a
# genuine per-angle receptive-field map — only real, verified left/right
# separation. Injection uses the *egocentric* bearing (target angle
# relative to current heading, computed from the allocentric
# bearing/heading already sent for FC/EPG) rather than an allocentric one:
# visual detection is inherently egocentric (where does the target appear
# in my current field of view), unlike the CX pathway's shared world-frame
# comparison — there is no PFL-style comparator neuron in this pathway to
# do that subtraction for us.
LC10_TYPE_PATTERN = re.compile(r"^LC10")
VISUAL_SCALE = 1.0
# Frontal acceptance zone: no turn command while the target sits within
# this angle of straight ahead. Real flies do exactly this — the classic
# Drosophila *fixation* response keeps a visual object in the frontal
# field and only corrects once it drifts off-centre, rather than
# demanding perfect alignment every instant. Here it is also what makes
# the controller's precision match the body's: on a 4-direction grid the
# best reachable heading can still be up to 45 degrees off the target, so
# without a deadzone the fly gets a strong "turn!" command (sin 41 deg =
# 0.66) even when it is already pointed as well as it possibly can be —
# it turns, the next heading is ~49 degrees off the other way, and it
# turns again. Traced live in the browser before adding this: the snake
# cycled (-1,0) -> (0,1) -> (1,0) -> (0,-1) endlessly with the egocentric
# bearing repeating the same four values (-138, -52, +41, +135 degrees),
# none near zero — a limit cycle, which is precisely the "circles" this
# demo kept drawing.
VISUAL_DEADZONE = np.pi / 4

# Phase 3.5: real trajectories showed the fly making one lucky early
# approach, then drifting away and wandering in a distant region for the
# rest of the episode without ever correcting back. First hypothesis
# tried: direction alone (however reliable in isolation, see LC10's t=88
# above) doesn't guarantee net progress — so a reward-like "search
# intensity" signal was built (rising when a fast/slow EMA crossover of
# distance-to-target showed no progress, decaying when it did, scaling the
# LC10 injection's magnitude — modeling real area-restricted-search
# behavior, documented in Drosophila and much of the foraging-animal
# literature as octopaminergic/dopaminergic modulation of search vigor).
# It measurably changed behavior (search_intensity swung across its full
# 0-4 range) but did *not* fix the drift — the actual bug was elsewhere.
#
# Root cause, found by testing MOTOR_EMA_TAU_MS directly: at the original
# 150ms, the motor readout lagged far enough behind each turn that the
# fly systematically overshot the target's true bearing every correction,
# a classic feedback-lag oscillation — reducing it to 15ms (still an EMA,
# not raw instantaneous spikes, just a much shorter one) let the readout
# track the post-turn bearing almost immediately. This alone, with no
# reward/vigor mechanism at all, took real gameplay from 2/24 episodes
# eating an apple (no goal information) or 0/24 (goal information, 150ms
# lag) to 20/24 (goal information, 15ms lag) — confirming the lag, not
# signal reliability or search vigor, was the actual bottleneck all along.
# The search_intensity mechanism was removed after this: measured to not
# help and to slightly hurt once the lag was fixed (a fast, accurate
# controller doesn't benefit from extra gain — it just overshoots more).


class LifSimulation:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data = np.load(DATA_PROCESSED / "subset.npz", allow_pickle=True)
        self.n = int(data["n_neurons"])
        edge_pre = torch.from_numpy(data["edge_pre"].astype(np.int64))
        edge_post = torch.from_numpy(data["edge_post"].astype(np.int64))
        # Phase 3.3: real per-neuron neurotransmitter sign (see
        # prepare_subset.py's INHIBITORY_NT / nt_sign — GABA/glutamate = -1,
        # everything else = +1, standard fly-connectome convention). Sign is
        # a property of the presynaptic neuron releasing the transmitter,
        # not of the connection, so index nt_sign by edge_pre. Previously
        # every edge here was positive regardless of real biology (Phase 0's
        # limitation, noted since Phase 2) — checked directly against this
        # subset before fixing it: Delta7 (real inhibitory ring-attractor-
        # sharpening interneuron in the compass circuit) was 100% wrongly
        # excitatory, and the central-complex cluster overall was ~32%
        # wrongly-signed (994/3137 real GABA/glutamate neurons).
        nt_sign = torch.from_numpy(data["nt_sign"].astype(np.float32))
        edge_weight = torch.from_numpy(data["edge_weight"].astype(np.float32)) * WEIGHT_SCALE * nt_sign[edge_pre]

        indices = torch.stack([edge_post, edge_pre])
        self.weight_matrix = torch.sparse_coo_tensor(
            indices, edge_weight, size=(self.n, self.n), device=self.device
        ).coalesce()

        cluster = data["cluster"]
        neuron_type = data["neuron_type"]
        direction_letter = np.array(
            [(m.group(1) if (m := DIRECTION_TYPE_PATTERN.match(t)) else "") for t in neuron_type]
        )
        self.direction_masks = {
            letter: torch.from_numpy((direction_letter == letter).astype(np.float32)).to(self.device)
            for letter in ("a", "b", "c", "d")
        }

        # FC and PFL get their own homeostatic groups, split out of "cx"
        # instead of sharing its blanket inhibition — measured directly
        # (test_diagnostic.py): injecting a goal bump into FC pushed the
        # whole cx-cluster average far over TARGET_RATE, and the resulting
        # cluster-wide proportional inhibition (INHIB_GAIN["cx"]=400) then
        # crushed PFL's activity to ~0 too (it fell from a baseline ~2e-4 to
        # 1.4e-69), even though PFL receives strong real synaptic drive from
        # FC (1585 edges) — the very mechanism keeping the recurrent network
        # stable was silently strangling the goal-signal relay it shares a
        # cluster with. Splitting them into independent homeostatic pools
        # lets FC's own overshoot get clamped without touching PFL.
        is_fc = np.array([str(t).startswith("FC") for t in neuron_type])
        is_pfl = np.array([str(t).startswith("PFL") for t in neuron_type])
        # EPG gets the same treatment as FC/PFL above, for the same reason:
        # sustained heading injection would otherwise push the shared "cx"
        # cluster average up and have its homeostatic inhibition crush
        # whatever else is left in "cx" (or, if EPG stayed lumped in with
        # PFL/FC's own pools, crush EPG's own output the moment it fires
        # enough to be useful).
        is_epg = np.array([str(t).startswith("EPG") for t in neuron_type])
        # LC10 gets the same treatment, split out of "motion" this time
        # (LC10 matches MOTION_PATTERN's "LC\d" in prepare_subset.py) — same
        # reasoning as fc/pfl/epg: sustained visual-target injection would
        # otherwise push the whole motion-cluster average up and have its
        # homeostatic inhibition crush LC10's own output, or (if motion's
        # T4/T5 neurons stayed lumped in with it) crush the real optic-flow
        # signal Phase 3's turning already depends on.
        is_lc10 = np.array([bool(LC10_TYPE_PATTERN.match(t)) for t in neuron_type])
        cx_other = (cluster == "cx") & ~is_fc & ~is_pfl & ~is_epg
        motion_other = (cluster == "motion") & ~is_lc10
        self.cluster_masks = {
            "motion": torch.from_numpy(motion_other.astype(np.float32)).to(self.device),
            "cx": torch.from_numpy(cx_other.astype(np.float32)).to(self.device),
            "dn": torch.from_numpy((cluster == "dn").astype(np.float32)).to(self.device),
            "fc": torch.from_numpy(is_fc.astype(np.float32)).to(self.device),
            "pfl": torch.from_numpy(is_pfl.astype(np.float32)).to(self.device),
            "epg": torch.from_numpy(is_epg.astype(np.float32)).to(self.device),
            "lc10": torch.from_numpy(is_lc10.astype(np.float32)).to(self.device),
        }

        fc_column = data["fc_column"]
        fc_onehot = np.zeros((FC_COLUMNS, self.n), dtype=np.float32)
        valid = fc_column >= 0
        fc_onehot[fc_column[valid], np.where(valid)[0]] = 1.0
        self.fc_column_matrix = torch.from_numpy(fc_onehot).to(self.device)  # (FC_COLUMNS, n)
        self._fc_positions = np.arange(FC_COLUMNS)

        heading_ring = data["heading_ring"]
        heading_onehot = np.zeros((HEADING_RING_SIZE, self.n), dtype=np.float32)
        heading_valid = heading_ring >= 0
        heading_onehot[heading_ring[heading_valid], np.where(heading_valid)[0]] = 1.0
        self.heading_ring_matrix = torch.from_numpy(heading_onehot).to(self.device)  # (HEADING_RING_SIZE, n)
        self._heading_positions = np.arange(HEADING_RING_SIZE)

        soma_side = data["soma_side"]
        is_steering_dn = np.array([bool(STEERING_DN_TYPE_PATTERN.match(t)) for t in neuron_type])
        self.dn_left_mask = torch.from_numpy((is_steering_dn & (soma_side == "L")).astype(np.float32)).to(
            self.device
        )
        self.dn_right_mask = torch.from_numpy((is_steering_dn & (soma_side == "R")).astype(np.float32)).to(
            self.device
        )
        self.dn_left_count = max(1.0, float(self.dn_left_mask.sum().item()))
        self.dn_right_count = max(1.0, float(self.dn_right_mask.sum().item()))

        self.lc10_left_mask = torch.from_numpy((is_lc10 & (soma_side == "L")).astype(np.float32)).to(self.device)
        self.lc10_right_mask = torch.from_numpy((is_lc10 & (soma_side == "R")).astype(np.float32)).to(self.device)

        # Extra named populations tracked only for the frontend decision-flow
        # panel (not used to drive any dynamics beyond the homeostatic split
        # above) — real per-type-group activity so the panel shows genuine
        # values for genuine anatomical populations, not a decorative
        # animation. Motion is broken into its real T4/T5 a/b/c/d direction
        # subtypes (same self.direction_masks already used for sensory
        # injection) instead of one aggregate, matching the original Phase 1
        # panel's per-type-column layout; epg/fc/pfl/dna_left/dna_right reuse
        # the same masks as the homeostatic cluster split above.
        self.group_masks = {
            "motion_a": self.direction_masks["a"],
            "motion_b": self.direction_masks["b"],
            "motion_c": self.direction_masks["c"],
            "motion_d": self.direction_masks["d"],
            "lc10_left": self.lc10_left_mask,
            "lc10_right": self.lc10_right_mask,
            "epg": self.cluster_masks["epg"],
            "fc": self.cluster_masks["fc"],
            "pfl": self.cluster_masks["pfl"],
            "dna_left": self.dn_left_mask,
            "dna_right": self.dn_right_mask,
        }
        self.group_counts = {name: max(1.0, float(mask.sum().item())) for name, mask in self.group_masks.items()}
        self.group_activity_ema = {name: 0.0 for name in self.group_masks}

        # Phase 3.5: every per-population spike rate this class reads each
        # step (7 homeostatic clusters + 11 display groups + 2 motor) used
        # to be its own `(spikes * mask).sum().item()` — 20 separate
        # GPU->CPU syncs per step, 400 per 20-step broadcast batch, each one
        # stalling the pipeline. Measured: that was the dominant cost, and
        # it kept the simulation *below real time* (0.71x), which in turn
        # meant the fly's brain lived slower than the game it was playing.
        # Stacking every mask into one matrix and reading all 20 rates with
        # a single matmul + single sync measured 3x faster (to ~5x real
        # time), which is what lets the game's wall-clock tick rate and the
        # brain's simulated time actually correspond.
        self._cluster_names = list(self.cluster_masks.keys())
        self._group_names = list(self.group_masks.keys())
        self._readout_matrix = torch.stack(
            [self.cluster_masks[n] for n in self._cluster_names]
            + [self.group_masks[n] for n in self._group_names]
            + [self.dn_left_mask, self.dn_right_mask]
        )

        self.leak_decay = float(np.exp(-DT_MS / LEAK_TAU_MS))
        self.drive_decay = float(np.exp(-DT_MS / DRIVE_DECAY_TAU_MS))
        self.activity_ema_decay = float(np.exp(-DT_MS / ACTIVITY_EMA_TAU_MS))
        self.motor_ema_decay = float(np.exp(-DT_MS / MOTOR_EMA_TAU_MS))
        self.refractory_steps = int(round(REFRACTORY_MS / DT_MS))

        self.dn_left_ema = 0.0
        self.dn_right_ema = 0.0
        self.current_turn = "straight"

        self.v = torch.zeros(self.n, device=self.device)
        self.external_drive = torch.zeros(self.n, device=self.device)
        self.spikes = torch.zeros(self.n, device=self.device)
        self.refractory = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.cluster_counts = {name: max(1.0, float(mask.sum().item())) for name, mask in self.cluster_masks.items()}
        self.cluster_activity_ema = {name: 0.0 for name in self.cluster_masks}

    # Phase 1 added blanket per-event drives ("move"/"eat"/"collide") back when
    # the brain panel was decorative and just needed to flicker in time with
    # the game. All of them are gone now, because once the brain actually
    # steers they stop being harmless decoration:
    #   - "move" fired every game tick (~7/s), dumping 0.6 into all 18,433
    #     motion neurons, competing directly with the real visual steering
    #     signal — measured live, removing it (with the retina DC fix in
    #     frontend/js/retina.js) tripled the apple-eating rate. It modelled
    #     nothing either: a real fly's self-motion cue is optic flow, which is
    #     what the T4/T5 retina path already supplies.
    #   - "eat"/"collide" injected 0.8-1.2 straight into the *descending
    #     neurons*, i.e. the very population the steering decision is decoded
    #     from, corrupting it for ~300ms (DRIVE_DECAY_TAU_MS) at exactly the
    #     moment the fly had just reached an apple and needed to pick a new
    #     heading. A real appetitive signal goes to reward circuitry (mushroom
    #     body / PAM dopaminergic neurons), not to steering DNs — and this
    #     subset contains no such neurons to send it to.
    # The brain panel still lights up, from real spikes rather than injected
    # flashes.
    def inject_sensory(self, values: dict) -> None:
        for letter in ("a", "b", "c", "d"):
            amount = values.get(letter, 0.0)
            if amount:
                self.external_drive += self.direction_masks[letter] * (amount * SENSORY_SCALE)
        # Both "bearing" (apple angle) and "heading" (snake's own facing
        # angle) are allocentric — measured against the same fixed world
        # axis, not against each other — so PFL's real EPG+FC synapses can
        # compute the heading-vs-goal comparison themselves. See the
        # Phase 3.2b comment above HEADING_RING_SIZE for why this replaced
        # injecting a pre-computed relative bearing into FC alone.
        if "bearing" in values:
            self.inject_goal(values["bearing"])
        if "heading" in values:
            self.inject_heading(values["heading"])
        if "bearing" in values and "heading" in values:
            ego_bearing = np.arctan2(
                np.sin(values["bearing"] - values["heading"]), np.cos(values["bearing"] - values["heading"])
            )
            self.inject_visual_target(float(ego_bearing))

    def inject_visual_target(self, ego_bearing: float) -> None:
        # Egocentric: 0 = target straight ahead, +-pi/2 = directly to the
        # side. A smooth sin-based split rather than a hard left/right
        # switch, since we only have real L/R separation to work with (no
        # retinotopic position label — see the Phase 3.4 comment above
        # LC10_TYPE_PATTERN) — this still gives zero drive when the target
        # is dead ahead or directly behind and maximum drive when it's
        # squarely to one side, without an arbitrary discontinuity at 0.
        if abs(ego_bearing) <= VISUAL_DEADZONE:
            return
        right_amount = max(0.0, float(np.sin(ego_bearing))) * VISUAL_SCALE
        left_amount = max(0.0, float(-np.sin(ego_bearing))) * VISUAL_SCALE
        if right_amount:
            self.external_drive += self.lc10_right_mask * right_amount
        if left_amount:
            self.external_drive += self.lc10_left_mask * left_amount

    def inject_goal(self, bearing: float) -> None:
        target = (bearing / (2 * np.pi)) * FC_COLUMNS
        diff = np.abs(self._fc_positions - target)
        circular_dist = np.minimum(diff, FC_COLUMNS - diff)
        weights = np.exp(-(circular_dist**2) / (2 * GOAL_SIGMA**2)) * GOAL_SCALE
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        self.external_drive += weights_t @ self.fc_column_matrix

    def inject_heading(self, heading: float) -> None:
        target = (heading / (2 * np.pi)) * HEADING_RING_SIZE
        diff = np.abs(self._heading_positions - target)
        circular_dist = np.minimum(diff, HEADING_RING_SIZE - diff)
        weights = np.exp(-(circular_dist**2) / (2 * HEADING_SIGMA**2)) * HEADING_SCALE
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        self.external_drive += weights_t @ self.heading_ring_matrix

    def _update_motor(self, left_rate: float, right_rate: float) -> None:
        self.dn_left_ema = self.dn_left_ema * self.motor_ema_decay + left_rate * (1 - self.motor_ema_decay)
        self.dn_right_ema = self.dn_right_ema * self.motor_ema_decay + right_rate * (1 - self.motor_ema_decay)

        diff = self.dn_right_ema - self.dn_left_ema
        # "more right-side steering-DN activity -> turn right" is a
        # consistent convention we chose, not something derivable from the
        # data alone (we don't have the actual sign of the DNa02-leg-motor
        # mapping) — but the *population* being read is now the real,
        # specifically identified steering DN family, not an arbitrary cut.
        if self.current_turn == "straight":
            if diff > TURN_ON_THRESH:
                self.current_turn = "right"
            elif diff < -TURN_ON_THRESH:
                self.current_turn = "left"
        elif abs(diff) < TURN_OFF_THRESH:
            self.current_turn = "straight"

    def read_motor(self) -> str:
        return self.current_turn

    def read_groups(self) -> dict:
        return dict(self.group_activity_ema)

    def step(self) -> torch.Tensor:
        input_current = torch.sparse.mm(self.weight_matrix, self.spikes.unsqueeze(1)).squeeze(1)
        noise = torch.randn(self.n, device=self.device) * NOISE_STD

        inhibition = torch.zeros(self.n, device=self.device)
        for name, mask in self.cluster_masks.items():
            gain = INHIB_GAIN[name] * max(0.0, self.cluster_activity_ema[name] - TARGET_RATE)
            if gain:
                inhibition += mask * gain

        self.v = self.v * self.leak_decay + input_current + noise + self.external_drive - inhibition
        self.external_drive *= self.drive_decay

        can_fire = self.refractory <= 0
        self.spikes = ((self.v > THRESHOLD) & can_fire).float()
        self.v = torch.where(self.spikes > 0, torch.full_like(self.v, RESET), self.v)
        self.refractory = torch.where(
            self.spikes > 0, torch.full_like(self.refractory, self.refractory_steps), self.refractory - 1
        )
        self.refractory.clamp_(min=0)

        # One matmul, one GPU->CPU sync for all 20 population rates (see the
        # _readout_matrix comment in __init__ for why this matters).
        counts = self._readout_matrix @ self.spikes
        rates = counts.tolist()

        n_clusters = len(self._cluster_names)
        for i, name in enumerate(self._cluster_names):
            rate = rates[i] / self.cluster_counts[name]
            self.cluster_activity_ema[name] = (
                self.cluster_activity_ema[name] * self.activity_ema_decay + rate * (1 - self.activity_ema_decay)
            )
        for j, name in enumerate(self._group_names):
            rate = rates[n_clusters + j] / self.group_counts[name]
            self.group_activity_ema[name] = (
                self.group_activity_ema[name] * self.activity_ema_decay + rate * (1 - self.activity_ema_decay)
            )
        self._update_motor(rates[-2] / self.dn_left_count, rates[-1] / self.dn_right_count)
        return self.spikes

    def step_batch(self, n_steps: int) -> list[int]:
        spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            spiked |= self.step() > 0
        return spiked.nonzero().squeeze(1).tolist()
