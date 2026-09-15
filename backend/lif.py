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
EVENT_DRIVE = {
    "move": {"motion": 0.6},
    "eat": {"cx": 0.8, "dn": 0.8},
    "collide": {"dn": 1.2},
}

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
INHIB_GAIN = {"motion": 40.0, "cx": 400.0, "dn": 40.0, "fc": 400.0, "pfl": 400.0, "epg": 400.0}
ACTIVITY_EMA_TAU_MS = 20.0

# Phase 3: real per-neuron structure used for sensory-in/motor-out, not made
# up. T4/T5 subtypes a/b/c/d are real (Drosophila literature: these four
# subtypes are each tuned to one of the four cardinal motion directions;
# T4 = ON-edge motion, T5 = OFF-edge, same directional tuning, so each
# direction letter combines both).
DIRECTION_TYPE_PATTERN = re.compile(r"^T[45]([abcd])")
SENSORY_SCALE = 0.5
MOTOR_EMA_TAU_MS = 150.0

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
# instance strings (see prepare_subset.py) — 277 real neurons. The
# readout uses all identified DNa* (numbered) steering descending neurons (32
# neurons, 16 L / 16 R) rather than DNa02 alone (2 neurons, too few to read
# a rate from in this uncalibrated LIF) or the full unrelated-DN-diluted
# 1308-neuron aggregate.
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
HEADING_RING_SIZE = 18
HEADING_SCALE = 1.0
HEADING_SIGMA = 3.0  # bump width in ring positions (double GOAL_SIGMA: ring is 2x FC_COLUMNS)

# Set from direct multi-trial measurement against this specific readout
# population (see the standalone test run before committing this), not
# guessed or carried over from the old aggregate's noise floor.
TURN_ON_THRESH = 0.0001
TURN_OFF_THRESH = 0.00003


class LifSimulation:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data = np.load(DATA_PROCESSED / "subset.npz", allow_pickle=True)
        self.n = int(data["n_neurons"])
        edge_pre = torch.from_numpy(data["edge_pre"].astype(np.int64))
        edge_post = torch.from_numpy(data["edge_post"].astype(np.int64))
        edge_weight = torch.from_numpy(data["edge_weight"].astype(np.float32)) * WEIGHT_SCALE

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
        cx_other = (cluster == "cx") & ~is_fc & ~is_pfl & ~is_epg
        self.cluster_masks = {
            "motion": torch.from_numpy((cluster == "motion").astype(np.float32)).to(self.device),
            "cx": torch.from_numpy(cx_other.astype(np.float32)).to(self.device),
            "dn": torch.from_numpy((cluster == "dn").astype(np.float32)).to(self.device),
            "fc": torch.from_numpy(is_fc.astype(np.float32)).to(self.device),
            "pfl": torch.from_numpy(is_pfl.astype(np.float32)).to(self.device),
            "epg": torch.from_numpy(is_epg.astype(np.float32)).to(self.device),
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
            "epg": self.cluster_masks["epg"],
            "fc": self.cluster_masks["fc"],
            "pfl": self.cluster_masks["pfl"],
            "dna_left": self.dn_left_mask,
            "dna_right": self.dn_right_mask,
        }
        self.group_counts = {name: max(1.0, float(mask.sum().item())) for name, mask in self.group_masks.items()}
        self.group_activity_ema = {name: 0.0 for name in self.group_masks}

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

    def inject_event(self, kind: str) -> None:
        boosts = EVENT_DRIVE.get(kind)
        if not boosts:
            return
        for cluster_name, amount in boosts.items():
            self.external_drive += self.cluster_masks[cluster_name] * amount

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

    def _update_motor(self) -> None:
        left_rate = (self.spikes * self.dn_left_mask).sum().item() / self.dn_left_count
        right_rate = (self.spikes * self.dn_right_mask).sum().item() / self.dn_right_count
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

        for name, mask in self.cluster_masks.items():
            rate = (self.spikes * mask).sum().item() / self.cluster_counts[name]
            self.cluster_activity_ema[name] = (
                self.cluster_activity_ema[name] * self.activity_ema_decay + rate * (1 - self.activity_ema_decay)
            )
        for name, mask in self.group_masks.items():
            rate = (self.spikes * mask).sum().item() / self.group_counts[name]
            self.group_activity_ema[name] = (
                self.group_activity_ema[name] * self.activity_ema_decay + rate * (1 - self.activity_ema_decay)
            )
        self._update_motor()
        return self.spikes

    def step_batch(self, n_steps: int) -> list[int]:
        spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            spiked |= self.step() > 0
        return spiked.nonzero().squeeze(1).tolist()
