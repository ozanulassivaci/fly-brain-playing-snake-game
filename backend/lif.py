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
INHIB_GAIN = {"motion": 40.0, "cx": 400.0, "dn": 40.0}
ACTIVITY_EMA_TAU_MS = 20.0

# Phase 3: real per-neuron structure used for sensory-in/motor-out, not made
# up. T4/T5 subtypes a/b/c/d are real (Drosophila literature: these four
# subtypes are each tuned to one of the four cardinal motion directions;
# T4 = ON-edge motion, T5 = OFF-edge, same directional tuning, so each
# direction letter combines both). Descending-neuron soma side (L/R) is also
# real, roughly 50/50 in this dataset. What's a *simplification*: mapping
# "more right-side DN activity" to a specific turn direction — the real
# circuit has specific identified steering DNs (e.g. DNa01/DNa02), not "all
# DNs on one side", which this subset-wide aggregate can't distinguish.
DIRECTION_TYPE_PATTERN = re.compile(r"^T[45]([abcd])")
SENSORY_SCALE = 0.5
MOTOR_EMA_TAU_MS = 150.0
# DN fires sparsely even at target rate (~650 neurons/side at ~0.0003/step),
# so the L/R difference is small and noisy by nature (measured: baseline
# noise alone reaches +-0.0001-0.0002; sustained one-sided sensory drive
# shifts it by a similar order of magnitude, not a clean separation).
# Thresholds are set relative to that real noise floor, not to some larger
# assumed signal — this is the honest, real granularity of this
# aggregate DN readout (see class docstring note on identified steering DNs
# we don't have). The first value tried (0.00015) made the snake turn only
# ~2 times per 20s — mostly running straight into walls, which read as
# "not playing" rather than "reactive but erratic". Lowered so real noise
# alone produces a visible turn every ~0.4s (measured ~46 transitions/20s
# at this value) — genuinely more responsive, not just a cosmetic tweak.
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
        self.cluster_masks = {
            name: torch.from_numpy((cluster == name).astype(np.float32)).to(self.device)
            for name in ("motion", "cx", "dn")
        }

        neuron_type = data["neuron_type"]
        direction_letter = np.array(
            [(m.group(1) if (m := DIRECTION_TYPE_PATTERN.match(t)) else "") for t in neuron_type]
        )
        self.direction_masks = {
            letter: torch.from_numpy((direction_letter == letter).astype(np.float32)).to(self.device)
            for letter in ("a", "b", "c", "d")
        }

        soma_side = data["soma_side"]
        dn_mask = cluster == "dn"
        self.dn_left_mask = torch.from_numpy((dn_mask & (soma_side == "L")).astype(np.float32)).to(self.device)
        self.dn_right_mask = torch.from_numpy((dn_mask & (soma_side == "R")).astype(np.float32)).to(self.device)
        self.dn_left_count = max(1.0, float(self.dn_left_mask.sum().item()))
        self.dn_right_count = max(1.0, float(self.dn_right_mask.sum().item()))

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

    def _update_motor(self) -> None:
        left_rate = (self.spikes * self.dn_left_mask).sum().item() / self.dn_left_count
        right_rate = (self.spikes * self.dn_right_mask).sum().item() / self.dn_right_count
        self.dn_left_ema = self.dn_left_ema * self.motor_ema_decay + left_rate * (1 - self.motor_ema_decay)
        self.dn_right_ema = self.dn_right_ema * self.motor_ema_decay + right_rate * (1 - self.motor_ema_decay)

        diff = self.dn_right_ema - self.dn_left_ema
        # Simplification: "more right-side DN activity -> turn right" is a
        # consistent convention, not a claim about the real steering circuit
        # (see the module docstring note on identified steering DNs).
        if self.current_turn == "straight":
            if diff > TURN_ON_THRESH:
                self.current_turn = "right"
            elif diff < -TURN_ON_THRESH:
                self.current_turn = "left"
        elif abs(diff) < TURN_OFF_THRESH:
            self.current_turn = "straight"

    def read_motor(self) -> str:
        return self.current_turn

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
        self._update_motor()
        return self.spikes

    def step_batch(self, n_steps: int) -> list[int]:
        spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            spiked |= self.step() > 0
        return spiked.nonzero().squeeze(1).tolist()
