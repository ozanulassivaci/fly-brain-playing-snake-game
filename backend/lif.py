"""
Leaky integrate-and-fire simulation of the Phase 0 candidate subset, run on
GPU with PyTorch. Real connectivity and weights (from
data/processed/subset.npz), synthetic dynamics — there is no biologically
calibrated conductance value for "MaleCNS synapse count", so WEIGHT_SCALE
below is an approximate, empirically-tuned knob, not a claimed-accurate
parameter. See docs/architecture-plan.md for the subset's provenance.
"""

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
TARGET_RATE = 0.0003  # target fraction of neurons spiking per 1ms step
INHIB_GAIN = 40.0
ACTIVITY_EMA_TAU_MS = 20.0


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

        self.leak_decay = float(np.exp(-DT_MS / LEAK_TAU_MS))
        self.drive_decay = float(np.exp(-DT_MS / DRIVE_DECAY_TAU_MS))
        self.activity_ema_decay = float(np.exp(-DT_MS / ACTIVITY_EMA_TAU_MS))
        self.refractory_steps = int(round(REFRACTORY_MS / DT_MS))

        self.v = torch.zeros(self.n, device=self.device)
        self.external_drive = torch.zeros(self.n, device=self.device)
        self.spikes = torch.zeros(self.n, device=self.device)
        self.refractory = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.activity_ema = 0.0

    def inject_event(self, kind: str) -> None:
        boosts = EVENT_DRIVE.get(kind)
        if not boosts:
            return
        for cluster_name, amount in boosts.items():
            self.external_drive += self.cluster_masks[cluster_name] * amount

    def step(self) -> torch.Tensor:
        input_current = torch.sparse.mm(self.weight_matrix, self.spikes.unsqueeze(1)).squeeze(1)
        noise = torch.randn(self.n, device=self.device) * NOISE_STD
        inhibition = INHIB_GAIN * max(0.0, self.activity_ema - TARGET_RATE)

        self.v = self.v * self.leak_decay + input_current + noise + self.external_drive - inhibition
        self.external_drive *= self.drive_decay

        can_fire = self.refractory <= 0
        self.spikes = ((self.v > THRESHOLD) & can_fire).float()
        self.v = torch.where(self.spikes > 0, torch.full_like(self.v, RESET), self.v)
        self.refractory = torch.where(
            self.spikes > 0, torch.full_like(self.refractory, self.refractory_steps), self.refractory - 1
        )
        self.refractory.clamp_(min=0)

        rate = self.spikes.mean().item()
        self.activity_ema = self.activity_ema * self.activity_ema_decay + rate * (1 - self.activity_ema_decay)
        return self.spikes

    def step_batch(self, n_steps: int) -> list[int]:
        spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            spiked |= self.step() > 0
        return spiked.nonzero().squeeze(1).tolist()
