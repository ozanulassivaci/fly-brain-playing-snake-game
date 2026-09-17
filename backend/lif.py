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
INHIB_GAIN = {
    "motion": 40.0,
    "cx": 400.0,
    "dn": 40.0,
    "fc": 400.0,
    "pfl": 400.0,
    "epg": 400.0,
    "lc10": 400.0,
    "lplc1": 400.0,
    "pn": 400.0,
    "lh": 400.0,
    "kc": 400.0,
    "mbon": 400.0,
    "pam": 400.0,
    "apl": 400.0,
    # Phase 3.13's premotor cluster (LAL / PS / AOTU / VES). Its own pool,
    # like every other population here — see the note above: a cluster
    # sharing a pool with a driven one gets crushed by that one's overshoot.
    "lal": 400.0,
}
ACTIVITY_EMA_TAU_MS = 20.0

# Phase 3: real per-neuron structure used for sensory-in/motor-out, not made
# up. T4/T5 subtypes a/b/c/d are real (Drosophila literature: these four
# subtypes are each tuned to one of the four cardinal motion directions;
# T4 = ON-edge motion, T5 = OFF-edge, same directional tuning, so each
# direction letter combines both).
DIRECTION_TYPE_PATTERN = re.compile(r"^T[45]([abcd])")
SENSORY_SCALE = 0.5
# How long the steering readout is averaged over, and the second knob in
# this file to be right, then wrong, then right again as the rest changed.
#
# Phase 3.5 cut it from 150ms to 15ms because the lag was making the fly
# act on its pre-turn bearing and orbit the apple — true at the time,
# when a held decision was applied on *every* tick. Phase 3.10 rate-limited
# turning to one per two ticks, which removes most of that lag penalty and
# leaves the other side of the trade-off exposed: only 16 neurons per side
# feed this readout, so a 15ms window is extremely noisy. Measured
# directly, signal (apple 50 degrees off) against noise (no input at all):
#
#     tau    noise p90    signal median    correctly signed
#      15     0.00154        0.00104             89%
#      40     0.00123        0.00144             97%
#      80     0.00072        0.00138             98%
#     150     0.00061        0.00135            100%
#
# At 15ms the noise p90 is *above* the signal median — the fly was turning
# on noise as often as on the apple, which is exactly the aimless circling
# that kept being reported. At 150ms the signal is correctly signed every
# time. With turning rate-limited, the lag costs little and the
# signal-to-noise is worth far more.
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
# Set above the measured noise floor of the readout rather than at the old
# weak-signal noise level. At the previous 0.0001 the no-input |diff|
# crossed the threshold 42% of the time, so the fly was committing to
# turns on noise; 0.003 sits above the 150ms noise p90 (0.00061) and below
# the driven signal, so only a real bearing error turns it.
TURN_ON_THRESH = 0.003
TURN_OFF_THRESH = 0.003

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
# Raised from 1.0 in Phase 3.11 once the turn threshold was lifted clear of
# the noise, then cut back to 10.0 in Phase 3.12 — the third knob in this
# file to be right, then wrong, then right again as what reads it changed.
# At 30 the drive was so strong that the steering readout saturated: the
# signed left-right difference, measured against a held bearing with the
# deadzone off, was
#
#     bearing off-axis   0     15    30    45    60    90   120   150   180
#     scale 30       0.0002 .0093 .0171 .0194 .0195 .0200 .0194 .0168 .0000
#     scale 10       0.0001 .0031 .0061 .0087 .0099 .0123 .0104 .0060 .0000
#
# i.e. at 30 the readout is flat from 30 degrees out — an on/off command
# carrying no information about *how far* off the target is, confirmed in
# play (median signed difference varied only 0.0106-0.0124 across every
# bearing bin). At 10 it grades smoothly all the way out to 90 degrees and
# a 15-degree error still sits 5x above the readout's own noise floor
# (p90 0.0006). The turn command only became a rate worth grading once
# frontend/js/snake-game.js started integrating it into a heading rather
# than thresholding it, which is why this is only now the right value.
VISUAL_SCALE = 10.0
# Phase 3.6 added a 45-degree frontal deadzone here — no turn command while
# the target sat within that angle of straight ahead — and Phase 3.12
# removed it. It was never about LC10: real LC10 tiles the whole visual
# field, front included, and has no hole in the middle. It was a patch for
# the *body*: on a 4-direction grid the best reachable heading can be 45
# degrees off the target, so a fly that insisted on perfect alignment got a
# strong "turn!" command (sin 41 deg = 0.66) while already pointed as well
# as it could be, turned, ended up 49 degrees off the other way, and turned
# again — a limit cycle, traced live as the snake looping (-1,0) -> (0,1)
# -> (1,0) -> (0,-1) with the egocentric bearing repeating -138, -52, +41,
# +135 and never nearing zero. The continuous heading in
# frontend/js/snake-game.js removes the quantisation that caused it, so the
# patch can go: with it gone the fly steers correctly 0.81 of the time when
# the apple is 120-180 degrees behind it, against 0.51 with the deadzone in
# place. Keeping a blind spot pointed exactly where the food is was costing
# more than it saved.

# Phase 3.7: obstacle avoidance via LPLC1, the real loom-sensitive visual
# projection neuron. Once the fly started actually eating, the snake grew,
# and it died almost immediately afterwards — it had no channel at all
# through which its own lengthening body could be perceived, so it curled
# straight into itself. Real flies do have one: looming-sensitive lobula
# columnar cells driving both escape and avoidance steering.
#
# Checked against this dataset before choosing LPLC1 (not assumed): of the
# loom-associated types present, LPLC2 / LC4 / LC6 / LC16 / LC11 have
# *zero* connectivity to DNa* steering neurons — they feed the giant-fiber
# takeoff escape instead, which matches the real division of labour.
# LPLC1 (134 neurons, 68 L / 66 R, all cholinergic) does both: it reaches
# the classic escape descending neurons (DNp03 3602, DNp35 3194, DNp06
# 2773, DNp11 1041) *and* DNa07 (398) — a steering DN already inside this
# project's readout — and its DNa projection is perfectly ipsilateral
# (LPLC1_L -> DNa_L 272, LPLC1_L -> DNa_R 0, mirrored on the other side),
# the same clean structure LC10 has.
#
# Because that projection is ipsilateral and driving DNa_L turns the fly
# left (the arbitrary-but-consistent convention documented at
# _update_motor), a threat on the *left* is injected into LPLC1_*right*,
# so the turn goes away from it. That side assignment is ours, exactly as
# the DNa-to-turn convention already is; everything downstream of the
# injection is real anatomy. Approach (LC10) and avoidance (LPLC1) drives
# then simply sum at the descending neurons, which is how competing
# steering drives resolve in a real brain too.
LPLC1_TYPE_PATTERN = re.compile(r"^LPLC1$")
OBSTACLE_SCALE = 2.5

# Reading the avoidance signal off DNa alone did not work, and the reason
# is real rather than a tuning failure: only ~4% of LPLC1's descending
# output reaches DNa07 (398 weight). The other ~96% goes to the escape
# descending neurons below (DNp03 1988+1614, DNp35 1750+1444, DNp06
# 1604+1169, DNp11 542+499, DNp103 390+608 — all perfectly ipsilateral,
# zero crosstalk, same clean structure as LC10). In a real fly that output
# triggers *escape*: takeoff, or a backward lunge. Snake has no such
# action — the body can only go left, right or straight — so the fly's
# strongest collision response had almost nowhere to express itself, and
# the apple-approach drive (measured ~0.006 at DNa, vs 0.0016 from the
# LPLC1 route) simply outvoted it every time.
#
# So the escape command is read out separately and mapped onto the only
# evasive action this body has: a hard turn away. That mapping is ours and
# is the same kind of choice as the DNa-to-turn convention; the pathway,
# its weights and its lateralisation are the connectome's.
ESCAPE_DN_TYPE_PATTERN = re.compile(r"^DNp(03|06|11|35|103)$")
# Sized to put the two drives on equal footing rather than picked by feel:
# a close threat drives the escape readout to ~0.18, while the apple drive
# reaches ~0.006 at DNa, so 0.01 makes neither able to simply steamroll the
# other. Swept empirically too — higher values do stop the fly dying almost
# entirely (at 0.4, 12/12 episodes survive the full 60s) but it then hovers
# safely in open space and never commits to an apple, which is the classic
# approach-avoidance failure and not what we want.
ESCAPE_GAIN = 0.01

# Phase 3.8: odour. Real flies locate food mainly by smell, and unlike the
# frontal visual channel it works in every direction — including behind,
# where LC10 is blind and where an apple that has just respawned often is.
# Injected at the antennal-lobe projection neurons rather than at ORNs,
# because not one of this dataset's 2635 ORNs has a soma position (they
# are in the antenna, outside the reconstructed volume) — see
# prepare_subset.py. PN activity is the odour representation the rest of
# the brain receives, which is the honest level for this model.
#
# Bilateral, because that is how a walking fly localises an odour source:
# the two antennae sample slightly different concentrations and the
# difference steers the turn. The concentrations themselves are computed
# from real geometry in snake-game.js, exactly as the apple bearing is.
ODOUR_SCALE = 30.0
# Real olfactory neurons adapt: they report a *change* in concentration,
# not its absolute level. That matters here because it is the whole
# mechanism by which a walking fly finds a smell — concentration rising
# means "the way I am going is working", falling means "it is not" — and
# the first attempt at odour ignored it, injecting raw concentration
# bilaterally and asking the mushroom body for a left/right steering
# command it structurally cannot give (each Kenyon cell samples glomeruli
# at random, so the bilateral difference does not survive; LH->DNa* is
# only 66 weight). Subtracting a slowly-tracking baseline makes PN encode
# the derivative instead, which is both what the real neurons do and what
# the behaviour actually needs.
ODOUR_ADAPT_TAU_MS = 1500.0

# How the odour signal reaches behaviour: not as a turn direction, but as
# a turn *suppressor*, read off the lateral horn. LH is olfaction's innate
# output (PN->LH 378,010 weight; LH->descending neurons 22,810) as opposed
# to the mushroom body's learned one, and it is bilateral and symmetric —
# it raises both sides together rather than tilting them apart. Read as
# common mode rather than difference, that is exactly the klinokinesis a
# real fly uses: while the smell is getting stronger, hold your course;
# when it stops getting stronger, start turning again and search. The
# direction of any turn still comes entirely from the visual LC10/LPLC1
# drives and their real ipsilateral wiring.
#
# Measured first with the mushroom body in this role, which failed for a
# real reason worth keeping: PN responds beautifully (0.050 approaching
# vs 0.00015 receding, a 300x separation from the adaptation above), but
# MBON activity *drops* when odour rises — the mushroom body is a
# sparse-coding layer with APL inhibition and genuinely inhibitory MBONs,
# so it suppresses rather than relays. That is the learned-valence
# pathway doing its job, not a bug, and it is the wrong half of olfaction
# for innate food seeking.
ODOUR_TURN_SUPPRESSION = 10000.0

# Phase 3.9: dopamine. Eating an apple drives the PAM cluster — the real
# dopaminergic reward neurons, 314 of them, which is where reward goes in
# a fly and is a correction of Phase 1's version of this event, which
# drove the descending neurons directly and corrupted steering at exactly
# the moment the fly had just reached an apple.
#
# What is deliberately *not* here is mushroom-body plasticity. It was
# built and measured first, because that is what dopamine actually does:
# PAM projects onto the Kenyon cells (109,735 PAM->KC edges here), gating
# the KC->MBON synapses (59,709 edges, 460,343 weight), and the canonical
# Drosophila rule is that dopamine coinciding with recent KC activity
# depresses them. It worked mechanically — a reward burst measurably
# depressed the traced synapses by 22% — and made the fly slightly worse
# (14 apples over ten minutes against 21 with the rule off), with the
# plastic weights collapsing to their floor at 25%.
#
# That is not a tuning failure, it is the task: associative learning needs
# something to associate. This game has exactly one odour, the apple, and
# it is always rewarded. With no second cue to discriminate against, the
# only thing the rule can do is depress everything uniformly, which is a
# gain change rather than a memory. Worth revisiting if the game ever
# gains a second smell worth telling apart.
REWARD_PAM_DRIVE = 1.5

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
        # LPLC1 gets its own homeostatic pool for the same reason LC10 does:
        # a sustained looming injection would otherwise drag the whole
        # motion-cluster average up and have that cluster's inhibition
        # crush the very signal being injected.
        is_lplc1 = np.array([bool(LPLC1_TYPE_PATTERN.match(t)) for t in neuron_type])
        cx_other = (cluster == "cx") & ~is_fc & ~is_pfl & ~is_epg
        motion_other = (cluster == "motion") & ~is_lc10 & ~is_lplc1
        self.cluster_masks = {
            "motion": torch.from_numpy(motion_other.astype(np.float32)).to(self.device),
            "cx": torch.from_numpy(cx_other.astype(np.float32)).to(self.device),
            "dn": torch.from_numpy((cluster == "dn").astype(np.float32)).to(self.device),
            "fc": torch.from_numpy(is_fc.astype(np.float32)).to(self.device),
            "pfl": torch.from_numpy(is_pfl.astype(np.float32)).to(self.device),
            "epg": torch.from_numpy(is_epg.astype(np.float32)).to(self.device),
            "lc10": torch.from_numpy(is_lc10.astype(np.float32)).to(self.device),
            "lplc1": torch.from_numpy(is_lplc1.astype(np.float32)).to(self.device),
            "pn": torch.from_numpy((cluster == "pn").astype(np.float32)).to(self.device),
            "lh": torch.from_numpy((cluster == "lh").astype(np.float32)).to(self.device),
            "kc": torch.from_numpy((cluster == "kc").astype(np.float32)).to(self.device),
            "mbon": torch.from_numpy((cluster == "mbon").astype(np.float32)).to(self.device),
            "pam": torch.from_numpy((cluster == "pam").astype(np.float32)).to(self.device),
            "apl": torch.from_numpy((cluster == "apl").astype(np.float32)).to(self.device),
            "lal": torch.from_numpy((cluster == "lal").astype(np.float32)).to(self.device),
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
        self.lplc1_left_mask = torch.from_numpy((is_lplc1 & (soma_side == "L")).astype(np.float32)).to(self.device)
        self.lplc1_right_mask = torch.from_numpy((is_lplc1 & (soma_side == "R")).astype(np.float32)).to(self.device)

        is_pn = cluster == "pn"
        self.pn_left_mask = torch.from_numpy((is_pn & (soma_side == "L")).astype(np.float32)).to(self.device)
        self.pn_right_mask = torch.from_numpy((is_pn & (soma_side == "R")).astype(np.float32)).to(self.device)

        is_escape_dn = np.array([bool(ESCAPE_DN_TYPE_PATTERN.match(t)) for t in neuron_type])
        self.escape_left_mask = torch.from_numpy((is_escape_dn & (soma_side == "L")).astype(np.float32)).to(
            self.device
        )
        self.escape_right_mask = torch.from_numpy((is_escape_dn & (soma_side == "R")).astype(np.float32)).to(
            self.device
        )
        self.escape_left_count = max(1.0, float(self.escape_left_mask.sum().item()))
        self.escape_right_count = max(1.0, float(self.escape_right_mask.sum().item()))

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
            "lplc1_left": self.lplc1_left_mask,
            "lplc1_right": self.lplc1_right_mask,
            "escape_left": self.escape_left_mask,
            "escape_right": self.escape_right_mask,
            "pn_left": self.pn_left_mask,
            "pn_right": self.pn_right_mask,
            "lh": self.cluster_masks["lh"],
            "mbon": self.cluster_masks["mbon"],
            "kc": self.cluster_masks["kc"],
            "epg": self.cluster_masks["epg"],
            "fc": self.cluster_masks["fc"],
            "pfl": self.cluster_masks["pfl"],
            "lal": self.cluster_masks["lal"],
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
            + [self.dn_left_mask, self.dn_right_mask, self.escape_left_mask, self.escape_right_mask]
        )

        self.leak_decay = float(np.exp(-DT_MS / LEAK_TAU_MS))
        self.drive_decay = float(np.exp(-DT_MS / DRIVE_DECAY_TAU_MS))
        self.activity_ema_decay = float(np.exp(-DT_MS / ACTIVITY_EMA_TAU_MS))
        self.motor_ema_decay = float(np.exp(-DT_MS / MOTOR_EMA_TAU_MS))
        self.odour_adapt_decay = float(np.exp(-DT_MS / ODOUR_ADAPT_TAU_MS))
        self.refractory_steps = int(round(REFRACTORY_MS / DT_MS))

        self.dn_left_ema = 0.0
        self.dn_right_ema = 0.0
        self.escape_left_ema = 0.0
        self.escape_right_ema = 0.0
        self.odour_baseline = None

        self.current_turn = "straight"
        self.turn_diff = 0.0

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
        if "odour_left" in values or "odour_right" in values:
            self.inject_odour(float(values.get("odour_left", 0.0)), float(values.get("odour_right", 0.0)))
        if "threat_left" in values or "threat_right" in values:
            self.inject_obstacle(float(values.get("threat_left", 0.0)), float(values.get("threat_right", 0.0)))
        if "bearing" in values and "heading" in values:
            ego_bearing = np.arctan2(
                np.sin(values["bearing"] - values["heading"]), np.cos(values["bearing"] - values["heading"])
            )
            self.inject_visual_target(float(ego_bearing))

    def inject_reward(self) -> None:
        # Reward goes where reward goes in a fly: the PAM dopaminergic
        # cluster. Phase 1's version of this event drove the descending
        # neurons directly, which corrupted steering at exactly the wrong
        # moment; this drives the neurons whose job it actually is.
        self.external_drive += self.cluster_masks["pam"] * REWARD_PAM_DRIVE

    def inject_odour(self, odour_left: float, odour_right: float) -> None:
        # Adapting drive: PN sees how much the smell has strengthened since
        # its recent baseline, not how strong it is. Getting closer to the
        # apple therefore produces a real transient; sitting still at any
        # distance, however strong the smell, produces nothing — which is
        # how olfactory receptor neurons genuinely behave.
        mean_conc = (odour_left + odour_right) / 2.0
        if self.odour_baseline is None:
            self.odour_baseline = mean_conc
            return
        rise = mean_conc - self.odour_baseline
        self.odour_baseline += (mean_conc - self.odour_baseline) * (1 - self.odour_adapt_decay)
        if rise <= 0:
            return
        # Split by antenna so the side nearer the source still gets more,
        # even though what drives behaviour is the common-mode rise.
        total = odour_left + odour_right
        share_left = odour_left / total if total > 0 else 0.5
        self.external_drive += self.pn_left_mask * (rise * share_left * 2.0 * ODOUR_SCALE)
        self.external_drive += self.pn_right_mask * (rise * (1 - share_left) * 2.0 * ODOUR_SCALE)

    def inject_obstacle(self, threat_left: float, threat_right: float) -> None:
        # Contralateral by construction: a threat on the left drives the
        # right-hand LPLC1 population, whose real ipsilateral projection to
        # DNa turns the fly right, away from it. See the LPLC1_TYPE_PATTERN
        # comment for why that side assignment is ours and everything after
        # the injection is the connectome's.
        if threat_left > 0:
            self.external_drive += self.lplc1_right_mask * (threat_left * OBSTACLE_SCALE)
        if threat_right > 0:
            self.external_drive += self.lplc1_left_mask * (threat_right * OBSTACLE_SCALE)

    def inject_visual_target(self, ego_bearing: float) -> None:
        # Egocentric: 0 = target straight ahead, +-pi/2 = directly to the
        # side. A smooth sin-based split rather than a hard left/right
        # switch, since we only have real L/R separation to work with (no
        # retinotopic position label — see the Phase 3.4 comment above
        # LC10_TYPE_PATTERN) — this still gives zero drive when the target
        # is dead ahead or directly behind and maximum drive when it's
        # squarely to one side, without an arbitrary discontinuity at 0.
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

    def _update_motor(self, left_rate: float, right_rate: float, esc_left: float, esc_right: float) -> None:
        self.dn_left_ema = self.dn_left_ema * self.motor_ema_decay + left_rate * (1 - self.motor_ema_decay)
        self.dn_right_ema = self.dn_right_ema * self.motor_ema_decay + right_rate * (1 - self.motor_ema_decay)
        self.escape_left_ema = self.escape_left_ema * self.motor_ema_decay + esc_left * (1 - self.motor_ema_decay)
        self.escape_right_ema = self.escape_right_ema * self.motor_ema_decay + esc_right * (1 - self.motor_ema_decay)

        # Approach (DNa, driven by LC10) and escape (DNp, driven by LPLC1)
        # summed into one steering decision — competing drives resolving at
        # the descending level, which is where they meet in a real brain too.
        diff = (self.dn_right_ema - self.dn_left_ema) + ESCAPE_GAIN * (
            self.escape_right_ema - self.escape_left_ema
        )
        # Klinokinesis: while the mushroom body reports a strengthening
        # smell, hold course; the threshold to commit to a turn rises with
        # it. See ODOUR_TURN_SUPPRESSION.
        # Measured against the resting rate the homeostasis holds every
        # population at, so an unchanging smell (however strong) suppresses
        # nothing and only a genuine rise does.
        lh_rise = max(0.0, self.group_activity_ema.get("lh", 0.0) - TARGET_RATE)
        turn_on = TURN_ON_THRESH * (1.0 + ODOUR_TURN_SUPPRESSION * lh_rise)
        # "more right-side steering-DN activity -> turn right" is a
        # consistent convention we chose, not something derivable from the
        # data alone (we don't have the actual sign of the DNa02-leg-motor
        # mapping) — but the *population* being read is now the real,
        # specifically identified steering DN family, not an arbitrary cut.
        if self.current_turn == "straight":
            if diff > turn_on:
                self.current_turn = "right"
            elif diff < -turn_on:
                self.current_turn = "left"
        elif abs(diff) < TURN_OFF_THRESH:
            self.current_turn = "straight"

        # Phase 3.12: the signed difference itself, published alongside the
        # thresholded left/right/straight above, because a real descending
        # neuron pair encodes turn *velocity* in its firing-rate difference
        # rather than a discrete choice of direction. The thresholded form
        # is what the decision panel displays; this is what actually steers
        # (frontend/js/snake-game.js integrates it into a heading).
        #
        # A first attempt reported a normalised 0-1 "strength" relative to
        # turn_on instead, so that a strong command could turn twice in a
        # row. It measured far worse (2 apples against 26 over 10 seeds),
        # and logging showed why: dividing by an odour-modulated threshold
        # scrambled the quantity, and the strength that came out was
        # *anti*-correlated with the steering error — 0.90 mean with the
        # apple within 30 degrees, 0.28 with it 150-180 degrees behind. The
        # raw difference, once VISUAL_SCALE came down out of saturation, is
        # the graded quantity that attempt was reaching for.
        self.turn_diff = diff

    def read_motor(self) -> str:
        return self.current_turn

    def read_turn_rate(self) -> float:
        """Signed right-minus-left steering difference: a turn velocity."""
        return self.turn_diff

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
        self._update_motor(
            rates[-4] / self.dn_left_count,
            rates[-3] / self.dn_right_count,
            rates[-2] / self.escape_left_count,
            rates[-1] / self.escape_right_count,
        )
        return self.spikes

    def step_batch(self, n_steps: int) -> list[int]:
        spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            spiked |= self.step() > 0
        return spiked.nonzero().squeeze(1).tolist()
