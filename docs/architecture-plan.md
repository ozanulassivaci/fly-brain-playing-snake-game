# Architecture Plan

Status: brainstorming phase, no implementation started yet. This document
records the architectural decisions made so far and the phase breakdown we
agreed on. It will be revised as we learn more (especially once real
MaleCNS data is downloaded and inspected).

## Vision

A live 3D demo: a biomechanical fly (NeuroMechFly v2 model) sits at an
arcade cabinet and plays Snake by pressing a joystick/buttons. The demo
shows, side by side, the game screen, the fly's physical movement, and a
visualization of brain activity driven by the MaleCNS v1.0 connectome
(166,700 neurons, 125M synapses).

The goal is a visually compelling real-time demo, not a research-grade
biological simulation. Biological plausibility is a means to an
impressive visual, not an end in itself.

## Key architectural decisions

### 1. Compute split: local Python/CUDA backend + browser frontend

The user has a local Ubuntu machine with an RTX 4070 and wants to run
everything locally.

Decision: the neuron simulation runs as a separate Python process using
CUDA/PyTorch for sparse matrix operations, streaming spike/activity data
to the browser over WebSocket. The browser (Three.js) handles rendering
only.

Rejected alternative: running the LIF simulation directly in-browser via
WebGPU compute shaders. This would keep the "single browser tab" demo
property from the original vision, but requires writing sparse neural
simulation at a much lower level than PyTorch/CUDA gives us for free.
Revisit only if the two-process setup proves inconvenient.

### 2. Reuse strategy: cherry-pick assets from fly-parking-lab, not a full fork

We cloned and inspected `powerOFMAX/fly-parking-lab` (NeuroMechFly v2 +
MaleCNS + MuJoCo WASM + Three.js, a fly driving a car). Findings:

**Real and reusable (Apache 2.0 licensed):**
- `public/nmf/shared/vendor/mujoco/` — working MuJoCo WASM build
  (mujoco.js, mujoco.wasm, mujoco.d.ts)
- `public/nmf/game/assets/model_meta.json` — real NeuroMechFly v2 MuJoCo
  model: 73 generalized coordinates, 48 actuators, full joint/actuator
  definitions
- `solve2BoneIK()` (game.js:961) — real, working analytical 2-bone IK
  solver used per limb

**Advertised in the README but not actually present:**
- No `connectome/` directory exists. No real MaleCNS connectivity data
  is loaded anywhere in the codebase.
- The "brain visualization" (`_buildNeurons()`, game.js:333) is a
  hand-authored set of a few dozen neuron definitions with fixed
  positions, fixed colors, and fixed synthetic connections — decorative,
  not derived from real connectome data.
- The car is actually driven by `autopilot.mjs`, a scripted
  non-holonomic waypoint planner. It never reads from the "connectome."
  The brain panel just animates in sync with the autopilot's decisions.

Decision: cherry-pick the MuJoCo WASM build, the NeuroMechFly model
assets, and the IK solver. Do not fork the surrounding Next.js/Drizzle
web-app scaffold (routing, DB, leaderboard forms — irrelevant to this
project). Do not reuse the decorative brain visualization or the
autopilot logic; both need to be built for real.

Implication: the hardest part of this project — actually loading MaleCNS
data and using it to drive behavior — is not solved by any reference
project we've found. Budget for it as net-new work regardless of what we
reuse for body/physics/rendering plumbing.

### 3. Control loop scope: functional subset drives the game, full connectome is decorative-only

Decision: a functional subset of the connectome (optic lobe + central
complex + relevant descending neurons) is simulated in real time and
actually drives gameplay. The full 166k-neuron graph is used only for
the static/background visualization in the brain panel (structure shown
at all times, real subset's activity highlighted on top).

This substantially de-risks the real-time performance problem while
keeping the "real 166k-neuron connectome" visual claim honest (it's
genuinely loaded and shown, just not all of it is simulated live).

### 4. MVP physics: scripted animation before MuJoCo contact physics

Decision: Phase 1 uses scripted/procedural leg animation triggered by
discrete motor commands (walk forward, turn left/right, press button).
MuJoCo contact physics (fly limbs actually gripping a joystick/buttons)
is deferred to Phase 3, once the sensorimotor loop is being closed for
real.

## Data access findings (Phase 0 research, 2026-09-14)

Source: web research on MaleCNS v1.0 / neuPrint / FlyEM. See citations
below; items marked "inferred" were not directly confirmed against
male-cns-specific documentation.

- **Access method**: both a neuPrint API (`neuprint-python`, requires an
  account/API token) and a static bulk export
  (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, Arrow/
  Feather tables, no account needed) exist for `male-cns:v1.0`. Plan to
  use the bulk export as the primary source (versioned, doesn't depend
  on server uptime); use the API only for ad hoc exploration.
- **Schema**: three separate object types, not one flat table —
  `Neuron` (bodyId, type, instance, roiInfo, somaLocation, class_,
  cellBodyFiber, exitNerve, ...), `Synapse` (x/y/z, pre/post type,
  confidence, roi), `Connection`/`:ConnectsTo` (weight, per-ROI weight
  breakdown).
- **Filtering**: optic lobe (`ME(R)`, `LO(R)`, ...) and central complex
  (`EB`, `FB`, `PB`, `NO`) are filterable via the standard FlyEM
  `roiInfo` ROI ontology (confirmed pattern across FlyEM datasets,
  inferred to apply identically to male-cns). Descending neurons have no
  simple live filter — the community relies on published, hand-curated
  DN type catalogs (2025 Nature comparative connectomics paper; 2025
  bioRxiv split-GAL4 DN driver catalogue) cross-referenced against
  male-cns type names.
- **Realistic subset size (superseded by empirical findings below)**:
  literature-based estimate was optic lobe ~53,000 neurons, central
  complex ~3,000 (hemibrain analogy), descending neurons ~1,300
  (MANC-specific). Real male-cns data (see below) shows the optic lobe
  is actually much bigger than this estimate.
- **License**: CC-BY, attribution required (FlyEM/HHMI Janelia +
  Cambridge/MRC LMB + Google Research; cite the *Cell* paper, bioRxiv
  DOI 10.1101/2025.10.09.680999). No stated commercial-use or
  redistribution restriction beyond attribution. Exact license version
  (e.g. 4.0) not directly confirmed from a formal ToS page — worth a
  final check before redistributing any derived/filtered dataset.
  This is separate from and in addition to the NeuroMechFly model's own
  Apache 2.0 license, which only covers the biomechanical model assets.

## Empirical findings from real MaleCNS v1.0 data (2026-09-14)

Downloaded directly from the public bulk export
(`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, no
account needed, confirmed publicly listable via the GCS JSON API):
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (211,577 bodies,
36 annotation columns including `type`, `superclass`, `status`) and
`connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather`
(25.6M weighted connections, columns `body_pre`, `body_post`, `weight`,
`type_pre`, `type_post`). Files kept locally under `data/raw/`
(gitignored, not committed — see licensing note above on attribution
requirements if this data or derivatives are ever published).

**Real neuron counts** (filtering to `status == 'Traced'`, 165,122
neurons total — matches the project's stated 166,700 closely):

| Group | Filter used | Count |
| --- | --- | --- |
| Optic lobe (all visual superclasses) | `superclass` in `{ol_intrinsic, ol_sensory, visual_projection, visual_centrifugal}` | 103,268 |
| Motion/looming pathway only | `type` matches `T4`, `T5`, `LC*`, `LPLC*`, `LT*` | 18,457 |
| — of which core motion detectors | `type` matches `T4`, `T5` | 13,585 |
| Central complex (ROI-based, cross-checked) | `roiInfo` has synweight >= 5 in EB, FB, PB(any glomerulus), or NO | 3,137 |
| Descending neurons | `superclass == 'descending_neuron'` | 1,314 |

The full optic lobe (103k) is ~2x the literature-based estimate and is
~62% of the entire traced connectome — far too large for a first
real-time simulation target. The motion/looming pathway (T4/T5 + LC/
LPLC/LT) is a much smaller, biologically meaningful proxy for what a
Snake-playing fly actually needs (direction of movement, collision/
approach cues) rather than full visual acuity.

**Candidate functional subset: motion pathway + CX + DN = 22,893
neurons** (~14% of the full connectome). Its internal connectivity:
1,183,489 edges fully inside the subset, average out-degree ~51.7.

**Important caveat found while checking this candidate subset**: it is
not a self-contained circuit. Checking what fraction of each group's
inputs come from within the 22,893-neuron subset itself:

| Group | Inputs from within subset |
| --- | --- |
| Motion pathway (T4/T5/LC/LPLC/LT) | 25.9% |
| Central complex | 72.5% |
| Descending neurons | 17.0% |

The motion pathway's missing ~74% of input comes from earlier optic
lobe layers (Mi1, Tm1-9, etc.) that were deliberately excluded — this
is expected and acceptable, since Phase 3's plan already calls for
injecting a synthetic "motion energy" signal derived from the Snake
screen directly into this layer rather than simulating the full deep
visual pathway. Central complex is well-preserved (72.5% internal) —
its real ring-attractor/heading-integration dynamics are worth actually
simulating. Descending neurons are the weak point: only 17.0% of their
real input is captured, so simulating realistic DN spiking dynamics
from this subset alone would be misleading. Decision: treat CX output →
DN as a simplified categorical mapping (CX heading/turn signal → a
handful of well-known steering-related DN types, e.g. DNa01/DNa02-like)
rather than expecting biologically faithful DN spike trains. This is
consistent with the project's own framing of the reward/steering signal
as behavioral shaping, not real learning.

**ROI-based cross-check of the central complex selection (done).**
Central complex was initially selected by type-name prefix matching
(EPG, PEN, PEG, PFN, PFL, PFR, PFG, hDelta, vDelta, Delta7, ExR, FB,
EB, PB, NO, LNO, SpsP, IbSpsP, FS, FC, FR against the `type` column):
2,643 neurons. To verify this against the dataset's actual ROI
structure, downloaded the full `Neuprint_Neurons.feather` (4.6 GB,
from `v1.0/database/neuprint-inputs/` — the neuPrint Neo4j import
table, distinct from the smaller `flat-connectome` annotations file
and the only place `roiInfo` actually lives) and filtered by `roiInfo`
synapse weight (synweight >= 5) in EB, FB, any PB glomerulus, or NO —
the confirmed FlyEM ROI names for the central complex, verified
directly from real roiInfo keys in this dataset. Result: 3,137
neurons, 2,631 of which overlap with the type-name set (99.5%
agreement on the type-name side). Two real corrections came out of
this:
- 506 neurons the type-name heuristic missed, almost all `ER*` types
  (ellipsoid body ring neurons) plus `EL`/`SA*` — a real class of
  central-complex ring-attractor neurons the original prefix list
  simply didn't include.
- 12 false positives in the type-name set (`Nod1`-`Nod5`), which
  matched the `NO` prefix by name coincidence but are actually
  `visual_projection` neurons unrelated to the noduli.
The central complex count and all totals above already use the
corrected, ROI-based figure (3,137).

## Phase breakdown

**Phase 0 — Data discovery (done).** Confirmed real access method
(public GCS bulk export, no account needed), schema (separate
annotation/weight tables), and — critically — real neuron/connectivity
counts by downloading and querying the actual data (see empirical
findings above). Landed on a concrete candidate subset: motion pathway
(T4/T5/LC/LPLC/LT, 18,457) + central complex (2,643) + descending
neurons (1,314) = 22,414 neurons, with the input-completeness caveat
noted above.

**Phase 1 — Visual skeleton (done).** Cherry-picked MuJoCo/NeuroMechFly
body, arcade cabinet scene, Snake screen (scripted/keyboard-controlled),
a decorative brain animation synced to game events (similar in spirit to
fly-parking-lab's, explicitly not real neural data yet). Leg movement is
scripted/procedural, no MuJoCo contact physics. Goal: visually
convincing, no real neural data involved yet.

Implemented as a static, no-build-step page under `frontend/` (plain ES
modules + import map — no bundler, per the tooling decision above). The
fly's idle animation plays back NeuroMechFly's own recorded
walking-cycle joint angles (`model_meta.json`'s `preprogrammed.legs`)
via `mj_forward` only (kinematics, no `mj_step`/contact dynamics).
Noteworthy find: the cherry-picked `fly.xml` bundles fly-parking-lab's
driving-course markers (`gate*`, `ground_plane`, `start_pole_*`) in the
same MJCF as the fly body — `frontend/js/fly-body.js` filters these out
by geom name prefix (`nmf/` vs. everything else) before rendering.
Worth remembering if a later phase touches `fly.xml` again.

**Phase 1 polish (done).** After first-look feedback, pulled two small
pieces of later phases forward, both still decorative/scripted rather
than closing a real loop: (1) the front-left leg now reaches toward the
joystick on every snake direction change, via a numerical
coordinate-descent solver in `fly-body.js` (adjusts coxa-yaw/coxa-pitch/
femur-pitch qpos to minimize the tarsus tip's distance to the joystick
ball, verified against real `mj_forward` output rather than trusting
closed-form trig on the joints' local axis conventions) — the joystick
itself also tilts toward the current direction
(`cabinet-scene.js`'s `setJoystickTilt`); (2) the brain panel
(`brain-viz.js`) now renders a real 3D point cloud using actual MaleCNS
soma positions for the Phase 0 candidate subset
(`frontend/assets/brain-subset.json`, ~22.7k points, derived locally
from already-downloaded data — see THIRD_PARTY_NOTICES.md) instead of a
synthetic 2D layout, still only decoratively pulsing on game events.

Follow-up fixes after first-look feedback on this round: the fly was
lying on its side and later facing parallel to the screen rather than
into it — root cause was that once the Z-up-to-Y-up correction fixes
one Euler axis at -90°, Three.js's default 'XYZ' Euler order makes the
remaining "heading" axis rotate in the world X-Y plane instead of
turning left/right, so no single-Euler value could ever face the
screen. Fixed by composing two separate quaternions (stand-upright,
then heading around world Y) instead of one combined Euler — verified
with a temporary `ArrowHelper` on the head axis read from a straight
top-down camera, since an oblique view had given false confidence
twice already. Once facing the screen, the fly's long axis pointed in
Z instead of X, so it clipped into the control panel until
repositioned; the reach also had to move from the front-left to the
front-right leg (now the side nearer the joystick) and blend from the
leg's pose at trigger time rather than the live gait angle, since the
0.7 Hz walking cycle was otherwise fighting the reach and making it
look jittery.

**Phase 2 — Real connectome simulation (done).** `backend/` runs a real
LIF (leaky integrate-and-fire) simulation of the Phase 0 candidate
subset on the local RTX 4070 (PyTorch/CUDA), streaming spikes to the
browser over a FastAPI WebSocket (`backend/server.py`, `backend/lif.py`).
`backend/scripts/prepare_subset.py` is now the single reproducible
source for both the frontend's `brain-subset.json` and the backend's
`data/processed/subset.npz` — they must share the same neuron index
order, which the earlier ad hoc script didn't guarantee. `brain-viz.js`
renders real per-neuron activity (per-point vertex colors, flash-and-
decay) instead of the old decorative per-cluster pulse; it still
degrades gracefully to a static structure if the backend isn't running.

Per the user's choice, the simulation isn't purely spontaneous — Snake's
move/eat/collide events are sent to the backend over the same WebSocket
and inject extra current into the matching neuron cluster — a small,
contained step toward Phase 3, not the closed loop itself (Snake is
still human-controlled; the brain sim doesn't drive gameplay).

Real, load-bearing caveat found while tuning this: the weight table has
no excitatory/inhibitory sign (we didn't pull neurotransmitter-type
data in Phase 0), so every connection in this subset is positive. A
purely excitatory recurrent network at this connectivity density
saturates almost immediately — every tested weight/noise combination
without correction drove 15-85% of the whole network to spike every
20ms. Added a simple population-level homeostatic control loop
(proportional negative feedback toward a target firing rate,
`TARGET_RATE`/`INHIB_GAIN` in `backend/lif.py`) standing in for the
real inhibitory neurons this subset doesn't model — documented as an
approximate stabilizer, not a biologically derived mechanism. All of
`WEIGHT_SCALE`, `NOISE_STD`, `TARGET_RATE`, etc. are empirically-tuned
knobs for a visually plausible sparse firing rate, not calibrated
parameters.

Also found mid-implementation: rendering ~18k+ points per cluster with
additive blending saturates to solid white well before any real
activity is involved, purely from point density — this had nothing to
do with Phase 2's new activity data (confirmed by testing with the
backend off). Fixed by further lowering per-cluster point size/opacity/
base-color-dimness (`BASE_SIZE`/`BASE_OPACITY`/`DIM_FACTOR` in
`brain-viz.js`) — worth remembering before adding more points later.

Testing note: headless-Chrome's `--virtual-time-budget` (used for every
prior phase's screenshot testing) hangs/spins indefinitely once the
page holds an open WebSocket — the virtual clock never reaches network
idle. Real-time testing against a live WebSocket needs a plain
`--remote-debugging-port` + a real `sleep` + a CDP
`Page.captureScreenshot` call instead.

**Phase 3 — Closed loop (done, MuJoCo contact physics deferred per user
choice).** Snake is now fully brain-controlled — no keyboard input at
all. `frontend/js/retina.js` runs a real Hassenstein-Reichardt
correlator (the textbook elementary-motion-detector model, not a
learned/deep model) over the Snake canvas, sending four motion-energy
scalars (a/b/c/d) to the backend every ~100ms. `backend/lif.py` injects
that current into the real T4/T5 a/b/c/d subtype masks (genuine
per-neuron structure — these four real subtypes are each tuned to one
of the four cardinal motion directions in the Drosophila literature)
and separately tracks left- vs right-soma-side descending-neuron
activity (also real `somaSide` data) to decode a turn-left/turn-right/
straight decision, sent back and applied via `snake-game.js`'s new
`applyTurn()` (rotates the current heading 90°; the keyboard handler
was removed entirely). The existing joystick-tilt/leg-reach logic in
`main.js` needed no changes — it already keyed off `snake.getDirection()`
changes, so it now fires from brain-driven turns instead of keyboard
ones.

**Real, load-bearing finding from this phase**: the per-cluster
homeostatic control added late in Phase 2 testing turned out to still
be too coarse — checking actual spike counts (not just the visualized
sparse indices) showed a subset of CX neurons cycling at the max
refractory-limited rate (~400k spike-events across CX over 4000 steps)
while DN sat almost silent (~2.7e-5 spikes/neuron/step) even though the
*global* average looked "on target". Fixed by giving each cluster
(motion/cx/dn) its own independent homeostatic feedback loop instead of
one shared pool; CX needed roughly 10x the inhibition gain of the other
two clusters to actually reach the target rate, confirming its
recurrent excitation really is much stronger. After the fix, all three
clusters sit close to the same real per-neuron firing rate. The
resulting DN left/right signal is still small and genuinely noisy
(same order of magnitude as sensory-driven signal), so the turn
decision thresholds were set relative to that real noise floor rather
than an assumed larger separation — this is the actual, honest
granularity this simplified subset's aggregate DN readout provides, not
an oversight.

As anticipated in this phase's own planning: because T4/T5 are real
motion-only detectors, the fly's "vision" cannot perceive the
stationary apple, only movement (mostly the snake's own body/head). Play
looks like real-neural-activity-driven reactive turning, not
intentional food-seeking — the honest result of this design, not a bug.

**Phase 3.2 — Goal-directed spatial memory: FC→PFL3→DNa02, with the
original 2D decision panel back underneath the 3D brain view (done,
mixed result, reported honestly).** User's explicit requirement:
increase apple-eating success using genuinely emergent brain activity —
no scripted/external control pretending to be the brain (rejected
outright, having confirmed firsthand that at least one well-known
online "fly plays a game" demo is a scripted autopilot behind a
decorative brain panel).

*First attempt (in this phase) and why it failed*: injected the apple's
egocentric bearing into FC (fan-shaped-body columnar neurons; real
goal-representation cell type, column position parsed from real
`_C{1-9}_` instance labels — `backend/scripts/prepare_subset.py`), read
out via all identified `DNa\d+` steering descending neurons (32
neurons, 16 L / 16 R — chosen over DNa02 alone, too few to read a rate
from, and over the full 1308-neuron DN aggregate Phase 3.1 used, which
dilutes the signal with unrelated escape/flight/grooming/feeding DNs).
This pathway is real and strong (checked against this dataset's
connectome-weights table before building it: FC→PFL 1585 edges/16296
weight; PFL→DNa02 specifically 28 edges at 17-51 weight each; all 24
PFL3 neurons connect to DNa02). Multi-trial testing
(`test_diag_repeat.py`, 6 independent `LifSimulation` instances) found
a real bug first: FC and PFL share the "cx" homeostatic cluster, so
injecting a goal bump into FC pushed the cx-cluster average far over
`TARGET_RATE` and the resulting proportional inhibition crushed PFL's
own activity to ~0 (measured: PFL fell from a baseline ~2e-4 to 1.4e-69
spikes/neuron/step) — the stability mechanism from Phase 2 was silently
strangling the exact relay Phase 3.2 needed. Fixed by giving FC, PFL,
and (later) EPG their own independent homeostatic pools instead of
sharing "cx"'s. After the fix, FC and PFL both responded strongly and
reliably to injection — but the DNa* L/R difference still came out
wrong-signed in 5 of 6 independent trials. Not noise-floor-sized this
time (FC/PFL activity moved by 1-2 orders of magnitude), but
unreliable in sign — a different, more interesting failure than
Phase 3.1's.

*Root cause identified*: real PFL3 neurons don't relay the goal signal
on its own — they compare it against *current heading* (from the EPG
compass ring) via their real anatomical dendrite geometry. Injecting
only the goal gives PFL nothing to compare against, so no reliable
lateral (L/R) signal should be expected from that alone. Checked
`EPG->PFL` in the same connectome-weights table before building this:
247 edges, weight 2756 — a real, substantial pathway, comparable in
scale to `FC->PFL`.

*Second attempt*: added `heading_ring` (EPG-only, real PB-glomerulus
ring position, 50 neurons / 25 L / 25 R / glomeruli 1-9 — the same ring
structure Phase 3.1 used, but restricted to the actual compass cell
type EPG instead of "anything with a PB-glomerulus label", and used for
*heading* injection rather than goal injection this time). Both signals
are now injected in the same allocentric (world-fixed) reference frame
— `snake-game.js`'s `getHeadingAngle()`/`getGoalAngle()` replaced the
old egocentric `getGoalBearing()` — so PFL's real synaptic wiring
computes the heading-vs-goal comparison itself, rather than the JS side
pre-computing a relative bearing and only ever telling the brain half
of that comparison. Testing (`test_comparator.py`, mirrored heading=0
with apple at +pi/2 vs -pi/2, 8 trials each) showed a real directional
trend (apple-right-of-heading case: 6/8 trials correctly negative
DNa* L/R shift) but not a statistically solid one (Welch t≈1.2,
short of significance at n=8); sweeping the injection scale 1x-8x
(`test_comparator_scaled.py`) did not make the trend firm up or grow
monotonically, which is evidence this is a real ceiling of the
approach, not a tuning gap — most likely because the real PFL3
comparator depends on precise synaptic *sign* (excitatory vs
inhibitory) that this dataset's weight table doesn't carry (Phase 0
limitation, noted since Phase 2), and a purely-excitatory approximation
cannot reproduce a clean subtraction operation.

*What actually matters — measured game-level effect*: since the
neuron-level signal is weak-but-real rather than absent, and a weak
bias sustained continuously over hundreds of real gameplay ticks can
behave differently than one static 3-second lab measurement, ran full
simulated gameplay episodes (`test_gameplay.py`, coupling `LifSimulation`
to the same game rules as `snake-game.js`, 60 simulated seconds/episode)
comparing the finished Phase 3.2 circuit against no goal information at
all. First batch (n=10 each) looked promising: 2/10 episodes ate an
apple with goal+heading vs. 0/10 without. Exactly the lesson from
Phase 3.1 repeated itself, though: a second, larger batch (n=16 each,
different seeds) reversed it — 1/16 (mean 0.0625 apples/episode) with
the circuit vs. 2/16 (mean 0.125) without. Combined across both
batches: 3/26 apples with the circuit vs. 2/26 without — differences
of one or two apples on counts this small carry no statistical weight
either way. **The honest conclusion is that this circuit does not
measurably improve apple-eating success over having no goal
information at all**, despite being real anatomy, correctly wired, and
no longer sabotaged by the homeostasis bug found along the way.

**Honest summary of Phase 3.2**: the FC→PFL→DNa02→heading-comparator
circuit is real anatomy, correctly wired, and produces a measurable,
correctly-directed *trend* at the neuron level in isolated tests — but
that trend is too weak and unreliable to survive into an actual
measured improvement in gameplay success. The most likely fix (real
excitatory/inhibitory neurotransmitter sign per neuron, which MaleCNS
does publish separately from what Phase 0 pulled in — a purely
excitatory network structurally cannot compute a clean subtraction,
which is what the real PFL3 comparator needs) is a real avenue for a
future phase, not attempted here. What shipped and is worth keeping
regardless of the success-rate result: the fix for a genuine bug (FC's
own homeostasis was crushing PFL, unrelated to whether the circuit
would ultimately work), a completed (not half-built) biological
comparator circuit for any future attempt to build on, and a 2D
decision panel (`frontend/js/decision-panel.js`, see the follow-up
below for its final form) driven by the same real `group_activity_ema`
rates the backend actually computes, not a decorative animation.
Apple-eating success in this project remains low; that is reported
here plainly rather than dressed up, matching how every other
measurement in this project has been handled.

**Phase 3.2 follow-up — spinning-in-circles bug and the decision panel's
final layout (done).** After Phase 3.2 shipped, real gameplay showed
the snake repeatedly spinning in tight circles in one direction. Root
cause: `backend/lif.py`'s `_update_motor()` hysteresis holds a
"left"/"right" decision for a while once triggered (measured directly:
anywhere from ~10 to ~227 broadcasts, i.e. up to ~4.5 real seconds, out
of a representative 60-second run with 90 total state segments) rather
than flipping every broadcast — that debouncing is intentional. But
`frontend/js/snake-game.js`'s `applyTurn()`/`step()` applied a 90-degree
turn on *every game tick* for as long as that decoded state held, so a
single ~4.5-second hold (spanning ~30 ticks at `TICK_SECONDS=0.15`)
compounded into ~30 repeated 90-degree turns — a tight spinning circle,
not a bug in the neural signal itself. Fixed by making turning
edge-triggered: `applyTurn()` now only arms a turn on an actual
straight→left or straight→right transition in the decoded decision,
and `step()` consumes and clears that single-shot flag on the very next
tick, regardless of how many more broadcasts keep reporting the same
held state.

Separately, revisited the initial Phase 3.2 decision panel after
feedback that it should look like Phase 1's original node-link diagram
— one column per real brain region, several individually labeled real
neuron types glowing per column — rather than a generic 5-box flow
diagram. Rebuilt `decision-panel.js` on that layout: **Motion**
(T4/T5 a/b/c/d direction subtypes, individually tracked via the
existing `direction_masks` — previously only exposed as one aggregate
rate), **Central Cx** (EPG heading, FC goal, PFL compare), **Steering
DN** (DNa L, DNa R), plus a turn readout — all driven by the same real
per-population spike rates as before, just laid out and labeled to
match the original's per-type-column style instead of abstracted into
five stage names.

**Phase 3.3 — real neurotransmitter sign, ring topology correction, turn
cadence tuning, and a final honest measurement (done).** Triggered by
direct user feedback after playing: the snake sometimes moved toward the
apple but turned too late and hit a wall, and sometimes went in the exact
opposite direction. Investigated three real, independent issues rather
than re-tuning constants blindly:

1. **Neurotransmitter sign.** docs previously assumed no excitatory/
   inhibitory sign data existed for this dataset. Checked again, properly:
   the full `Neuprint_Neurons.feather` (already downloaded for the Phase 0
   ROI cross-check) carries real per-neuron `consensusNt` predictions.
   Delta7 — the real, well-known inhibitory ring-attractor-sharpening
   interneuron in the fly compass circuit — is 100% glutamate in this data
   (42/42 neurons), and the central-complex cluster overall is ~32% GABA
   or glutamate (994/3137), all previously simulated as excitatory like
   everything else. Fixed: `prepare_subset.py` now merges `consensusNt`
   and computes `nt_sign` (GABA/glutamate = -1, else +1, standard
   fly-connectome convention), and `lif.py` indexes it by each edge's
   presynaptic neuron. Network stability re-verified after the change (no
   runaway, no dead clusters).

2. **Ring topology.** Tried making FC's goal code side-aware (an 18-slot
   code mirroring the heading ring) on the theory that a side-blind FC
   code and a side-aware EPG code were mismatched coordinate systems —
   checked real connectivity before committing to it, found FC_L and
   FC_R project to PFL_L/PFL_R almost identically (e.g. FC_L->PFL_L 4199
   vs FC_L->PFL_R 3859), meaning side isn't a meaningful axis for FC.
   Reverted. Separately checked the heading ring itself: the PB's 18
   glomeruli are a real, published "double-wrapped" ring (Wolff & Rubin;
   Turner-Evans et al.) — the same 9 angular positions appear once per
   hemisphere, both jointly representing one heading, not 18 independent
   positions. Confirmed directly: EPG_L#k projects to EPG_R#k (matching
   glomerulus number) at ~8x the average weight of EPG_L#k to a
   different-numbered EPG_R#k2. Both FC and EPG now use the same
   9-position space, injecting into both hemispheres' matching
   column/glomerulus together for one coherent bump.

3. **Turn cadence.** Edge-triggered turning (the Phase 3.2 follow-up fix)
   stopped the circling bug but created a different problem: the snake
   travels straight for as long as the backend's decision holds (measured
   up to ~4.5 real seconds), often long enough to hit a wall on this
   project's 20x20 grid before the next transition ever comes — this is
   exactly the "turns too late" bug reported. Replaced with a cooldown
   (`MIN_TURN_TICKS = 2` in `snake-game.js`): a sustained non-straight
   decision re-executes a turn every 2 ticks for as long as it holds,
   instead of never again (too rare) or every tick (spins — confirmed by
   testing `min_turn_ticks=1`, which reproduces the original circling
   bug). Chosen as a reasonable middle ground between two known-bad
   extremes.

**Honest final measurement.** After all three fixes, ran repeated
60-second-episode gameplay simulations (goal+heading circuit vs. no goal
information at all, matching the real frontend's exact turning logic —
an earlier round of this same testing was invalidated when the test
harness turned out to still use the pre-fix circling logic, a bug in the
test itself, not the app). Results across independent batches of 24
episodes each, same configuration, different random seeds:
- Batch 1: 8 apples (goal+heading) vs. 4 apples (no goal) — looked like
  a real 2x improvement.
- Batch 2 (replication): 1 apple (goal+heading) vs. 5 apples (no goal) —
  the *opposite* pattern, same configuration.

This is the same "don't trust one run" lesson this project has hit
before (Phase 3.1), now confirmed by direct replication at the full
gameplay level: **none of Phase 3.3's fixes, individually or combined,
produced a measurable, reproducible improvement in apple-eating success
over having no goal information at all.** Every fix made in this phase
is real, defensible, and correct on its own biological/logical merits
(the neurotransmitter sign was genuinely wrong before; the ring topology
was genuinely mismatched; the turn cadence genuinely needed to be
somewhere between two bad extremes) — but the compounding uncertainty of
an uncalibrated LIF network (arbitrary `WEIGHT_SCALE`, no real
conductance data, small population counts of 16-50 neurons per readout
group) appears to dominate over whatever real signal the anatomy
provides, at least at the sample sizes tractable in this project. A
genuinely reliable fix would likely require calibrating the network
against real physiological firing-rate/response data rather than
tuning constants by hand, which is out of scope here.

**Phase 3.3 follow-up — the circling bug was still visible after the
cooldown mitigation (done).** Direct testing after the above showed the
snake still circling. The `MIN_TURN_TICKS` cooldown reduced turn
frequency but didn't fix the actual cause: the motor hysteresis band
(`TURN_ON_THRESH` vs. a much smaller `TURN_OFF_THRESH`) held a
left/right decision for a long time once triggered (median ~5.3 game
ticks, tail to ~41 ticks / ~6 real seconds) — even a 2-tick cooldown
still re-executes ~15+ turns during a hold that long, which looks like
continuous spinning on a grid regardless of the exact interval. Fixed
at the source instead of working around it in the game layer:
`TURN_OFF_THRESH` now equals `TURN_ON_THRESH` (no hysteresis band),
measured to shrink hold length to a median of ~1.6 ticks / max ~6.8
ticks — short enough that `snake-game.js` went back to the simplest
possible mapping (plain once-per-tick turning, no cooldown or
edge-trigger workaround needed). Verified this actually worked, not
just in theory, via a death-cause breakdown across 24 episodes:
self-collision (circling into the snake's own body — what "circling"
actually kills you with) dropped to 2/24, down from being the dominant
failure mode; wall collision is now the overwhelming majority cause of
death (21/24), a separate, still-open problem — the snake survives a
reasonable while (median ~95 ticks, ~14 real seconds) without spinning
into itself, but still has no reliable way to see a wall coming in
time to avoid it, consistent with everything already measured about
the weak/unreliable steering signal. Apple count remained low and
comparable with/without goal information (3 vs. 4 across 24 episodes)
even with circling fixed — the steering signal's reliability, not the
turn-execution mechanism, remains the actual bottleneck.

**On the user's suggestion to add DOOM/parking-demo-style guidance,
answered directly and not implemented (open question for a future
phase):** explicitly declined to add a scripted/external algorithm that
computes the correct move and overrides or fakes the brain's decision —
that would violate this project's foundational, repeatedly-reaffirmed
constraint (no external algorithm playing the game behind a decorative
brain panel, confirmed as exactly what at least one real "fly plays a
game" demo online actually does). The one remaining lever consistent
with "genuinely emergent, not scripted" that hasn't been tried: real
synaptic plasticity (a Hebbian/reward-modulated weight-update rule that
strengthens the pathway active just before eating an apple and weakens
the one active just before a collision) — this would still be the
network's own real connectome-derived structure adapting through real
experience, not a fake path-following algorithm, but is a substantial
new mechanism (online learning across repeated episodes) not yet
designed or attempted, and is a reasonable candidate for a Phase 4 if
this project continues.

## Open risks / unresolved questions

- Descending neurons only receive 17.0% of their real input from within
  the candidate subset — decided to treat CX → DN as a simplified
  categorical mapping rather than simulating realistic DN spiking (see
  empirical findings above). Revisit if this looks too artificial once
  something is actually running.
- Confirm exact MaleCNS license version/terms before any public
  redistribution of derived data.
- WebGPU in-browser simulation was rejected for now but could be
  revisited if the two-process (Python backend + browser) setup proves
  operationally inconvenient.
