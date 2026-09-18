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

**Phase 3.4 — LC10 direct visual pursuit, the anatomically correct
circuit for "sees it, flies straight at it" (done, real and large
behavioral effect, on a different metric than expected).** User's
framing directly identified a real conceptual error: how a fly finds
food (a directly visible object) is not the same behavior as central
complex path integration (FC/EPG/PFL, used since Phase 3.2) — that
system is for returning to a *remembered* location (e.g. a nest), not
steering toward something currently in view. Real-time visual target
pursuit in Drosophila is a separate, well-characterized circuit: LC10,
published as driving directed courtship pursuit (Ribeiro et al. 2018 —
a male fly steers toward a visually detected female via LC10
projecting to DNp11).

Checked directly against this dataset before using it (not assumed):
every LC10 subtype present (a, b, c-1, c-2, d, e — 960 neurons total)
projects to DNa* steering neurons with a *perfectly* ipsilateral,
zero-crosstalk pattern — e.g. LC10a_L -> DNa_L is 377 weight, LC10a_L ->
DNa_R is exactly 0, and the mirror image for LC10a_R. This is
qualitatively different from every CX-pathway measurement all project:
no delicate goal-vs-heading subtraction is needed, just a direct,
dedicated, real wire from "target detected on this side" to "steer
toward this side." DNa10 (the single strongest LC10 target) is already
inside the existing DNa* readout, so no new readout population was
needed. LC10 has no real retinotopic position label in this dataset
(unlike FC's column or EPG's glomerulus), so injection uses the
*egocentric* bearing (target angle relative to current heading,
computed from the already-sent allocentric bearing/heading) split
smoothly into LC10's real left/right populations — visual detection is
inherently egocentric, unlike the CX pathway's shared-world-frame
comparison.

Measured far more reliable than anything found with the CX pathway:
12/12 correctly-signed trials in both directions (target right vs.
left), t=88 (compare the CX circuit's best result, t~1-2), signal-to-
noise ~17-20x — the cleanest, least ambiguous real signal found in this
entire project.

**Gameplay result — real, large, and reproducible, but on survival
rather than apple-eating.** Across three independent 24-episode
batches: episodes surviving the full 60 seconds with zero collisions
went from 0/24 (no goal information, every batch) to 12/24, 13/24, and
15/24 with the LC10 circuit active — a dramatic, consistently
reproducible effect, unlike every other gameplay measurement in this
project's history. Apple-eating itself and closest-approach-distance
were also checked but showed smaller, less consistent improvements
across the same batches (e.g. closest-approach mean 5.38 vs. 7.42 in
one batch, 4.96 vs. 5.33 in the replication). The interpretation: a
strong, correctly-signed, always-on steering bias makes the snake
continuously curve back toward the apple's general direction instead
of ever traveling straight into a wall, which is exactly what the
survival metric captures — but the same coarse, oscillating,
90-degree-grid-turn approach doesn't reliably achieve the exact-cell
precision apple-eating requires on final approach. A real, substantial,
honestly-measured improvement in the fly's steering behavior, just not
the one metric (apples eaten) most directly asked for.

**Phase 3.5 — the actual bottleneck was motor lag, not signal quality
or search vigor (done, apple-eating finally, measurably, reproducibly
improved).** User pushback after Phase 3.4 was exactly right on two
counts: surviving 60 seconds without hitting anything doesn't mean the
fly is actually pursuing the apple (it could just be wandering in a
way that avoids edges), and it was still failing to eat. Direct
trajectory logging confirmed the first point: a representative episode
showed the fly closing distance from 12 to 8 early on, then drifting
steadily away to a final distance of 20+ and staying there — not
oscillating near the target, genuinely leaving and never returning,
despite LC10's isolated t=88 reliability.

First attempt at a fix implemented the user's specific proposal: a
persistent internal signal ("search_intensity") that rises when a
fast (~0.3s) vs. slow (~2s) EMA crossover of real distance-to-target
shows no progress and decays when it does, scaling the LC10
injection's magnitude — modeling a real, published Drosophila/foraging-
animal behavior (area-restricted search, normally attributed to
octopaminergic/dopaminergic modulation of locomotor vigor; this
dataset's subset has no identified PAM-cluster dopaminergic neurons to
simulate directly, so this was implemented at the same systems level as
existing knobs like GOAL_SCALE). It measurably worked as designed
(search_intensity swung across its full 0-4 range in response to real
stalled progress) but did not fix the drift-away trajectory.

Root cause, found by testing `MOTOR_EMA_TAU_MS` directly rather than
tuning the new mechanism further: at the original 150ms (chosen
without measurement back in Phase 3), the motor readout lagged behind
each turn long enough that the decoded decision still reflected the
*pre-turn* bearing for a meaningful fraction of every tick — a classic
feedback-lag oscillation, not a problem with the underlying signal.
Direct trajectory comparison confirmed it starkly: the same scenario
that orbited at distance 8-12 from the target at 150ms converged to
distance 0-2 at 20ms, with no other change. Confirmed at the full
gameplay level (24 episodes each, matching frontend/js/snake-game.js's
real turning logic exactly): apple-eating went from 3/24 episodes (no
goal information) or 0/24 (goal information, 150ms lag) to 17-21/24
across several batches (goal information, 15-20ms lag) — the first
configuration in this entire project to measurably and reproducibly
beat having no goal information at all, and by a wide margin (roughly
6-7x). `MOTOR_EMA_TAU_MS` is now 15ms. `search_intensity` was removed
after this: measured to not help and to slightly hurt once the lag was
fixed (a fast, accurate controller doesn't benefit from extra gain — it
just overshoots more) — tried in good faith, tested rigorously, found
unnecessary once the real bug was fixed, and removed rather than left
in as dead weight, matching how every other non-performing addition in
this project has been handled.

Honest remaining picture: the fly now reliably approaches and eats the
apple in most episodes (roughly 70-85% depending on the exact batch),
a categorical change from every measurement in this project's history
up through Phase 3.4. Death is now dominated by self-collision (the
snake growing long enough to run into its own body — an expected
consequence of actually succeeding at Snake, not a steering failure)
rather than wall collision or aimless wandering.

**Phase 3.6 — the harness was lying; four real defects found by
instrumenting the live browser (done).** User testing kept showing the
snake circling and never eating while the Python harness reported
17-21/24 episodes succeeding. The harness was not a faithful model of
the app, and chasing neuroscience explanations for a discrepancy that
turned out to be engineering cost a lot of this project's time. Lesson
recorded here deliberately: **when the user's observation and the test
disagree, instrument the thing the user is actually running.** Four
separate defects, all found that way:

1. *The brain ran slower than the game.* `step()` performed 20 separate
   GPU→CPU syncs per step (one `.item()` per cluster/group/motor
   readout — 400 per broadcast batch), and `server.py` slept
   `BROADCAST_INTERVAL_S` *after* that compute rather than accounting
   for it. Measured 0.42x real time: the fly's brain lived at under
   half the speed of the game it was playing, so every reaction landed
   a tick late. The harness advanced simulated and game time together
   1:1, so it never saw this. Fixed by stacking all 20 population masks
   into one matrix (single matmul, single sync — measured 3x faster,
   ~5.8x real time) and giving the server loop a real deadline;
   verified over the wire at exactly 1.00x.
2. *The controller demanded precision the body cannot deliver.* Traced
   live: the snake cycled `(-1,0) → (0,1) → (1,0) → (0,-1)` endlessly
   with the egocentric bearing repeating the same four values (−138°,
   −52°, +41°, +135°), never near zero. On a 4-direction grid the best
   reachable heading can still be 45° off, yet `sin(41°) = 0.66` still
   commands a hard turn — so the fly turned *away* from its own best
   heading, forever. That limit cycle is precisely the "circles" being
   reported. Fixed with a frontal acceptance zone (`VISUAL_DEADZONE`,
   45°): no turn command while the target is roughly ahead — which is
   also what real Drosophila fixation behaviour does.
3. *The retina fed the brain pure DC.* When the scene barely changes
   between samples, `Σ cur[i]·prev[i−1]` and `Σ cur[i−1]·prev[i]` are
   the *same sum* — measured live, all four T4/T5 channels came back
   byte-identical (0.13016532164177314) every single sample: a constant
   excitatory load carrying zero directional information. Now reports
   opponent (common-mode-subtracted) motion energy, matching how real
   T4/T5 is read downstream.
4. *Phase 1's decorative event pulses were competing with the real
   signal.* `move` fired every game tick, dumping 0.6 into all 18,433
   motion neurons; `eat`/`collide` injected 0.8-1.2 straight into the
   descending neurons the steering decision is decoded from, corrupting
   it for ~300ms at exactly the moment the fly had just reached an apple
   and needed a new heading. Suppressing them live tripled the eating
   rate; all three are now removed.

Also added an apples/best score readout (sidebar + game screen) — the
instrument that made measuring any of this in the real app possible.

**Measured in the real browser, end to end: 0 apples in 150s before,
5-7 apples in 200s after.** The fly now approaches and eats; it still
usually dies within a few seconds of eating (the new apple can spawn
behind it, and a 90°-per-tick body has no graceful way to reverse), so
`best` sits at 1. That is the honest current state.

### Answers to two recurring questions

- *Would downloading the full 166,700-neuron connectome help?* Not for
  any bottleneck found so far — every one of them was engineering
  (loop timing, control deadband, a degenerate sensory channel, leftover
  decorative drives), and the LC10→DNa circuit measured t=88 reliability
  in isolation with the subset we already have. It would also make
  performance ~7x worse, on a simulation that was until now running
  below real time. What the full connectome *would* add that this subset
  genuinely lacks: the olfactory pathway (real flies find food primarily
  by smell), and the mushroom body with its PAM dopaminergic reward
  neurons — the only way to implement a "reward/hunger" signal out of
  real neurons rather than as a systems-level parameter.
- *Would a reward/"happiness" signal help?* Tried and measured (see
  Phase 3.5): a progress-tracking search-intensity signal worked exactly
  as designed and did not fix the drift, because the real cause was
  elsewhere. It is a real mechanism (area-restricted search) and worth
  revisiting if a future phase adds the PAM neurons to implement it
  properly, but on the evidence it is not what stands between this fly
  and the apple.

**Phase 3.7 — collision avoidance via LPLC1 (done, partial).** Once the
fly started eating, the snake grew and it died almost immediately after
— it had no channel at all through which its own lengthening body could
be perceived. Real flies do: looming-sensitive lobula columnar cells.
Checked before choosing one: LPLC2 / LC4 / LC6 / LC16 / LC11 have *zero*
connectivity to DNa* steering neurons (they feed the giant-fibre takeoff
escape instead — the real division of labour), while **LPLC1** (134
neurons, all cholinergic, already in the subset) reaches both the escape
descending neurons (DNp03 1988+1614, DNp35 1750+1444, DNp06 1604+1169,
DNp11, DNp103) *and* DNa07, every one perfectly ipsilateral with zero
crosstalk.

Reading avoidance off DNa alone did not work, for a real reason: only
~4% of LPLC1's descending output reaches DNa07. The other ~96% goes to
escape neurons, whose real meaning is *takeoff* — an action Snake does
not have, so the fly's strongest collision response had nowhere to go
and the apple drive (≈0.006 at DNa vs 0.0016 from LPLC1) outvoted it
every time. The escape command is now read out separately and mapped
onto the only evasive action this body has, a hard turn away.

Two further corrections came from measurement, not theory: the threat
field was initially frontal-only, which missed the fatal case entirely
(after one turn the snake's own body sits *beside* the head, invisible
to a forward field, and the next turn drives into it), and it used
1/d², which in a logged death sat at 0.06 for seven ticks then jumped to
0.56 one tick before impact — far too late for a 150ms tick plus the
motor EMA to act on. Now panoramic, weighted by where the body can
actually end up, 1/d over 6 cells.

`ESCAPE_GAIN` is sized to put the two drives on equal footing (close
threat drives escape to ~0.18, apple drive reaches ~0.006). Result
across 14 episodes: **wall deaths eliminated (6 → 0)** and some episodes
now survive the full 60s at the same apple rate. Higher gains stop
almost all deaths (12/12 survive at 0.4) but the fly then hovers safely
and never commits to an apple — the classic approach-avoidance failure.
**Self-collision remains the dominant death and best score is still 1:
not solved.**

**Phase 3.8 — olfaction and dopamine: added, measured, and honestly not
what steers this fly (done).** Requested on the reasonable grounds that
real flies find food by smell and that dopamine rises on approach. Both
turned out to be one circuit, and the whole of it was already on disk —
the "subset" was only ever our filter, so nothing needed downloading.

Added: antennal-lobe projection neurons (541, 273 L / 273 R) → Kenyon
cells (4050) → MBON (90) → DNa* steering at 1145 weight (the same order
as LC10's ~970 into the same readout), plus PAM (314 dopaminergic
neurons) exactly where it sits in the animal — on the KC→MBON synapses —
and APL. Subset 22,853 → 27,850 neurons, still 2.8x real time. The 2635
ORNs are excluded because not one has a soma position (0/2635): they are
in the antenna, outside the reconstructed volume, so odour enters
bilaterally at the PN stage instead, which is the odour representation
the rest of the brain actually receives.

**Measured result: odour drives PN hard (0.05-0.08) and propagates
through KC and MBON, but produces no reliable lateralised steering
signal — DNa shifts ~0, inconsistent in sign across trials.** This is
anatomy, not tuning:
- The mushroom body deliberately *discards* spatial information. Each
  Kenyon cell samples glomeruli at random; its output answers "is this
  odour good or bad", not "which way". A bilateral difference does not
  survive that mixing.
- The lateral horn, olfaction's other output, agrees: PN→LH is 378,010
  weight and LH→any DN is 22,810, but **LH→DNa\* is just 66**, and barely
  lateralised (23/15, 6/22).
- Which matches real behaviour: odour tracking in a *walking* fly
  modulates walking speed and turn rate (klinokinesis), it does not
  command a direction. This body has no speed to modulate — only left,
  right and straight.

The same finding disposes of dopamine as a fix: PAM's role is plasticity
at KC→MBON, on a pathway measured not to reach steering. Implementing
mushroom-body learning would faithfully shape a signal that does not move
the snake. Worth doing for authenticity some day; not worth doing to make
the fly eat.

Both are kept — they are real, active parts of the brain being simulated
and now have their own "Smell" column in the decision panel — but the
panel comment and this document both say plainly that they are not what
drives the fly.

**Phase 3.9 — odour made to work, and dopamine measured and removed
(done).** Phase 3.8 concluded odour could not guide the fly. That
conclusion was wrong in an instructive way: the *mechanism* was wrong,
not the idea. It injected raw bilateral concentration and asked the
mushroom body for a left/right steering command, which that structure
cannot produce — each Kenyon cell samples glomeruli at random, so a
bilateral difference does not survive it. What a real fly does, and what
was being asked for, is gradient following over time.

Two changes turned it into a working channel:

1. **Adaptation.** Real olfactory neurons report a *change* in
   concentration, not its level, so PN drive is now concentration minus a
   slowly tracking baseline (1.5s). Measured: approaching gives PN 0.075,
   receding 0.00015 — a 300x separation that simply did not exist before.
2. **The lateral horn** (2028 neurons, added to the subset). This is
   olfaction's *innate* output, as opposed to the mushroom body's learned
   one: PN→LH is 378,010 weight and LH→descending 22,810. Leaving it out
   was why the signal had nowhere to go but the MB — which actively
   suppresses it (MBON activity *drops* when odour rises: sparse coding,
   APL inhibition, genuinely inhibitory MBONs; that is the learned-valence
   half doing its job, not a bug). LH now rises to 0.0011 approaching
   against 0.0002 receding.

The coupling is **klinokinesis**, read as common mode rather than
difference: while the smell strengthens, the threshold to commit to a
turn rises, so the fly holds course; when it stops strengthening, the
threshold drops and it turns and searches. Direction still comes
entirely from LC10/LPLC1 and their real ipsilateral wiring — odour never
says which way, only whether to keep going.

Measured across five independent 12-20 episode batches, odour off vs on:
7/11, 15/15, 13/17, 9/12, 14/16 — **58 apples against 71 over 88
episodes, about +22%, positive in four batches and negative in none**.
Best single-episode score reached 3, up from 1. The exact gain sits
inside the noise; 10000 is chosen because it roughly doubles the turn
threshold on a typical measured rise.

**Dopamine: built, measured, removed.** Reward now drives PAM, the real
dopaminergic cluster (correcting Phase 1's blanket event, which hit the
descending neurons directly). Mushroom-body plasticity was then
implemented properly — PAM→KC gating the 59,709 KC→MBON synapses, with
the canonical rule that dopamine coinciding with recent Kenyon-cell
activity depresses them, plus an eligibility trace and a slow decay back
to the connectome's own weights. It worked mechanically (a reward burst
depressed the traced synapses by 22%) and made the fly slightly *worse*:
14 apples over ten minutes against 21 with the rule off, the plastic
weights collapsing to their 25% floor.

That is the task, not the tuning: **associative learning needs something
to associate.** This game has exactly one odour, the apple, and it is
always rewarded. With no second cue to discriminate against, the only
thing the rule can do is depress everything uniformly — a gain change,
not a memory. The plasticity was removed and the reasoning recorded in
`lif.py`; it is worth revisiting only if the game ever gains a second
smell worth telling apart.

**Phase 3.10 — the turn-execution policy, which mattered more than any
circuit (done).** The report was that the fly still wandered in circles
and took tens of seconds to reach an apple instead of going at it
directly, and would even orbit an apple it had reached without eating
it. Both turned out to come from the same place, and it was not the
sensing: it was how a held left/right decision becomes grid moves.

The backend holds a decision for a median of ~1.6 ticks. Applying it on
*every* tick of that hold therefore means typically two 90-degree turns
back to back — a 180-degree reversal. That is the circling, and with a
short snake it is also a self-collision. The orbiting-next-to-the-apple
case has the same root with a geometric twist: a diagonally adjacent
apple sits at exactly 45°, right on the edge of `VISUAL_DEADZONE`, so
the fly turned, overshot, and went round.

All three policies measured on identical seeds, 16 episodes each,
everything else unchanged:

| policy | apples | best episode | self-collisions |
| --- | --- | --- | --- |
| turn every tick (was) | 10 | 1 | 9/16 |
| turn once per decision | 8 | 2 | 3/16 |
| **at most one per 2 ticks** | **33** | **5** | **1/16** |

Worth noting *why* this was missed for so long: the middle policy was
tried back in Phase 3.6 and rejected, correctly, because the hysteresis
was long then (median 5.3 ticks) so one turn per decision left the fly
running straight into walls. The rate-limited policy was tried too, in
the same era, and also failed — for the same reason. Once Phase 3.6
shortened the hysteresis, the combination that had never been tested was
short hysteresis *plus* rate limiting, and that is the one that works.
A reminder that a rejected option can be worth re-testing after the
thing that made it fail has changed.

With the cooldown in place, the obstacle-avoidance gain was re-swept in
the new regime; `ESCAPE_GAIN = 0.01` is still the best point (36 apples
against 19 at 0.05 and 13 at 0.15 — raising it makes the fly survive
more and eat much less, the same approach-avoidance trade-off as before).

Live in the browser: the fly now eats two apples inside the first thirty
seconds and survives over two minutes on a single life, against roughly
one apple per 30-90 seconds and near-immediate death before.

**Phase 3.11 — the steering readout was losing to its own noise (done).**
After Phase 3.10 the fly still circled and still took tens of seconds per
apple. Logging a long run showed it executing 22-25 turns per 50 ticks —
the maximum the cooldown allows — essentially never going straight.
Measuring the readout explained it:

| | \|diff\| |
| --- | --- |
| no input at all, median | 0.000038 |
| no input at all, p90 | 0.00205 |
| no input, fraction over the 0.0001 threshold | **42%** |
| apple 50° off, median | 0.00109 |

The turn threshold sat *below the readout's own noise floor*. The fly was
committing to turns on noise as often as on the apple — not aiming at
anything, just a biased random walk that eventually stumbled onto food,
which is exactly what "wanders in circles, takes tens of seconds" looks
like. Raising only the threshold made it worse (14/14 straight into a
wall), because the signal was small too.

Three things had to move together, and the first is another case of a
setting that was right, then wrong, then right again:

- **`MOTOR_EMA_TAU_MS` back to 150ms.** Phase 3.5 cut it to 15ms because
  the lag made the fly act on its pre-turn bearing — true while a held
  decision was applied on every tick. Phase 3.10's rate limiting removed
  most of that penalty and left only the noise cost, and with just 16
  neurons per side a 15ms window is very noisy. Measured by window:

  | tau | noise p90 | signal median | correctly signed |
  | --- | --- | --- | --- |
  | 15ms | 0.00154 | 0.00104 | 89% |
  | 40ms | 0.00123 | 0.00144 | 97% |
  | 80ms | 0.00072 | 0.00138 | 98% |
  | 150ms | 0.00061 | 0.00135 | **100%** |

- **Thresholds to 0.003**, above the 150ms noise floor and below the
  driven signal.
- **`VISUAL_SCALE` to 30**, so a real bearing error clears the higher bar.

Result: 42-52 apples per 14 episodes against 39 for the old low-signal,
low-threshold combination — but the number that matters is the shape of
the trajectory, which is now aimed rather than wandering: distance to the
apple falls 8→7→6→5→4→3→2→1, and five apples land inside 55 ticks. Live
in the browser the fly eats one every four to eight seconds, and took
four inside a six-second stretch, against roughly one per 30-90 seconds
before. Escape gain and turn cooldown were re-swept in the new regime and
0.01 / 2 ticks remain the best points.

Three separate times now, the fix has been a parameter that was correct
when it was set and became wrong when something else changed —
`MOTOR_EMA_TAU_MS` twice, the turn policy once. Worth re-testing rejected
options whenever the thing that made them fail has moved.

## Phase 3.12 — the fly gets a body: an integrated heading instead of 90° turns

The complaint this phase started from: away from the board edges the fly
slaloms, and around an apple it locks into a wide orbit — at one point a
9×9 box with the apple in the middle. Plus a direct question: is the
brain's decision out of sync with the game?

### Two fixes proposed from first principles, both measured, both rejected

**A — graded turn rate as a permission gate.** Report how far past its
commit threshold the steering signal is, and let a strong command turn on
consecutive ticks while a weak one keeps the rate limit. Measured **2
apples against 26** over ten seeds. Logging the strength against the real
steering error showed the premise was simply false:

| \|ego\| | mean strength | correct turn | wrong turn | straight |
| --- | --- | --- | --- | --- |
| 0-29° | 0.90 | 0.34 | **0.65** | 0.01 |
| 30-59° | 0.88 | 0.52 | 0.44 | 0.04 |
| 60-89° | 0.76 | **0.80** | 0.10 | 0.10 |
| 90-119° | 0.54 | 0.79 | 0.09 | 0.12 |
| 120-149° | 0.54 | 0.62 | 0.22 | 0.17 |
| 150-179° | 0.28 | 0.36 | 0.48 | 0.16 |

The strength was *anti*-correlated with the error: strongest when already
aligned, weakest with the apple behind — the opposite of what the fix
needed. LC10 drive goes as sin(ego), so the rear is where it is weakest,
and dividing by an odour-modulated threshold scrambled what was left.

**B — phase-lock sensory sampling to the game tick.** Sampling ran on a
free 100ms timer against a 150ms tick: not harmonic, so the age of the
data a decision was made on drifted between 0 and 100ms. Real, and worth
fixing, but on its own it measured *worse* (10-15 apples against 26): with
the drive decaying over 300ms, less frequent sampling meant a weaker drive.
It is kept, because the winning configuration below includes it.

Also swept and rejected: `DRIVE_DECAY_TAU_MS` (300 → 120/60/30, with and
without gain compensation; 60ms cuts the command's release time from 860ms
to 140ms but the fly then flies straight into walls, 10/10), asymmetric
hysteresis (`TURN_OFF` at 2-3× `TURN_ON`: 28-29 against 26, inside the
noise), and raising the obstacle gain (below).

### The actual defect: a 90° body executing an analog command

Every configuration shared one signature — with the apple within 60° of
straight ahead the fly turned correctly about as often as wrongly
(0.49-0.54) and went straight on 2-4% of ticks. That is not a tuning
failure. With the apple 30° to the right, "straight" is the correct move
on a grid, and a controller whose only outputs are two 90° turns cannot
express it. It turns, overshoots by 60°, is told to turn back, and
overshoots again: the slalom. At larger errors the same effect closes
into the orbit.

So the fix is not in the brain. It is the missing physics between a
steering command and a grid: **the fly now carries a continuous heading**.
The descending-neuron left-right difference is integrated into it as a
turn velocity (capped at 90°/tick, so a fatal reversal cannot happen in
one step), and the snake moves along whichever cardinal that heading
points nearest to. Nothing plans a route; the connectome still decides
everything. A weak command rotates the heading slightly and never crosses
into the next cardinal — which is how "go straight" finally became
expressible.

Two supporting changes fall out of it:

- **`VISUAL_SCALE` 30 → 10.** For a rate to mean anything the readout has
  to be out of saturation. Measured against a held bearing:

  | bearing off-axis | 0° | 15° | 30° | 45° | 60° | 90° | 120° | 150° | 180° |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | scale 30 | .0002 | .0093 | .0171 | .0194 | .0195 | .0200 | .0194 | .0168 | .0000 |
  | scale 10 | .0001 | .0031 | .0061 | .0087 | .0099 | .0123 | .0104 | .0060 | .0000 |

  At 30 the signal is flat from 30° out — an on/off command. At 10 it
  grades to 90°, and a 15° error still sits 5× over the noise floor
  (p90 0.0006).

- **The 45° frontal deadzone is gone.** It was never about LC10, which
  tiles the frontal field in a real fly; it was a patch for the grid
  quantisation the heading now removes. With it gone, steering is correct
  0.81 of the time with the apple 120-180° behind, against 0.51 with it.

### Result

A methodological note first: the LIF noise is not seeded, and run-to-run
spread on ten episodes turned out to be wider than most of the effects
being chased — the same configuration measured 25 and 33 apples on
consecutive runs. Two earlier comparisons in this phase were re-run under
fixed noise seeds before any of the above was believed. Forty episodes of
500 ticks per configuration, twenty game seeds × two noise seeds:

| | total apples | best in one life |
| --- | --- | --- |
| rate-limited 90° turns | 83 | 5, 5 |
| **integrated heading** | **112** | **8, 8** |
| integrated heading, obstacle gain ×4 | 83 | 7, 5 |
| integrated heading, obstacle gain ×12 | 47 | 4, 7 |

The old build never once passed five apples in a life across all forty
episodes; the new one reached eight under both noise seeds and repeated
its total exactly (56/56), where the old one swung 50/33. Raising the
obstacle gain on top of it is harmful, so LPLC1 keeps its scale.

Confirmed live in the browser over 150 seconds rather than only in the
harness: ten lives, an apple every 1.5-5 seconds, five apples in a life
four separate times.

Wall collisions are now the dominant death (53 of 60 short episodes). The
fly flies straighter, so the obstacle pathway has less slack to save it by
accident — and a higher gain is not the answer. That is the next thing to
work on.

## Phase 3.13 — where the deaths actually come from, and the 97% we were not using

Three questions this phase started from: the fly dies at walls but must
still be able to eat an apple sitting against one; it dies against its own
body once it has grown; and is the brain out of sync with the game.

### Where the deaths come from

Sixty episodes of 500 ticks, three noise seeds, logging the state at every
death rather than only the totals:

| apple's distance from an edge | share of spawns | share of wall deaths | ratio |
| --- | --- | --- | --- |
| 0 (against the wall) | 7.8% | **22%** | **2.8x** |
| 1 | 13.8% | **32%** | **2.3x** |
| 2 | 10% | 19% | 1.9x |
| 3+ | 68% | 27% | 0.4x |

The fly does not wander into walls. It flies into them **chasing an apple
that is next to one**: 54% of wall deaths happen while the apple sits in
the outer two rings, which are only 21.6% of spawns. Self-collisions are a
different phase of the same life — they cluster at snake length 8 (9 of
23) with a median of 5 apples already eaten, where wall deaths have a
median of 1. Survive the wall phase, then the growth phase.

### The obstacle pathway, measured in isolation

Holding a threat on one side with no apple, against the same protocol used
for LC10:

| | steering readout |
| --- | --- |
| apple 15° off axis (the weakest useful signal) | +0.0031 |
| apple 90° off axis | +0.0123 |
| **wall to one side** | **+0.00089** |
| **wall dead ahead** | **+0.00002** |

A lateral wall is 3.5x weaker than the weakest useful apple signal. A wall
**dead ahead produces nothing at all** — it is bilaterally symmetric and
the readout is a left-right difference, so the one thing that reaches the
motor output cancels exactly. Trajectory logs show it directly: heading
straight at a wall, `L-R` reads +0.060, +0.016, +0.0009, 0.0000, 0.0000
while the total threat climbs past 2.0. The fly was not ignoring the wall;
it was never told.

### Six attempts at that, all measured, all rejected

- **Looming** (threat as a rate of rise against an adapting baseline, the
  same construction that made odour work). 160 apples against 209, and it
  did not reduce wall deaths. The reason is arithmetic: it raises the gain
  on a lateral difference that is *zero* in the case it was built for.
- **Flipping the threat side to ipsilateral.** The anatomy argued for it —
  LPLC1 → escape DNp is 11,608 ipsilateral and escape DNp → DNa* is 544
  at 96% crossing, against only 575 for LPLC1 → DNa* direct, so the
  dominant route inverts our convention. Measured in isolation, the
  current contralateral convention is correct (threat left → +0.00089,
  turning away) and the flip is exactly backwards (−0.00074, turning
  *into* it). A good argument that would have broken the fly.
- **LPLC2, driven bilaterally.** 185 of them in the subset, unused, and
  anatomically the right cell: it is the dedicated collision-course
  detector, its only route to steering is the escape DNs, and escape DN
  input is systematically left-biased (DNp03 10,665/8,100; DNp11
  11,641/9,739), so a symmetric loom should still resolve to a side.
  Produces **+0.00002 at drive scale 100** — nothing. Driving the escape
  DNs directly produces nothing either. In this subset the escape circuit
  cannot deliver a steering signal at all.
- **Steeper distance falloff** (1/d² and 1/d³, re-tested because the
  reason 1/d was chosen — a controller too slow to act on a late signal —
  went away with the heading integrator). Best 149 against 148; the rest
  worse.
- **Weighting the snake's own body above the wall**, which LC/LPLC cells
  argue for since they are small-object selective and a 13-cell wall line
  is not the stimulus they respond to. This one *worked at what it aimed
  at*: self-collisions fall 16 → 13 → 3 → 0 as the weight goes 1 → 3 → 10
  → 30. But wall deaths rise 24 → 27 → 37 → 40 and the apple total does
  not move.
- **Raising the obstacle gain**, in every combination above.

That last pattern is the finding, not the failure: **every one of these
trades one kind of death for another.** Apple attraction, wall repulsion
and body repulsion all compete for the same scalar — 16 DNa neurons per
side — and the budget is fixed.

### The 97%

So the question became what actually drives that readout. The 32 DNa*
neurons receive 326,741 total input weight, and everything this project
injects arrives through about **2.9%** of it — LC10 1.2%, PFL 0.9%, MBON
0.4%, LPLC1 0.2%, LPLC2 0. The rest is LAL, PS, AOTU and VES: the premotor
network, which in insects is where steering commands are assembled. The
subset had 8 of the LAL's 204 types and none of PS's 266.

Every pathway here had a real route we were not using:

| | → DNa* direct | → LAL/PS/AOTU | |
| --- | --- | --- | --- |
| LC10 (visual target) | 3,863 | **379,073** | 98x |
| LH (odour) | 66 | **16,672** | 253x |
| PFL (central complex) | 2,832 | 40,010 | 14x |
| MBON (mushroom body) | 1,145 | 25,164 | 22x |
| LPLC1 (obstacle) | 575 | 5,126 | 9x |

Phase 3.9 got odour working over a 66-weight connection while its real
16,672-weight route was not in the model. That is also the measured reason
raising any of these gains only ever moved deaths around: they were all
competing inside the 3% of the readout we could reach.

Adding LAL/PS/AOTU/VES costs 2,185 neurons (29,876 → 32,061) and 0.36M
edges (2.44M → 2.81M), and the simulation still runs at 2.44x real time,
1.00x over the wire. It makes the readout substantially better:

| apple off-axis | 15° | 45° | 90° | noise p90 |
| --- | --- | --- | --- | --- |
| before | 0.0031 | 0.0087 | 0.0123 | 0.00060 |
| after | 0.0096 | 0.0254 | 0.0335 | 0.00033 |

2.7x the signal, half the noise, graded out to 90° instead of flattening
at 60°. `TURN_RATE_GAIN` had to come down 850 → 310 to match — at 850 the
fly spins (48 apples, 47 of 60 deaths by self-collision), so this was a
required part of the change rather than tuning on top of it.

**Result: apples unchanged, variance gone.** 200 against 209 over 60
episodes, well inside the spread — but 67/66/67 across three noise seeds
where the old build gave 66/82/61. Deaths stay 38 wall / 22 self against
37 / 23. Confirmed live in the browser: ten lives in 180 seconds, an apple
every 2.5-4 seconds.

Kept on the evidence of faithfulness and reproducibility rather than
score, and stated as such: the premotor stage genuinely belongs in a model
that claims to read a decision off descending neurons, the readout is
measurably cleaner, and the obstacle pathway's real route into steering is
now present, which anything that improves avoidance will need.

### Synchronisation

There was a real problem and Phase 3.12 fixed it (sampling now runs on the
game tick instead of a free 100ms timer against a 150ms tick). What
remains is latency, not phase: injected current decays over 300ms and the
motor readout averages over 150ms, so a decision rests on geometry one to
two cells old. `MOTOR_EMA_TAU_MS` was measured as harmful to shorten in
Phase 3.11 — but that was when the readout was a threshold crossing and
now it is a rate that a heading integrator smooths anyway. Not yet
re-tested; it is the fourth instance of a setting that may have become
wrong when what reads it changed.

## Phase 3.14 — the slalom is a control oscillation, and T4/T5 went nowhere

Reported from playing it: most deaths are now self-collisions, the fly
travels in a slalom, and once it has eaten six or seven apples the body is
long enough that the weave runs into itself. Also still the occasional
straight-into-a-wall.

### The slalom, measured

Sixty episodes of 1000 ticks, recording the commanded turn rate every tick:

- the sign of the command reverses every ~6.7 ticks
- same-direction runs peak sharply at 4-5 ticks (53.6% of all runs)
- median run 5 ticks, so the **oscillation period is 10 ticks = 1.50s**

A pure integrator with delayed proportional feedback oscillates at roughly
four times the loop delay. The loop delay here is injected current decaying
over 300ms, plus a 150ms motor average, plus the 150ms tick — call it
375ms, and 4 x 375ms = 1.50s. The measurement and the model agree to the
tick. **The slalom is not something the fly decides; it is what a delayed
feedback loop does.**

And the self-collision lengths line up with the report exactly: 6, 6, 6, 6,
7, 7, then eight of them at length 8, then 9, 9, 9, 10, 11, 11, 11, 11. A
5-tick half-period weave has a wavelength around 10 cells, so a body of 8
is the first that can re-enter it.

### Shortening the delay works, and the fly's own trick does not

Cutting the loop delay shortens the period exactly as predicted, and the
self-collisions go with it — but the drive decay also sets how firmly the
fly commits to any turn, so cutting that costs wall safety:

| drive / motor window | period | self | wall | apples |
| --- | --- | --- | --- | --- |
| 300 / 150 | 10 ticks | 15 | 25 | 133 |
| 300 / 60 | 8 | 8 | 32 | 139 |
| 150 / 60 | 6 | **0** | 40 | 113 |
| 80 / 40 | 4 | **0** | 40 | 33 |

A real fly does not solve this by being faster. It adds rate feedback: the
**optomotor reflex**, where self-generated wide-field motion opposes an
unintended turn. That is derivative damping, which kills the oscillation
without weakening the pull toward the apple — and it needs the lobula
plate, which was missing.

### T4/T5 had no route to steering at all

Checked after Phase 3.13, because the premotor network was in and the
motion pathway still did nothing: **T4/T5 → DNa\* is 0 weight, and T4/T5 →
LAL/PS/AOTU/VES is also 0.** T4/T5 are early cells that project to the
lobula plate tangential cells, and not one of those matched
`MOTION_PATTERN`. So the retina has been driving 13,581 neurons — 42% of
the entire subset — into a dead end since Phase 3, lighting up the brain
panel and reaching the motor readout with nothing whatsoever.

The missing stage is 350 neurons (HS 6, VS 18, H1/H2 4, LPT 302, Am1 2):

| | weight |
| --- | --- |
| T4/T5 → it | 589,447 |
| it → LAL/PS/AOTU/VES | 34,002 |
| it → DNa* direct | 1,186 (mostly ipsilateral) |

Twice the obstacle pathway's direct route and six times its premotor one.
Cost: 350 neurons and 118k edges; still 1.00x over the wire.

### Reafferent optomotor damping: measured twice, rejected twice

With the pathway connected, the fly was given the visual consequence of its
own turning — the board is world-fixed and never rotates, so its retina
cannot produce this itself.

**First attempt, bilateral**, on the reasoning that a yaw sweeps the whole
panorama one way. The oscillation period did not move from 10 ticks at any
drive strength up to 30. The reason is the same one that killed the LPLC2
attempt in Phase 3.13: a bilaterally symmetric input to a left-minus-right
readout cancels exactly.

**Second attempt, antisymmetric**, which is the correct physics — turning
right, the right eye sees back-to-front and the left eye front-to-back, and
T4/T5a is the front-to-back subtype against T4/T5b's back-to-front (Maisak
et al. 2013). The two eyes see *opposite* eye-centric directions and that
asymmetry is the whole content of the signal. Period still 10 ticks at
every strength up to 100, apples flat. Removed from the code; the negative
result lives here instead.

So the optomotor route does not carry damping in this model, for the same
structural reason as everything else that enters through a thin pathway:
the perturbation that survives the homeostatic pools to reach 16 DNa
neurons a side is too small to matter.

### What did work

`MOTOR_EMA_TAU_MS` 150ms → 40ms — the fourth time a constant in this file
has been right, then wrong, then re-derived. Phase 3.11 measured 150ms as
necessary against a *threshold* readout with a 2.7x weaker signal; it is
now a rate that the heading integrator smooths anyway, so the window was
only adding lag. Sixty episodes of 700 ticks, three noise seeds:

| window | apples | best in one life | self-collisions | period |
| --- | --- | --- | --- | --- |
| **150ms** | **199** | **7** | **25** | **10 ticks** |
| 100ms | 208 | 8 | 24 | 8 |
| 60ms | 215 | 11 | 20 | 8 |
| **40ms** | **252** | **12** | **18** | **6** |
| 25ms | 217 | 12 | 11 | 6 |
| 15ms | 200 | 13 | 14 | 8 |
| 8ms | 177 | 7 | 17 | 8 |

**+27% apples, best in one life 7 → 12, self-collisions 25 → 18.** The
period tracks the window as the oscillation model says it should, and
below 40ms the old noise argument reasserts itself.

`TURN_RATE_GAIN` was re-swept at the new window and stays at 310. 450
scores more apples (269 against 252) but takes self-collisions from 18 to
40; the extra apples are inside the spread and the self-collisions are the
reported problem.

Live in the browser: 68 apples in 180 seconds, against 45 for the previous
build.

Self-collisions are reduced, not solved. What the sweep shows is that they
can be driven to zero by shortening the loop further — at a cost in wall
deaths that is worse than the gain. Splitting that trade needs the two
threats to stop competing for one scalar, which is still the open problem.

## Phase 3.15 — a self-critique, and the plan it produced, measured and rejected

Prompted by the report that the Phase 3.14 build hits walls more often (true:
wall deaths went from 35 to 42 of 60 in the harness, which the Phase 3.14
write-up underplayed) and by the sense that the project had stalled.

### Is the connectome doing the work at all?

Three conditions on identical seeds, 60 episodes of 700 ticks:

| | apples |
| --- | --- |
| real MaleCNS wiring | 252 |
| weight matrix zeroed | 0 (60/60 wall deaths) |
| same graph, neuron identities permuted | **0** (60/60 wall deaths) |

The permuted graph keeps every statistic — degrees, weights, signs,
sparsity — and scrambles only which biological cell is which. It cannot eat
a single apple. So the behaviour depends on the real anatomy, not on having
a large recurrent network with the right statistics. (The zeroed condition
is close to tautological, since DNa's only input is the network; the
permuted one is the informative control.)

### How much of the released data is used

15.3% of the 211,577 neurons, 11.4% of the 25.6M edges. Of the real input
weight onto the populations that matter, reproduced inside the subset:
DNa* 62.1%, lobula plate 68.6%, MBON 81.3%, LH 56.3%, LAL/PS 54.0%, LC10
38.9%, **LPLC1 23.8%, PN 14.5%** — the two worst-covered being the two
channels that have repeatedly failed.

What the model takes from the dataset is, in practice, the connectivity
matrix and each neuron's transmitter sign. Everything else is ours: uniform
LIF parameters for every cell, one global weight scale, no synaptic delays,
no conductance synapses, no modulation, a sensory interface that hands over
the apple's bearing as a number, and a homeostatic clamp with 16 hand-set
gains holding every population at a fixed rate.

### The plan, and why each step failed

**Visual local inhibition** (Li, LoVC, LoVP, TuTu, MeTu: +3,387 neurons,
1.2x edges). The case for it was good on paper: these cells act directly on
LC10 and LPLC1, 1,230 of the 1,558 Li cells are inhibitory, TuTu forms a
closed glutamatergic loop with LC10 (47,839 in, 37,288 back), LC10 coverage
would go 38.9% -> 60.8%, and the model's inhibition/excitation coverage
ratio 0.82 -> 0.99. The success criterion set in advance was that LC10's
response curve should stop saturating at VISUAL_SCALE 30. It did not
change (0.0544 / 0.0515 / 0.0557 at 45 / 60 / 90 degrees, against 0.0560 /
0.0549 / 0.0565 without). Behaviour: 219 apples against 252, and 236 with
the new cells' own clamp removed.

**Loosening the clamp**, which the balanced E/I ratio was supposed to
permit. All gains halved: 131 apples, 56 of 60 deaths by self-collision, an
oscillation period of 4 ticks — the fly spins. The clamp cannot be weakened.

**Antennal-lobe local neurons** (315 cells, 158 inhibitory, reciprocal with
the PNs at ~330,000 weight each way). 244 apples against 252; with their
clamp removed, 221, and the LNs run at 368x the target rate.

**Retinotopy** was gated on the first two and was not attempted. Two
arguments made for it earlier were also wrong on inspection: it would not
fix the head-on wall case (an object dead ahead is bilaterally symmetric
on a retinotopic map too, and the readout is still left minus right), and
the full hex-mapped retina layer *worsens* the E/I ratio (0.73).

**Neuromodulators.** The dataset has 101 octopaminergic, 48 serotonergic
and 396 dopaminergic cells (21, 8 and 344 of them already in the subset).
Octopamine is the most Snake-relevant — the flight/arousal modulator, with
OA-VUMa1 sending 17,716 weight into the LAL — but this simulator has no
notion of modulation: added as cells, they would be ordinary excitatory
neurons, and making them act as a hormone would be a mechanism written by
us, not read from the data. Not attempted.

**A looming veto on pursuit.** The looming pathway has no inhibitory route
at all onto LC10, the LAL or DNa (1,434 / 11,093 / 1,513 weight, all
excitatory). No anatomical basis; not attempted.

All three structural changes were reverted; the build is Phase 3.14's.

### What the diagnostics say about the regime the model is in

The injected populations run far above the target rate — LC10 at 36-66x,
PN at 35x — while everything downstream is pinned near 1x by the clamp. So
the operating picture is: saturated input layer -> clamped relay ->
left-minus-right readout. The only information that survives is which side
of a saturated input population is more active. Every addition this phase
landed in the clamped middle, where by construction it cannot change the
sustained activity of anything.

That is the honest summary of where the project stands. Adding real
anatomy helped while it opened routes that did not exist (Phase 3.13's
premotor network made the readout signal 2.7x stronger). It stops helping
once the routes exist, because the dynamics the anatomy runs on — uniform
parameters and a proportional clamp in place of real inhibition — flatten
whatever the extra wiring could express. Further gains most likely need a
change in model class (per-type dynamics, synaptic time constants, real
inhibitory balance) rather than more of the connectome in the current one.

## Phase 3.16 — which sense the fly plays with, and the first model-class probes

### It plays on the handed-over bearing, not on smell

Withholding senses, 60 episodes of 700 ticks each:

| the brain receives | apples |
| --- | --- |
| everything | 252 |
| **smell only** (no apple bearing to LC10 or FC) | **0** (60/60 wall) |
| vision only (no smell) | 239 |
| neither | 0 |

The odour channel works the way a real one should on the input side —
two antennae 1.2 cells apart, concentration `1/(1+d^2)` rising on
approach — but it contributes nothing measurable to play: removing it
changes 252 to 239, inside the spread. Held in isolation (fly advancing a
cell per tick past an apple 2, 4 or 6 cells to one side), the steering
readout's median response is 0.00002-0.00054, under the 0.00146 noise
floor and about 100x weaker than a 15-degree visual bearing, with peaks
of inconsistent sign.

Found while checking why: the klinokinesis mechanism
(`ODOUR_TURN_SUPPRESSION`, holding course while the lateral horn reports a
rising smell) only raises the threshold of the left/right/straight label.
Since Phase 3.12 made the heading integrator read the raw steering rate,
that label no longer steers anything, so klinokinesis has been silently
disconnected from behaviour since then.

### Model-class probes, all rejected

**Slowing the homeostatic clamp** (correct on a 200ms or 1s average
instead of 20ms, same gains) — the direct test of the Phase 3.15 diagnosis
that the clamp erases every sustained change: 17 and 114 apples against
252, every death a wall. Smell only stays at 0.

**Spike-frequency adaptation** on every neuron, meant to compress the
saturated input populations: 3 apples at the smallest increment (0.1), 0
at 0.3 and 1.0, with or without the clamp halved. It flattens the
steering difference in the relay stage along with everything else.

**A stronger odour input** (smell only, ODOUR_SCALE x10, x100, x1000): 0
apples at every strength, and the mean life is exactly 10 ticks — the
distance from the start position to the wall straight ahead. The fly
never turns once.

All reverted.

### The finding underneath

That last number ties the unsolved failures together. This fly turns only
when an injected signal differs between its left and right halves. It has
no spontaneous turning, no casting, no search. A wall dead ahead, an odour
gradient too shallow to differ between two antennae, and an apple it
cannot see all present the same thing — a symmetric input — and all
produce the same behaviour: straight on. Real flies turn spontaneously;
the LAL's bistable steering units generate saccades with no stimulus at
all, and odour-guided search is built on that (turn often while the smell
falls, rarely while it rises). The circuit is in this subset since Phase
3.13, but a population pinned at a fixed rate on a 20ms average cannot be
bistable.

## Phase 3.17 — why smell cannot steer, and whether this brain can turn on its own

### Smell: the gradient is there, the route to steering is not

The two antennae carry a strong lateral gradient: with the apple 2, 5 and
12 cells to one side the near antenna reads 90%, 46% and 20% more than the
far one. `inject_odour` discards nearly all of it — it drives the PNs only
while the *mean* concentration is rising, scaled by that small rise, so the
side-to-side difference only ever arrives multiplied by a few thousandths.

Feeding each antenna to its own side's PNs instead (its concentration, or
the normalised side-to-side contrast, at drive scales up to 10x apart, and
in one case one side only) produced a steering readout indistinguishable
from zero in every condition. The connectome explains it:

| route | weight into the premotor network | side-preserving |
| --- | --- | --- |
| vision: LC10 | 379,215 | 97% |
| smell: lateral horn | 21,222 | 70% |
| smell: MBON | 27,423 | 58% |

(PN -> LH itself is 96% side-preserving; it is the next hop that loses it.)
The net lateralised odour route is about 40x weaker than vision's. That is a
property of the fly, not of this model: the lateral horn reports what an
odour is and how strong, not where it is. Real flies mostly localise odour
over time — hold course while it strengthens, turn while it weakens — which
needs a fly that turns without a lateral cue.

### Can the network turn without input?

No sensory input, 30 s per run, three noise seeds; a "turn" counted only
above |0.005| (an apple ~10 degrees off, ~90 deg/s at the game's gain).

- **Clamp intact:** above threshold 0.0-0.3% of the time, 0-1 side changes.
- **Clamp removed from the premotor and descending populations:** nothing
  changes. Unclamped, the LAL sits at 1.0-1.3x the target rate. The clamp
  was not suppressing it; nothing drives it. The hypothesis that the clamp
  was hiding bistable LAL dynamics is refuted.
- **More background noise** does produce spontaneous, balanced alternation
  through the connectome: at NOISE_STD 0.3, 6-13 side changes per 30 s
  with a median dwell of 1.4-2.3 s; at 0.5, 26-30 with 0.65-0.87 s. With
  the premotor clamp also removed it runs away (LAL 55x target, dwell
  140 ms).

In play it does not help. Smell only stays at 0 apples with a mean life of
10-11 ticks — the excursions peak around 0.011, about 29 degrees per tick,
and reverse before the heading accumulates the 45 degrees needed to change
grid direction. With all senses, noise 0.3 gives 171 apples and 0.5 gives
126, against 252.

Nothing changed in the code. Tested and rejected, with the whole-brain
alternative measured rather than estimated: the full MaleCNS graph
(164,740 connected neurons, 25.6M edges) fits in 2.4 GiB of the laptop
GPU, but its synaptic matmul alone runs at 0.20x real time in the COO
format lif.py uses and 0.61x in CSR. CSR is also 3x faster than COO on
this operation, which the current subset would benefit from too.

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
