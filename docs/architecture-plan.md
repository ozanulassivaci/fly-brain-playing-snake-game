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

**Phase 1 — Visual skeleton.** Cherry-picked MuJoCo/NeuroMechFly body,
arcade cabinet scene, Snake screen (scripted/keyboard-controlled),
a decorative brain animation synced to game events (similar in spirit to
fly-parking-lab's, explicitly not real neural data yet). Leg movement is
scripted/procedural, no MuJoCo contact physics. Goal: visually
convincing, no real neural data involved yet.

**Phase 2 — Real connectome visualization.** Python/CUDA backend
simulates the real MaleCNS functional subset (LIF), streams spikes over
WebSocket. Brain panel renders real activity on the full-connectome
structure. Game is still scripted/human-controlled — the brain sim is an
observer, not yet a driver.

**Phase 3 — Closed loop.** Snake screen → simplified retina reduction →
visual neurons → LIF sim → descending neurons → IK targets → joystick/
button animation → real key input. MuJoCo contact physics replaces the
Phase 1 scripted animation here.

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
