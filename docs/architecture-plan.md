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
- **Realistic subset size**: optic lobe ~53,000 neurons / 732 types
  (male-cns-specific, confirmed). Central complex ~3,000 neurons
  (hemibrain analogy, not male-cns-specific). Descending neurons
  ~1,300 (MANC-specific count). So "optic lobe + CX + DN" is still
  ~55–60k neurons — too large for a first real-time target. We will
  likely need a much smaller sub-subset (specific types most relevant to
  motion/turning, probably low thousands or fewer) for the actual live
  simulation; the rest of the loaded subset can remain structural/
  decorative. Synapse count for this subset is unconfirmed for
  male-cns specifically; order-of-magnitude tens of millions based on
  hemibrain's ratio (extrapolation, not a published figure).
- **License**: CC-BY, attribution required (FlyEM/HHMI Janelia +
  Cambridge/MRC LMB + Google Research; cite the *Cell* paper, bioRxiv
  DOI 10.1101/2025.10.09.680999). No stated commercial-use or
  redistribution restriction beyond attribution. Exact license version
  (e.g. 4.0) not directly confirmed from a formal ToS page — worth a
  final check before redistributing any derived/filtered dataset.
  This is separate from and in addition to the NeuroMechFly model's own
  Apache 2.0 license, which only covers the biomechanical model assets.

Open question for the next Phase 0 step: what is the real neuron/type
distribution once we actually query or download the data, and which
concrete types should the live-simulated sub-subset consist of?

## Phase breakdown

**Phase 0 — Data discovery (current phase).** Confirm real access
method, schema, and realistic subset size by actually pulling MaleCNS
data (bulk export or a neuPrint sample query), not just reading docs.
Decide the concrete list of neuron types that will be live-simulated.

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

- Exact neuron types to include in the live-simulated sub-subset (needs
  real data inspection, Phase 0 next step).
- Whether the ROI ontology and DN `class` field used by hemibrain/MANC
  apply identically to male-cns (assumed, not yet confirmed against the
  actual downloaded schema).
- Confirm exact MaleCNS license version/terms before any public
  redistribution of derived data.
- WebGPU in-browser simulation was rejected for now but could be
  revisited if the two-process (Python backend + browser) setup proves
  operationally inconvenient.
