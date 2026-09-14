# Third-Party Notices

This project vendors and adapts assets and code from other open-source
projects, listed below.

## NeuroMechFly v2 (fly biomechanical model)

- Files: `frontend/assets/fly-model/` (MJCF model, meshes, model metadata)
- License: Apache License 2.0. Copyright 2023-2026 The NeuroMechFly v2
  Authors. Full text: `frontend/assets/fly-model/LICENSE-NeuroMechFly.txt`.
- Sourced via `powerOFMAX/fly-parking-lab` (public/nmf/game/assets/model/),
  itself built on the official NeuroMechFly v2 release.
- Not modified.

## MuJoCo (WebAssembly build)

- Files: `frontend/vendor/mujoco/`
- MuJoCo itself is licensed under the Apache License 2.0 by Google
  DeepMind. This specific WebAssembly compilation was sourced via
  `powerOFMAX/fly-parking-lab` (public/nmf/shared/vendor/mujoco/); its own
  build provenance is not separately documented there. Not modified.

## Three.js

- Files: `frontend/vendor/three/` (`three.module.js`, `OrbitControls.js`)
- Version: 0.169.0
- License: MIT License, Copyright (c) 2010-2026 three.js authors.
- Sourced via `powerOFMAX/fly-parking-lab` (public/nmf/shared/vendor/three/).
  Not modified.

## Reused/adapted code pattern

- `frontend/js/scene-loader.js` is adapted from
  `powerOFMAX/fly-parking-lab`'s `public/nmf/shared/scene.js` (Apache
  License 2.0). See the file header for the specific modifications made.

## MaleCNS v1.0 connectome data

- The raw dataset itself is not redistributed in this repository (see
  `.gitignore`'s `data/` entry) — downloaded directly from Janelia's
  public bulk export for local analysis only.
- `frontend/assets/brain-subset.json` **is** committed and **is**
  derived from this dataset: real 3D soma positions (from
  `body-annotations-male-cns-v1.0-minconf-0.5.feather`'s `somaLocation`
  field) for the ~22.7k-neuron candidate functional subset identified in
  Phase 0 (motion pathway / central complex / descending neurons — see
  `docs/architecture-plan.md`), centered and rescaled, with each point
  tagged only by its cluster (motion/cx/dn) — no per-neuron type names,
  IDs, or connectivity are included. Used by `frontend/js/brain-viz.js`
  to render a real, anatomically-shaped point cloud; the pulsing shown
  on game events remains decorative/scripted, not simulated activity.
- License: CC-BY. Attribution: FlyEM/HHMI Janelia, University of
  Cambridge/MRC LMB, and Google Research. Citation: "Sexual dimorphism in
  the complete connectome of the Drosophila male central nervous
  system," bioRxiv DOI 10.1101/2025.10.09.680999.
- A handful of real neuron type names from this dataset (e.g. T4, T5,
  EPG, DNa01) were also used as labels in the brain panel's earlier 2D
  version; the current 3D point cloud omits per-point labels.
