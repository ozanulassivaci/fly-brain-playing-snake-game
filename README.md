# Fly Brain Plays Snake

A spiking simulation of 32,411 neurons from the real male *Drosophila* connectome (MaleCNS v1.0) steers a Snake game, with no scripted controller between the brain and the game.

<!-- screenshot: the arcade cabinet with the fly, the 3D brain panel and the decision panel -->

## Features

- **Leaky integrate-and-fire network on GPU.** 32,411 neurons and 2,923,473 synapses taken from the MaleCNS v1.0 connectome. Synapse sign comes from each neuron's consensus neurotransmitter (GABA and glutamate are inhibitory). It runs at 2.3x real time on an RTX 4070 Laptop GPU.
- **Real circuits carry the signals.**
  - LC10 cells (visual target pursuit) receive the apple's direction.
  - LPLC1 cells (looming) receive nearby walls and the snake's own body.
  - Antennal-lobe projection neurons receive a two-antenna odour gradient. The odour signal reaches steering through the lateral horn and mushroom body.
  - The motor decision is read from the DNa steering descending neurons. The lateral accessory lobe and posterior slope premotor network sits upstream of them.
- **No hidden autopilot.** The game only turns when the descending-neuron activity says so.
  - We tested whether the real wiring matters. The fly keeps every graph statistic (degrees, weights, signs, sparsity) but has its neuron identities randomly permuted. That fly eats 0 apples in 60 lives. The real wiring eats 252.
- **Live visualisation.**
  - A Three.js scene shows a NeuroMechFly body model (MuJoCo, WebAssembly) at an arcade cabinet.
  - A point cloud shows real soma positions and flashes on real spikes.
  - A decision panel shows population rates stage by stage, from sensory input to the turn command.

Measured performance, over 60 episodes of 700 game ticks (20 game seeds × 3 noise seeds):

- 252 apples in total
- best single life: 12 apples
- deaths: 42 into a wall, 18 into its own body

In the browser it eats about one apple every 2.5 to 3 seconds.

## Tech stack

- Simulation: Python 3.10, PyTorch (CUDA), NumPy
- Server: FastAPI + Uvicorn. The server streams spikes and the motor command over a WebSocket at 50 Hz.
- Data preparation: pandas, pyarrow
- Frontend: plain ES modules, Three.js 0.169, MuJoCo compiled to WebAssembly. Everything is vendored, so no build step and no CDN.

## Installation

Requires an NVIDIA GPU with CUDA and about 5.2 GB of disk for the raw connectome data.

1. Install PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/). For CUDA 12.4:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

2. Install the rest of the dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Download the MaleCNS v1.0 data from Janelia's public bulk export (no account needed). The files are 14 MB, 502 MB and 4.6 GB:

   ```bash
   mkdir -p data/raw
   BASE=https://storage.googleapis.com/flyem-male-cns/v1.0
   curl -L -o data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
     $BASE/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
   curl -L -o data/raw/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather \
     $BASE/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather
   curl -L -o data/raw/Neuprint_Neurons.feather \
     $BASE/database/neuprint-inputs/Neuprint_Neurons.feather
   ```

4. Build the simulated subset. This step writes `data/processed/subset.npz` and `frontend/assets/brain-subset.json`:

   ```bash
   python3 backend/scripts/prepare_subset.py
   ```

## Usage

Run each command from the repository root, in its own terminal.

1. Start the brain server:

   ```bash
   uvicorn backend.server:app --port 8765
   ```

2. Serve the frontend:

   ```bash
   python3 -m http.server 8080 --directory frontend
   ```

3. Open http://localhost:8080.

The frontend needs the backend on `ws://localhost:8765`. Without it, the brain panel shows the static structure and the snake never turns.

## Project structure

```
backend/
  lif.py                     LIF simulation: sensory injection, homeostasis, motor readout
  server.py                  WebSocket server, keeps simulated time locked to wall-clock time
  scripts/prepare_subset.py  selects the neuron subset from the raw connectome
frontend/
  index.html
  js/snake-game.js           the game, the fly's continuous heading, threat and odour geometry
  js/main.js                 wires the game, the brain stream and the 3D scene together
  js/retina.js               Reichardt-style motion detector over the game canvas (feeds T4/T5)
  js/brain-viz.js            3D point cloud of real soma positions, flashed by real spikes
  js/decision-panel.js       per-stage population rates
  js/fly-body.js, js/cabinet-scene.js, js/scene-loader.js
  assets/                    NeuroMechFly model, brain-subset.json
  vendor/                    MuJoCo (WebAssembly), Three.js
docs/architecture-plan.md    design notes and every measurement, including rejected approaches
```

## Limitations

- **Some sensory input is computed by the game, not seen by the fly.**
  - The apple's direction relative to the fly, the fly's heading, the wall and body proximity, and the odour concentration at each antenna are all calculated in JavaScript and injected into the right neuron populations.
  - Only the T4/T5 motion signal is computed from pixels.
  - The connectome makes the decision; the senses are simplified.
- **The dynamics are generic.**
  - Every neuron has the same threshold, leak and refractory period.
  - A single global constant converts synapse count to current.
  - There are no synaptic delays and no neuromodulation.
- **The activity is held stable by an invented feedback loop.**
  - The subset keeps proportionally less of the real inhibition than of the real excitation.
  - To stop the network from running away, a proportional homeostatic feedback loop holds each population near a target firing rate.
  - That loop is not in the data, and weakening it makes the fly spin.
- **The fly still dies often.** About two thirds of deaths are wall collisions, often while chasing an apple next to a wall. Once the snake reaches about 8 segments, it also runs into its own body.
- **The subset is 15.3% of the released neurons and 11.4% of its synapses.** Coverage of real input is 62% for the steering neurons, 24% for the looming population and 15% for the olfactory projection neurons.
- **The game spawns apples away from the edges more often than classic Snake does.**

## License

No license has been chosen for this project's own code yet. Third-party assets and data carry their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md):

- NeuroMechFly v2 and MuJoCo: Apache 2.0
- Three.js: MIT
- MaleCNS v1.0 connectome: CC-BY, attribution to FlyEM/HHMI Janelia, University of Cambridge/MRC LMB and Google Research
