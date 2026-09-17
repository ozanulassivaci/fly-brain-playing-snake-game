"""
WebSocket server streaming real LIF spike activity for the Phase 0 subset
to the frontend's brain panel. Run with:

    uvicorn backend.server:app --port 8765

Requires data/processed/subset.npz — generate it first with
backend/scripts/prepare_subset.py.
"""

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.lif import LifSimulation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("brain-server")

app = FastAPI()
sim = LifSimulation()
log.info("LIF simulation running on %s (%d neurons)", sim.device, sim.n)

STEPS_PER_BROADCAST = 20  # 20ms of simulated time per broadcast, ~50Hz
BROADCAST_INTERVAL_S = 0.02

clients: set[WebSocket] = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    log.info("client connected (%d total)", len(clients))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "sensory":
                sim.inject_sensory(msg)
            elif msg.get("type") == "reward":
                sim.inject_reward()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        log.info("client disconnected (%d total)", len(clients))


async def simulation_loop():
    # The sleep below has to account for how long the batch itself took,
    # or simulated time falls behind wall-clock time: each batch advances
    # exactly STEPS_PER_BROADCAST ms of simulated brain time, so sleeping a
    # further BROADCAST_INTERVAL_S *after* the compute made the brain run at
    # 20/(20+compute) of real speed — measured at ~0.42x before the
    # single-sync readout optimization in lif.py. A fly whose brain runs at
    # half the speed of the game it is playing reacts to every event a tick
    # late, which is not a property of the connectome, just of our loop.
    loop = asyncio.get_running_loop()
    next_deadline = loop.time()
    while True:
        spiked = sim.step_batch(STEPS_PER_BROADCAST)
        if clients:
            payload = json.dumps(
                {
                    "type": "spikes",
                    "indices": spiked,
                    "motor": {"turn": sim.read_motor(), "rate": sim.read_turn_rate()},
                    "groups": sim.read_groups(),
                }
            )
            dead = []
            for ws in clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
        next_deadline += BROADCAST_INTERVAL_S
        delay = next_deadline - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Fell behind (GPU hiccup, or a machine too slow for this
            # subset): don't try to catch up by spinning, just resync so
            # simulated time keeps tracking wall-clock time going forward.
            next_deadline = loop.time()
            await asyncio.sleep(0)


@app.on_event("startup")
async def start_simulation_loop():
    asyncio.create_task(simulation_loop())
