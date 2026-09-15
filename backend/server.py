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
            if msg.get("type") == "event":
                sim.inject_event(msg.get("kind", ""))
            elif msg.get("type") == "sensory":
                sim.inject_sensory(msg)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        log.info("client disconnected (%d total)", len(clients))


async def simulation_loop():
    while True:
        spiked = sim.step_batch(STEPS_PER_BROADCAST)
        if clients:
            payload = json.dumps(
                {
                    "type": "spikes",
                    "indices": spiked,
                    "motor": {"turn": sim.read_motor()},
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
        await asyncio.sleep(BROADCAST_INTERVAL_S)


@app.on_event("startup")
async def start_simulation_loop():
    asyncio.create_task(simulation_loop())
