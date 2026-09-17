import * as THREE from 'three';

// Real MaleCNS v1.0 soma positions (Phase 0 data, see docs/architecture-plan.md)
// for the candidate functional subset — motion pathway / central complex /
// descending neurons — rendered as a glowing point cloud. Point order in
// frontend/assets/brain-subset.json is the same canonical neuron index the
// backend LIF simulation uses (backend/scripts/prepare_subset.py generates
// both together), so a spike index from the WebSocket maps directly onto a
// point here. If the backend isn't running, the panel just shows the real
// structure with no activity — it degrades gracefully, doesn't fail.

const CLUSTER_COLORS = {
  motion: [0.31, 0.71, 1.0],
  cx: [1.0, 0.78, 0.31],
  dn: [1.0, 0.39, 0.51],
};

const BASE_SIZE = { motion: 0.01, cx: 0.03, dn: 0.04 };
const BASE_OPACITY = { motion: 0.25, cx: 0.4, dn: 0.45 };
const DIM_FACTOR = 0.12;
const FLASH_DECAY = 0.82; // multiplicative color decay per render frame

const WS_URL = 'ws://localhost:8765/ws';

function makeGlowTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.4, 'rgba(255,255,255,0.55)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

export async function createBrainViz(canvas, { onMotor, onGroups } = {}) {
  const data = await fetch('./assets/brain-subset.json').then((r) => r.json());
  const byCluster = { motion: [], cx: [], dn: [] };
  data.points.forEach((p, globalIndex) => {
    byCluster[p.c]?.push({ ...p, globalIndex });
  });

  const width = canvas.width,
    height = canvas.height;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(width, height, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1018);
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 10);
  camera.position.set(0, 0.25, 2.1);

  const glowTexture = makeGlowTexture();
  const clusters = {};
  const indexLookup = new Map(); // globalIndex -> { cluster, local }

  for (const key of Object.keys(byCluster)) {
    const pts = byCluster[key];
    const [r, g, b] = CLUSTER_COLORS[key];
    const dim = [r * DIM_FACTOR, g * DIM_FACTOR, b * DIM_FACTOR];
    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      colors[i * 3] = dim[0];
      colors[i * 3 + 1] = dim[1];
      colors[i * 3 + 2] = dim[2];
      indexLookup.set(p.globalIndex, { cluster: key, local: i });
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const colorAttr = new THREE.BufferAttribute(colors, 3);
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('color', colorAttr);
    const material = new THREE.PointsMaterial({
      size: BASE_SIZE[key],
      map: glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: BASE_OPACITY[key],
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    scene.add(new THREE.Points(geometry, material));
    clusters[key] = { colorAttr, dim, bright: [r, g, b] };
  }

  function flashGlobalIndex(globalIndex) {
    const loc = indexLookup.get(globalIndex);
    if (!loc) return;
    const { colorAttr, bright } = clusters[loc.cluster];
    const arr = colorAttr.array;
    arr[loc.local * 3] = bright[0];
    arr[loc.local * 3 + 1] = bright[1];
    arr[loc.local * 3 + 2] = bright[2];
    colorAttr.needsUpdate = true;
  }

  let ws = null;
  try {
    ws = new WebSocket(WS_URL);
    ws.addEventListener('message', (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'spikes') {
        for (const idx of msg.indices) flashGlobalIndex(idx);
        if (msg.motor) onMotor?.(msg.motor.turn, msg.motor.rate ?? 0);
        if (msg.groups) onGroups?.(msg.groups, msg.motor?.turn);
      }
    });
    ws.addEventListener('error', () => {
      console.warn('brain-viz: backend WebSocket unavailable, showing static structure only');
    });
  } catch (err) {
    console.warn('brain-viz: could not open WebSocket', err);
  }

  function send(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function sendSensory(values) {
    send({ type: 'sensory', ...values });
  }

  function sendReward() {
    send({ type: 'reward' });
  }

  function render(nowSec) {
    scene.rotation.y = nowSec * 0.15;
    for (const key of Object.keys(clusters)) {
      const { colorAttr, dim } = clusters[key];
      const arr = colorAttr.array;
      let changed = false;
      for (let i = 0; i < arr.length; i += 3) {
        for (let c = 0; c < 3; c++) {
          const target = dim[c];
          const cur = arr[i + c];
          if (cur > target + 0.001) {
            arr[i + c] = target + (cur - target) * FLASH_DECAY;
            changed = true;
          }
        }
      }
      if (changed) colorAttr.needsUpdate = true;
    }
    renderer.render(scene, camera);
  }

  return { render, sendSensory, sendReward };
}
