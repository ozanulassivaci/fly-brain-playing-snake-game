import * as THREE from 'three';

// Real MaleCNS v1.0 soma positions (Phase 0 data, see docs/architecture-plan.md)
// for the candidate functional subset — motion pathway / central complex /
// descending neurons — rendered as a glowing point cloud. The STRUCTURE here
// is real; the pulsing on game events is decorative, not simulated activity
// (that's Phase 2/3). See frontend/assets/brain-subset.json.

const CLUSTER_COLORS = {
  motion: 0x50b4ff,
  cx: 0xffc850,
  dn: 0xff6482,
};

const BASE_SIZE = { motion: 0.016, cx: 0.045, dn: 0.055 };
const BASE_OPACITY = { motion: 0.4, cx: 0.6, dn: 0.65 };

const PULSE_TARGETS = {
  move: ['motion'],
  eat: ['cx', 'dn'],
  collide: ['dn'],
};

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

export async function createBrainViz(canvas) {
  const data = await fetch('./assets/brain-subset.json').then((r) => r.json());
  const byCluster = { motion: [], cx: [], dn: [] };
  for (const p of data.points) byCluster[p.c]?.push(p);

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
  for (const key of Object.keys(byCluster)) {
    const pts = byCluster[key];
    const positions = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: CLUSTER_COLORS[key],
      size: BASE_SIZE[key],
      map: glowTexture,
      transparent: true,
      opacity: BASE_OPACITY[key],
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    clusters[key] = { material, pulse: 0 };
  }

  function pulse(kind) {
    const targets = PULSE_TARGETS[kind];
    if (!targets) return;
    targets.forEach((key) => {
      if (clusters[key]) clusters[key].pulse = 1;
    });
  }

  function render(nowSec) {
    scene.rotation.y = nowSec * 0.15;
    for (const key of Object.keys(clusters)) {
      const c = clusters[key];
      c.pulse = Math.max(0, c.pulse - 0.02);
      c.material.size = BASE_SIZE[key] * (1 + c.pulse * 1.8);
      c.material.opacity = Math.min(1, BASE_OPACITY[key] + c.pulse * 0.5);
    }
    renderer.render(scene, camera);
  }

  return { render, pulse };
}
