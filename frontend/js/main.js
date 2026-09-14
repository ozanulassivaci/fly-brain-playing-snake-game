import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadScene, makeFailOverlay } from './scene-loader.js';
import { createFlyBody } from './fly-body.js';
import { createCabinetScene } from './cabinet-scene.js';
import { createSnakeGame } from './snake-game.js';
import { createBrainViz } from './brain-viz.js';

const overlay = document.getElementById('overlay');
const viewport = document.getElementById('viewport');
const fail = makeFailOverlay(overlay);

const TARGET_FLY_HEIGHT = 0.38; // meters, sized relative to the cabinet's control panel

async function main() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11141c);

  const camera = new THREE.PerspectiveCamera(45, viewport.clientWidth / viewport.clientHeight, 0.01, 100);
  camera.position.set(1.5, 1.5, 2.3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(-0.05, 0.9, 0.3);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(2, 3, 2);
  scene.add(sun);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 6),
    new THREE.MeshStandardMaterial({ color: 0x1a1d24, roughness: 0.9 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const { group: cabinetGroup, screen, joystickBall, setJoystickTilt } = createCabinetScene();
  scene.add(cabinetGroup);

  overlay.textContent = 'Loading MuJoCo (WebAssembly)…';
  const { mj, model, data, meta } = await loadScene({
    assetsDir: './assets/fly-model',
    xmlName: 'fly.xml',
    onStage: (text) => (overlay.textContent = text),
  });

  const fly = createFlyBody(mj, model, data, meta);
  fly.update(0);

  const flyWrapper = new THREE.Group();
  flyWrapper.add(fly.group);
  scene.add(flyWrapper);

  const box = new THREE.Box3().setFromObject(fly.group);
  const size = box.getSize(new THREE.Vector3());
  const scale = TARGET_FLY_HEIGHT / Math.max(size.x, size.y, size.z, 1e-6);
  const center = box.getCenter(new THREE.Vector3());
  // MuJoCo's model is Z-up; Three.js is Y-up. rotationX corrects for that
  // (without it the fly renders lying on its side, on top of a Y-heading
  // rotation that just spins it while still lying down — confirmed by
  // comparing screenshots across both axes). rotationY then sets which way
  // it faces once actually standing upright.
  const rotationX = -Math.PI / 2;
  const rotationY = 0;
  const euler = new THREE.Euler(rotationX, rotationY, 0);
  flyWrapper.scale.setScalar(scale);
  flyWrapper.rotation.copy(euler);
  // position is set so the *rotated, scaled* bounding-box center (not the
  // raw local center) lands at the target world point next to the joystick.
  const targetPosition = new THREE.Vector3(-0.24, 0.82, 0.56);
  const centerOffset = center.clone().multiplyScalar(scale).applyEuler(euler);
  flyWrapper.position.copy(targetPosition).sub(centerOffset);

  const brainCanvas = document.getElementById('brain-canvas');
  const brainViz = await createBrainViz(brainCanvas);

  const snake = createSnakeGame({
    onMove: () => brainViz.pulse('move'),
    onEat: () => brainViz.pulse('eat'),
    onCollide: () => brainViz.pulse('collide'),
  });
  const screenTexture = new THREE.CanvasTexture(snake.canvas);
  screenTexture.colorSpace = THREE.SRGBColorSpace;
  screen.material = new THREE.MeshBasicMaterial({ map: screenTexture });

  overlay.classList.add('hidden');

  // On every snake direction change: tilt the joystick that way and have the
  // front-left leg reach toward it for a moment (see fly-body.js's
  // solveReach). Purely cosmetic sync with the game, not a real control loop
  // (that's Phase 3) — but real IK against the real skeleton, not a canned pose.
  let lastDir = [0, 0];
  let reachTriggeredAt = -Infinity;
  const REACH_RAMP = 0.15,
    REACH_HOLD = 0.15,
    REACH_DECAY = 0.4;
  function reachEnvelope(elapsed) {
    if (elapsed < 0) return 0;
    if (elapsed < REACH_RAMP) return elapsed / REACH_RAMP;
    if (elapsed < REACH_RAMP + REACH_HOLD) return 1;
    if (elapsed < REACH_RAMP + REACH_HOLD + REACH_DECAY) return 1 - (elapsed - REACH_RAMP - REACH_HOLD) / REACH_DECAY;
    return 0;
  }

  const _targetWorld = new THREE.Vector3();
  let last = performance.now() / 1000;
  function frame(nowMs) {
    requestAnimationFrame(frame);
    const now = nowMs / 1000;
    const dt = Math.min(0.1, now - last);
    last = now;

    const dir = snake.getDirection();
    if (dir[0] !== lastDir[0] || dir[1] !== lastDir[1]) {
      lastDir = dir;
      setJoystickTilt(dir[0], dir[1]);
      joystickBall.getWorldPosition(_targetWorld);
      const targetLocal = flyWrapper.worldToLocal(_targetWorld.clone());
      fly.triggerReach(targetLocal);
      reachTriggeredAt = now;
    }

    fly.update(now, reachEnvelope(now - reachTriggeredAt));
    snake.update(dt);
    screenTexture.needsUpdate = true;
    brainViz.render(now);

    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);

  window.addEventListener('resize', () => {
    camera.aspect = viewport.clientWidth / viewport.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  });
}

main().catch((err) => fail('Failed to start the scene.', err));
