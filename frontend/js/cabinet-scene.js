import * as THREE from 'three';

const CABINET_COLOR = 0x2a2f3a;
const TRIM_COLOR = 0x1a1d24;

export function createCabinetScene() {
  const group = new THREE.Group();

  const bodyMat = new THREE.MeshStandardMaterial({ color: CABINET_COLOR, roughness: 0.6 });
  const trimMat = new THREE.MeshStandardMaterial({ color: TRIM_COLOR, roughness: 0.5 });

  const body = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.5, 0.55), bodyMat);
  body.position.y = 0.75;
  group.add(body);

  const marquee = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.18, 0.2), trimMat);
  marquee.position.set(0, 1.55, -0.15);
  group.add(marquee);

  const screenBezel = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.42, 0.04), trimMat);
  screenBezel.position.set(0, 1.15, 0.27);
  group.add(screenBezel);

  const screenGeometry = new THREE.PlaneGeometry(0.46, 0.36);
  const screenMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
  const screen = new THREE.Mesh(screenGeometry, screenMaterial);
  screen.position.set(0, 1.15, 0.295);
  group.add(screen);

  const panel = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.05, 0.3), trimMat);
  panel.position.set(0, 0.75, 0.4);
  panel.rotation.x = -0.35;
  group.add(panel);

  const joystickBase = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.035, 0.03, 16), trimMat);
  joystickBase.position.set(-0.12, 0.79, 0.46);
  group.add(joystickBase);

  // Stick + ball pivot at the base, so tilting is a single local rotation.
  const joystickPivot = new THREE.Group();
  joystickPivot.position.set(-0.12, 0.805, 0.46);
  group.add(joystickPivot);

  const joystickStick = new THREE.Mesh(
    new THREE.CylinderGeometry(0.008, 0.008, 0.12, 12),
    new THREE.MeshStandardMaterial({ color: 0x444444 }),
  );
  joystickStick.position.set(0, 0.06, 0);
  joystickPivot.add(joystickStick);

  const joystickBall = new THREE.Mesh(
    new THREE.SphereGeometry(0.022, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0xdd2222 }),
  );
  joystickBall.position.set(0, 0.12, 0);
  joystickPivot.add(joystickBall);

  const MAX_TILT = 0.35; // radians
  function setJoystickTilt(dx, dy) {
    joystickPivot.rotation.z = -dx * MAX_TILT;
    joystickPivot.rotation.x = dy * MAX_TILT;
  }

  const buttonColors = [0xdddd22, 0x22aadd, 0x22dd66];
  buttonColors.forEach((color, i) => {
    const button = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.015, 16),
      new THREE.MeshStandardMaterial({ color }),
    );
    button.position.set(0.05 + i * 0.06, 0.775, 0.46);
    group.add(button);
  });

  return { group, screen, joystickBall, setJoystickTilt };
}
