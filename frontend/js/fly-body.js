import { buildMeshes, syncMeshes } from './scene-loader.js';

// Order must match model_meta.json's preprogrammed.legs keys and
// ctrl_index_by_leg_dof's row order (both generated together from the same
// NeuroMechFly asset pipeline).
const LEG_NAMES = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];

const CYCLES_PER_SECOND = 0.7;

// Front-left leg's yaw/pitch/femur-pitch qpos indices, confirmed directly from
// model_meta.json's actuators list (id 0/1/3 -> qposadr 7/8/10) rather than
// through ctrl_index_by_leg_dof's permutation, whose dof-order convention
// isn't independently confirmed to match preprogrammed.legs' column order.
const REACH_QPOSADR = [7, 8, 10];
const REACH_TIP_BODY = 'nmf/lf_tarsus1';

// fly.xml (copied from fly-parking-lab) bundles the fly body together with
// that game's driving-course markers (gate*/ground_plane/start_pole_*) in one
// MJCF. All real fly body geoms are named "nmf/...": drop everything else.
function pruneNonFlyGeoms(group) {
  for (const item of [...group.userData.items]) {
    if (!item.mesh.userData.name.startsWith('nmf/')) group.remove(item.mesh);
  }
  group.userData.items = group.userData.items.filter((item) => item.mesh.userData.name.startsWith('nmf/'));
}

// Builds the fly's Three.js mesh group and an update(nowSec) function that
// plays back NeuroMechFly's own recorded walking-cycle joint angles via pure
// forward kinematics (mj_forward only, never mj_step) — no contact physics,
// no time integration, just posing a real skeleton with real gait data.
export function createFlyBody(mj, model, data, meta) {
  const group = buildMeshes(model, meta);
  pruneNonFlyGeoms(group);

  const qposadrByActuatorId = meta.actuators.map((a) => a.qposadr);
  const legDofQposadr = meta.ctrl_index_by_leg_dof.map((row) => row.map((actuatorId) => qposadrByActuatorId[actuatorId]));
  const legAngles = LEG_NAMES.map((name) => meta.preprogrammed.legs[name].angles);
  const nSamples = meta.preprogrammed.n_samples;

  data.qpos.set(meta.neutral_qpos);

  let reachAngles = null;

  // Numerically solves for the front-left leg's yaw/coxa-pitch/femur-pitch
  // angles that bring its tarsus tip closest to targetLocalPos (in the
  // model's own coordinate space, i.e. already converted out of Three.js
  // world/flyWrapper space by the caller). Coordinate-descent against real
  // mj_forward output rather than closed-form trig, since the joints' local
  // axis conventions aren't ones we can trust blindly (already got bitten
  // once this session by a wrong axis assumption for the whole-body
  // rotation) — this way it's self-correcting regardless of sign/axis
  // convention. Cheap: a few dozen mj_forward calls on a tiny model, run
  // only once per direction change, not per frame.
  function solveReach(targetLocalPos) {
    const savedQpos = data.qpos.slice();
    const angles = REACH_QPOSADR.map((adr) => data.qpos[adr]);

    const apply = () => {
      REACH_QPOSADR.forEach((adr, i) => {
        data.qpos[adr] = angles[i];
      });
      mj.mj_forward(model, data);
    };
    const tipError = () => {
      const p = data.body(REACH_TIP_BODY).xpos;
      const dx = p[0] - targetLocalPos.x,
        dy = p[1] - targetLocalPos.y,
        dz = p[2] - targetLocalPos.z;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    };

    apply();
    let bestErr = tipError();
    let step = 0.35;
    for (let pass = 0; pass < 6; pass++) {
      for (let i = 0; i < angles.length; i++) {
        for (const delta of [step, -step]) {
          const prev = angles[i];
          angles[i] = prev + delta;
          apply();
          const err = tipError();
          if (err < bestErr) {
            bestErr = err;
          } else {
            angles[i] = prev;
          }
        }
      }
      step *= 0.5;
    }

    data.qpos.set(savedQpos);
    reachAngles = angles;
  }

  function update(nowSec, reachWeight = 0) {
    const frame = Math.floor(nowSec * CYCLES_PER_SECOND * nSamples) % nSamples;
    for (let leg = 0; leg < LEG_NAMES.length; leg++) {
      const angles = legAngles[leg][frame];
      const qposadr = legDofQposadr[leg];
      for (let dof = 0; dof < qposadr.length; dof++) {
        data.qpos[qposadr[dof]] = angles[dof];
      }
    }
    if (reachAngles && reachWeight > 0) {
      REACH_QPOSADR.forEach((adr, i) => {
        data.qpos[adr] = data.qpos[adr] * (1 - reachWeight) + reachAngles[i] * reachWeight;
      });
    }
    mj.mj_forward(model, data);
    syncMeshes(group, data);
  }

  return { group, update, triggerReach: solveReach };
}
