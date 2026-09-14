import { buildMeshes, syncMeshes } from './scene-loader.js';

// Order must match model_meta.json's preprogrammed.legs keys and
// ctrl_index_by_leg_dof's row order (both generated together from the same
// NeuroMechFly asset pipeline).
const LEG_NAMES = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];

const CYCLES_PER_SECOND = 0.7;

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

  function update(nowSec) {
    const frame = Math.floor(nowSec * CYCLES_PER_SECOND * nSamples) % nSamples;
    for (let leg = 0; leg < LEG_NAMES.length; leg++) {
      const angles = legAngles[leg][frame];
      const qposadr = legDofQposadr[leg];
      for (let dof = 0; dof < qposadr.length; dof++) {
        data.qpos[qposadr[dof]] = angles[dof];
      }
    }
    mj.mj_forward(model, data);
    syncMeshes(group, data);
  }

  return { group, update };
}
