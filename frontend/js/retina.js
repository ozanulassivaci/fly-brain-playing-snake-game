// A minimal Hassenstein-Reichardt correlator — the textbook elementary
// motion detector model that real T4/T5 research is built on — not a
// learned/deep vision model. Downsamples the Snake canvas to a small grid,
// compares it to the previous grid, and reports four scalar motion-energy
// values (a/b/c/d) matching the four T4/T5 direction-tuned subtypes.
//
// Honest limitation (see docs/architecture-plan.md): this only detects
// *motion*, exactly like the real neurons it targets. It cannot perceive
// the stationary apple — only the snake's own moving body/head.

const GRID_SIZE = 8;

export function createRetina() {
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = GRID_SIZE;
  sampleCanvas.height = GRID_SIZE;
  const ctx = sampleCanvas.getContext('2d', { willReadFrequently: true });

  let previous = null;

  function sampleMotion(sourceCanvas) {
    ctx.drawImage(sourceCanvas, 0, 0, GRID_SIZE, GRID_SIZE);
    const { data } = ctx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const current = new Float32Array(GRID_SIZE * GRID_SIZE);
    for (let i = 0; i < current.length; i++) {
      const r = data[i * 4],
        g = data[i * 4 + 1],
        b = data[i * 4 + 2];
      current[i] = (r + g + b) / (3 * 255);
    }

    if (!previous) {
      previous = current;
      return { a: 0, b: 0, c: 0, d: 0 };
    }

    let right = 0,
      left = 0,
      down = 0,
      up = 0;
    for (let y = 1; y < GRID_SIZE - 1; y++) {
      for (let x = 1; x < GRID_SIZE - 1; x++) {
        const idx = y * GRID_SIZE + x;
        right += current[idx] * previous[idx - 1];
        left += current[idx - 1] * previous[idx];
        down += current[idx] * previous[idx - GRID_SIZE];
        up += current[idx - GRID_SIZE] * previous[idx];
      }
    }

    previous = current;
    // a/b/c/d <-> {right, left, down, up}: an arbitrary but fixed mapping
    // (the exact real a/b/c/d<->cardinal-direction correspondence for this
    // dataset isn't independently verified — see docs/architecture-plan.md).
    return { a: right, b: left, c: down, d: up };
  }

  return { sampleMotion };
}
