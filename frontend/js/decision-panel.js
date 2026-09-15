// The original Phase 1 "decision tree" panel, rebuilt for Phase 3.2 to show
// the *real* current pipeline instead of a generic diagram: five boxes for
// the actual real-neuron populations driving steering (see backend/lif.py's
// group_activity_ema / read_groups), fed by the same WebSocket broadcast
// already driving the 3D brain panel above it. Brightness is a real spike
// rate, not a decorative animation — but the per-node reference scales below
// are a display-only calibration (these five populations have wildly
// different baseline/active rates — FC can reach ~0.1 spikes/neuron/step
// under goal injection, DNa*-steering barely reaches ~2e-3 — so each needs
// its own brightness scale to be visible at all; this affects only how
// bright a box draws, never any simulation dynamics).
const NODES = [
  { key: 'motion', label: 'Motion', ref: 0.001 },
  { key: 'fc', label: 'Goal (FC)', ref: 0.05 },
  { key: 'pfl', label: 'Compare (PFL)', ref: 0.0015 },
  { key: 'dna', label: 'Steer (DNa)', ref: 0.002 },
  { key: 'turn', label: 'Turn', ref: 1 },
];

const BOX_COLOR = [0.31, 0.71, 1.0];

export function createDecisionPanel(canvas) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  let groups = { motion: 0, fc: 0, pfl: 0, dna_left: 0, dna_right: 0 };
  let turn = 'straight';

  function update(newGroups, newTurn) {
    groups = newGroups;
    if (newTurn) turn = newTurn;
  }

  function drawArrow(x1, y, x2) {
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y);
    ctx.lineTo(x2, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y);
    ctx.lineTo(x2 - 6, y - 4);
    ctx.lineTo(x2 - 6, y + 4);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fill();
  }

  function drawBox(cx, cy, w, h, intensity, label, sublabel) {
    const [r, g, b] = BOX_COLOR;
    const alpha = 0.15 + intensity * 0.7;
    ctx.fillStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${alpha})`;
    ctx.strokeStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},0.6)`;
    ctx.lineWidth = 1;
    const x = cx - w / 2,
      y = cy - h / 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, cx, cy - (sublabel ? 7 : 0));
    if (sublabel) {
      ctx.font = '10px sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText(sublabel, cx, cy + 9);
    }
  }

  function render() {
    ctx.fillStyle = '#0c1018';
    ctx.fillRect(0, 0, width, height);

    const boxW = 56,
      boxH = 40;
    const n = NODES.length;
    const margin = boxW / 2 + 4;
    const spacing = (width - margin * 2) / (n - 1);
    const cy = height / 2;

    for (let i = 0; i < n - 1; i++) {
      const x1 = margin + i * spacing + boxW / 2;
      const x2 = margin + (i + 1) * spacing - boxW / 2;
      drawArrow(x1, cy, x2);
    }

    NODES.forEach((node, i) => {
      const cx = margin + i * spacing;
      if (node.key === 'dna') {
        const leftIntensity = Math.min(1, (groups.dna_left ?? 0) / node.ref);
        const rightIntensity = Math.min(1, (groups.dna_right ?? 0) / node.ref);
        drawBox(cx - 16, cy, boxW / 2 - 2, boxH, leftIntensity, 'L', '');
        drawBox(cx + 16, cy, boxW / 2 - 2, boxH, rightIntensity, 'R', '');
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(node.label, cx, cy + boxH / 2 + 12);
      } else if (node.key === 'turn') {
        drawBox(cx, cy, boxW, boxH, 0.5, node.label, turn.toUpperCase());
      } else {
        const intensity = Math.min(1, (groups[node.key] ?? 0) / node.ref);
        drawBox(cx, cy, boxW, boxH, intensity, node.label, '');
      }
    });
  }

  return { update, render };
}
