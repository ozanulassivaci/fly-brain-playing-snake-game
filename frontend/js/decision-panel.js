// The original Phase 1 panel (frontend/js/brain-viz.js, now replaced by the
// real 3D point cloud above this one) was a stylized node-link diagram: one
// column per real brain region, each column showing several real neuron
// type names as individually glowing nodes, wired together. That layout is
// rebuilt here — same per-region columns, same individually labeled real
// types — but every node's brightness now comes from a genuinely simulated
// spike rate (backend/lif.py's group_activity_ema, broadcast as "groups"
// alongside the 3D panel's spike data) instead of a decorative shimmer.
//
// Reference scales below are display-only calibration: these populations
// have wildly different baseline/active firing rates (FC/EPG can reach
// ~0.03-0.13 spikes/neuron/step under goal/heading injection; DNa*-steering
// barely reaches ~2e-3) so each needs its own brightness scale to be visible
// at all. This affects only how bright a node draws, never simulation
// dynamics.
const COLUMNS = [
  {
    title: 'Motion',
    nodes: [
      { key: 'motion_a', label: 'a', ref: 0.001 },
      { key: 'motion_b', label: 'b', ref: 0.001 },
      { key: 'motion_c', label: 'c', ref: 0.001 },
      { key: 'motion_d', label: 'd', ref: 0.001 },
    ],
  },
  {
    title: 'Central Cx',
    nodes: [
      { key: 'epg', label: 'EPG (heading)', ref: 0.05 },
      { key: 'fc', label: 'FC (goal)', ref: 0.05 },
      { key: 'pfl', label: 'PFL (compare)', ref: 0.0015 },
    ],
  },
  {
    title: 'Steering DN',
    nodes: [
      { key: 'dna_left', label: 'DNa L', ref: 0.002 },
      { key: 'dna_right', label: 'DNa R', ref: 0.002 },
    ],
  },
];

export function createDecisionPanel(canvas) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  let groups = {};
  let turn = 'straight';

  const layout = [];
  const colSpacing = width / (COLUMNS.length + 1);
  COLUMNS.forEach((col, ci) => {
    const cx = colSpacing * (ci + 1);
    const nodeCount = col.nodes.length;
    const topMargin = 34;
    const bottomMargin = 36;
    const usable = height - topMargin - bottomMargin;
    col.nodes.forEach((node, ni) => {
      const cy = topMargin + (nodeCount === 1 ? usable / 2 : (usable / (nodeCount - 1)) * ni);
      layout.push({ ...node, x: cx, y: cy, phase: Math.random() * Math.PI * 2 });
    });
  });

  function update(newGroups, newTurn) {
    groups = newGroups || {};
    if (newTurn) turn = newTurn;
  }

  function render(nowSec = 0) {
    ctx.fillStyle = '#0c1018';
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 1;
    for (let ci = 0; ci < COLUMNS.length - 1; ci++) {
      const from = layout.filter((n) => COLUMNS[ci].nodes.some((c) => c.key === n.key));
      const to = layout.filter((n) => COLUMNS[ci + 1].nodes.some((c) => c.key === n.key));
      from.forEach((a) => {
        to.forEach((b) => {
          const glow = Math.max((groups[a.key] ?? 0) / a.ref, (groups[b.key] ?? 0) / b.ref);
          ctx.strokeStyle = `rgba(140,160,190,${0.1 + Math.min(1, glow) * 0.2})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        });
      });
    }

    ctx.textAlign = 'center';
    COLUMNS.forEach((col, ci) => {
      ctx.fillStyle = '#8892a4';
      ctx.font = '10px sans-serif';
      ctx.fillText(col.title, colSpacing * (ci + 1), 14);
    });

    layout.forEach((node) => {
      const rate = groups[node.key] ?? 0;
      const intensity = Math.min(1, rate / node.ref);
      const shimmer = 0.05 * Math.sin(nowSec * 1.5 + node.phase);
      const [r, g, b] = [80, 180, 255];
      const radius = 4 + intensity * 6;

      ctx.beginPath();
      ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + Math.max(0, intensity + shimmer) * 0.7})`;
      ctx.shadowColor = `rgba(${r},${g},${b},${intensity})`;
      ctx.shadowBlur = 3 + intensity * 10;
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#c7cede';
      ctx.font = '9px sans-serif';
      ctx.fillText(node.label, node.x, node.y + radius + 10);
    });

    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(`turn: ${turn.toUpperCase()}`, width / 2, height - 6);
  }

  return { update, render };
}
