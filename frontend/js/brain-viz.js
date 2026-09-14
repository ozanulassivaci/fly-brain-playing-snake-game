// A stylized, hand-laid-out node-link diagram — NOT a 3D anatomical
// reconstruction and NOT driven by real neural activity. That's Phase 2/3
// (see /docs/architecture-plan.md). The group labels and the handful of type
// names shown here are real (drawn from the Phase 0 MaleCNS v1.0 data
// exploration), used only for authenticity; positions and pulses are
// decorative, synced to game events rather than simulated.

const CLUSTERS = [
  { key: 'motion', title: 'Motion pathway', color: [80, 180, 255], types: ['T4', 'T5', 'LC11', 'LPLC2'] },
  { key: 'cx', title: 'Central complex', color: [255, 200, 80], types: ['EPG', 'PEN', 'ER4d', 'hDeltaB'] },
  { key: 'dn', title: 'Descending neurons', color: [255, 100, 130], types: ['DNa01', 'DNa02', 'DNp01', 'DNp03'] },
];

const PULSE_TARGETS = {
  move: ['motion'],
  eat: ['cx', 'dn'],
  collide: ['dn'],
};

export function createBrainViz(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width,
    h = canvas.height;

  const nodes = [];
  const colSpacing = w / (CLUSTERS.length + 1);
  CLUSTERS.forEach((cluster, ci) => {
    const cx = colSpacing * (ci + 1);
    cluster.types.forEach((label, ti) => {
      const cy = (h / (cluster.types.length + 1)) * (ti + 1);
      nodes.push({
        cluster: cluster.key,
        color: cluster.color,
        label,
        x: cx + (Math.random() - 0.5) * 18,
        y: cy,
        phase: Math.random() * Math.PI * 2,
        pulse: 0,
      });
    });
  });

  const edges = [];
  for (let ci = 0; ci < CLUSTERS.length - 1; ci++) {
    const from = nodes.filter((n) => n.cluster === CLUSTERS[ci].key);
    const to = nodes.filter((n) => n.cluster === CLUSTERS[ci + 1].key);
    from.forEach((a) => {
      to.forEach((b) => {
        if (Math.random() < 0.35) edges.push([a, b]);
      });
    });
  }

  function pulse(kind) {
    const targets = PULSE_TARGETS[kind];
    if (!targets) return;
    nodes.forEach((n) => {
      if (targets.includes(n.cluster)) n.pulse = 1;
    });
  }

  function render(nowSec) {
    ctx.fillStyle = '#0c1018';
    ctx.fillRect(0, 0, w, h);

    ctx.lineWidth = 1;
    edges.forEach(([a, b]) => {
      const glow = Math.max(a.pulse, b.pulse);
      ctx.strokeStyle = `rgba(140,160,190,${0.12 + glow * 0.3})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    ctx.textAlign = 'center';
    CLUSTERS.forEach((cluster, ci) => {
      ctx.fillStyle = '#8892a4';
      ctx.font = '11px sans-serif';
      ctx.fillText(cluster.title, colSpacing * (ci + 1), 16);
    });

    nodes.forEach((n) => {
      n.pulse = Math.max(0, n.pulse - 0.02);
      const shimmer = 0.15 + 0.1 * Math.sin(nowSec * 1.5 + n.phase);
      const intensity = Math.min(1, shimmer + n.pulse);
      const [r, g, b] = n.color;
      const radius = 5 + n.pulse * 4;

      ctx.beginPath();
      ctx.fillStyle = `rgba(${r},${g},${b},${0.35 + intensity * 0.65})`;
      ctx.shadowColor = `rgba(${r},${g},${b},${intensity})`;
      ctx.shadowBlur = 4 + n.pulse * 12;
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#c7cede';
      ctx.font = '10px sans-serif';
      ctx.fillText(n.label, n.x, n.y + 16);
    });
  }

  return { render, pulse };
}
