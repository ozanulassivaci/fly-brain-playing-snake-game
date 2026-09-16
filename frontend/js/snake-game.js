const TICK_SECONDS = 0.15;
const RESTART_DELAY_SECONDS = 1.5;
const THREAT_RADIUS = 6; // cells the looming channel can see (see getThreat)
const ANTENNA_OFFSET = 0.6; // cells either side of the head (see getOdour)
// Brain-controlled only (Phase 3) — no keyboard input. Earlier attempts to
// fix a spinning-in-circles bug worked around it in this game layer
// (edge-triggering, then a fixed turn cooldown) instead of fixing the real
// cause: backend/lif.py's motor hysteresis held a "left"/"right" decision
// for a long time once triggered (measured: median ~5.3 game ticks, tail
// out to ~41 ticks / ~6 real seconds, at the old TURN_ON/OFF_THRESH gap),
// long enough that even a capped cooldown still produced many repeated
// same-direction turns during one hold — still visibly circling — or, at
// the other extreme, one turn followed by many seconds straight into a
// wall. Fixed at the source instead: TURN_OFF_THRESH now equals
// TURN_ON_THRESH (no hysteresis band), which measured out to a median
// hold of ~1.6 ticks and a max of ~6.8 ticks — short enough that plain
// once-per-tick turning below doesn't need a workaround.
function rotateLeft([dx, dy]) {
  return [dy, -dx];
}
function rotateRight([dx, dy]) {
  return [-dy, dx];
}

export function createSnakeGame({ cols = 20, rows = 20, cellSize = 24, onEat, onCollide } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d');

  let snake, dir, currentMotorTurn, apple, alive, tickAcc, restartAcc, score;
  // Survives reset() so a death doesn't erase what the fly has managed —
  // the run-to-run record is the number worth watching.
  let bestScore = 0;

  function reset() {
    snake = [
      { x: Math.floor(cols / 2), y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 1, y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 2, y: Math.floor(rows / 2) },
    ];
    dir = [1, 0];
    currentMotorTurn = 'straight';
    alive = true;
    tickAcc = 0;
    restartAcc = 0;
    score = 0;
    spawnApple();
  }

  function spawnApple() {
    do {
      apple = { x: Math.floor(Math.random() * cols), y: Math.floor(Math.random() * rows) };
    } while (snake.some((s) => s.x === apple.x && s.y === apple.y));
  }

  function applyTurn(turn) {
    currentMotorTurn = turn;
  }

  function step() {
    if (currentMotorTurn === 'left') dir = rotateLeft(dir);
    else if (currentMotorTurn === 'right') dir = rotateRight(dir);
    const head = { x: snake[0].x + dir[0], y: snake[0].y + dir[1] };

    const hitWall = head.x < 0 || head.x >= cols || head.y < 0 || head.y >= rows;
    const hitSelf = snake.some((s) => s.x === head.x && s.y === head.y);
    if (hitWall || hitSelf) {
      alive = false;
      onCollide?.();
      return;
    }

    snake.unshift(head);
    if (head.x === apple.x && head.y === apple.y) {
      score++;
      bestScore = Math.max(bestScore, score);
      onEat?.();
      spawnApple();
    } else {
      snake.pop();
    }
  }

  function draw() {
    ctx.fillStyle = '#0a1410';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#ff5555';
    ctx.fillRect(apple.x * cellSize + 2, apple.y * cellSize + 2, cellSize - 4, cellSize - 4);

    snake.forEach((s, i) => {
      ctx.fillStyle = i === 0 ? '#7CFC9A' : '#3fae63';
      ctx.fillRect(s.x * cellSize + 1, s.y * cellSize + 1, cellSize - 2, cellSize - 2);
    });

    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.font = `${Math.floor(cellSize * 0.7)}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText(`apples ${score}   best ${bestScore}`, 6, cellSize);

    if (!alive) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fff';
      ctx.font = `${Math.floor(cellSize * 0.9)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('game over', canvas.width / 2, canvas.height / 2);
    }
  }

  function update(dtSeconds) {
    if (alive) {
      tickAcc += dtSeconds;
      while (tickAcc >= TICK_SECONDS) {
        tickAcc -= TICK_SECONDS;
        step();
        if (!alive) break;
      }
    } else {
      restartAcc += dtSeconds;
      if (restartAcc >= RESTART_DELAY_SECONDS) reset();
    }
    draw();
  }

  function getDirection() {
    return dir;
  }

  // Phase 3.2b: both angles are allocentric (measured against the same
  // fixed grid axis), not relative to each other — the backend's real EPG
  // (heading) and FC (goal) neurons each get their own real ring/column
  // injection, and the actual PFL synapses in the connectome compute the
  // heading-vs-goal comparison, the way the real circuit does. Phase 3.1
  // pre-computed a relative bearing in JS and only ever fed the brain "half"
  // of that comparison (goal, no heading) — see docs/architecture-plan.md.
  function getHeadingAngle() {
    return Math.atan2(dir[1], dir[0]);
  }

  function getGoalAngle() {
    const head = snake[0];
    return Math.atan2(apple.y - head.y, apple.x - head.x);
  }

  // Obstacle proximity in the left and right halves of the visual field —
  // the sensory quantity a real fly's looming pathway (LPLC1) reports,
  // computed here rather than seen, because retina.js downsamples a 20x20
  // board to 8x8 and a one-cell body segment simply does not survive that.
  // Same arrangement as the apple bearing: the geometry is measured
  // honestly here, the *decision* stays with the real LPLC1 -> DN synapses
  // in the connectome (see backend/lif.py's inject_obstacle). Segment 1 is
  // skipped because the neck always trails the head and is never avoidable.
  function getThreat() {
    const head = snake[0];
    const [hx, hy] = dir;
    const rx = -hy, // egocentric "right" = rotateRight(dir)
      ry = hx;
    let left = 0,
      right = 0;

    const consider = (ox, oy) => {
      const dx = ox - head.x,
        dy = oy - head.y;
      const d2 = dx * dx + dy * dy;
      if (d2 === 0 || d2 > THREAT_RADIUS * THREAT_RADIUS) return;
      const d = Math.sqrt(d2);
      const forward = (dx * hx + dy * hy) / d; // -1 behind .. +1 ahead
      const lateral = (dx * rx + dy * ry) / d; // -1 left .. +1 right
      // Weighted by where the body could actually end up: straight ahead is
      // the worst (it gets hit by doing nothing), the sides matter because
      // one 90-degree turn reaches them, and directly behind is
      // unreachable, so it counts for nothing. Restricting this to the
      // frontal sector was the first thing tried and it missed the fatal
      // case entirely: after one turn the snake's own body sits *beside*
      // the head, invisible to a forward-only field, and the next turn
      // drives straight into it. Real lobula columnar cells see nearly
      // panoramically, so the broad field is also the more faithful one.
      // 1/d, not 1/d^2: with the steeper falloff only an already-adjacent
      // cell registered at all (measured in a logged death — the threat
      // sat at 0.06 for seven ticks and then jumped to 0.56 one tick
      // before the collision, far too late for a 150ms tick plus the motor
      // EMA and turn threshold to act on). Real looming responses build up
      // over the whole approach rather than firing at the last instant.
      const reachable = (1 + forward) / 2;
      const weight = reachable / d;
      right += weight * (0.5 + lateral / 2);
      left += weight * (0.5 - lateral / 2);
    };

    for (let i = 2; i < snake.length; i++) consider(snake[i].x, snake[i].y);
    for (let k = -THREAT_RADIUS; k <= THREAT_RADIUS; k++) {
      consider(-1, head.y + k);
      consider(cols, head.y + k);
      consider(head.x + k, -1);
      consider(head.x + k, rows);
    }
    return { left, right };
  }

  // Bilateral odour concentration at two "antennae" offset either side of
  // the head — the cue a real fly actually uses to find food, and the one
  // channel here that works in every direction, including behind, where
  // the frontal visual pathway is blind and a freshly respawned apple
  // often is. Inverse-square falloff from a point source, which is the
  // standard idealisation of a still-air odour field.
  function getOdour() {
    const head = snake[0];
    const [hx, hy] = dir;
    const rx = -hy,
      ry = hx;
    const conc = (ax, ay) => {
      const dx = apple.x - ax,
        dy = apple.y - ay;
      return 1 / (1 + dx * dx + dy * dy);
    };
    return {
      left: conc(head.x - rx * ANTENNA_OFFSET, head.y - ry * ANTENNA_OFFSET),
      right: conc(head.x + rx * ANTENNA_OFFSET, head.y + ry * ANTENNA_OFFSET),
    };
  }

  reset();
  function getScore() {
    return { score, bestScore, alive };
  }

  return {
    canvas,
    update,
    getDirection,
    applyTurn,
    getHeadingAngle,
    getGoalAngle,
    getThreat,
    getOdour,
    getScore,
  };
}
