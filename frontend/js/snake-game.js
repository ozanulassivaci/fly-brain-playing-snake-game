const TICK_SECONDS = 0.15;
const RESTART_DELAY_SECONDS = 1.5;
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

  reset();
  function getScore() {
    return { score, bestScore, alive };
  }

  return { canvas, update, getDirection, applyTurn, getHeadingAngle, getGoalAngle, getScore };
}
