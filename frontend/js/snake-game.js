const TICK_SECONDS = 0.15;
const RESTART_DELAY_SECONDS = 1.5;
// Minimum game ticks between two executed turns. The backend's own
// hysteresis (see backend/lif.py's _update_motor) holds a "left"/"right"
// decision for a while once triggered (measured: anywhere from ~10 to
// ~200+ broadcasts, i.e. up to several real seconds) rather than flipping
// every broadcast. Turning once per *tick* the decision holds spins the
// snake in tight circles (measured and fixed); turning only once per
// straight->left/right *transition* and then going straight until the next
// one was tried next and made the snake travel straight for long stretches,
// often into a wall, before the next transition ever came (this project's
// 20x20 grid is smaller than a multi-second hold period at
// TICK_SECONDS=0.15). This cooldown re-executes a turn every
// MIN_TURN_TICKS ticks for as long as the decision holds, instead of never
// again (too rare) or every tick (spins) — a reasonable middle ground for
// avoiding both known-bad extremes, not a value with a measured effect on
// apple-eating success: repeated multi-trial testing (see
// docs/architecture-plan.md's Phase 3.3 section) found no configuration of
// this project's circuit, cadence included, that reliably beat having no
// goal information at all — results replicated in the opposite direction
// as often as not.
const MIN_TURN_TICKS = 2;
function rotateLeft([dx, dy]) {
  return [dy, -dx];
}
function rotateRight([dx, dy]) {
  return [-dy, dx];
}

export function createSnakeGame({ cols = 20, rows = 20, cellSize = 24, onMove, onEat, onCollide } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d');

  let snake, dir, pendingMotorTurn, ticksSinceTurn, apple, alive, tickAcc, restartAcc;

  function reset() {
    snake = [
      { x: Math.floor(cols / 2), y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 1, y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 2, y: Math.floor(rows / 2) },
    ];
    dir = [1, 0];
    pendingMotorTurn = null;
    ticksSinceTurn = MIN_TURN_TICKS;
    alive = true;
    tickAcc = 0;
    restartAcc = 0;
    spawnApple();
  }

  function spawnApple() {
    do {
      apple = { x: Math.floor(Math.random() * cols), y: Math.floor(Math.random() * rows) };
    } while (snake.some((s) => s.x === apple.x && s.y === apple.y));
  }

  function applyTurn(turn) {
    pendingMotorTurn = turn !== 'straight' ? turn : null;
  }

  function step() {
    ticksSinceTurn++;
    if (pendingMotorTurn && ticksSinceTurn >= MIN_TURN_TICKS) {
      dir = pendingMotorTurn === 'left' ? rotateLeft(dir) : rotateRight(dir);
      ticksSinceTurn = 0;
    }
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
      onEat?.();
      spawnApple();
    } else {
      snake.pop();
    }
    onMove?.();
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
  return { canvas, update, getDirection, applyTurn, getHeadingAngle, getGoalAngle };
}
