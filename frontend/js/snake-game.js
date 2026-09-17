const TICK_SECONDS = 0.15;
const RESTART_DELAY_SECONDS = 1.5;
const THREAT_RADIUS = 6; // cells the looming channel can see (see getThreat)
const ANTENNA_OFFSET = 0.6; // cells either side of the head (see getOdour)
// Brain-controlled only (Phase 3) — no keyboard input.
//
// Phase 3.12: the fly has a *body*. It carries a continuous heading, the
// descending-neuron signal is integrated into that heading as a turn
// velocity, and the snake moves along whichever cardinal direction the
// heading currently points nearest to. Nothing here plans a route or looks
// at where the apple is; all of that stays in the connectome. This is only
// the physics between a steering command and a grid.
//
// It replaces a rate limiter (at most one 90-degree turn per two ticks)
// that was the best of three policies measured at the time, but that was
// papering over a quantisation problem it could not fix. The brain would
// say "the apple is 30 degrees to your right" and the game executed a
// 90-degree turn, overshooting by 60; next tick it said "now 60 degrees to
// your left" and the game overshot back. That is the slalom, and at larger
// errors the same effect closes into an orbit. Measured over four
// configurations, the fly turned wrongly about as often as rightly
// whenever the apple was within 60 degrees of straight ahead (0.49-0.54
// correct) and went straight on 2-4% of ticks — because on a grid, with
// the apple 30 degrees off, *straight* is the correct move and the
// controller had no way to express it.
//
// Integrating a rate expresses it: a weak command rotates the heading a
// little and never crosses into the next cardinal, so the snake goes
// straight; a strong one crosses quickly. The turn rate saturates at 90
// degrees per tick, so a reversal can never happen in a single step.
//
// Measured against the rate-limited build, 40 episodes of 500 ticks each
// per configuration, over two independent noise seeds (the LIF noise is
// not seeded by default, and run-to-run spread on ten episodes turned out
// to be wider than most of the effects being chased — this is measured on
// twenty game seeds x two noise seeds, and the result repeats):
//
//     rate-limited 90-degree turns   83 apples, best 5 and 5
//     integrated heading            112 apples, best 8 and 8
//
// The old build never once passed 5 apples in a life across all 40
// episodes. Raising the obstacle gain on top of this was tried and made it
// worse (83, then 47, at 4x and 12x), so LPLC1 keeps its old scale.
// rad/s per unit of DN left-right difference. Cut from 850 in Phase 3.13,
// not as tuning but because the quantity being multiplied changed size:
// adding the premotor network (backend/scripts/prepare_subset.py) made the
// steering readout 2.7x stronger for the same visual drive, so the old gain
// over-rotated. Shipping the new subset with the old value would have been
// a regression, not a neutral change -- measured over 60 episodes:
//
//     gain 200   133 apples    gain 310   200 apples
//     gain 500   110 apples    gain 850    48 apples, 47 of 60 deaths by
//                                          self-collision (it spins)
const TURN_RATE_GAIN = 310.0;
const MAX_TURN_RATE = Math.PI / (2 * TICK_SECONDS); // 90 degrees per tick
const CARDINALS = [
  [1, 0],
  [0, 1],
  [-1, 0],
  [0, -1],
];

function nearestCardinal(heading) {
  const k = ((Math.round(heading / (Math.PI / 2)) % 4) + 4) % 4;
  return CARDINALS[k];
}

export function createSnakeGame({ cols = 20, rows = 20, cellSize = 24, onEat, onCollide, onTick } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d');

  let snake, dir, heading, currentTurnRate, apple, alive, tickAcc, restartAcc, score;
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
    heading = 0; // radians, matching dir — the fly's own facing, not the grid's
    currentTurnRate = 0;
    alive = true;
    tickAcc = 0;
    restartAcc = 0;
    score = 0;
    spawnApple();
  }

  // Classic Snake picks uniformly among free cells, and this did too. The
  // trouble is arithmetic rather than fairness: on a 20x20 board the two
  // outermost rings are 36% of the squares, so more than a third of apples
  // land where reaching one means flying at a wall and pulling away inside
  // a cell or two. Rejection sampling thins those rings to about 22% —
  // still reachable often enough to matter, no longer the common case.
  // This changes the game rather than the fly, deliberately and at the
  // user's request; nothing here touches how the apple is found.
  function spawnApple() {
    for (;;) {
      const x = Math.floor(Math.random() * cols);
      const y = Math.floor(Math.random() * rows);
      const margin = Math.min(x, y, cols - 1 - x, rows - 1 - y);
      if (Math.random() > Math.min(1, (margin + 1) / 3)) continue;
      if (snake.some((s) => s.x === x && s.y === y)) continue;
      apple = { x, y };
      return;
    }
  }

  // The signed descending-neuron difference, straight from the backend.
  function applyTurn(rate) {
    currentTurnRate = rate;
  }

  function step() {
    const rate = Math.max(-MAX_TURN_RATE, Math.min(MAX_TURN_RATE, TURN_RATE_GAIN * currentTurnRate));
    heading += rate * TICK_SECONDS;
    heading = Math.atan2(Math.sin(heading), Math.cos(heading));
    const next = nearestCardinal(heading);
    // A reversal is instant death against the neck, and the rate cap makes
    // one impossible in a single step anyway — but the heading can wrap
    // past two cardinals if a frame is ever dropped, so refuse it outright.
    if (next[0] !== -dir[0] || next[1] !== -dir[1]) dir = next;
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
    // onTick fires once per *game* step, after the board has been redrawn,
    // and is what drives sensory sampling. It used to run off its own 100ms
    // timer in main.js while the game stepped every 150ms: not harmonic, so
    // the phase between "what the fly last saw" and "when the game asks it
    // to move" drifted continuously, and a decision was made against data
    // anywhere from 0 to 100ms stale. Sampling here pins that lag to one
    // constant — and the sample sees the board the move just produced,
    // because draw() has already run.
    let stepped = false;
    if (alive) {
      tickAcc += dtSeconds;
      while (tickAcc >= TICK_SECONDS) {
        tickAcc -= TICK_SECONDS;
        step();
        stepped = true;
        if (!alive) break;
      }
    } else {
      restartAcc += dtSeconds;
      if (restartAcc >= RESTART_DELAY_SECONDS) reset();
    }
    draw();
    if (stepped && alive) onTick?.();
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
  // The fly's own continuous heading, which is what its brain should get —
  // not the quantised grid direction the body happens to be moving along.
  // The two can differ by up to 45 degrees, and that difference is exactly
  // the steering error the connectome needs to see in order to close it.
  function getHeadingAngle() {
    return heading;
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
