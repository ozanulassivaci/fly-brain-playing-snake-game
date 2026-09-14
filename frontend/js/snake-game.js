const TICK_SECONDS = 0.15;
const RESTART_DELAY_SECONDS = 1.5;

const DIRECTIONS = {
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  w: [0, -1],
  s: [0, 1],
  a: [-1, 0],
  d: [1, 0],
};

export function createSnakeGame({ cols = 20, rows = 20, cellSize = 24, onMove, onEat, onCollide } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d');

  let snake, dir, nextDir, apple, alive, tickAcc, restartAcc;

  function reset() {
    snake = [
      { x: Math.floor(cols / 2), y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 1, y: Math.floor(rows / 2) },
      { x: Math.floor(cols / 2) - 2, y: Math.floor(rows / 2) },
    ];
    dir = [1, 0];
    nextDir = [1, 0];
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

  window.addEventListener('keydown', (e) => {
    const d = DIRECTIONS[e.key];
    if (!d) return;
    if (d[0] === -dir[0] && d[1] === -dir[1]) return; // no reversing into self
    nextDir = d;
  });

  function step() {
    dir = nextDir;
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

  reset();
  return { canvas, update };
}
