let currentUser = "";
let state = null;
let cellSize = 36.0;
let camX = 26.0, camY = 26.0;
let isPanning = false, panStartX = 0, panStartY = 0;
let chartInstance = null;

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  };
}

function clampCamera() {
  const boardPx = 18 * cellSize;
  const margin = 1.0 * cellSize;
  const minX = canvas.width - boardPx - margin;
  const maxX = margin;
  camX = minX > maxX ? (canvas.width - boardPx) / 2.0 : Math.max(minX, Math.min(maxX, camX));
  const minY = canvas.height - boardPx - margin;
  const maxY = margin;
  camY = minY > maxY ? (canvas.height - boardPx) / 2.0 : Math.max(minY, Math.min(maxY, camY));
}

canvas.addEventListener("contextmenu", e => e.preventDefault());

canvas.addEventListener("mousedown", e => {
  const pos = getCanvasCoords(e);
  if (e.button === 1 || e.button === 2) {
    isPanning = true;
    panStartX = pos.x;
    panStartY = pos.y;
  } else if (e.button === 0) {
    const c = Math.floor((pos.x - camX) / cellSize);
    const r = Math.floor((pos.y - camY) / cellSize);
    if (r >= 0 && r < 18 && c >= 0 && c < 18) {
      sendAct("click_tile", r, c);
    }
  }
});

window.addEventListener("mouseup", () => {
  isPanning = false;
});

canvas.addEventListener("mousemove", e => {
  if (!isPanning) return;
  const pos = getCanvasCoords(e);
  camX += (pos.x - panStartX);
  camY += (pos.y - panStartY);
  panStartX = pos.x;
  panStartY = pos.y;
  clampCamera();
  drawBoard();
});

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const pos = getCanvasCoords(e);
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  const oldSize = cellSize;
  cellSize = Math.max(24.0, Math.min(92.0, cellSize * factor));
  const ratio = cellSize / oldSize;
  camX = pos.x - (pos.x - camX) * ratio;
  camY = pos.y - (pos.y - camY) * ratio;
  clampCamera();
  drawBoard();
});

async function switchProfile() {
  currentUser = document.getElementById("username-inp").value.trim() || "Commander";
  await fetchState();
}

async function fetchState() {
  try {
    const res = await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action_type: "get" })
    });
    if (!res.ok) throw new Error("HTTP error " + res.status);
    state = await res.json();
    updateUI();
  } catch (err) {
    console.error("Помилка завантаження стану:", err);
  }
}

async function sendAct(action_type, r = null, c = null, param = null) {
  try {
    const res = await fetch("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action_type, r, c, param })
    });
    if (!res.ok) throw new Error("HTTP error " + res.status);
    state = await res.json();
    updateUI();
  } catch (err) {
    console.error("Помилка виконання дії:", err);
  }
}

function updateUI() {
  if (!state) return;

  document.getElementById("telemetry-text").textContent =
    `Гравець: ${state.username} | Партія: #${state.matches_played + 1} | Хід: ${state.turn}/${state.max_turns}\n` +
    `Вінрейт профілю: ${state.winrate.toFixed(2)}% | Моделей у пулі: ${state.pool_size} (Макс: ${state.pool_max_rating.toFixed(2)})\n` +
    `----------------------------------------------------\n` +
    `СИНІ (ВИ) : Казна ${state.p2.gold.toFixed(2)}$ | Землі: ${state.p2.tiles} (+${state.p2.inc.toFixed(2)}$)\n` +
    `            Армія: ${state.p2.army} (Утрим: -${state.p2.upk.toFixed(2)}$) | Баланс: ${state.p2.net >= 0 ? '+' : ''}${state.p2.net.toFixed(2)}$/х\n` +
    `ЧЕРВОНІ ШІ: Казна ${state.p1.gold.toFixed(2)}$ | Землі: ${state.p1.tiles} (+${state.p1.inc.toFixed(2)}$)\n` +
    `            Армія: ${state.p1.army} (Утрим: -${state.p1.upk.toFixed(2)}$) | Баланс: ${state.p1.net >= 0 ? '+' : ''}${state.p1.net.toFixed(2)}$/х`;

  document.getElementById("gen-live-badge").textContent =
    `Зараз грає: ${state.active_gen_label} | Рейтинг: ${state.active_rating.toFixed(2)} (Базовий: ${state.base_rating.toFixed(2)})`;

  const evoEl = document.getElementById("evo-banner");
  let logLines = [];
  if (state.event_log && state.event_log.length > 0) {
    logLines = state.event_log.slice(-4);
  }
  if (state.last_evolution_report) {
    const rep = state.last_evolution_report;
    if (rep.status === "training") {
      logLines.push(`[Еволюція] Навчання нового покоління (${rep.samples} зразків)...`);
    } else if (rep.status === "completed") {
      logLines.push(`[Пул] Додано ${rep.new_gen_label} | Рейтинг: ${rep.new_gen_rating.toFixed(2)}`);
    }
  }
  evoEl.textContent = logLines.join("\n");

  document.getElementById("inspector-text").textContent = state.inspector.lines.join("\n");
  document.getElementById("b-heal").disabled = !state.inspector.can_heal;
  document.getElementById("b-harv").disabled = !state.inspector.can_harvest;
  document.getElementById("b-tow").disabled = !state.inspector.can_build_tower;
  document.getElementById("b-bar").disabled = !state.inspector.can_build_barracks;
  document.getElementById("b-swk").disabled = !state.inspector.can_spawn_worker;
  document.getElementById("b-swr").disabled = !state.inspector.can_spawn_warrior;
  document.getElementById("b-sar").disabled = !state.inspector.can_spawn_archer;

  drawBoard();
}

function drawUnitIcon(cx, cy, rad, uType, faction) {
  ctx.save();
  ctx.strokeStyle = faction === 1 ? "#ffffff" : "#141b24";
  ctx.fillStyle = faction === 1 ? "#feca57" : "#ffffff";
  ctx.lineWidth = Math.max(2, rad * 0.16);

  if (uType === "worker") {
    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.4, cy + rad * 0.4);
    ctx.lineTo(cx + rad * 0.35, cy - rad * 0.35);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.15, cy - rad * 0.48);
    ctx.quadraticCurveTo(cx + rad * 0.3, cy - rad * 0.3, cx + rad * 0.48, cy + rad * 0.15);
    ctx.stroke();
  } else if (uType === "warrior") {
    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.42, cy + rad * 0.42);
    ctx.lineTo(cx + rad * 0.48, cy - rad * 0.48);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.38, cy);
    ctx.lineTo(cx, cy + rad * 0.38);
    ctx.stroke();
  } else {
    ctx.beginPath();
    ctx.arc(cx - rad * 0.1, cy, rad * 0.48, -Math.PI * 0.4, Math.PI * 0.4);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.45, cy);
    ctx.lineTo(cx + rad * 0.5, cy);
    ctx.stroke();
  }
  ctx.restore();
}

function drawBoard() {
  if (!state) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cs = cellSize, ox = camX, oy = camY, boardPx = 18 * cs;

  ctx.strokeStyle = "#2c3e50";
  ctx.lineWidth = 2;
  ctx.strokeRect(ox - cs, oy - cs, boardPx + 2 * cs, boardPx + 2 * cs);
  ctx.strokeStyle = "#feca57";
  ctx.strokeRect(ox - 2, oy - 2, boardPx + 4, boardPx + 4);

  for (let r = 0; r < 18; r++) {
    for (let c = 0; c < 18; c++) {
      const x = ox + c * cs, y = oy + r * cs;
      const tType = state.terrain[r][c], owner = state.territory[r][c];
      ctx.fillStyle = tType === 0 ? "#3d3d3d" : (owner === 1 ? "#632227" : (owner === 2 ? "#1b3b6f" : "#2b343b"));
      ctx.fillRect(x, y, cs, cs);
      ctx.strokeStyle = "#4b6584";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cs, cs);

      if (tType === 0) {
        ctx.fillStyle = "#718093";
        ctx.beginPath();
        ctx.moveTo(x + cs * 0.18, y + cs * 0.84);
        ctx.lineTo(x + cs * 0.5, y + cs * 0.18);
        ctx.lineTo(x + cs * 0.82, y + cs * 0.84);
        ctx.closePath();
        ctx.fill();
      } else {
        const val = state.eff_yields[r][c];
        if (val > 0 && cs >= 30) {
          ctx.fillStyle = val >= 0.30 ? "#feca57" : (val >= 0.12 ? "#1dd1a1" : "#8395a7");
          ctx.font = `bold ${Math.max(8, Math.floor(cs * 0.20))}px Arial`;
          ctx.fillText(val.toFixed(2), x + 3, y + cs * 0.26);
        }
      }
    }
  }

  for (const st of state.structs) {
    const x = ox + st.c * cs, y = oy + st.r * cs;
    ctx.fillStyle = st.owner === 1 ? "#c23616" : "#0097e6";
    ctx.fillRect(x + cs * 0.14, y + cs * 0.22, cs * 0.72, cs * 0.66);
    ctx.strokeStyle = "#feca57";
    ctx.strokeRect(x + cs * 0.14, y + cs * 0.22, cs * 0.72, cs * 0.66);
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.max(8, Math.floor(cs * 0.19))}px Arial`;
    ctx.fillText((st.type === 1 ? "ВЖ " : "КЗ ") + st.hp.toFixed(0), x + cs * 0.20, y + cs * 0.60);
  }

  const castles = [
    { r: 17, c: 0, hp: state.p1.castle_hp, col: "#83142c", lbl: "ЗМ ШІ" },
    { r: 0, c: 17, hp: state.p2.castle_hp, col: "#0a3d62", lbl: "ВАШ ЗМ" }
  ];
  for (const csObj of castles) {
    const x = ox + csObj.c * cs, y = oy + csObj.r * cs;
    ctx.fillStyle = csObj.col;
    ctx.fillRect(x + 2, y + 4, cs - 4, cs - 6);
    ctx.strokeStyle = "#feca57";
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 2, y + 4, cs - 4, cs - 6);
    ctx.fillStyle = "#feca57";
    ctx.font = `bold ${Math.max(8, Math.floor(cs * 0.18))}px Arial`;
    ctx.fillText(`${csObj.lbl} ${csObj.hp.toFixed(0)}`, x + 4, y + cs * 0.75);
  }

  // Візуалізація траєкторій останнього ходу ШІ (пунктирні лінії)
  if (state.ai_traces) {
    for (const tr of state.ai_traces) {
      const x1 = ox + tr.from[1] * cs + cs * 0.5;
      const y1 = oy + tr.from[0] * cs + cs * 0.5;
      const x2 = ox + tr.to[1] * cs + cs * 0.5;
      const y2 = oy + tr.to[0] * cs + cs * 0.5;
      ctx.save();
      ctx.strokeStyle = tr.kind === "attack" ? "#ff4757" : "#ff9f43";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(x1, y1, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // Жовті клітинки можливого ходу обраного юніта
  for (const [mr, mc] of state.valid_moves) {
    const x = ox + mc * cs, y = oy + mr * cs;
    ctx.strokeStyle = "#f1c40f";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
    ctx.fillStyle = "#f1c40f";
    ctx.beginPath();
    ctx.arc(x + cs * 0.5, y + cs * 0.5, Math.max(3, cs * 0.1), 0, Math.PI * 2);
    ctx.fill();
  }

  // Фіолетові клітинки цілей для атаки
  for (const [tr, tc] of state.valid_attacks) {
    const x = ox + tc * cs, y = oy + tr * cs;
    ctx.strokeStyle = "#e056fd";
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
  }

  // Малювання юнітів
  for (const u of state.units) {
    const cx = ox + u.c * cs + cs * 0.5, cy = oy + u.r * cs + cs * 0.48, rad = cs * 0.34;
    ctx.fillStyle = u.faction === 1 ? "#eb3b5a" : "#0fbcf9";
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = (u.faction === 2 && u.ready) ? "#2ecc71" : "#dcdde1";
    ctx.lineWidth = 2;
    ctx.stroke();
    drawUnitIcon(cx, cy, rad, u.type, u.faction);

    const bw = cs * 0.72, bh = Math.max(3, cs * 0.09);
    ctx.fillStyle = "#1e272e";
    ctx.fillRect(cx - bw / 2, cy + rad + 1, bw, bh);
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(cx - bw / 2, cy + rad + 1, bw * (u.hp / u.max_hp), bh);
  }

  // Рамка обраної клітинки
  if (state.selected_pos) {
    const [sr, sc] = state.selected_pos;
    ctx.strokeStyle = "#00d2d3";
    ctx.lineWidth = 3;
    ctx.strokeRect(ox + sc * cs + 2, oy + sr * cs + 2, cs - 4, cs - 4);
  }
}

function openStatsModal() {
  if (!state || !state.stats) return;
  document.getElementById("modal-overlay").style.display = "flex";
  const st = state.stats;
  document.getElementById("stats-summary").textContent =
    `Поточна ефективність Вашої Фракції: ${st.player_current.toFixed(2)} (Медіана: ${st.player_median.toFixed(2)}) | ` +
    `Поточна ефективність ШІ: ${st.ai_current.toFixed(2)} (Медіана: ${st.ai_median.toFixed(2)})`;

  const pMedLine = st.turns.map(() => st.player_median);
  const aiMedLine = st.turns.map(() => st.ai_median);

  if (chartInstance) chartInstance.destroy();
  const cctx = document.getElementById("effChart").getContext("2d");
  chartInstance = new Chart(cctx, {
    type: "line",
    data: {
      labels: st.turns.map(t => `Хід ${t}`),
      datasets: [
        { label: "Ефективність Гравця (Сині)", data: st.player_eff, borderColor: "#0fbcf9", borderWidth: 2.5, tension: 0.2 },
        { label: `Медіана Гравця (${st.player_median.toFixed(2)})`, data: pMedLine, borderColor: "#0fbcf9", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
        { label: "Ефективність Нейромережі (Червоні)", data: st.ai_eff, borderColor: "#eb3b5a", borderWidth: 2.5, tension: 0.2 },
        { label: `Медіана ШІ (${st.ai_median.toFixed(2)})`, data: aiMedLine, borderColor: "#eb3b5a", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#c8d6e5" } },
        y: { ticks: { color: "#c8d6e5" } }
      }
    }
  });
}

function closeStatsModal() {
  document.getElementById("modal-overlay").style.display = "none";
}

async function initApp() {
  try {
    const res = await fetch("/api/init_user");
    if (!res.ok) throw new Error("HTTP error " + res.status);
    const data = await res.json();
    
    // Автоматично присвоєний або збережений нікнейм для цього IP
    currentUser = data.username;
    
    // Оновлюємо поле вводу
    document.getElementById("username-inp").value = currentUser;
    
    // Завантажуємо ігрову дошку
    await fetchState();
  } catch (err) {
    console.error("Помилка ініціалізації гостя:", err);
  }
}

clampCamera();
initApp();

setInterval(() => {
  if (state && state.is_training_new_gen) {
    fetchState();
  }
}, 1500);