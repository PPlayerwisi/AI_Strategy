let currentUser = "";
let currentLang = "en";
let state = null;
let cellSize = 36.0;
let camX = 26.0, camY = 26.0;
let isPanning = false, panStartX = 0, panStartY = 0;
let chartInstance = null;

const UI_LOC = {
  en: {
    boxProfile: "USER PROFILE",
    placeholderUser: "Username...",
    btnLogin: "Login",
    btnLangToggle: "UK",
    boxTelemetry: "SESSION TELEMETRY AND AI STATUS",
    boxInspector: "OBJECT INSPECTOR",
    boxOrders: "UNIT ORDERS AND CONSTRUCTION",
    btnHeal: "Heal Unit (+25.00% max HP)",
    btnHarv: "Extra Tile Harvest (+1.50x)",
    btnTow: "Build Defense Tower (4.00$)",
    btnBar: "Build Forward Barracks (5.00$)",
    btnDem: "Demolish Allied Structure (+50.00% refund)",
    btnDis: "Disband Selected Unit (+40.00% refund)",
    btnSwk: "Recruit Worker (3.00$ | Upkeep: 0.40$)",
    btnSwr: "Recruit Warrior (6.00$ | Upkeep: 1.10$)",
    btnSar: "Recruit Archer (8.50$ | Upkeep: 1.60$)",
    btnStats: "Efficiency Statistics (Chart and Median)",
    btnEnd: "End Turn (Auto-Heal and AI Turn)",
    modalTitle: "Efficiency Growth Dynamics and Median from Match Start",
    btnClose: "Close",
    playerLabel: "Player",
    matchLabel: "Match",
    turnLabel: "Turn",
    winrateLabel: "Profile Winrate",
    poolModelsLabel: "Pool Models",
    maxLabel: "Max",
    blueLabel: "BLUE (YOU)",
    redLabel: "RED (AI)  ",
    goldLabel: "Gold",
    landLabel: "Land",
    armyLabel: "Army",
    upkeepLabel: "Upkeep",
    netLabel: "Net",
    perTurn: "/t",
    activeGenPrefix: "Active AI",
    ratingLabel: "Rating",
    baseLabel: "Base",
    evoTraining: "[Evolution] Training new generation on elite units >=100 eff",
    evoSamples: "samples",
    evoAdded: "[Pool] Added",
    evoDoc: "Doctrine",
    towerShort: "TW ",
    barracksShort: "BR ",
    aiCastleShort: "AI CST",
    yourCastleShort: "YOUR CST",
    statsPlayerCurr: "Your Faction Current Efficiency",
    statsAiCurr: "AI Current Efficiency",
    medianWord: "Median",
    chartTurn: "Turn",
    chartPlayerEff: "Player Efficiency (Blue)",
    chartPlayerMed: "Player Median",
    chartAiEff: "AI Efficiency (Red)",
    chartAiMed: "AI Median"
  },
  uk: {
    boxProfile: "ПРОФІЛЬ КОРИСТУВАЧА",
    placeholderUser: "Нікнейм...",
    btnLogin: "Увійти",
    btnLangToggle: "EN",
    boxTelemetry: "ТЕЛЕМЕТРІЯ СЕСІЇ ТА СТАН ШІ",
    boxInspector: "ІНСПЕКТОР ОБ'ЄКТА",
    boxOrders: "КЕРУВАННЯ ПІДРОЗДІЛАМИ ТА БУДІВНИЦТВО",
    btnHeal: "Лікувати юніта (+25.00% макс. HP)",
    btnHarv: "Додаткова обробка клітинки (+1.50x)",
    btnTow: "Звести оборонну вежу (4.00$)",
    btnBar: "Звести передову казарму (5.00$)",
    btnDem: "Знести союзну споруду (+50.00% повернення)",
    btnDis: "Розпустити обраного юніта (+40.00% повернення)",
    btnSwk: "Найняти робітника (3.00$ | Утрим: 0.40$)",
    btnSwr: "Найняти піхотинця (6.00$ | Утрим: 1.10$)",
    btnSar: "Найняти стрільця (8.50$ | Утрим: 1.60$)",
    btnStats: "Статистика ефективності (Графік та медіана)",
    btnEnd: "Кінець ходу (Авто-лікування та хід ШІ)",
    modalTitle: "Динаміка ефективності та медіана від початку партії",
    btnClose: "Закрити",
    playerLabel: "Гравець",
    matchLabel: "Партія",
    turnLabel: "Хід",
    winrateLabel: "Вінрейт профілю",
    poolModelsLabel: "Моделей у пулі",
    maxLabel: "Макс",
    blueLabel: "СИНІ (ВИ) ",
    redLabel: "ЧЕРВОНІ ШІ",
    goldLabel: "Казна",
    landLabel: "Землі",
    armyLabel: "Армія",
    upkeepLabel: "Утрим",
    netLabel: "Баланс",
    perTurn: "/х",
    activeGenPrefix: "Зараз грає",
    ratingLabel: "Рейтинг",
    baseLabel: "Базовий",
    evoTraining: "[Еволюція] Навчання нового покоління на еліті >=100 еф.",
    evoSamples: "зразків",
    evoAdded: "[Пул] Додано",
    evoDoc: "Доктрина",
    towerShort: "ВЖ ",
    barracksShort: "КЗ ",
    aiCastleShort: "ЗМ ШІ",
    yourCastleShort: "ВАШ ЗМ",
    statsPlayerCurr: "Поточна ефективність Вашої Фракції",
    statsAiCurr: "Поточна ефективність ШІ",
    medianWord: "Медіана",
    chartTurn: "Хід",
    chartPlayerEff: "Ефективність Гравця (Сині)",
    chartPlayerMed: "Медіана Гравця",
    chartAiEff: "Ефективність Нейромережі (Червоні)",
    chartAiMed: "Медіана ШІ"
  }
};

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

function applyStaticLabels() {
  const t = UI_LOC[currentLang] || UI_LOC.en;
  document.getElementById("lbl-box-profile").textContent = t.boxProfile;
  document.getElementById("username-inp").placeholder = t.placeholderUser;
  document.getElementById("btn-login").textContent = t.btnLogin;
  document.getElementById("btn-lang").textContent = t.btnLangToggle;
  document.getElementById("lbl-box-telemetry").textContent = t.boxTelemetry;
  document.getElementById("lbl-box-inspector").textContent = t.boxInspector;
  document.getElementById("lbl-box-orders").textContent = t.boxOrders;
  document.getElementById("b-heal").textContent = t.btnHeal;
  document.getElementById("b-harv").textContent = t.btnHarv;
  document.getElementById("b-tow").textContent = t.btnTow;
  document.getElementById("b-bar").textContent = t.btnBar;
  document.getElementById("b-dem").textContent = t.btnDem;
  document.getElementById("b-dis").textContent = t.btnDis;
  document.getElementById("b-swk").textContent = t.btnSwk;
  document.getElementById("b-swr").textContent = t.btnSwr;
  document.getElementById("b-sar").textContent = t.btnSar;
  document.getElementById("b-stats").textContent = t.btnStats;
  document.getElementById("b-end").textContent = t.btnEnd;
  document.getElementById("lbl-modal-title").textContent = t.modalTitle;
  document.getElementById("btn-modal-close").textContent = t.btnClose;
}

async function toggleLanguage() {
  const nextLang = currentLang === "en" ? "uk" : "en";
  currentLang = nextLang;
  applyStaticLabels();
  await sendAct("set_lang", null, null, currentLang);
}

async function switchProfile() {
  currentUser = document.getElementById("username-inp").value.trim() || "Guest1";
  await fetchState();
}

async function fetchState() {
  try {
    const res = await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action_type: "get", param: currentLang })
    });
    if (!res.ok) throw new Error("HTTP error " + res.status);
    state = await res.json();
    if (state.lang && (state.lang === "en" || state.lang === "uk")) {
      currentLang = state.lang;
      applyStaticLabels();
    }
    updateUI();
  } catch (err) {
    console.error("State load error:", err);
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
    console.error("Action execution error:", err);
  }
}

function updateUI() {
  if (!state) return;
  const t = UI_LOC[currentLang] || UI_LOC.en;

  document.getElementById("telemetry-text").textContent =
    `${t.playerLabel}: ${state.username} | ${t.matchLabel}: #${state.matches_played + 1} | ${t.turnLabel}: ${state.turn}/${state.max_turns}\n` +
    `${t.winrateLabel}: ${state.winrate.toFixed(2)}% | ${t.poolModelsLabel}: ${state.pool_size} (${t.maxLabel}: ${state.pool_max_rating.toFixed(2)})\n` +
    `----------------------------------------------------\n` +
    `${t.blueLabel}: ${t.goldLabel} ${state.p2.gold.toFixed(2)}$ | ${t.landLabel}: ${state.p2.tiles} (+${state.p2.inc.toFixed(2)}$)\n` +
    `            ${t.armyLabel}: ${state.p2.army} (${t.upkeepLabel}: -${state.p2.upk.toFixed(2)}$) | ${t.netLabel}: ${state.p2.net >= 0 ? '+' : ''}${state.p2.net.toFixed(2)}$${t.perTurn}\n` +
    `${t.redLabel}: ${t.goldLabel} ${state.p1.gold.toFixed(2)}$ | ${t.landLabel}: ${state.p1.tiles} (+${state.p1.inc.toFixed(2)}$)\n` +
    `            ${t.armyLabel}: ${state.p1.army} (${t.upkeepLabel}: -${state.p1.upk.toFixed(2)}$) | ${t.netLabel}: ${state.p1.net >= 0 ? '+' : ''}${state.p1.net.toFixed(2)}$${t.perTurn}`;

  document.getElementById("gen-live-badge").textContent =
    `${t.activeGenPrefix}: ${state.active_gen_label} | ${t.ratingLabel}: ${state.active_rating.toFixed(2)} (${t.baseLabel}: ${state.base_rating.toFixed(2)})`;

  const evoEl = document.getElementById("evo-banner");
  let logLines = [];
  if (state.event_log && state.event_log.length > 0) {
    logLines = state.event_log.slice(-4);
  }
  if (state.last_evolution_report) {
    const rep = state.last_evolution_report;
    if (rep.status === "training") {
      logLines.push(`${t.evoTraining} (${rep.samples} ${t.evoSamples})...`);
    } else if (rep.status === "completed") {
      const docVal = rep.doctrine_alignment !== undefined ? rep.doctrine_alignment.toFixed(2) : "100.00";
      logLines.push(`${t.evoAdded} ${rep.new_gen_label} | ${t.ratingLabel}: ${rep.new_gen_rating.toFixed(2)} | ${t.evoDoc}: ${docVal}%`);
    }
  }
  evoEl.textContent = logLines.join("\n");

  document.getElementById("inspector-text").textContent = state.inspector.lines.join("\n");
  document.getElementById("b-heal").disabled = !state.inspector.can_heal;
  document.getElementById("b-harv").disabled = !state.inspector.can_harvest;
  document.getElementById("b-tow").disabled = !state.inspector.can_build_tower;
  document.getElementById("b-bar").disabled = !state.inspector.can_build_barracks;
  document.getElementById("b-dem").disabled = !state.inspector.can_demolish;
  document.getElementById("b-dis").disabled = !state.inspector.can_disband;
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
  const t = UI_LOC[currentLang] || UI_LOC.en;
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
    ctx.fillText((st.type === 1 ? t.towerShort : t.barracksShort) + st.hp.toFixed(0), x + cs * 0.18, y + cs * 0.60);
  }

  const castles = [
    { r: 17, c: 0, hp: state.p1.castle_hp, col: "#83142c", lbl: t.aiCastleShort },
    { r: 0, c: 17, hp: state.p2.castle_hp, col: "#0a3d62", lbl: t.yourCastleShort }
  ];
  for (const csObj of castles) {
    const x = ox + csObj.c * cs, y = oy + csObj.r * cs;
    ctx.fillStyle = csObj.col;
    ctx.fillRect(x + 2, y + 4, cs - 4, cs - 6);
    ctx.strokeStyle = "#feca57";
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 2, y + 4, cs - 4, cs - 6);
    ctx.fillStyle = "#feca57";
    ctx.font = `bold ${Math.max(7, Math.floor(cs * 0.16))}px Arial`;
    ctx.fillText(`${csObj.lbl} ${csObj.hp.toFixed(0)}`, x + 3, y + cs * 0.75);
  }

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

  for (const [tr, tc] of state.valid_attacks) {
    const x = ox + tc * cs, y = oy + tr * cs;
    ctx.strokeStyle = "#e056fd";
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
  }

  for (const u of state.units) {
    const cx = ox + u.c * cs + cs * 0.5, cy = oy + u.r * cs + cs * 0.48, rad = cs * 0.34;
    ctx.fillStyle = u.faction === 1 ? "#eb3b5a" : "#0fbcf9";
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = (u.faction === 2 && u.ready) ? "#2ecc71" : (u.efficiency >= 100.0 ? "#feca57" : "#dcdde1");
    ctx.lineWidth = 2;
    ctx.stroke();
    drawUnitIcon(cx, cy, rad, u.type, u.faction);

    const bw = cs * 0.72, bh = Math.max(3, cs * 0.09);
    ctx.fillStyle = "#1e272e";
    ctx.fillRect(cx - bw / 2, cy + rad + 1, bw, bh);
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(cx - bw / 2, cy + rad + 1, bw * (u.hp / u.max_hp), bh);
  }

  if (state.selected_pos) {
    const [sr, sc] = state.selected_pos;
    ctx.strokeStyle = "#00d2d3";
    ctx.lineWidth = 3;
    ctx.strokeRect(ox + sc * cs + 2, oy + sr * cs + 2, cs - 4, cs - 4);
  }
}

function openStatsModal() {
  if (!state || !state.stats) return;
  const t = UI_LOC[currentLang] || UI_LOC.en;
  document.getElementById("modal-overlay").style.display = "flex";
  const st = state.stats;
  document.getElementById("stats-summary").textContent =
    `${t.statsPlayerCurr}: ${st.player_current.toFixed(2)} (${t.medianWord}: ${st.player_median.toFixed(2)}) | ` +
    `${t.statsAiCurr}: ${st.ai_current.toFixed(2)} (${t.medianWord}: ${st.ai_median.toFixed(2)})`;

  const pMedLine = st.turns.map(() => st.player_median);
  const aiMedLine = st.turns.map(() => st.ai_median);

  if (chartInstance) chartInstance.destroy();
  const cctx = document.getElementById("effChart").getContext("2d");
  chartInstance = new Chart(cctx, {
    type: "line",
    data: {
      labels: st.turns.map(turnNum => `${t.chartTurn} ${turnNum}`),
      datasets: [
        { label: t.chartPlayerEff, data: st.player_eff, borderColor: "#0fbcf9", borderWidth: 2.5, tension: 0.2 },
        { label: `${t.chartPlayerMed} (${st.player_median.toFixed(2)})`, data: pMedLine, borderColor: "#0fbcf9", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
        { label: t.chartAiEff, data: st.ai_eff, borderColor: "#eb3b5a", borderWidth: 2.5, tension: 0.2 },
        { label: `${t.chartAiMed} (${st.ai_median.toFixed(2)})`, data: aiMedLine, borderColor: "#eb3b5a", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 }
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
    currentUser = data.username;
    currentLang = data.lang || "en";
    document.getElementById("username-inp").value = currentUser;
    applyStaticLabels();
    await fetchState();
  } catch (err) {
    console.error("Guest init error:", err);
  }
}

clampCamera();
initApp();

setInterval(() => {
  if (state && state.is_training_new_gen) {
    fetchState();
  }
}, 1500);