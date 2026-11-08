let currentUser = "";
let currentLang = "en";
let state = null;
let cellSize = 34.0;
let camX = 24.0, camY = 24.0;
let isPanning = false, panMoved = false, panStartX = 0, panStartY = 0;
let chartInstance = null;

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
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
  if (e.button === 0) {
    const c = Math.floor((pos.x - camX) / cellSize);
    const r = Math.floor((pos.y - camY) / cellSize);
    if (r >= 0 && r < 18 && c >= 0 && c < 18) {
      sendAct("select_tile", r, c);
    }
  } else if (e.button === 1 || e.button === 2) {
    isPanning = true;
    panMoved = false;
    panStartX = pos.x;
    panStartY = pos.y;
  }
});

window.addEventListener("mouseup", e => {
  if (isPanning && e.button === 2 && !panMoved) {
    const pos = getCanvasCoords(e);
    const c = Math.floor((pos.x - camX) / cellSize);
    const r = Math.floor((pos.y - camY) / cellSize);
    if (r >= 0 && r < 18 && c >= 0 && c < 18) {
      sendAct("order_tile", r, c);
    }
  }
  isPanning = false;
});

canvas.addEventListener("mousemove", e => {
  if (!isPanning) return;
  const pos = getCanvasCoords(e);
  const dx = pos.x - panStartX;
  const dy = pos.y - panStartY;
  if (Math.hypot(dx, dy) > 5) {
    panMoved = true;
  }
  if (panMoved) {
    camX += dx;
    camY += dy;
    panStartX = pos.x;
    panStartY = pos.y;
    clampCamera();
    drawBoard();
  }
});

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const pos = getCanvasCoords(e);
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  const oldSize = cellSize;
  cellSize = Math.max(22.0, Math.min(92.0, cellSize * factor));
  const ratio = cellSize / oldSize;
  camX = pos.x - (pos.x - camX) * ratio;
  camY = pos.y - (pos.y - camY) * ratio;
  clampCamera();
  drawBoard();
});

window.addEventListener("beforeunload", () => {
  if (currentUser) {
    const blob = new Blob([JSON.stringify({ username: currentUser })], { type: "application/json" });
    navigator.sendBeacon("/api/leave_beacon", blob);
  }
});

function applyStaticLabels(t) {
  if (!t) return;
  document.title = t.pageTitle || "Faction Warfare RTS";
  document.getElementById("lbl-auth-title").textContent = t.authTitle;
  document.getElementById("lbl-auth-subtitle").textContent = t.authSubtitle;
  document.getElementById("auth-user").placeholder = t.usernameLabel;
  document.getElementById("auth-pass").placeholder = t.passwordLabel;
  document.getElementById("lbl-btn-login").textContent = t.btnLogin;
  document.getElementById("lbl-btn-reg").textContent = t.btnRegister;
  document.getElementById("lbl-btn-guest").textContent = t.btnGuest;

  document.getElementById("lbl-box-profile").textContent = t.boxProfile;
  document.getElementById("btn-guide").textContent = t.btnGuide || "Guide";
  document.getElementById("btn-logout").textContent = t.btnLogout || "Logout";
  document.getElementById("btn-lang").textContent = currentLang === "en" ? "UK" : "EN";

  document.getElementById("lbl-box-createroom").textContent = t.createRoomTitle;
  document.getElementById("room-name-inp").placeholder = t.roomNamePlaceholder;
  document.getElementById("room-pass-inp").placeholder = t.roomPasswordPlaceholder;
  document.getElementById("lbl-mode-2p").textContent = t.mode2P;
  document.getElementById("lbl-mode-4p").textContent = t.mode4P;
  document.getElementById("lbl-fill-ai").textContent = t.fillAiCheck;
  document.getElementById("lbl-btn-createroom").textContent = t.btnCreateRoom;
  document.getElementById("lbl-btn-quickai").textContent = t.btnQuickAi;
  document.getElementById("lbl-box-roomslist").textContent = t.roomsListTitle;
  document.getElementById("btn-leave-room").textContent = t.btnLeaveRoom;

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
  document.getElementById("b-surrender").textContent = t.btnSurrender || "Surrender";
  document.getElementById("b-backlobby").textContent = t.btnBackLobby || "Leave";

  document.getElementById("lbl-modal-title").textContent = t.modalTitle;
  document.getElementById("btn-modal-close").textContent = t.btnClose;
  document.getElementById("lbl-guide-title").textContent = t.guideModalTitle || "Guide";
  document.getElementById("btn-guide-close").textContent = t.btnClose;
  if (t.guideHtml) {
    document.getElementById("guide-content").innerHTML = t.guideHtml;
  }
}

async function initApp() {
  try {
    const res = await fetch("/api/init_user");
    const data = await res.json();
    if (data.suggested_guest) {
      document.getElementById("auth-user").value = data.suggested_guest;
    }
    if (data.lang) {
      currentLang = data.lang;
    }
    if (data.ui_text) {
      applyStaticLabels(data.ui_text);
    }
  } catch (err) {
    console.error("Init error:", err);
  }
}

async function submitAuth(mode) {
  const user = document.getElementById("auth-user").value.trim();
  const pass = document.getElementById("auth-pass").value.trim();
  const errEl = document.getElementById("auth-error");
  errEl.textContent = "";

  try {
    const res = await fetch("/api/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, username: user, password: pass, lang: currentLang })
    });
    const data = await res.json();
    if (!data.ok) {
      errEl.textContent = data.error || "Authentication failed";
      return;
    }
    currentUser = data.state.username;
    currentLang = data.state.lang || currentLang;
    document.getElementById("auth-screen").style.display = "none";
    document.getElementById("app-container").style.display = "flex";
    state = data.state;
    updateUI();
  } catch (err) {
    errEl.textContent = "Network error";
  }
}

async function logoutUser() {
  if (currentUser) {
    await roomAction("leave");
  }
  currentUser = "";
  document.getElementById("app-container").style.display = "none";
  document.getElementById("auth-screen").style.display = "flex";
  document.getElementById("auth-pass").value = "";
}

async function toggleLanguage() {
  currentLang = currentLang === "en" ? "uk" : "en";
  await sendAct("set_lang", null, null, currentLang);
}

async function fetchState() {
  if (!currentUser) return;
  try {
    const res = await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action_type: "get", param: currentLang })
    });
    state = await res.json();
    if (state.lang) currentLang = state.lang;
    updateUI();
  } catch (err) {
    console.error("Fetch state error:", err);
  }
}

async function roomAction(action, extra = {}) {
  try {
    const res = await fetch("/api/rooms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action, ...extra })
    });
    const data = await res.json();
    if (!data.ok) {
      alert(data.error || "Room action error");
      return;
    }
    state = data.state;
    updateUI();
  } catch (err) {
    console.error("Room action error:", err);
  }
}

function createRoom() {
  const name = document.getElementById("room-name-inp").value.trim();
  const pass = document.getElementById("room-pass-inp").value.trim();
  const modeVal = document.querySelector('input[name="room-mode"]:checked').value;
  const fillAi = document.getElementById("room-fill-ai").checked;

  roomAction("create", {
    room_name: name,
    max_players: parseInt(modeVal),
    is_private: pass.length > 0,
    password: pass,
    fill_ai: fillAi
  });
}

function quickBotMatch() {
  const modeVal = document.querySelector('input[name="room-mode"]:checked').value;
  roomAction("quick_bot_match", { max_players: parseInt(modeVal), fill_ai: true });
}

async function joinRoom(rid, isPriv) {
  let pass = "";
  if (isPriv) {
    pass = prompt("Enter room password:") || "";
  }
  await roomAction("join", { room_id: rid, password: pass });
}

async function confirmSurrender() {
  const msg = (state && state.ui_text && state.ui_text.surrenderConfirm)
    ? state.ui_text.surrenderConfirm
    : "Are you sure you want to surrender?";
  if (confirm(msg)) {
    await sendAct("surrender");
  }
}

async function sendAct(action_type, r = null, c = null, param = null) {
  if (!currentUser) return;
  try {
    const res = await fetch("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: currentUser, action_type, r, c, param })
    });
    state = await res.json();
    if (state.lang) currentLang = state.lang;
    updateUI();
  } catch (err) {
    console.error("Action error:", err);
  }
}

function updateUI() {
  if (!state || !state.ui_text) return;
  const t = state.ui_text;
  applyStaticLabels(t);

  document.getElementById("current-user-display").textContent = `${state.username} (${t.baseLabel}: ${state.base_rating.toFixed(2)})`;

  const screen = state.screen || "lobby";
  const lobbyPanel = document.getElementById("lobby-panel");
  const roomLobbyPanel = document.getElementById("room-lobby-panel");
  const gamePanel = document.getElementById("game-control-panel");

  if (screen === "lobby") {
    lobbyPanel.style.display = "block";
    roomLobbyPanel.style.display = "none";
    gamePanel.style.display = "none";

    const listCont = document.getElementById("rooms-list-container");
    listCont.innerHTML = "";
    if (!state.rooms || state.rooms.length === 0) {
      listCont.innerHTML = `<div style="color: #8395a7; font-size: 12px; text-align:center; padding: 20px;">${t.emptyRooms}</div>`;
    } else {
      for (const rm of state.rooms) {
        const card = document.createElement("div");
        card.className = "room-card";
        const stText = rm.status === "waiting" ? t.statusWaiting : (rm.status === "playing" ? t.statusPlaying : t.statusFinished);
        card.innerHTML = `
          <div class="room-info">
            <h4>${rm.name} (${rm.max_players}P) ${rm.is_private ? '🔒' : ''}</h4>
            <p>${t.playersCount}: ${rm.filled_slots}/${rm.max_players} | ${t.hostLabel}: ${rm.host} | ${stText}</p>
          </div>
          <button class="auth-btn" style="padding: 6px 14px; font-size: 11px;" onclick="joinRoom('${rm.room_id}', ${rm.is_private})">${t.btnJoin}</button>
        `;
        listCont.appendChild(card);
      }
    }
  } else if (screen === "room_lobby") {
    lobbyPanel.style.display = "none";
    roomLobbyPanel.style.display = "block";
    gamePanel.style.display = "none";

    const rm = state.room;
    document.getElementById("room-lobby-title").textContent = `${t.roomLobbyTitle} ${rm.name} (${rm.max_players}P)`;
    const slotsCont = document.getElementById("room-slots-container");
    slotsCont.innerHTML = "";

    for (const slot of rm.slots) {
      const row = document.createElement("div");
      row.className = "slot-row";
      let statusDesc = t.slotEmpty;
      if (slot.kind === "human") {
        statusDesc = `👤 ${slot.username}`;
      } else if (slot.kind === "ai") {
        statusDesc = `🤖 ${slot.username} (R: ${slot.ai_rating.toFixed(2)})`;
      }

      let btnHtml = "";
      if (slot.kind === "empty") {
        btnHtml = `<button class="auth-btn" style="font-size:10px; padding:4px 8px;" onclick="roomAction('switch_slot', {slot_faction: ${slot.faction}})">${t.btnTakeSlot}</button>`;
        if (state.is_host) {
          btnHtml += ` <button class="auth-btn" style="font-size:10px; padding:4px 8px; background:#f39c12;" onclick="roomAction('add_ai', {slot_faction: ${slot.faction}})">${t.btnAddAi}</button>`;
        }
      } else if (state.is_host && slot.username !== state.username) {
        btnHtml = `<button class="auth-btn" style="font-size:10px; padding:4px 8px; background:#eb3b5a;" onclick="roomAction('kick_slot', {slot_faction: ${slot.faction}})">${t.btnKick}</button>`;
      }

      row.innerHTML = `<span>Faction #${slot.faction}: <b>${statusDesc}</b></span> <span>${btnHtml}</span>`;
      slotsCont.appendChild(row);
    }

    const startBtn = document.getElementById("btn-start-game");
    if (state.is_host) {
      startBtn.style.display = "block";
      startBtn.disabled = !state.can_start;
      startBtn.textContent = rm.status === "finished" ? "Rematch" : t.btnStartGame;
    } else {
      startBtn.style.display = "none";
    }
  } else if (screen === "game") {
    lobbyPanel.style.display = "none";
    roomLobbyPanel.style.display = "none";
    gamePanel.style.display = "flex";

    const bannerEl = document.getElementById("turn-status-banner");
    if (state.match_finished) {
      bannerEl.className = "turn-banner turn-finished";
      bannerEl.textContent = `${t.matchFinishedBanner}${state.winner_name || ("F" + state.winner_faction)}`;
    } else if (state.my_alive === false) {
      bannerEl.className = "turn-banner turn-dead";
      bannerEl.textContent = t.eliminatedBanner;
    } else if (state.is_my_turn) {
      bannerEl.className = "turn-banner turn-my";
      bannerEl.textContent = t.yourTurnBanner;
    } else {
      bannerEl.className = "turn-banner turn-wait";
      bannerEl.textContent = `${t.waitingTurnBanner}${state.current_turn_user} (F${state.current_turn_faction})`;
    }

    const fKeys = state.factions ? Object.keys(state.factions).sort((a, b) => parseInt(a) - parseInt(b)) : [];
    let facLines = [];
    for (const fk of fKeys) {
      const fd = state.factions[fk];
      const isMe = (parseInt(fk) === state.my_faction);
      const isTurn = (parseInt(fk) === state.current_turn_faction && !state.match_finished);
      const turnMark = isTurn ? "▶ " : "  ";
      const meTag = isMe ? ` [${t.youTag}]` : "";
      if (!fd.alive) {
        facLines.push(`${turnMark}F${fk} (${fd.controller})${meTag}: [${t.deadTag}]`);
      } else {
        facLines.push(
          `${turnMark}F${fk} (${fd.controller})${meTag}: ${t.goldLabel} ${fd.gold.toFixed(2)}$ | ${t.landLabel} ${fd.tiles} (+${fd.inc.toFixed(2)}$) | ${t.armyLabel} ${fd.army} (-${fd.upk.toFixed(2)}$) | ${fd.net >= 0 ? '+' : ''}${fd.net.toFixed(2)}$${t.perTurn} | HQ: ${fd.castle_hp.toFixed(0)}`
        );
      }
    }

    document.getElementById("telemetry-text").textContent =
      `${t.playerLabel}: ${state.username} | ${t.turnLabel}: ${state.turn}/${state.max_turns} | ${t.winrateLabel}: ${state.winrate.toFixed(2)}%\n` +
      `------------------------------------------------------------\n` +
      facLines.join("\n");

    document.getElementById("gen-live-badge").textContent =
      `${t.activeGenPrefix}: ${state.active_gen_label} | ${t.ratingLabel}: ${state.active_rating.toFixed(2)}`;

    const evoEl = document.getElementById("evo-banner");
    let logLines = state.event_log ? state.event_log.slice(-4) : [];
    if (state.last_evolution_report) {
      const rep = state.last_evolution_report;
      if (rep.status === "training") {
        logLines.push(`${t.evoTraining} (${rep.samples} ${t.evoSamples})...`);
      } else if (rep.status === "completed") {
        logLines.push(`${t.evoAdded} ${rep.new_gen_label} | ${t.ratingLabel}: ${rep.new_gen_rating.toFixed(2)}`);
      }
    }
    evoEl.textContent = logLines.join("\n");

    document.getElementById("inspector-text").textContent = state.inspector.lines.join("\n");
    if (state.inspector.disband_refund_val > 0) {
      document.getElementById("b-dis").textContent = `${t.btnDis.split('(')[0].trim()} (+${state.inspector.disband_refund_val.toFixed(2)}$)`;
    }
    if (state.inspector.demolish_refund_val > 0) {
      document.getElementById("b-dem").textContent = `${t.btnDem.split('(')[0].trim()} (+${state.inspector.demolish_refund_val.toFixed(2)}$)`;
    }

    const turnActive = state.is_my_turn && !state.match_finished && state.my_alive !== false;
    document.getElementById("b-heal").disabled = !turnActive || !state.inspector.can_heal;
    document.getElementById("b-harv").disabled = !turnActive || !state.inspector.can_harvest;
    document.getElementById("b-tow").disabled = !turnActive || !state.inspector.can_build_tower;
    document.getElementById("b-bar").disabled = !turnActive || !state.inspector.can_build_barracks;
    document.getElementById("b-dem").disabled = !turnActive || !state.inspector.can_demolish;
    document.getElementById("b-dis").disabled = !turnActive || !state.inspector.can_disband;
    document.getElementById("b-swk").disabled = !turnActive || !state.inspector.can_spawn_worker;
    document.getElementById("b-swr").disabled = !turnActive || !state.inspector.can_spawn_warrior;
    document.getElementById("b-sar").disabled = !turnActive || !state.inspector.can_spawn_archer;
    document.getElementById("b-end").disabled = !turnActive;
    document.getElementById("b-surrender").disabled = state.match_finished || state.my_alive === false;
  }

  drawBoard();
  if (document.getElementById("modal-overlay").style.display === "flex") {
    openStatsModal();
  }
}

function drawUnitIcon(cx, cy, rad, uType, faction) {
  ctx.save();
  ctx.strokeStyle = faction === 1 ? "#ffffff" : "#141b24";
  ctx.fillStyle = faction === 1 ? "#feca57" : "#ffffff";
  ctx.lineWidth = Math.max(2, rad * 0.16);

  if (uType === "worker") {
    ctx.beginPath();
    ctx.moveTo(cx - rad * 0.44, cy + rad * 0.44);
    ctx.lineTo(cx + rad * 0.18, cy - rad * 0.18);
    ctx.stroke();

    ctx.save();
    ctx.translate(cx + rad * 0.18, cy - rad * 0.18);
    ctx.rotate(-Math.PI / 4);
    const hw = rad * 0.34;
    const hh = rad * 0.72;
    ctx.fillRect(-hw / 2, -hh / 2, hw, hh);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(-hw / 2, -hh / 2, hw, hh);
    ctx.restore();
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
  if (!state || !state.terrain || !state.ui_text) return;
  const t = state.ui_text;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cs = cellSize, ox = camX, oy = camY, boardPx = 18 * cs;

  ctx.strokeStyle = "#2c3e50";
  ctx.lineWidth = 2;
  ctx.strokeRect(ox - cs, oy - cs, boardPx + 2 * cs, boardPx + 2 * cs);
  ctx.strokeStyle = (state.is_my_turn && !state.match_finished) ? "#2ecc71" : "#feca57";
  ctx.lineWidth = (state.is_my_turn && !state.match_finished) ? 3.5 : 2;
  ctx.strokeRect(ox - 2, oy - 2, boardPx + 4, boardPx + 4);

  const factionColors = { 1: "#eb3b5a", 2: "#0fbcf9", 3: "#2ecc71", 4: "#feca57" };
  const terrColors = { 1: "#632227", 2: "#1b3b6f", 3: "#1e5128", 4: "#6c5ce7" };

  for (let r = 0; r < 18; r++) {
    for (let c = 0; c < 18; c++) {
      const x = ox + c * cs, y = oy + r * cs;
      const tType = state.terrain[r][c], owner = state.territory[r][c];
      ctx.fillStyle = tType === 0 ? "#3d3d3d" : (terrColors[owner] || "#2b343b");
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
    ctx.fillStyle = factionColors[st.owner] || "#0097e6";
    ctx.fillRect(x + cs * 0.14, y + cs * 0.22, cs * 0.72, cs * 0.66);
    ctx.strokeStyle = "#feca57";
    ctx.strokeRect(x + cs * 0.14, y + cs * 0.22, cs * 0.72, cs * 0.66);
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.max(8, Math.floor(cs * 0.19))}px Arial`;
    ctx.fillText((st.type === 1 ? t.towerShort : t.barracksShort) + st.hp.toFixed(0), x + cs * 0.18, y + cs * 0.60);
  }

  const castles = [
    { r: 17, c: 0, hp: state.factions?.["1"]?.castle_hp || 0, col: "#83142c", lbl: "F1" },
    { r: 0, c: 17, hp: state.factions?.["2"]?.castle_hp || 0, col: "#0a3d62", lbl: "F2" },
    { r: 0, c: 0, hp: state.factions?.["3"]?.castle_hp || 0, col: "#1e5128", lbl: "F3" },
    { r: 17, c: 17, hp: state.factions?.["4"]?.castle_hp || 0, col: "#4a3b8c", lbl: "F4" }
  ];
  for (const csObj of castles) {
    if (csObj.hp <= 0) continue;
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

  const minEff = (state.config && state.config.min_training_eff) ? state.config.min_training_eff : 100.0;
  for (const u of state.units) {
    const cx = ox + u.c * cs + cs * 0.5, cy = oy + u.r * cs + cs * 0.48, rad = cs * 0.34;
    ctx.fillStyle = factionColors[u.faction] || "#0fbcf9";
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();

    let ringCol = "#dcdde1";
    if (u.faction === state.my_faction && u.ready) {
      ringCol = (u.moved && !u.attacked) ? "#e056fd" : "#2ecc71";
    } else if (u.efficiency >= minEff) {
      ringCol = "#feca57";
    }
    ctx.strokeStyle = ringCol;
    ctx.lineWidth = 2.2;
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
  if (!state || !state.stats || !state.ui_text) return;
  const t = state.ui_text;
  document.getElementById("modal-overlay").style.display = "flex";
  const st = state.stats;

  const factionColors = { "1": "#eb3b5a", "2": "#0fbcf9", "3": "#2ecc71", "4": "#feca57" };
  const fKeys = st.factions_eff ? Object.keys(st.factions_eff).sort((a, b) => parseInt(a) - parseInt(b)) : ["1", "2"];

  let summaryParts = [];
  let datasets = [];

  for (const fk of fKeys) {
    const arr = st.factions_eff ? st.factions_eff[fk] : (fk === "2" ? st.player_eff : st.ai_eff);
    const med = st.factions_median ? st.factions_median[fk] : (fk === "2" ? st.player_median : st.ai_median);
    const curr = (arr && arr.length > 0) ? arr[arr.length - 1] : 0.0;
    const fInfo = state.factions && state.factions[fk] ? state.factions[fk] : null;
    const fLabel = fInfo ? `F${fk} (${fInfo.controller})` : `Faction ${fk}`;
    const col = factionColors[fk] || "#ffffff";

    summaryParts.push(`<b>${fLabel}:</b> ${curr.toFixed(2)} (${t.medianWord}: ${med.toFixed(2)})`);

    datasets.push({
      label: fLabel,
      data: arr,
      borderColor: col,
      borderWidth: 2.4,
      tension: 0.2
    });
    datasets.push({
      label: `${fLabel} ${t.medianWord} (${med.toFixed(2)})`,
      data: st.turns.map(() => med),
      borderColor: col,
      borderDash: [6, 4],
      borderWidth: 1.3,
      pointRadius: 0
    });
  }

  document.getElementById("stats-summary").innerHTML = summaryParts.join(" &nbsp;|&nbsp; ");

  if (chartInstance) chartInstance.destroy();
  const cctx = document.getElementById("effChart").getContext("2d");
  chartInstance = new Chart(cctx, {
    type: "line",
    data: {
      labels: st.turns.map(turnNum => `${t.chartTurn} ${turnNum}`),
      datasets: datasets
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#fff", boxWidth: 14, font: { size: 11 } } } },
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

function openGuideModal() {
  document.getElementById("guide-overlay").style.display = "flex";
}

function closeGuideModal() {
  document.getElementById("guide-overlay").style.display = "none";
}

clampCamera();
initApp();

setInterval(() => {
  if (state && (state.is_training_new_gen || state.screen === "game" || state.screen === "room_lobby" || state.screen === "lobby")) {
    fetchState();
  }
}, 2000);