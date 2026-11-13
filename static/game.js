let currentUser = "";
let currentLang = "en";
let state = null;
let cellSize = 34.0;
let minCellSize = 24.0;
let camX = 34.0, camY = 34.0;
let isPanning = false, panMoved = false, panStartX = 0, panStartY = 0;
let touchMode = null;
let touchMoved = false;
let touchStartX = 0, touchStartY = 0;
let pinchLastDist = 0;
let chartInstance = null;
let boardInitializedForMatch = false;

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

const DOCK_ICONS = {
  heal: `<svg viewBox="0 0 24 24"><path d="M10 3h4v7h7v4h-7v7h-4v-7H3v-4h7V3z"/></svg>`,
  harvest: `<svg viewBox="0 0 24 24"><path d="M12 2L4 9l8 13 8-13-8-7zm0 3.2L16.6 9 12 16.8 7.4 9 12 5.2z"/></svg>`,
  tower: `<svg viewBox="0 0 24 24"><path d="M5 3h3v3h2V3h4v3h2V3h3v6l-2 2v10H7V11L5 9V3zm5 10v5h4v-5h-4z"/></svg>`,
  barracks: `<svg viewBox="0 0 24 24"><path d="M12 3L2 11h3v9h5v-6h4v6h5v-9h3L12 3zm0 4.5l3 2.5H9l3-2.5z"/></svg>`,
  demolish: `<svg viewBox="0 0 24 24"><path d="M13 2l-2 6-6-2 4 5-5 4 6 1-1 6 4-5 5 4-1-6 6-2-5-4 2-5-6 2-1-4z"/></svg>`,
  disband: `<svg viewBox="0 0 24 24"><path d="M12 2C7.5 2 4 5.5 4 10c0 2.8 1.5 5.2 3.8 6.6L8 21h8l.2-4.4C18.5 15.2 20 12.8 20 10c0-4.5-3.5-8-8-8zm-3 8a1.5 1.5 0 110-3 1.5 1.5 0 010 3zm6 0a1.5 1.5 0 110-3 1.5 1.5 0 010 3zm-5 5h4v2h-4v-2z"/></svg>`,
  worker: `<svg viewBox="0 0 24 24"><path d="M15.5 4l4.5 4.5-2.8 2.8-1.7-1.7-8.5 8.5-2.1-2.1 8.5-8.5-1.7-1.7L15.5 4z"/></svg>`,
  warrior: `<svg viewBox="0 0 24 24"><path d="M19 3l2 2-10 10 2 2-1.5 1.5-2.5-2.5-4 4L3 18l4-4-2.5-2.5L6 10l2 2L18 2l1 1z"/></svg>`,
  archer: `<svg viewBox="0 0 24 24"><path d="M4 4c7 1 14 8 15 15h-2c-1-6-7-12-13-13V4zm15 1L7 17l-2-2L17 3l2 2z"/></svg>`,
  fire: `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" fill="none" stroke="#fff" stroke-width="2.2"/><circle cx="12" cy="12" r="2.5"/><path d="M12 1v4M12 19v4M1 12h4M19 12h4" stroke="#fff" stroke-width="2.2"/></svg>`
};

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function getTouchCanvasCoords(touch) {
  const rect = canvas.getBoundingClientRect();
  return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
}

function isTileInList(r, c, list) {
  if (!list) return false;
  for (let i = 0; i < list.length; i++) {
    if (list[i][0] === r && list[i][1] === c) return true;
  }
  return false;
}

function clampCamera() {
  const boardPx = 18.0 * cellSize;
  const margin = 1.0 * cellSize;
  const minX = canvas.width - boardPx - margin;
  const maxX = margin;
  camX = minX > maxX ? (canvas.width - boardPx) / 2.0 : Math.max(minX, Math.min(maxX, camX));

  const minY = canvas.height - boardPx - margin;
  const maxY = margin;
  camY = minY > maxY ? (canvas.height - boardPx) / 2.0 : Math.max(minY, Math.min(maxY, camY));
}

function resizeCanvas() {
  const pane = document.getElementById("left-pane");
  if (!pane || pane.style.display === "none") return;
  const rect = pane.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  const side = Math.max(260, Math.floor(Math.min(rect.width - 4, rect.height - 4)));
  const prevMin = minCellSize;
  canvas.width = side;
  canvas.height = side;

  minCellSize = side / 20.0;
  if (!boardInitializedForMatch || cellSize < minCellSize || Math.abs(cellSize - prevMin) < 0.5) {
    cellSize = minCellSize;
    camX = cellSize;
    camY = cellSize;
    boardInitializedForMatch = true;
  }
  clampCamera();
  drawBoard();
}

window.addEventListener("resize", resizeCanvas);

canvas.addEventListener("contextmenu", e => e.preventDefault());

canvas.addEventListener("mousedown", e => {
  const pos = getCanvasCoords(e);
  if (e.button === 0) {
    const c = Math.floor((pos.x - camX) / cellSize);
    const r = Math.floor((pos.y - camY) / cellSize);
    if (r >= 0 && r < 18 && c >= 0 && c < 18) {
      if (state && state.selected_pos && state.selected_pos[0] === r && state.selected_pos[1] === c) {
        state.selected_pos = null;
        state.valid_moves = [];
        state.valid_attacks = [];
        updateUI();
        return;
      }
      sendAct("select_tile", r, c);
    } else if (state) {
      state.selected_pos = null;
      state.valid_moves = [];
      state.valid_attacks = [];
      updateUI();
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
  const maxSize = Math.max(96.0, minCellSize * 3.0);
  cellSize = Math.max(minCellSize, Math.min(maxSize, cellSize * factor));
  const ratio = cellSize / oldSize;
  camX = pos.x - (pos.x - camX) * ratio;
  camY = pos.y - (pos.y - camY) * ratio;
  clampCamera();
  drawBoard();
}, { passive: false });

canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  if (e.touches.length === 1) {
    const pos = getTouchCanvasCoords(e.touches[0]);
    touchMode = "single";
    touchMoved = false;
    touchStartX = pos.x;
    touchStartY = pos.y;
  } else if (e.touches.length === 2) {
    touchMode = "pinch";
    touchMoved = true;
    const p1 = getTouchCanvasCoords(e.touches[0]);
    const p2 = getTouchCanvasCoords(e.touches[1]);
    pinchLastDist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
  }
}, { passive: false });

canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  if (touchMode === "single" && e.touches.length === 1) {
    const pos = getTouchCanvasCoords(e.touches[0]);
    const dx = pos.x - touchStartX;
    const dy = pos.y - touchStartY;
    if (Math.hypot(dx, dy) > 6) {
      touchMoved = true;
    }
    if (touchMoved) {
      camX += dx;
      camY += dy;
      touchStartX = pos.x;
      touchStartY = pos.y;
      clampCamera();
      drawBoard();
    }
  } else if (touchMode === "pinch" && e.touches.length === 2) {
    const p1 = getTouchCanvasCoords(e.touches[0]);
    const p2 = getTouchCanvasCoords(e.touches[1]);
    const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    if (pinchLastDist > 10 && dist > 10) {
      const midX = (p1.x + p2.x) / 2.0;
      const midY = (p1.y + p2.y) / 2.0;
      const factor = dist / pinchLastDist;
      const oldSize = cellSize;
      const maxSize = Math.max(96.0, minCellSize * 3.0);
      cellSize = Math.max(minCellSize, Math.min(maxSize, cellSize * factor));
      const ratio = cellSize / oldSize;
      camX = midX - (midX - camX) * ratio;
      camY = midY - (midY - camY) * ratio;
      clampCamera();
      drawBoard();
    }
    pinchLastDist = dist;
  }
}, { passive: false });

canvas.addEventListener("touchend", e => {
  e.preventDefault();
  if (touchMode === "single" && !touchMoved && e.changedTouches.length > 0) {
    const pos = getTouchCanvasCoords(e.changedTouches[0]);
    const c = Math.floor((pos.x - camX) / cellSize);
    const r = Math.floor((pos.y - camY) / cellSize);
    if (r >= 0 && r < 18 && c >= 0 && c < 18) {
      if (state && (isTileInList(r, c, state.valid_moves) || isTileInList(r, c, state.valid_attacks))) {
        sendAct("order_tile", r, c);
      } else if (state && state.selected_pos && state.selected_pos[0] === r && state.selected_pos[1] === c) {
        state.selected_pos = null;
        state.valid_moves = [];
        state.valid_attacks = [];
        updateUI();
      } else {
        sendAct("select_tile", r, c);
      }
    } else if (state) {
      state.selected_pos = null;
      state.valid_moves = [];
      state.valid_attacks = [];
      updateUI();
    }
  }
  if (e.touches.length === 0) {
    touchMode = null;
  }
}, { passive: false });

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
  document.getElementById("lbl-btn-quickai").textContent = t.btnQuickAi || "AI Simulation";
  document.getElementById("lbl-box-roomslist").textContent = t.roomsListTitle;
  document.getElementById("btn-leave-room").textContent = t.btnLeaveRoom;

  document.getElementById("lbl-box-telemetry").textContent = t.boxTelemetry;
  document.getElementById("lbl-box-inspector").textContent = t.boxInspector;
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

function renderActionDock() {
  const dockEl = document.getElementById("unit-action-dock");
  if (!dockEl) return;

  if (!state || state.screen !== "game" || !state.selected_pos || !state.inspector || !state.ui_text) {
    dockEl.classList.add("hidden");
    dockEl.innerHTML = "";
    return;
  }

  const t = state.ui_text;
  const insp = state.inspector;
  const [sr, sc] = state.selected_pos;
  const turnActive = Boolean(state.is_my_turn && !state.match_finished && state.my_alive !== false);

  const selectedUnit = state.units ? state.units.find(u => u.r === sr && u.c === sc) : null;
  const selectedStruct = state.structs ? state.structs.find(s => s.r === sr && s.c === sc) : null;

  let isCastle = false;
  let castleOwner = null;
  if (state.factions) {
    for (const fk of Object.keys(state.factions)) {
      const fd = state.factions[fk];
      if (fd.alive && fd.castle_pos && fd.castle_pos[0] === sr && fd.castle_pos[1] === sc) {
        isCastle = true;
        castleOwner = parseInt(fk);
        break;
      }
    }
  }

  let items = [];

  if (selectedUnit) {
    const isMyUnit = (selectedUnit.faction === state.my_faction);
    const disRefund = insp.disband_refund_val > 0 ? `+${insp.disband_refund_val.toFixed(1)}$` : "";
    const demRefund = insp.demolish_refund_val > 0 ? `+${insp.demolish_refund_val.toFixed(1)}$` : "";

    items.push({
      icon: DOCK_ICONS.heal,
      label: t.dockHeal || "Heal",
      available: Boolean(turnActive && isMyUnit && insp.can_heal),
      onClick: () => sendAct("heal")
    });

    if (selectedUnit.type === "worker") {
      items.push({
        icon: DOCK_ICONS.harvest,
        label: t.dockHarv || "Harvest",
        available: Boolean(turnActive && isMyUnit && insp.can_harvest),
        onClick: () => sendAct("harvest")
      });
      items.push({
        icon: DOCK_ICONS.tower,
        label: t.dockTow || "Tower (4$)",
        available: Boolean(turnActive && isMyUnit && insp.can_build_tower),
        onClick: () => sendAct("build", null, null, "tower")
      });
      items.push({
        icon: DOCK_ICONS.barracks,
        label: t.dockBar || "Barracks (5$)",
        available: Boolean(turnActive && isMyUnit && insp.can_build_barracks),
        onClick: () => sendAct("build", null, null, "barracks")
      });
      items.push({
        icon: DOCK_ICONS.demolish,
        label: `${t.dockDem || "Demolish"} ${demRefund}`.trim(),
        available: Boolean(turnActive && isMyUnit && insp.can_demolish),
        onClick: () => sendAct("demolish")
      });
    }

    items.push({
      icon: DOCK_ICONS.disband,
      label: `${t.dockDis || "Disband"} ${disRefund}`.trim(),
      available: Boolean(turnActive && isMyUnit && insp.can_disband),
      onClick: () => sendAct("disband")
    });
  } else if (isCastle || (selectedStruct && selectedStruct.type === 2)) {
    const isMySpawner = isCastle ? (castleOwner === state.my_faction) : (selectedStruct.owner === state.my_faction);
    items.push({
      icon: DOCK_ICONS.worker,
      label: t.dockSwk || "Worker (3$)",
      available: Boolean(turnActive && isMySpawner && insp.can_spawn_worker),
      onClick: () => sendAct("spawn", null, null, "worker")
    });
    items.push({
      icon: DOCK_ICONS.warrior,
      label: t.dockSwr || "Warrior (6$)",
      available: Boolean(turnActive && isMySpawner && insp.can_spawn_warrior),
      onClick: () => sendAct("spawn", null, null, "warrior")
    });
    items.push({
      icon: DOCK_ICONS.archer,
      label: t.dockSar || "Archer (8.5$)",
      available: Boolean(turnActive && isMySpawner && insp.can_spawn_archer),
      onClick: () => sendAct("spawn", null, null, "archer")
    });
  } else if (selectedStruct && selectedStruct.type === 1) {
    const canFire = Boolean(
      turnActive &&
      selectedStruct.owner === state.my_faction &&
      selectedStruct.ready &&
      state.valid_attacks &&
      state.valid_attacks.length > 0
    );
    items.push({
      icon: DOCK_ICONS.fire,
      label: t.dockTowerFire || "Fire (RMB)",
      available: canFire,
      onClick: () => {
        if (canFire && state.valid_attacks.length > 0) {
          const [tr, tc] = state.valid_attacks[0];
          sendAct("order_tile", tr, tc);
        }
      }
    });
  }

  if (items.length === 0) {
    dockEl.classList.add("hidden");
    dockEl.innerHTML = "";
    return;
  }

  dockEl.innerHTML = "";
  for (const item of items) {
    const wrap = document.createElement("div");
    wrap.className = "dock-item";
    wrap.innerHTML = `
      <div class="dock-circle ${item.available ? 'available' : 'unavailable'}">${item.icon}</div>
      <div class="dock-label">${item.label}</div>
    `;
    wrap.addEventListener("click", e => {
      e.stopPropagation();
      if (item.available) {
        item.onClick();
      }
    });
    dockEl.appendChild(wrap);
  }
  dockEl.classList.remove("hidden");
}

function aiSimulationPlaceholder() {
  const msg = currentLang === "uk"
    ? "Режим автономної ШІ Симуляції буде додано в наступному оновленні!"
    : "Autonomous AI Simulation mode is coming in a future update!";
  alert(msg);
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
    const newState = await res.json();
    if (state && state.selected_pos === null && newState.screen === "game") {
      newState.selected_pos = null;
      newState.valid_moves = [];
      newState.valid_attacks = [];
    }
    state = newState;
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
    if (action === "start" || action === "rematch") {
      boardInitializedForMatch = false;
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

async function joinRoom(rid, isPriv) {
  let pass = "";
  if (isPriv) {
    pass = prompt("Enter room password:") || "";
  }
  boardInitializedForMatch = false;
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
  const appCont = document.getElementById("app-container");
  const lobbyPanel = document.getElementById("lobby-panel");
  const roomLobbyPanel = document.getElementById("room-lobby-panel");
  const gamePanel = document.getElementById("game-control-panel");

  if (screen === "lobby") {
    appCont.className = "mode-menu";
    lobbyPanel.style.display = "grid";
    roomLobbyPanel.style.display = "none";
    gamePanel.style.display = "none";
    renderActionDock();

    const listCont = document.getElementById("rooms-list-container");
    listCont.innerHTML = "";
    if (!state.rooms || state.rooms.length === 0) {
      listCont.innerHTML = `<div style="color: #8395a7; font-size: 13px; text-align:center; padding: 30px 10px;">${t.emptyRooms}</div>`;
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
          <button class="auth-btn" style="padding: 7px 16px; flex: none;" onclick="joinRoom('${rm.room_id}', ${rm.is_private})">${t.btnJoin}</button>
        `;
        listCont.appendChild(card);
      }
    }
  } else if (screen === "room_lobby") {
    appCont.className = "mode-menu";
    lobbyPanel.style.display = "none";
    roomLobbyPanel.style.display = "flex";
    gamePanel.style.display = "none";
    renderActionDock();

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
        btnHtml = `<button class="auth-btn" style="font-size:11px; padding:5px 10px;" onclick="roomAction('switch_slot', {slot_faction: ${slot.faction}})">${t.btnTakeSlot}</button>`;
        if (state.is_host) {
          btnHtml += ` <button class="auth-btn" style="font-size:11px; padding:5px 10px; background:#f39c12;" onclick="roomAction('add_ai', {slot_faction: ${slot.faction}})">${t.btnAddAi}</button>`;
        }
      } else if (state.is_host && slot.username !== state.username) {
        btnHtml = `<button class="auth-btn" style="font-size:11px; padding:5px 10px; background:#eb3b5a; color:#fff;" onclick="roomAction('kick_slot', {slot_faction: ${slot.faction}})">${t.btnKick}</button>`;
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
    const switchedToGame = (appCont.className !== "mode-game");
    appCont.className = "mode-game";
    lobbyPanel.style.display = "none";
    roomLobbyPanel.style.display = "none";
    gamePanel.style.display = "flex";

    if (switchedToGame) {
      resizeCanvas();
    }

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

    const turnActive = state.is_my_turn && !state.match_finished && state.my_alive !== false;
    document.getElementById("b-end").disabled = !turnActive;
    document.getElementById("b-surrender").disabled = state.match_finished || state.my_alive === false;

    renderActionDock();
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
  if (!state || !state.terrain || !state.ui_text || state.screen !== "game") return;
  const t = state.ui_text;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cs = cellSize, ox = camX, oy = camY, boardPx = 18 * cs;

  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(ox - cs, oy - cs, boardPx + 2 * cs, boardPx + 2 * cs);
  ctx.strokeStyle = "#2c3e50";
  ctx.lineWidth = 2.5;
  ctx.strokeRect(ox - cs + 1, oy - cs + 1, boardPx + 2 * cs - 2, boardPx + 2 * cs - 2);

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
        if (val > 0 && cs >= 22) {
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
    ctx.font = `bold ${Math.max(7, Math.floor(cs * 0.19))}px Arial`;
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

initApp();

setInterval(() => {
  if (state && (state.is_training_new_gen || state.screen === "game" || state.screen === "room_lobby" || state.screen === "lobby")) {
    fetchState();
  }
}, 2000);