import os
import json
import time
import uuid
import math
import random
import zipfile
import hashlib
import threading
from typing import Optional
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import game_config as cfg
import ai_pool
import ai_strategist

GRID_SIZE = getattr(cfg, "GRID_SIZE", 18)
MAX_STEPS = getattr(cfg, "MAX_STEPS", getattr(cfg, "MAX_TURNS", 300))
INITIAL_GOLD = getattr(cfg, "INITIAL_GOLD", getattr(cfg, "STARTING_GOLD", 12.00))
CASTLE_MAX_HP = getattr(cfg, "CASTLE_MAX_HP", 25.00)
TILE_VALUES = getattr(cfg, "TILE_VALUES", getattr(cfg, "BASE_TILE_YIELDS", {0: 0.00, 1: 0.05, 2: 0.15, 3: 0.35}))
UNIT_STATS = getattr(cfg, "UNIT_STATS", {})
STRUCT_STATS = getattr(cfg, "STRUCT_STATS", {})
SUPPLY_LIMIT = getattr(cfg, "SUPPLY_LIMIT", getattr(cfg, "SUPPLY_FREE_LIMIT", 8))
SUPPLY_TAX_PER_UNIT = getattr(cfg, "SUPPLY_TAX_PER_UNIT", 0.15)
DISBAND_REFUND_RATIO = getattr(cfg, "DISBAND_REFUND_RATIO", 0.40)
DEMOLISH_REFUND_RATIO = getattr(cfg, "DEMOLISH_REFUND_RATIO", 0.50)
MIN_TRAINING_EFFICIENCY = getattr(cfg, "MIN_TRAINING_EFFICIENCY", 100.00)
MAX_WORKERS_PER_FACTION = getattr(cfg, "MAX_WORKERS_PER_FACTION", 4)
POOL_DIR = getattr(cfg, "POOL_DIR", "./selfplay_pool/")
USERS_DB_FILE = getattr(cfg, "USERS_DB_FILE", os.path.join(POOL_DIR, "users_db.json"))
PLAYER_INITIAL_RATING = getattr(cfg, "PLAYER_INITIAL_RATING", 10.00)

TERRAIN_MOUNTAIN = 0
TERRAIN_PLAIN = 1
TERRAIN_FERTILE = 2
TERRAIN_GOLD = 3
BASE_TILE_YIELDS = TILE_VALUES

PoolRegistry = getattr(ai_pool, "PoolRegistry", None)
OnlineEvolutionEngine = getattr(ai_pool, "OnlineEvolutionEngine", None)
encode_state_from_live = getattr(ai_strategist, "encode_state_from_live", lambda *a, **kw: np.zeros(128, dtype=np.float32))
compute_unit_efficiency = getattr(ai_strategist, "compute_unit_efficiency", lambda *a, **kw: 0.0)

app = FastAPI()
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("locales", exist_ok=True)
os.makedirs(POOL_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

USERS_LOCK = threading.Lock()
ROOMS_LOCK = threading.Lock()
LOCALE_CACHE = {}
DISCONNECT_TIMEOUT_SECONDS = 65.0

CASTLE_POSITIONS = {
    1: (GRID_SIZE - 1, 0),
    2: (0, GRID_SIZE - 1),
    3: (0, 0),
    4: (GRID_SIZE - 1, GRID_SIZE - 1)
}


def load_locale(lang: str) -> dict:
    lang_key = "uk" if str(lang).lower().startswith("uk") else "en"
    path = os.path.join("locales", f"{lang_key}.json")
    try:
        mtime = os.path.getmtime(path)
        cached = LOCALE_CACHE.get(lang_key)
        if cached and cached["mtime"] == mtime:
            return cached["data"]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        LOCALE_CACHE[lang_key] = {"mtime": mtime, "data": data}
        return data
    except Exception:
        return {"units": {}, "structs": {}, "messages": {}, "ui": {}}


def tr(lang: str, category: str, key: str, **kwargs) -> str:
    data = load_locale(lang)
    template = data.get(category, {}).get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return round(val, 4)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    return obj


class UserProfileManager:
    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_unlocked() -> dict:
        if not os.path.exists(USERS_DB_FILE):
            return {}
        try:
            with open(USERS_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_unlocked(db: dict):
        os.makedirs(POOL_DIR, exist_ok=True)
        tmp_path = USERS_DB_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(clean_for_json(db), f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, USERS_DB_FILE)

    @staticmethod
    def generate_guest_name() -> str:
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            for _ in range(100):
                cand = f"Commander_{random.randint(100, 999)}"
                if cand not in db:
                    return cand
            return f"Commander_{uuid.uuid4().hex[:5]}"

    @staticmethod
    def authenticate(mode: str, username: str, password: str, lang: str = "en") -> tuple[bool, str, dict]:
        uname = str(username or "").strip()
        pwd = str(password or "").strip()

        if mode == "guest":
            if not uname:
                uname = UserProfileManager.generate_guest_name()
            with USERS_LOCK:
                db = UserProfileManager._load_unlocked()
                existing = db.get(uname)
                if existing and existing.get("is_registered"):
                    uname = UserProfileManager.generate_guest_name()
                prof = db.get(uname, {
                    "username": uname,
                    "is_registered": False,
                    "base_rating": float(PLAYER_INITIAL_RATING),
                    "matches": 0,
                    "wins": 0,
                    "lang": lang
                })
                prof["lang"] = lang
                db[uname] = prof
                UserProfileManager._save_unlocked(db)
            return True, "", prof

        if len(uname) < 2 or len(uname) > 28:
            return False, "Username must be between 2 and 28 characters.", {}
        if len(pwd) < 3:
            return False, "Password must be at least 3 characters.", {}

        pwd_hash = UserProfileManager._hash_password(pwd)
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            existing = db.get(uname)

            if mode == "register":
                if existing and existing.get("is_registered"):
                    return False, "Username is already registered. Please log in.", {}
                prof = {
                    "username": uname,
                    "password_hash": pwd_hash,
                    "is_registered": True,
                    "base_rating": float(existing.get("base_rating", PLAYER_INITIAL_RATING)) if existing else float(PLAYER_INITIAL_RATING),
                    "matches": int(existing.get("matches", 0)) if existing else 0,
                    "wins": int(existing.get("wins", 0)) if existing else 0,
                    "lang": lang
                }
                db[uname] = prof
                UserProfileManager._save_unlocked(db)
                try:
                    from cloud_backup import CloudBackupManager
                    CloudBackupManager.sync_users_db_async()
                except Exception:
                    pass
                return True, "", prof

            if mode == "login":
                if not existing or not existing.get("is_registered"):
                    return False, "Account not found. Please register first.", {}
                if existing.get("password_hash") != pwd_hash:
                    return False, "Invalid password.", {}
                existing["lang"] = lang
                db[uname] = existing
                UserProfileManager._save_unlocked(db)
                return True, "", existing

        return False, "Unknown auth mode.", {}

    @staticmethod
    def get_profile(username: str) -> dict:
        uname = str(username or "").strip()
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            if uname not in db:
                prof = {
                    "username": uname,
                    "is_registered": False,
                    "base_rating": float(PLAYER_INITIAL_RATING),
                    "matches": 0,
                    "wins": 0,
                    "lang": "en"
                }
                db[uname] = prof
                UserProfileManager._save_unlocked(db)
                return dict(prof)
            return dict(db[uname])

    @staticmethod
    def update_profile(username: str, profile_data: dict):
        uname = str(username).strip()
        if not uname or uname.startswith("__"):
            return
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            existing = db.get(uname, {})
            merged = dict(existing)
            merged.update(clean_for_json(profile_data))
            if existing.get("password_hash") and not merged.get("password_hash"):
                merged["password_hash"] = existing["password_hash"]
                merged["is_registered"] = True
            db[uname] = merged
            UserProfileManager._save_unlocked(db)
        try:
            from cloud_backup import CloudBackupManager
            CloudBackupManager.sync_users_db_async()
        except Exception:
            pass


class Unit:
    def __init__(self, r: int, c: int, faction: int, u_type: str, ready: bool = True):
        self.r = int(r)
        self.c = int(c)
        self.faction = int(faction)
        self.u_type = str(u_type)
        stats = UNIT_STATS[self.u_type]
        self.max_hp = float(stats["max_hp"])
        self.hp = float(self.max_hp)
        self.move_range = int(stats["move_range"])
        self.atk_range = int(stats["atk_range"])
        self.dmg = float(stats["dmg"])
        self.cost = float(stats["cost"])
        self.upkeep = float(stats["upkeep"])
        self.moved = not ready
        self.attacked = not ready
        self.ready = bool(ready)
        self.dmg_dealt = 0.0
        self.tiles_captured = 0
        self.gold_harvested = 0.0
        self.builds_done = 0
        self.turns_alive = 0

    @property
    def efficiency(self) -> float:
        return compute_unit_efficiency(
            self.u_type,
            self.dmg_dealt,
            self.tiles_captured,
            self.gold_harvested,
            self.builds_done,
            self.turns_alive
        )


class Structure:
    def __init__(self, r: int, c: int, owner: int, s_type: int, ready: bool = False):
        self.r = int(r)
        self.c = int(c)
        self.owner = int(owner)
        self.s_type = int(s_type)
        stats = STRUCT_STATS[self.s_type]
        self.max_hp = float(stats["max_hp"])
        self.hp = float(self.max_hp)
        self.atk_range = int(stats["atk_range"])
        self.dmg = float(stats["dmg"])
        self.cost = float(stats["cost"])
        self.ready = bool(ready)


class GameSession:
    def __init__(self, room_id: str, max_players: int, slots_config: list):
        self.room_id = room_id
        self.max_players = int(max_players)
        self.slots_config = slots_config
        self.turn = 1
        self.max_turns = int(MAX_STEPS)
        self.match_finished = False
        self.winner_faction = None
        self.winner_name = ""
        self.event_log = []
        self.ai_traces = []
        self.selected_by_faction = {}
        self.factions = {}
        self.terrain = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.territory = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.units = {}
        self.structs = {}
        self.match_obs_by_faction = {}
        self.match_acts_by_faction = {}
        self.match_effs_by_faction = {}
        self.stats_turns = [1]
        self.stats_factions_eff = {}
        self.active_gen_label = "Gen-Base"
        self.active_rating = float(PLAYER_INITIAL_RATING)
        self.is_training_new_gen = False
        self.last_evolution_report = None
        self._init_board()

    def _log(self, msg: str):
        self.event_log.append(str(msg))
        if len(self.event_log) > 30:
            self.event_log = self.event_log[-30:]

    def _init_board(self):
        rng = np.random.default_rng()
        self.terrain = np.full((GRID_SIZE, GRID_SIZE), TERRAIN_PLAIN, dtype=np.int32)
        self.territory = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if r <= c:
                    p = rng.random()
                    if p < 0.12:
                        t_val = TERRAIN_MOUNTAIN
                    elif p < 0.34:
                        t_val = TERRAIN_FERTILE
                    elif p < 0.44:
                        t_val = TERRAIN_GOLD
                    else:
                        t_val = TERRAIN_PLAIN
                    self.terrain[r, c] = t_val
                    self.terrain[GRID_SIZE - 1 - c, GRID_SIZE - 1 - r] = t_val

        active_factions = [int(s["faction"]) for s in self.slots_config if s["kind"] != "empty"]
        for s in self.slots_config:
            if s["kind"] == "empty":
                continue
            f = int(s["faction"])
            cr, cc = CASTLE_POSITIONS[f]
            self.factions[f] = {
                "faction": f,
                "controller": s["username"],
                "kind": s["kind"],
                "ai_entry": s.get("ai_entry"),
                "gold": float(INITIAL_GOLD),
                "castle_hp": float(CASTLE_MAX_HP),
                "castle_pos": (cr, cc),
                "alive": True
            }
            if s["kind"] == "ai" and s.get("ai_entry"):
                self.active_gen_label = s["ai_entry"].get("label", "Gen-Base")
                self.active_rating = float(s["ai_entry"].get("rating", PLAYER_INITIAL_RATING))

            self.match_obs_by_faction[f] = []
            self.match_acts_by_faction[f] = []
            self.match_effs_by_faction[f] = []
            self.stats_factions_eff[str(f)] = [0.0]

            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if self.terrain[nr, nc] == TERRAIN_MOUNTAIN:
                            self.terrain[nr, nc] = TERRAIN_PLAIN
                        if abs(dr) + abs(dc) <= 2:
                            self.territory[nr, nc] = f

            spawn_offsets = {
                1: [(-1, 0, "worker"), (0, 1, "warrior"), (-1, 1, "archer")],
                2: [(1, 0, "worker"), (0, -1, "warrior"), (1, -1, "archer")],
                3: [(1, 0, "worker"), (0, 1, "warrior"), (1, 1, "archer")],
                4: [(-1, 0, "worker"), (0, -1, "warrior"), (-1, -1, "archer")]
            }
            for dr, dc, ut in spawn_offsets.get(f, []):
                ur, uc = cr + dr, cc + dc
                if 0 <= ur < GRID_SIZE and 0 <= uc < GRID_SIZE:
                    self.units[(ur, uc)] = Unit(ur, uc, f, ut, ready=True)
                    self.territory[ur, uc] = f

        self.turn_order = sorted(active_factions)
        self.turn_idx = 0
        self._log(tr("en", "messages", "match_started"))
        self._check_and_run_ai_turns()

    @property
    def current_turn_faction(self) -> int:
        if not self.turn_order:
            return 1
        return self.turn_order[self.turn_idx % len(self.turn_order)]

    def _effective_tile_yield(self, r: int, c: int) -> float:
        t_type = int(self.terrain[r, c])
        if t_type == TERRAIN_MOUNTAIN:
            return 0.0
        if (r, c) in self.structs:
            return 0.0
        for f, fd in self.factions.items():
            if fd["alive"] and fd["castle_pos"] == (r, c):
                return 0.0
        return float(BASE_TILE_YIELDS.get(t_type, 0.05))

    def _compute_economy_for_faction(self, faction: int) -> tuple[int, float, int, float, float]:
        if faction not in self.factions or not self.factions[faction]["alive"]:
            return 0, 0.0, 0, 0.0, 0.0
        tiles_cnt = 0
        inc = 0.0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.territory[r, c] == faction:
                    tiles_cnt += 1
                    inc += self._effective_tile_yield(r, c)
        f_units = [u for u in self.units.values() if u.faction == faction]
        army_cnt = len(f_units)
        upk = sum(u.upkeep for u in f_units)
        if army_cnt > SUPPLY_LIMIT:
            upk += (army_cnt - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT
        net = inc - upk
        return tiles_cnt, round(inc, 4), army_cnt, round(upk, 4), round(net, 4)

    def _compute_faction_total_efficiency(self, faction: int) -> float:
        if faction not in self.factions or not self.factions[faction]["alive"]:
            return 0.0
        f_units = [u for u in self.units.values() if u.faction == faction]
        u_eff = sum(u.efficiency for u in f_units)
        tiles_cnt, inc, _, _, net = self._compute_economy_for_faction(faction)
        return round(u_eff + tiles_cnt * 1.5 + max(0.0, net) * 8.0, 2)

    def _can_build_at(self, faction: int, r: int, c: int, b_type: int) -> bool:
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            return False
        if self.terrain[r, c] == TERRAIN_MOUNTAIN:
            return False
        if (r, c) in self.structs:
            return False
        for fd in self.factions.values():
            if fd["alive"] and fd["castle_pos"] == (r, c):
                return False
        u_on_tile = self.units.get((r, c))
        if u_on_tile and u_on_tile.faction != faction:
            return False
        terr_owner = int(self.territory[r, c])
        if terr_owner != 0 and terr_owner != faction:
            return False
        if b_type == 2:
            for fd in self.factions.values():
                if fd["alive"]:
                    cr, cc = fd["castle_pos"]
                    if abs(r - cr) + abs(c - cc) < 4:
                        return False
        return True

    def _get_valid_build_tiles(self, u: Unit, b_type: int) -> list:
        if not u or u.u_type != "worker" or not u.ready or u.moved or u.attacked:
            return []
        cost = float(STRUCT_STATS[b_type]["cost"])
        if self.factions[u.faction]["gold"] + 1e-9 < cost:
            return []
        valid = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            br, bc = u.r + dr, u.c + dc
            if self._can_build_at(u.faction, br, bc, b_type):
                valid.append([br, bc])
        return valid

    def _find_demolishable_struct(self, u: Unit) -> Optional[Structure]:
        if not u or u.u_type != "worker" or not u.ready or u.moved or u.attacked:
            return None
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = u.r + dr, u.c + dc
            st = self.structs.get((nr, nc))
            if st and st.owner == u.faction:
                return st
        return None

    def _get_valid_moves(self, u: Unit) -> list:
        if not u or not u.ready or u.moved:
            return []
        visited = {(u.r, u.c): 0}
        queue = [(u.r, u.c, 0)]
        valid = []
        while queue:
            cr, cc, dist = queue.pop(0)
            if dist >= u.move_range:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                    continue
                if self.terrain[nr, nc] == TERRAIN_MOUNTAIN:
                    continue
                st = self.structs.get((nr, nc))
                if st and st.owner != u.faction:
                    continue
                if st and st.owner == u.faction and st.s_type == 1:
                    continue
                enemy_castle = False
                for fk, fd in self.factions.items():
                    if fd["alive"] and fk != u.faction and fd["castle_pos"] == (nr, nc):
                        enemy_castle = True
                        break
                if enemy_castle:
                    continue
                other_u = self.units.get((nr, nc))
                if other_u and other_u.faction != u.faction:
                    continue
                if (nr, nc) not in visited or visited[(nr, nc)] > dist + 1:
                    visited[(nr, nc)] = dist + 1
                    queue.append((nr, nc, dist + 1))
                    if other_u is None:
                        valid.append([nr, nc])
        return valid

    def _get_valid_attacks_for_unit(self, u: Unit) -> list:
        if not u or not u.ready or u.attacked:
            return []
        targets = []
        for r in range(max(0, u.r - u.atk_range), min(GRID_SIZE, u.r + u.atk_range + 1)):
            for c in range(max(0, u.c - u.atk_range), min(GRID_SIZE, u.c + u.atk_range + 1)):
                dist = abs(r - u.r) + abs(c - u.c)
                if 1 <= dist <= u.atk_range:
                    other_u = self.units.get((r, c))
                    if other_u and other_u.faction != u.faction:
                        targets.append([r, c])
                        continue
                    st = self.structs.get((r, c))
                    if st and st.owner != u.faction:
                        targets.append([r, c])
                        continue
                    for fk, fd in self.factions.items():
                        if fd["alive"] and fk != u.faction and fd["castle_pos"] == (r, c):
                            targets.append([r, c])
                            break
        return targets

    def _get_valid_attacks_for_tower(self, st: Structure) -> list:
        if not st or st.s_type != 1 or not st.ready:
            return []
        targets = []
        for r in range(max(0, st.r - st.atk_range), min(GRID_SIZE, st.r + st.atk_range + 1)):
            for c in range(max(0, st.c - st.atk_range), min(GRID_SIZE, st.c + st.atk_range + 1)):
                dist = abs(r - st.r) + abs(c - st.c)
                if 1 <= dist <= st.atk_range:
                    other_u = self.units.get((r, c))
                    if other_u and other_u.faction != st.owner:
                        targets.append([r, c])
                        continue
                    other_st = self.structs.get((r, c))
                    if other_st and other_st.owner != st.owner:
                        targets.append([r, c])
                        continue
        return targets

    def _recompute_selection(self, faction: int):
        sel = self.selected_by_faction.get(faction)
        if not sel:
            return
        sr, sc = sel["pos"]
        u = self.units.get((sr, sc))
        st = self.structs.get((sr, sc))
        is_my_turn = (self.current_turn_faction == faction and not self.match_finished)
        if u and u.faction == faction and is_my_turn:
            sel["valid_moves"] = self._get_valid_moves(u)
            sel["valid_attacks"] = self._get_valid_attacks_for_unit(u)
        elif st and st.owner == faction and st.s_type == 1 and is_my_turn:
            sel["valid_moves"] = []
            sel["valid_attacks"] = self._get_valid_attacks_for_tower(st)
        else:
            sel["valid_moves"] = []
            sel["valid_attacks"] = []

    def _record_human_sample_if_valid(self, faction: int, action_idx: int, u_eff: float):
        try:
            obs = encode_state_from_live(self, faction)
            self.match_obs_by_faction[faction].append(obs)
            self.match_acts_by_faction[faction].append(int(action_idx))
            self.match_effs_by_faction[faction].append(float(max(u_eff, MIN_TRAINING_EFFICIENCY)))
        except Exception:
            pass

    def _apply_damage_at(self, attacker_faction: int, tr_pos: int, tc_pos: int, dmg: float, u_attacker: Optional[Unit] = None):
        target_u = self.units.get((tr_pos, tc_pos))
        if target_u and target_u.faction != attacker_faction:
            dealt = min(target_u.hp, float(dmg))
            target_u.hp = round(target_u.hp - dealt, 4)
            if u_attacker:
                u_attacker.dmg_dealt += dealt
            if target_u.hp <= 1e-5:
                del self.units[(tr_pos, tc_pos)]
            return dealt

        target_st = self.structs.get((tr_pos, tc_pos))
        if target_st and target_st.owner != attacker_faction:
            dealt = min(target_st.hp, float(dmg))
            target_st.hp = round(target_st.hp - dealt, 4)
            if u_attacker:
                u_attacker.dmg_dealt += dealt
            if target_st.hp <= 1e-5:
                del self.structs[(tr_pos, tc_pos)]
            return dealt

        for fk, fd in self.factions.items():
            if fd["alive"] and fk != attacker_faction and fd["castle_pos"] == (tr_pos, tc_pos):
                dealt = min(fd["castle_hp"], float(dmg))
                fd["castle_hp"] = round(fd["castle_hp"] - dealt, 4)
                if u_attacker:
                    u_attacker.dmg_dealt += dealt * 1.5
                if fd["castle_hp"] <= 1e-5:
                    fd["castle_hp"] = 0.0
                    self._eliminate_faction(fk)
                return dealt
        return 0.0

    def _eliminate_faction(self, faction: int):
        if faction not in self.factions or not self.factions[faction]["alive"]:
            return
        self.factions[faction]["alive"] = False
        self.factions[faction]["castle_hp"] = 0.0
        self.factions[faction]["gold"] = 0.0

        to_del_u = [pos for pos, u in self.units.items() if u.faction == faction]
        for pos in to_del_u:
            del self.units[pos]

        to_del_s = [pos for pos, st in self.structs.items() if st.owner == faction]
        for pos in to_del_s:
            del self.structs[pos]

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.territory[r, c] == faction:
                    self.territory[r, c] = 0

        alive_factions = [fk for fk, fd in self.factions.items() if fd["alive"]]
        if len(alive_factions) <= 1 and not self.match_finished:
            winner = alive_factions[0] if alive_factions else 1
            self._finalize_match(winner)

    def surrender_faction(self, faction: int, lang: str = "en", is_disconnect: bool = False):
        if self.match_finished or faction not in self.factions or not self.factions[faction]["alive"]:
            return
        p_name = self.factions[faction]["controller"]
        if is_disconnect:
            self._log(tr(lang, "messages", "player_disconnected_surrender", player=p_name, faction=faction))
        else:
            self._log(tr(lang, "messages", "player_surrendered", player=p_name, faction=faction))

        was_current = (self.current_turn_faction == faction)
        self._eliminate_faction(faction)

        if not self.match_finished and was_current:
            self._advance_turn_and_round(lang)

    def apply_player_action(self, faction: int, action_type: str, r=None, c=None, param=None, lang: str = "en") -> dict:
        if action_type == "set_lang":
            return self.get_state(for_faction=faction, lang=str(param or lang))

        if action_type == "surrender":
            self.surrender_faction(faction, lang=lang, is_disconnect=False)
            return self.get_state(for_faction=faction, lang=lang)

        if action_type == "select_tile":
            if r is not None and c is not None and 0 <= int(r) < GRID_SIZE and 0 <= int(c) < GRID_SIZE:
                self.selected_by_faction[faction] = {
                    "pos": [int(r), int(c)],
                    "valid_moves": [],
                    "valid_attacks": []
                }
                self._recompute_selection(faction)
            return self.get_state(for_faction=faction, lang=lang)

        if self.match_finished or self.current_turn_faction != faction or not self.factions.get(faction, {}).get("alive"):
            return self.get_state(for_faction=faction, lang=lang)

        if action_type == "end_turn":
            for u in self.units.values():
                if u.faction == faction and u.ready and not u.moved and not u.attacked and u.hp < u.max_hp:
                    gain = min(u.max_hp - u.hp, u.max_hp * 0.25)
                    u.hp = round(u.hp + gain, 4)
                if u.faction == faction:
                    u.ready = False
            self._advance_turn_and_round(lang)
            return self.get_state(for_faction=faction, lang=lang)

        sel = self.selected_by_faction.get(faction)
        if not sel:
            return self.get_state(for_faction=faction, lang=lang)

        sr, sc = sel["pos"]
        u = self.units.get((sr, sc))
        st = self.structs.get((sr, sc))

        if action_type in ("order_tile", "move"):
            if r is None or c is None:
                return self.get_state(for_faction=faction, lang=lang)
            tr_pos, tc_pos = int(r), int(c)
            self._recompute_selection(faction)
            v_moves = sel.get("valid_moves", [])
            v_atks = sel.get("valid_attacks", [])

            if [tr_pos, tc_pos] in v_atks:
                if u and u.faction == faction and u.ready and not u.attacked:
                    dealt = self._apply_damage_at(faction, tr_pos, tc_pos, u.dmg, u_attacker=u)
                    u.attacked = True
                    u.ready = False
                    self._record_human_sample_if_valid(faction, 2, u.efficiency)
                    self._log(tr(lang, "messages", "player_attack", dmg=dealt, r=tr_pos, c=tc_pos))
                    self._recompute_selection(faction)
                elif st and st.owner == faction and st.s_type == 1 and st.ready:
                    dealt = self._apply_damage_at(faction, tr_pos, tc_pos, st.dmg)
                    st.ready = False
                    self._log(tr(lang, "messages", "player_attack", dmg=dealt, r=tr_pos, c=tc_pos))
                    self._recompute_selection(faction)
                return self.get_state(for_faction=faction, lang=lang)

            if [tr_pos, tc_pos] in v_moves and u and u.faction == faction and u.ready and not u.moved:
                del self.units[(sr, sc)]
                u.r, u.c = tr_pos, tc_pos
                u.moved = True
                if self.territory[tr_pos, tc_pos] != faction:
                    self.territory[tr_pos, tc_pos] = faction
                    u.tiles_captured += 1
                self.units[(tr_pos, tc_pos)] = u
                sel["pos"] = [tr_pos, tc_pos]
                self._recompute_selection(faction)
                if not sel["valid_attacks"]:
                    u.ready = False
                self._record_human_sample_if_valid(faction, 1, u.efficiency)
                u_name = tr(lang, "units", u.u_type)
                self._log(tr(lang, "messages", "player_move", unit=u_name, r=tr_pos, c=tc_pos))
                return self.get_state(for_faction=faction, lang=lang)

        elif action_type == "heal":
            if u and u.faction == faction and u.ready and not u.moved and not u.attacked and u.hp < u.max_hp:
                gain = min(u.max_hp - u.hp, u.max_hp * 0.25)
                u.hp = round(u.hp + gain, 4)
                u.moved = True
                u.attacked = True
                u.ready = False
                self._log(tr(lang, "messages", "player_heal", gained=gain))
                self._recompute_selection(faction)

        elif action_type == "harvest":
            if u and u.faction == faction and u.u_type == "worker" and u.ready and not u.moved and not u.attacked:
                tile_y = self._effective_tile_yield(u.r, u.c)
                if tile_y > 0:
                    bonus = round(tile_y * 1.5, 4)
                    self.factions[faction]["gold"] = round(self.factions[faction]["gold"] + bonus, 4)
                    u.gold_harvested += bonus
                    u.moved = True
                    u.attacked = True
                    u.ready = False
                    self._record_human_sample_if_valid(faction, 3, u.efficiency)
                    self._log(tr(lang, "messages", "player_harvest", bonus=bonus))
                    self._recompute_selection(faction)

        elif action_type == "build":
            if not u or u.faction != faction or u.u_type != "worker" or not u.ready or u.moved or u.attacked:
                return self.get_state(for_faction=faction, lang=lang)
            b_type = 1 if param == "tower" else 2
            cost = float(STRUCT_STATS[b_type]["cost"])
            if self.factions[faction]["gold"] + 1e-9 < cost:
                return self.get_state(for_faction=faction, lang=lang)

            if r is not None and c is not None and 0 <= int(r) < GRID_SIZE and 0 <= int(c) < GRID_SIZE:
                cand_tiles = [(int(r), int(c))]
            else:
                cand_tiles = [
                    (u.r + dr, u.c + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    if 0 <= u.r + dr < GRID_SIZE and 0 <= u.c + dc < GRID_SIZE
                ]

            built_pos = None
            for br, bc in cand_tiles:
                if max(abs(br - u.r), abs(bc - u.c)) > 1:
                    continue
                if not self._can_build_at(faction, br, bc, b_type):
                    continue
                self.factions[faction]["gold"] = round(self.factions[faction]["gold"] - cost, 4)
                self.structs[(br, bc)] = Structure(br, bc, faction, b_type, ready=False)
                self.territory[br, bc] = faction
                u.moved = True
                u.attacked = True
                u.ready = False
                u.builds_done += 1
                built_pos = (br, bc)
                break

            if built_pos is not None:
                self._record_human_sample_if_valid(faction, 4 if b_type == 1 else 5, u.efficiency)
                st_name = tr(lang, "structs", str(b_type))
                self._log(tr(lang, "messages", "player_build", struct=st_name))
                self._recompute_selection(faction)

        elif action_type == "demolish":
            target_st = self._find_demolishable_struct(u)
            if target_st:
                refund = round(target_st.cost * DEMOLISH_REFUND_RATIO, 4)
                st_name = tr(lang, "structs", str(target_st.s_type))
                del self.structs[(target_st.r, target_st.c)]
                self.factions[faction]["gold"] = round(self.factions[faction]["gold"] + refund, 4)
                u.moved = True
                u.attacked = True
                u.ready = False
                self._log(tr(lang, "messages", "player_demolish", struct=st_name, refund=refund))
                self._recompute_selection(faction)

        elif action_type == "disband":
            if u and u.faction == faction:
                refund = round(u.cost * DISBAND_REFUND_RATIO, 4)
                u_name = tr(lang, "units", u.u_type)
                del self.units[(sr, sc)]
                self.factions[faction]["gold"] = round(self.factions[faction]["gold"] + refund, 4)
                self.selected_by_faction.pop(faction, None)
                self._log(tr(lang, "messages", "player_disband", unit=u_name, refund=refund))

        elif action_type == "spawn":
            u_type = str(param or "worker")
            if u_type not in UNIT_STATS:
                return self.get_state(for_faction=faction, lang=lang)
            is_my_castle = (self.factions[faction]["castle_pos"] == (sr, sc))
            is_my_barracks = (st is not None and st.owner == faction and st.s_type == 2)
            if not (is_my_castle or is_my_barracks):
                return self.get_state(for_faction=faction, lang=lang)
            if (sr, sc) in self.units:
                return self.get_state(for_faction=faction, lang=lang)
            if u_type == "worker":
                w_cnt = sum(1 for unit in self.units.values() if unit.faction == faction and unit.u_type == "worker")
                if w_cnt >= MAX_WORKERS_PER_FACTION:
                    return self.get_state(for_faction=faction, lang=lang)
            cost = float(UNIT_STATS[u_type]["cost"])
            if self.factions[faction]["gold"] + 1e-9 >= cost:
                self.factions[faction]["gold"] = round(self.factions[faction]["gold"] - cost, 4)
                self.units[(sr, sc)] = Unit(sr, sc, faction, u_type, ready=False)
                self.territory[sr, sc] = faction
                self._record_human_sample_if_valid(faction, 0, MIN_TRAINING_EFFICIENCY)
                u_name = tr(lang, "units", u_type)
                self._log(tr(lang, "messages", "player_spawn", unit=u_name))
                self._recompute_selection(faction)

        return self.get_state(for_faction=faction, lang=lang)

    def _advance_turn_and_round(self, lang: str = "en"):
        if self.match_finished:
            return
        alive_order = [f for f in self.turn_order if self.factions.get(f, {}).get("alive")]
        if len(alive_order) <= 1:
            winner = alive_order[0] if alive_order else 1
            self._finalize_match(winner)
            return

        curr_f = self.current_turn_faction
        if curr_f in alive_order:
            idx = alive_order.index(curr_f)
            next_idx = (idx + 1) % len(alive_order)
            wrapped_round = (next_idx == 0)
            next_f = alive_order[next_idx]
        else:
            next_f = alive_order[0]
            wrapped_round = False

        self.turn_idx = self.turn_order.index(next_f)

        if wrapped_round:
            self._process_global_round_end(lang)
            if self.match_finished:
                return

        self._prepare_faction_turn(self.current_turn_faction)
        self._check_and_run_ai_turns(lang)

    def _prepare_faction_turn(self, faction: int):
        for u in self.units.values():
            if u.faction == faction:
                u.moved = False
                u.attacked = False
                u.ready = True
                u.turns_alive += 1
        for st in self.structs.values():
            if st.owner == faction:
                st.ready = True

    def _process_global_round_end(self, lang: str = "en"):
        self.turn += 1
        for f, fd in self.factions.items():
            if not fd["alive"]:
                continue
            _, _, _, _, net = self._compute_economy_for_faction(f)
            fd["gold"] = round(fd["gold"] + net, 4)
            if fd["gold"] < -1e-5:
                fd["gold"] = 0.0
                to_del = [pos for pos, u in self.units.items() if u.faction == f]
                for pos in to_del:
                    del self.units[pos]
                self._log(tr(lang, "messages", "bankruptcy_player"))

        self.stats_turns.append(int(self.turn))
        for fk in self.stats_factions_eff.keys():
            f_int = int(fk)
            self.stats_factions_eff[fk].append(self._compute_faction_total_efficiency(f_int))

        if self.turn >= self.max_turns and not self.match_finished:
            best_f = max(
                [fk for fk, fd in self.factions.items() if fd["alive"]],
                key=lambda fk: (self.factions[fk]["castle_hp"], self._compute_faction_total_efficiency(fk)),
                default=1
            )
            self._finalize_match(best_f)

    def _check_and_run_ai_turns(self, lang: str = "en"):
        safety = 0
        while not self.match_finished and safety < 8:
            safety += 1
            curr_f = self.current_turn_faction
            fd = self.factions.get(curr_f)
            if not fd or not fd["alive"]:
                self._advance_turn_and_round(lang)
                break
            if fd["kind"] != "ai":
                break
            self._prepare_faction_turn(curr_f)
            self._execute_ai_turn(curr_f, lang)
            if self.match_finished:
                break
            self._advance_turn_and_round(lang)
            break

    def _execute_ai_turn(self, ai_faction: int, lang: str = "en"):
        self.ai_traces = []
        moves_cnt = 0
        atks_cnt = 0

        for pos, st in list(self.structs.items()):
            if st.owner == ai_faction and st.s_type == 1 and st.ready:
                targets = self._get_valid_attacks_for_tower(st)
                if targets:
                    tr_pos, tc_pos = targets[0]
                    dealt = self._apply_damage_at(ai_faction, tr_pos, tc_pos, st.dmg)
                    st.ready = False
                    atks_cnt += 1
                    self.ai_traces.append({"from": [st.r, st.c], "to": [tr_pos, tc_pos], "kind": "attack"})
                    self._log(tr(lang, "messages", "ai_tower_shot", r=tr_pos, c=tc_pos, dmg=dealt))
                    if self.match_finished:
                        return

        ai_units = [u for u in self.units.values() if u.faction == ai_faction and u.ready]
        for u in ai_units:
            if self.match_finished:
                return
            if (u.r, u.c) not in self.units:
                continue

            atks = self._get_valid_attacks_for_unit(u)
            if atks:
                tr_pos, tc_pos = atks[0]
                dealt = self._apply_damage_at(ai_faction, tr_pos, tc_pos, u.dmg, u_attacker=u)
                u.attacked = True
                u.ready = False
                atks_cnt += 1
                self.ai_traces.append({"from": [u.r, u.c], "to": [tr_pos, tc_pos], "kind": "attack"})
                continue

            if u.u_type == "worker":
                tile_y = self._effective_tile_yield(u.r, u.c)
                if tile_y >= 0.15 and random.random() < 0.55:
                    bonus = round(tile_y * 1.5, 4)
                    self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] + bonus, 4)
                    u.gold_harvested += bonus
                    u.moved = True
                    u.attacked = True
                    u.ready = False
                    continue

                if self.factions[ai_faction]["gold"] >= 6.0 and random.random() < 0.30:
                    b_type = 2 if (self.factions[ai_faction]["gold"] >= 7.5 and random.random() < 0.45) else 1
                    b_tiles = self._get_valid_build_tiles(u, b_type)
                    if b_tiles:
                        br, bc = b_tiles[0]
                        cost = float(STRUCT_STATS[b_type]["cost"])
                        self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] - cost, 4)
                        self.structs[(br, bc)] = Structure(br, bc, ai_faction, b_type, ready=False)
                        self.territory[br, bc] = ai_faction
                        u.moved = True
                        u.attacked = True
                        u.ready = False
                        u.builds_done += 1
                        continue

            v_moves = self._get_valid_moves(u)
            if v_moves:
                enemy_castles = [fd["castle_pos"] for fk, fd in self.factions.items() if fd["alive"] and fk != ai_faction]
                target_goal = enemy_castles[0] if enemy_castles else (GRID_SIZE // 2, GRID_SIZE // 2)

                best_m = min(
                    v_moves,
                    key=lambda m: (
                        abs(m[0] - target_goal[0]) + abs(m[1] - target_goal[1])
                        - (2.5 if self.territory[m[0], m[1]] != ai_faction else 0.0)
                        - self._effective_tile_yield(m[0], m[1]) * 4.0
                    )
                )
                old_r, old_c = u.r, u.c
                nr, nc = best_m
                del self.units[(old_r, old_c)]
                u.r, u.c = nr, nc
                u.moved = True
                if self.territory[nr, nc] != ai_faction:
                    self.territory[nr, nc] = ai_faction
                    u.tiles_captured += 1
                self.units[(nr, nc)] = u
                moves_cnt += 1
                self.ai_traces.append({"from": [old_r, old_c], "to": [nr, nc], "kind": "move"})

                post_atks = self._get_valid_attacks_for_unit(u)
                if post_atks:
                    tr_pos, tc_pos = post_atks[0]
                    dealt = self._apply_damage_at(ai_faction, tr_pos, tc_pos, u.dmg, u_attacker=u)
                    u.attacked = True
                    atks_cnt += 1
                    self.ai_traces.append({"from": [nr, nc], "to": [tr_pos, tc_pos], "kind": "attack"})
                u.ready = False

        spawners = [self.factions[ai_faction]["castle_pos"]] + [
            (st.r, st.c) for st in self.structs.values() if st.owner == ai_faction and st.s_type == 2
        ]
        w_cnt = sum(1 for unit in self.units.values() if unit.faction == ai_faction and unit.u_type == "worker")
        for sp_r, sp_c in spawners:
            if (sp_r, sp_c) in self.units:
                continue
            gold = self.factions[ai_faction]["gold"]
            choice = None
            if w_cnt < 2 and gold >= UNIT_STATS["worker"]["cost"]:
                choice = "worker"
                w_cnt += 1
            elif gold >= UNIT_STATS["archer"]["cost"] and random.random() < 0.45:
                choice = "archer"
            elif gold >= UNIT_STATS["warrior"]["cost"]:
                choice = "warrior"
            elif w_cnt < MAX_WORKERS_PER_FACTION and gold >= UNIT_STATS["worker"]["cost"] and random.random() < 0.35:
                choice = "worker"
                w_cnt += 1

            if choice:
                c_val = float(UNIT_STATS[choice]["cost"])
                self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] - c_val, 4)
                self.units[(sp_r, sp_c)] = Unit(sp_r, sp_c, ai_faction, choice, ready=False)
                self.territory[sp_r, sp_c] = ai_faction

        self._log(tr(lang, "messages", "ai_turn_summary", turn=self.turn, moves=moves_cnt, atks=atks_cnt))

    def _finalize_match(self, winner_faction: int):
        if self.match_finished:
            return
        self.match_finished = True
        self.winner_faction = int(winner_faction)
        w_fd = self.factions.get(self.winner_faction)
        self.winner_name = w_fd["controller"] if w_fd else f"Faction #{self.winner_faction}"

        for fk, fd in self.factions.items():
            if fd["kind"] != "human":
                continue
            uname = fd["controller"]
            prof = UserProfileManager.get_profile(uname)
            prof["matches"] = int(prof.get("matches", 0)) + 1
            won = (fk == self.winner_faction)
            if won:
                prof["wins"] = int(prof.get("wins", 0)) + 1
                prof["base_rating"] = round(float(prof.get("base_rating", PLAYER_INITIAL_RATING)) + 4.5, 2)
            else:
                prof["base_rating"] = round(max(10.0, float(prof.get("base_rating", PLAYER_INITIAL_RATING)) - 2.0), 2)
            UserProfileManager.update_profile(uname, prof)

            m_obs = self.match_obs_by_faction.get(fk, [])
            m_acts = self.match_acts_by_faction.get(fk, [])
            m_effs = self.match_effs_by_faction.get(fk, [])
            if m_obs:
                self.is_training_new_gen = True
                self.last_evolution_report = {"status": "training", "samples": len(m_obs)}

                def _bg_evolve(u_name=uname, p_rating=prof["base_rating"], obs_l=m_obs, acts_l=m_acts, effs_l=m_effs):
                    try:
                        entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(
                            username=u_name,
                            player_rating=p_rating,
                            match_obs=np.array(obs_l, dtype=np.float32),
                            match_actions=np.array(acts_l, dtype=np.int64),
                            match_efficiencies=np.array(effs_l, dtype=np.float32)
                        )
                        if entry:
                            self.active_gen_label = entry.get("label", self.active_gen_label)
                            self.active_rating = float(entry.get("rating", self.active_rating))
                            self.last_evolution_report = {
                                "status": "completed",
                                "new_gen_label": entry.get("label", "Gen-New"),
                                "new_gen_rating": float(entry.get("rating", p_rating))
                            }
                    except Exception as e:
                        print(f"[Evolution] Error: {e}")
                    finally:
                        self.is_training_new_gen = False

                threading.Thread(target=_bg_evolve, daemon=True).start()

    def _build_inspector(self, faction: int, lang: str) -> dict:
        sel = self.selected_by_faction.get(faction)
        res = {
            "lines": [tr(lang, "messages", "nothing_selected"), tr(lang, "messages", "click_prompt")],
            "can_heal": False,
            "can_harvest": False,
            "can_build_tower": False,
            "can_build_barracks": False,
            "can_demolish": False,
            "can_disband": False,
            "can_spawn_worker": False,
            "can_spawn_warrior": False,
            "can_spawn_archer": False,
            "disband_refund_val": 0.0,
            "demolish_refund_val": 0.0,
            "valid_tower_tiles": [],
            "valid_barracks_tiles": []
        }
        if not sel:
            return res

        sr, sc = sel["pos"]
        lines = []
        t_owner = int(self.territory[sr, sc])
        if t_owner == 0:
            owner_str = tr(lang, "messages", "owner_neutral")
        elif t_owner == faction:
            owner_str = tr(lang, "messages", "owner_player")
        else:
            owner_str = f"F{t_owner}"

        eff_y = self._effective_tile_yield(sr, sc)
        base_y = float(BASE_TILE_YIELDS.get(int(self.terrain[sr, sc]), 0.0))
        if (sr, sc) in self.structs or any(fd["alive"] and fd["castle_pos"] == (sr, sc) for fd in self.factions.values()):
            lines.append(tr(lang, "messages", "land_built", base=base_y, owner=owner_str))
        else:
            lines.append(tr(lang, "messages", "land_free", eff=eff_y, owner=owner_str))

        u = self.units.get((sr, sc))
        st = self.structs.get((sr, sc))
        my_gold = self.factions.get(faction, {}).get("gold", 0.0)

        if u:
            u_name = tr(lang, "units", u.u_type)
            st_key = "status_ready" if (u.ready and not u.moved and not u.attacked) else ("status_can_attack" if (u.ready and not u.attacked) else "status_moved")
            lines.append(tr(lang, "messages", "unit_header", name=u_name, faction=f"F{u.faction}", status=tr(lang, "messages", st_key)))
            lines.append(tr(lang, "messages", "hp_line", hp=u.hp, max_hp=u.max_hp, pct=(u.hp / u.max_hp) * 100.0))
            lines.append(tr(lang, "messages", "stats_line", move=u.move_range, atk=u.atk_range, dmg=u.dmg))
            lines.append(tr(lang, "messages", "upkeep_line", upk=u.upkeep, eff=u.efficiency))
            u_ref = round(u.cost * DISBAND_REFUND_RATIO, 2)
            lines.append(tr(lang, "messages", "unit_refund_line", refund=u_ref, pct=DISBAND_REFUND_RATIO * 100.0, cost=u.cost))

            if u.faction == faction:
                res["disband_refund_val"] = u_ref
                res["can_disband"] = True
                if u.ready and not u.moved and not u.attacked:
                    if u.hp < u.max_hp:
                        res["can_heal"] = True
                    if u.u_type == "worker":
                        if eff_y > 0:
                            res["can_harvest"] = True
                        tow_tiles = self._get_valid_build_tiles(u, 1)
                        bar_tiles = self._get_valid_build_tiles(u, 2)
                        res["valid_tower_tiles"] = tow_tiles
                        res["valid_barracks_tiles"] = bar_tiles
                        res["can_build_tower"] = len(tow_tiles) > 0
                        res["can_build_barracks"] = len(bar_tiles) > 0
                        dem_st = self._find_demolishable_struct(u)
                        if dem_st:
                            res["can_demolish"] = True
                            res["demolish_refund_val"] = round(dem_st.cost * DEMOLISH_REFUND_RATIO, 2)

        is_my_castle = (self.factions.get(faction, {}).get("castle_pos") == (sr, sc) and self.factions.get(faction, {}).get("alive"))
        is_my_barracks = (st is not None and st.owner == faction and st.s_type == 2)

        if is_my_castle:
            c_hp = self.factions[faction]["castle_hp"]
            lines.append(tr(lang, "messages", "your_castle", hp=c_hp, max_hp=CASTLE_MAX_HP))
        elif st:
            st_name = tr(lang, "structs", str(st.s_type))
            lines.append(tr(lang, "messages", "struct_line", name=st_name, hp=st.hp, max_hp=st.max_hp))
            s_ref = round(st.cost * DEMOLISH_REFUND_RATIO, 2)
            lines.append(tr(lang, "messages", "struct_refund_line", refund=s_ref, pct=DEMOLISH_REFUND_RATIO * 100.0, cost=st.cost))

        if (is_my_castle or is_my_barracks) and u is None:
            w_cnt = sum(1 for unit in self.units.values() if unit.faction == faction and unit.u_type == "worker")
            res["can_spawn_worker"] = (w_cnt < MAX_WORKERS_PER_FACTION and my_gold + 1e-9 >= UNIT_STATS["worker"]["cost"])
            res["can_spawn_warrior"] = (my_gold + 1e-9 >= UNIT_STATS["warrior"]["cost"])
            res["can_spawn_archer"] = (my_gold + 1e-9 >= UNIT_STATS["archer"]["cost"])

        res["lines"] = lines
        return res

    def get_state(self, for_faction: int, username: str = "", lang: str = "en") -> dict:
        self._recompute_selection(for_faction)
        sel = self.selected_by_faction.get(for_faction)

        eff_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                eff_grid[r, c] = self._effective_tile_yield(r, c)

        factions_out = {}
        for fk, fd in self.factions.items():
            tiles_cnt, inc, army_cnt, upk, net = self._compute_economy_for_faction(fk)
            factions_out[str(fk)] = {
                "controller": fd["controller"],
                "kind": fd["kind"],
                "gold": round(fd["gold"], 2),
                "tiles": tiles_cnt,
                "inc": round(inc, 2),
                "army": army_cnt,
                "upk": round(upk, 2),
                "net": round(net, 2),
                "castle_hp": round(fd["castle_hp"], 2),
                "castle_pos": list(fd["castle_pos"]),
                "alive": bool(fd["alive"])
            }

        units_out = [
            {
                "r": u.r, "c": u.c, "faction": u.faction, "type": u.u_type,
                "hp": round(u.hp, 2), "max_hp": round(u.max_hp, 2),
                "moved": u.moved, "attacked": u.attacked, "ready": u.ready,
                "efficiency": round(u.efficiency, 2)
            }
            for u in self.units.values()
        ]

        structs_out = [
            {
                "r": st.r, "c": st.c, "owner": st.owner, "type": st.s_type,
                "hp": round(st.hp, 2), "max_hp": round(st.max_hp, 2), "ready": st.ready
            }
            for st in self.structs.values()
        ]

        factions_median = {}
        for fk, arr in self.stats_factions_eff.items():
            factions_median[fk] = round(float(np.median(arr)), 2) if arr else 0.0

        prof = UserProfileManager.get_profile(username) if username else {"base_rating": PLAYER_INITIAL_RATING, "matches": 0, "wins": 0}
        matches = int(prof.get("matches", 0))
        wins = int(prof.get("wins", 0))
        winrate = (wins / matches * 100.0) if matches > 0 else 0.0

        curr_f = self.current_turn_faction
        curr_user = self.factions.get(curr_f, {}).get("controller", f"F{curr_f}")

        return clean_for_json({
            "screen": "game",
            "username": username,
            "lang": lang,
            "base_rating": round(float(prof.get("base_rating", PLAYER_INITIAL_RATING)), 2),
            "winrate": round(winrate, 2),
            "turn": self.turn,
            "max_turns": self.max_turns,
            "my_faction": for_faction,
            "my_alive": self.factions.get(for_faction, {}).get("alive", False),
            "current_turn_faction": curr_f,
            "current_turn_user": curr_user,
            "is_my_turn": (curr_f == for_faction and not self.match_finished),
            "match_finished": self.match_finished,
            "winner_faction": self.winner_faction,
            "winner_name": self.winner_name,
            "active_gen_label": self.active_gen_label,
            "active_rating": round(self.active_rating, 2),
            "is_training_new_gen": self.is_training_new_gen,
            "last_evolution_report": self.last_evolution_report,
            "factions": factions_out,
            "terrain": self.terrain.tolist(),
            "territory": self.territory.tolist(),
            "eff_yields": np.round(eff_grid, 2).tolist(),
            "units": units_out,
            "structs": structs_out,
            "selected_pos": sel["pos"] if sel else None,
            "valid_moves": sel["valid_moves"] if sel else [],
            "valid_attacks": sel["valid_attacks"] if sel else [],
            "ai_traces": self.ai_traces,
            "event_log": self.event_log,
            "inspector": self._build_inspector(for_faction, lang),
            "stats": {
                "turns": self.stats_turns,
                "factions_eff": self.stats_factions_eff,
                "factions_median": factions_median,
                "player_eff": self.stats_factions_eff.get(str(for_faction), [0.0]),
                "ai_eff": self.stats_factions_eff.get("1", [0.0]),
                "player_median": factions_median.get(str(for_faction), 0.0),
                "ai_median": factions_median.get("1", 0.0)
            },
            "config": {
                "min_training_eff": MIN_TRAINING_EFFICIENCY,
                "disband_ratio": DISBAND_REFUND_RATIO,
                "demolish_ratio": DEMOLISH_REFUND_RATIO,
                "unit_stats": UNIT_STATS,
                "struct_stats": {str(k): v for k, v in STRUCT_STATS.items()}
            },
            "ui_text": load_locale(lang).get("ui", {})
        })


class Room:
    def __init__(self, room_id: str, name: str, host: str, max_players: int, is_private: bool = False, password: str = "", fill_ai: bool = True):
        self.room_id = room_id
        self.name = name or f"Room #{room_id[:4]}"
        self.host = host
        self.max_players = 4 if int(max_players) == 4 else 2
        self.is_private = bool(is_private)
        self.password = str(password or "")
        self.status = "waiting"
        self.session: Optional[GameSession] = None
        self.last_seen = {host: time.time()}
        self.slots = []

        host_prof = UserProfileManager.get_profile(host)
        host_rating = float(host_prof.get("base_rating", PLAYER_INITIAL_RATING))

        for f in range(1, self.max_players + 1):
            if f == 2:
                self.slots.append({"faction": f, "kind": "human", "username": host, "ai_rating": 0.0, "ai_entry": None})
            elif fill_ai:
                ai_entry = PoolRegistry.select_opponent_for_rating(host_rating)
                ai_lbl = f"AI [{ai_entry.get('label', 'Gen-Base')}]"
                self.slots.append({
                    "faction": f,
                    "kind": "ai",
                    "username": ai_lbl,
                    "ai_rating": float(ai_entry.get("rating", PLAYER_INITIAL_RATING)),
                    "ai_entry": ai_entry
                })
            else:
                self.slots.append({"faction": f, "kind": "empty", "username": "", "ai_rating": 0.0, "ai_entry": None})


class RoomManager:
    rooms: dict[str, Room] = {}
    user_to_room: dict[str, str] = {}

    @staticmethod
    def touch_user(username: str):
        rid = RoomManager.user_to_room.get(username)
        if rid and rid in RoomManager.rooms:
            RoomManager.rooms[rid].last_seen[username] = time.time()

    @staticmethod
    def check_disconnects():
        now = time.time()
        for rid, rm in list(RoomManager.rooms.items()):
            for s in rm.slots:
                if s["kind"] == "human":
                    u = s["username"]
                    last_t = rm.last_seen.get(u, now)
                    if now - last_t > DISCONNECT_TIMEOUT_SECONDS:
                        RoomManager._remove_user_from_room_unlocked(u, is_disconnect=True)

    @staticmethod
    def _remove_user_from_room_unlocked(username: str, is_disconnect: bool = False):
        rid = RoomManager.user_to_room.pop(username, None)
        if not rid or rid not in RoomManager.rooms:
            return
        rm = RoomManager.rooms[rid]
        rm.last_seen.pop(username, None)

        user_faction = None
        for s in rm.slots:
            if s["kind"] == "human" and s["username"] == username:
                user_faction = s["faction"]
                s["kind"] = "empty"
                s["username"] = ""
                break

        if rm.status == "playing" and rm.session and user_faction is not None:
            rm.session.surrender_faction(user_faction, is_disconnect=is_disconnect)
            if rm.session.match_finished:
                rm.status = "finished"

        remaining_humans = [s["username"] for s in rm.slots if s["kind"] == "human" and s["username"]]
        if not remaining_humans:
            RoomManager.rooms.pop(rid, None)
        elif rm.host == username:
            rm.host = remaining_humans[0]

    @staticmethod
    def build_user_state(username: str, lang: str = "en") -> dict:
        RoomManager.check_disconnects()
        RoomManager.touch_user(username)
        prof = UserProfileManager.get_profile(username)
        lang_code = lang or prof.get("lang", "en")

        rid = RoomManager.user_to_room.get(username)
        if not rid or rid not in RoomManager.rooms:
            rooms_list = []
            for r_id, rm in RoomManager.rooms.items():
                filled = sum(1 for s in rm.slots if s["kind"] != "empty")
                rooms_list.append({
                    "room_id": r_id,
                    "name": rm.name,
                    "max_players": rm.max_players,
                    "filled_slots": filled,
                    "is_private": rm.is_private,
                    "host": rm.host,
                    "status": rm.status
                })
            return clean_for_json({
                "screen": "lobby",
                "username": username,
                "lang": lang_code,
                "base_rating": round(float(prof.get("base_rating", PLAYER_INITIAL_RATING)), 2),
                "rooms": rooms_list,
                "ui_text": load_locale(lang_code).get("ui", {})
            })

        rm = RoomManager.rooms[rid]
        my_slot = next((s for s in rm.slots if s["kind"] == "human" and s["username"] == username), None)
        my_faction = my_slot["faction"] if my_slot else 2

        if rm.status == "playing" and rm.session:
            st = rm.session.get_state(for_faction=my_faction, username=username, lang=lang_code)
            if rm.session.match_finished:
                rm.status = "finished"
            return st

        filled_cnt = sum(1 for s in rm.slots if s["kind"] != "empty")
        return clean_for_json({
            "screen": "room_lobby",
            "username": username,
            "lang": lang_code,
            "base_rating": round(float(prof.get("base_rating", PLAYER_INITIAL_RATING)), 2),
            "is_host": (rm.host == username),
            "can_start": (filled_cnt >= 2),
            "room": {
                "room_id": rm.room_id,
                "name": rm.name,
                "max_players": rm.max_players,
                "host": rm.host,
                "status": rm.status,
                "slots": [
                    {
                        "faction": s["faction"],
                        "kind": s["kind"],
                        "username": s["username"],
                        "ai_rating": round(float(s.get("ai_rating", 0.0)), 2)
                    }
                    for s in rm.slots
                ]
            },
            "ui_text": load_locale(lang_code).get("ui", {})
        })


class AuthPayload(BaseModel):
    mode: str
    username: Optional[str] = ""
    password: Optional[str] = ""
    lang: Optional[str] = "en"


class RoomActionPayload(BaseModel):
    username: str
    action: str
    room_id: Optional[str] = None
    room_name: Optional[str] = ""
    max_players: Optional[int] = 2
    is_private: Optional[bool] = False
    password: Optional[str] = ""
    fill_ai: Optional[bool] = True
    slot_faction: Optional[int] = None


class ActionPayload(BaseModel):
    username: str
    action_type: str
    r: Optional[int] = None
    c: Optional[int] = None
    param: Optional[str] = None


@app.on_event("startup")
def on_server_startup():
    try:
        from cloud_backup import CloudBackupManager
        CloudBackupManager.restore_from_github()
    except Exception as e:
        print(f"[Startup] Backup restore skipped: {e}")
    PoolRegistry.ensure_initialized()


@app.get("/api/init_user")
def api_init_user():
    guest_name = UserProfileManager.generate_guest_name()
    return {
        "suggested_guest": guest_name,
        "lang": "en",
        "ui_text": load_locale("en").get("ui", {})
    }


@app.post("/api/auth")
def api_auth(payload: AuthPayload):
    ok, err, prof = UserProfileManager.authenticate(
        payload.mode, payload.username or "", payload.password or "", payload.lang or "en"
    )
    if not ok:
        return {"ok": False, "error": err}
    with ROOMS_LOCK:
        st = RoomManager.build_user_state(prof["username"], lang=prof.get("lang", "en"))
    return {"ok": True, "state": st}


@app.post("/api/rooms")
def api_rooms(payload: RoomActionPayload):
    uname = payload.username.strip()
    prof = UserProfileManager.get_profile(uname)
    lang = prof.get("lang", "en")

    with ROOMS_LOCK:
        act = payload.action
        if act == "create":
            RoomManager._remove_user_from_room_unlocked(uname)
            rid = uuid.uuid4().hex[:8]
            rm = Room(
                room_id=rid,
                name=payload.room_name or f"{uname}'s Battle",
                host=uname,
                max_players=payload.max_players or 2,
                is_private=bool(payload.is_private),
                password=payload.password or "",
                fill_ai=bool(payload.fill_ai)
            )
            RoomManager.rooms[rid] = rm
            RoomManager.user_to_room[uname] = rid

        elif act == "quick_bot_match":
            RoomManager._remove_user_from_room_unlocked(uname)
            rid = uuid.uuid4().hex[:8]
            rm = Room(
                room_id=rid,
                name=f"{uname} vs AI",
                host=uname,
                max_players=payload.max_players or 2,
                is_private=False,
                fill_ai=True
            )
            rm.status = "playing"
            rm.session = GameSession(rid, rm.max_players, rm.slots)
            RoomManager.rooms[rid] = rm
            RoomManager.user_to_room[uname] = rid

        elif act == "join":
            target_rid = payload.room_id
            if not target_rid or target_rid not in RoomManager.rooms:
                return {"ok": False, "error": "Room not found."}
            rm = RoomManager.rooms[target_rid]
            if rm.is_private and rm.password != (payload.password or ""):
                return {"ok": False, "error": "Incorrect room password."}
            empty_slot = next((s for s in rm.slots if s["kind"] == "empty"), None)
            if not empty_slot:
                ai_slot = next((s for s in rm.slots if s["kind"] == "ai"), None)
                if ai_slot and rm.status == "waiting":
                    empty_slot = ai_slot
            if not empty_slot:
                return {"ok": False, "error": "Room is full."}
            RoomManager._remove_user_from_room_unlocked(uname)
            empty_slot["kind"] = "human"
            empty_slot["username"] = uname
            empty_slot["ai_entry"] = None
            rm.last_seen[uname] = time.time()
            RoomManager.user_to_room[uname] = target_rid

        elif act == "leave":
            RoomManager._remove_user_from_room_unlocked(uname, is_disconnect=False)

        elif act == "switch_slot":
            rid = RoomManager.user_to_room.get(uname)
            if rid and rid in RoomManager.rooms:
                rm = RoomManager.rooms[rid]
                target_f = int(payload.slot_faction or 0)
                target_s = next((s for s in rm.slots if s["faction"] == target_f), None)
                curr_s = next((s for s in rm.slots if s["kind"] == "human" and s["username"] == uname), None)
                if target_s and target_s["kind"] == "empty" and curr_s:
                    curr_s["kind"] = "empty"
                    curr_s["username"] = ""
                    target_s["kind"] = "human"
                    target_s["username"] = uname

        elif act == "add_ai":
            rid = RoomManager.user_to_room.get(uname)
            if rid and rid in RoomManager.rooms:
                rm = RoomManager.rooms[rid]
                if rm.host == uname:
                    target_f = int(payload.slot_faction or 0)
                    target_s = next((s for s in rm.slots if s["faction"] == target_f), None)
                    if target_s and target_s["kind"] == "empty":
                        ai_entry = PoolRegistry.select_opponent_for_rating(float(prof.get("base_rating", PLAYER_INITIAL_RATING)))
                        target_s["kind"] = "ai"
                        target_s["username"] = f"AI [{ai_entry.get('label', 'Gen-Base')}]"
                        target_s["ai_rating"] = float(ai_entry.get("rating", PLAYER_INITIAL_RATING))
                        target_s["ai_entry"] = ai_entry

        elif act == "kick_slot":
            rid = RoomManager.user_to_room.get(uname)
            if rid and rid in RoomManager.rooms:
                rm = RoomManager.rooms[rid]
                if rm.host == uname:
                    target_f = int(payload.slot_faction or 0)
                    target_s = next((s for s in rm.slots if s["faction"] == target_f), None)
                    if target_s and target_s["kind"] != "empty":
                        if target_s["kind"] == "human":
                            kicked_u = target_s["username"]
                            RoomManager.user_to_room.pop(kicked_u, None)
                            rm.last_seen.pop(kicked_u, None)
                        target_s["kind"] = "empty"
                        target_s["username"] = ""
                        target_s["ai_entry"] = None

        elif act == "start":
            rid = RoomManager.user_to_room.get(uname)
            if rid and rid in RoomManager.rooms:
                rm = RoomManager.rooms[rid]
                if rm.host == uname:
                    rm.status = "playing"
                    rm.session = GameSession(rid, rm.max_players, rm.slots)

        st = RoomManager.build_user_state(uname, lang=lang)
    return {"ok": True, "state": st}


@app.post("/api/state")
def api_state(payload: ActionPayload):
    uname = payload.username.strip()
    prof = UserProfileManager.get_profile(uname)
    lang = payload.param or prof.get("lang", "en")
    with ROOMS_LOCK:
        return RoomManager.build_user_state(uname, lang=lang)


@app.post("/api/action")
def api_action(payload: ActionPayload):
    uname = payload.username.strip()
    prof = UserProfileManager.get_profile(uname)
    lang = prof.get("lang", "en")

    if payload.action_type == "set_lang" and payload.param:
        lang = str(payload.param)
        prof["lang"] = lang
        UserProfileManager.update_profile(uname, prof)

    with ROOMS_LOCK:
        RoomManager.touch_user(uname)
        rid = RoomManager.user_to_room.get(uname)
        if not rid or rid not in RoomManager.rooms:
            return RoomManager.build_user_state(uname, lang=lang)
        rm = RoomManager.rooms[rid]
        if rm.status != "playing" or not rm.session:
            return RoomManager.build_user_state(uname, lang=lang)
        my_slot = next((s for s in rm.slots if s["kind"] == "human" and s["username"] == uname), None)
        my_faction = my_slot["faction"] if my_slot else 2
        st = rm.session.apply_player_action(
            faction=my_faction,
            action_type=payload.action_type,
            r=payload.r,
            c=payload.c,
            param=payload.param,
            lang=lang
        )
        if rm.session.match_finished:
            rm.status = "finished"
        return st


@app.post("/api/leave_beacon")
async def api_leave_beacon(request: Request):
    try:
        data = await request.json()
        uname = str(data.get("username", "")).strip()
        if uname:
            with ROOMS_LOCK:
                RoomManager._remove_user_from_room_unlocked(uname, is_disconnect=True)
    except Exception:
        pass
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin/export_models")
def export_models_archive():
    backup_path = os.path.join(POOL_DIR, "models_backup.zip")
    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(POOL_DIR):
            if fname.endswith(".zip") and fname != "models_backup.zip":
                zf.write(os.path.join(POOL_DIR, fname), arcname=fname)
            elif fname in ["pool_registry.json", "users_db.json", "human_telemetry.npz"]:
                zf.write(os.path.join(POOL_DIR, fname), arcname=fname)
    return FileResponse(backup_path, filename="microrts_models_backup.zip", media_type="application/zip")


if __name__ == "__main__":
    PoolRegistry.ensure_initialized()
    uvicorn.run(app, host="0.0.0.0", port=8000)