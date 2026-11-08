import os
import json
import time
import uuid
import hashlib
import threading
import zipfile
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np

from game_config import (
    UNIT_STATS, STRUCT_STATS, CASTLE_MAX_HP, MAX_STEPS,
    MAX_GOLD, SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT,
    MAX_WORKERS_PER_FACTION, RELATIVE_OFFSETS, USERS_DB_FILE, POOL_DIR,
    MIN_TRAINING_EFFICIENCY, DISBAND_REFUND_RATIO, DEMOLISH_REFUND_RATIO
)
from localization import (
    normalize_lang, get_text, get_unit_name, get_struct_name, get_ui_bundle
)
from microrts_env import MicroRTSEnv, PoolRegistry, OnlineEvolutionEngine

app = FastAPI(title="Faction Warfare RTS — Multi-User Evolutionary Platform")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

USERS_LOCK = threading.Lock()
ROOMS_LOCK = threading.RLock()
ROOMS = {}
USER_ACTIVE_ROOM = {}
DISCONNECT_TIMEOUT_SECONDS = 65.0

FACTION_META = {
    1: {"key": "red", "name_en": "Red Empire (Bottom-Left)", "name_uk": "Червона Імперія (Лівий Низ)", "color": "#eb3b5a"},
    2: {"key": "blue", "name_en": "Blue Alliance (Top-Right)", "name_uk": "Синій Альянс (Правий Верх)", "color": "#0fbcf9"},
    3: {"key": "green", "name_en": "Green Order (Top-Left)", "name_uk": "Зелений Орден (Лівий Верх)", "color": "#2ecc71"},
    4: {"key": "yellow", "name_en": "Golden Dynasty (Bottom-Right)", "name_uk": "Золота Династія (Правий Низ)", "color": "#feca57"}
}

def get_faction_display_name(faction: int, lang: str = "en") -> str:
    meta = FACTION_META.get(int(faction), FACTION_META[1])
    return meta["name_uk"] if normalize_lang(lang) == "uk" else meta["name_en"]

def hash_password(raw_password: str) -> str:
    return hashlib.sha256(str(raw_password).encode("utf-8")).hexdigest()

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

class UserProfileManager:
    @staticmethod
    def _load_unlocked() -> dict:
        os.makedirs(POOL_DIR, exist_ok=True)
        if os.path.exists(USERS_DB_FILE):
            try:
                with open(USERS_DB_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _save_unlocked(db: dict):
        os.makedirs(POOL_DIR, exist_ok=True)
        tmp_file = USERS_DB_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(clean_for_json(db), f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, USERS_DB_FILE)

    @staticmethod
    def load_all() -> dict:
        with USERS_LOCK:
            return UserProfileManager._load_unlocked()

    @staticmethod
    def create_guest_for_ip(ip: str, lang: str = "en") -> dict:
        norm_l = normalize_lang(lang)
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            ip_map = db.get("__ip_map__", {})
            if ip in ip_map and ip_map[ip] in db:
                prof = db[ip_map[ip]]
                if "lang" not in prof:
                    prof["lang"] = norm_l
                UserProfileManager._save_unlocked(db)
                return prof

            counter = int(db.get("__guest_counter__", 0)) + 1
            guest_name = f"Guest{counter}"
            while guest_name in db:
                counter += 1
                guest_name = f"Guest{counter}"

            ip_map[ip] = guest_name
            db["__ip_map__"] = ip_map
            db["__guest_counter__"] = counter

            prof = {
                "username": guest_name,
                "is_registered": False,
                "password_hash": "",
                "lang": norm_l,
                "base_rating": 35.00,
                "matches_played": 0,
                "recent_wins": [],
                "prev_winrate": 0.50,
                "created_generations": []
            }
            db[guest_name] = prof
            UserProfileManager._save_unlocked(db)
            return prof

    @staticmethod
    def authenticate_or_register(username: str, password: str, mode: str = "login", lang: str = "en") -> tuple:
        uname = str(username).strip()
        pwd = str(password).strip()
        norm_l = normalize_lang(lang)

        if not uname or len(uname) < 2 or uname.startswith("__"):
            msg = "Ім'я користувача має містити щонайменше 2 символи." if norm_l == "uk" else "Username must be at least 2 characters."
            return False, msg, None

        if not pwd or len(pwd) < 3:
            msg = "Пароль має містити щонайменше 3 символи." if norm_l == "uk" else "Password must be at least 3 characters."
            return False, msg, None

        pwd_hash = hash_password(pwd)

        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            existing = db.get(uname)

            if mode == "register":
                if existing and existing.get("is_registered") and existing.get("password_hash"):
                    msg = "Цей нікнейм вже зареєстровано. Увійдіть зі своїм паролем." if norm_l == "uk" else "Username already registered. Please log in."
                    return False, msg, None

                base_r = float(existing.get("base_rating", 35.00)) if existing else 35.00
                m_played = int(existing.get("matches_played", 0)) if existing else 0
                r_wins = list(existing.get("recent_wins", [])) if existing else []
                p_wr = float(existing.get("prev_winrate", 0.50)) if existing else 0.50
                c_gens = list(existing.get("created_generations", [])) if existing else []

                prof = {
                    "username": uname,
                    "is_registered": True,
                    "password_hash": pwd_hash,
                    "lang": norm_l,
                    "base_rating": round(base_r, 2),
                    "matches_played": m_played,
                    "recent_wins": r_wins,
                    "prev_winrate": round(p_wr, 4),
                    "created_generations": c_gens
                }
                db[uname] = prof
                UserProfileManager._save_unlocked(db)
                return True, "OK", prof

            if not existing:
                prof = {
                    "username": uname,
                    "is_registered": True,
                    "password_hash": pwd_hash,
                    "lang": norm_l,
                    "base_rating": 35.00,
                    "matches_played": 0,
                    "recent_wins": [],
                    "prev_winrate": 0.50,
                    "created_generations": []
                }
                db[uname] = prof
                UserProfileManager._save_unlocked(db)
                return True, "OK", prof

            stored_hash = existing.get("password_hash", "")
            if not stored_hash:
                existing["is_registered"] = True
                existing["password_hash"] = pwd_hash
                existing["lang"] = existing.get("lang", norm_l)
                db[uname] = existing
                UserProfileManager._save_unlocked(db)
                return True, "OK", existing

            if stored_hash != pwd_hash:
                msg = "Невірний пароль для цього акаунта." if norm_l == "uk" else "Invalid password for this account."
                return False, msg, None

            return True, "OK", existing

    @staticmethod
    def get_or_create(username: str, lang: str = "en") -> dict:
        uname = str(username).strip() or "Guest1"
        norm_l = normalize_lang(lang)
        with USERS_LOCK:
            db = UserProfileManager._load_unlocked()
            if uname not in db:
                db[uname] = {
                    "username": uname,
                    "is_registered": False,
                    "password_hash": "",
                    "lang": norm_l,
                    "base_rating": 35.00,
                    "matches_played": 0,
                    "recent_wins": [],
                    "prev_winrate": 0.50,
                    "created_generations": []
                }
                UserProfileManager._save_unlocked(db)
            elif "lang" not in db[uname]:
                db[uname]["lang"] = norm_l
                UserProfileManager._save_unlocked(db)
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

    @staticmethod
    def set_user_lang(username: str, lang: str) -> dict:
        uname = str(username).strip()
        norm_l = normalize_lang(lang)
        prof = UserProfileManager.get_or_create(uname, norm_l)
        prof["lang"] = norm_l
        UserProfileManager.update_profile(uname, prof)
        return prof

class GameRoom:
    def __init__(self, room_id: str, name: str, host: str, max_players: int = 2, is_private: bool = False, password: str = ""):
        self.room_id = room_id
        self.name = name.strip() or f"Room-{room_id}"
        self.host = host
        self.max_players = 4 if int(max_players) == 4 else 2
        self.is_private = bool(is_private and len(str(password).strip()) > 0)
        self.password_hash = hash_password(str(password).strip()) if self.is_private else ""
        self.status = "waiting"
        self.created_at = int(time.time())

        self.active_factions = [1, 2, 3, 4] if self.max_players == 4 else [1, 2]
        self.slots = {}
        for f in self.active_factions:
            self.slots[f] = {
                "faction": f,
                "kind": "empty",
                "username": None,
                "ai_rating": 35.00,
                "ai_label": "None"
            }

        self.slots[2] = {
            "faction": 2,
            "kind": "human",
            "username": host,
            "ai_rating": 0.00,
            "ai_label": "Human"
        }

        self.env: Optional[MicroRTSEnv] = None
        self.current_turn_faction = 2
        self.selected_by_user = {}
        self.last_seen_by_user = {host: time.time()}
        self.active_ai_ratings = {}
        self.is_training_new_gen = False
        self.last_evolution_report = None
        self.winner_faction = None
        self.winner_name = None

    def touch_user(self, username: str):
        if username:
            self.last_seen_by_user[username] = time.time()

    def check_disconnected_players(self):
        if self.status != "playing" or self.env is None:
            return
        now = time.time()
        for f in list(self.active_factions):
            s = self.slots.get(f, {})
            if s.get("kind") == "human" and s.get("username") and self.env.is_faction_alive(f):
                uname = s["username"]
                last_t = self.last_seen_by_user.get(uname, now)
                if (now - last_t) > DISCONNECT_TIMEOUT_SECONDS:
                    self.surrender_faction(f, is_disconnect=True)

    def get_user_faction(self, username: str) -> Optional[int]:
        for f in self.active_factions:
            s = self.slots[f]
            if s["kind"] == "human" and s["username"] == username:
                return f
        return None

    def is_full(self) -> bool:
        return all(self.slots[f]["kind"] in ("human", "ai") for f in self.active_factions)

    def human_count(self) -> int:
        return sum(1 for f in self.active_factions if self.slots[f]["kind"] == "human")

    def ai_count(self) -> int:
        return sum(1 for f in self.active_factions if self.slots[f]["kind"] == "ai")

    def filled_count(self) -> int:
        return sum(1 for f in self.active_factions if self.slots[f]["kind"] != "empty")

    def verify_password(self, password: str) -> bool:
        if not self.is_private:
            return True
        return hash_password(str(password or "").strip()) == self.password_hash

    def add_human_player(self, username: str, preferred_faction: int = None) -> tuple:
        self.touch_user(username)
        existing_f = self.get_user_faction(username)
        if existing_f is not None:
            return True, existing_f
        if self.status != "waiting":
            return False, None

        if preferred_faction in self.active_factions and self.slots[preferred_faction]["kind"] == "empty":
            target_f = int(preferred_faction)
        else:
            order = [2, 1, 3, 4] if self.max_players == 4 else [2, 1]
            empty_f = [f for f in order if f in self.active_factions and self.slots[f]["kind"] == "empty"]
            if not empty_f:
                return False, None
            target_f = empty_f[0]

        self.slots[target_f] = {
            "faction": target_f,
            "kind": "human",
            "username": username,
            "ai_rating": 0.00,
            "ai_label": "Human"
        }
        return True, target_f

    def add_ai_to_slot(self, slot_faction: int = None, ai_rating: float = None) -> bool:
        if self.status != "waiting":
            return False

        target_f = None
        if slot_faction in self.active_factions and self.slots[int(slot_faction)]["kind"] == "empty":
            target_f = int(slot_faction)
        else:
            order = [1, 3, 4, 2] if self.max_players == 4 else [1, 2]
            for f in order:
                if f in self.active_factions and self.slots[f]["kind"] == "empty":
                    target_f = f
                    break

        if target_f is None:
            return False

        host_prof = UserProfileManager.get_or_create(self.host)
        r_val = float(ai_rating) if ai_rating is not None else float(host_prof.get("base_rating", 35.00))
        r_val = round(float(np.clip(r_val, 0.00, max(100.00, PoolRegistry.get_max_rating()))), 2)
        entry = PoolRegistry.select_closest_generation(r_val)
        lbl = str(entry.get("label", f"Gen-{int(round(r_val))}"))

        self.slots[target_f] = {
            "faction": target_f,
            "kind": "ai",
            "username": f"AI [{lbl}]",
            "ai_rating": r_val,
            "ai_label": lbl
        }
        return True

    def clear_slot(self, slot_faction: int) -> Optional[str]:
        if self.status != "waiting":
            return None
        f = int(slot_faction)
        if f not in self.active_factions:
            return None
        removed_user = self.slots[f]["username"] if self.slots[f]["kind"] == "human" else None
        self.slots[f] = {
            "faction": f,
            "kind": "empty",
            "username": None,
            "ai_rating": 35.00,
            "ai_label": "None"
        }
        return removed_user

    def start_match(self) -> bool:
        if not self.is_full():
            return False

        host_prof = UserProfileManager.get_or_create(self.host)
        lang = normalize_lang(host_prof.get("lang", "en"))

        self.env = MicroRTSEnv(mode="manual_p2", record_telemetry=True, lang=lang, num_players=self.max_players)
        self.env.ai_controlled_factions = {f for f in self.active_factions if self.slots[f]["kind"] == "ai"}
        self.active_ai_ratings = {}
        self.selected_by_user = {}
        self.winner_faction = None
        self.winner_name = None

        now = time.time()
        for f in self.active_factions:
            s = self.slots[f]
            if s["kind"] == "ai":
                r_val = float(s.get("ai_rating", 35.00))
                entry = self.env.set_faction_ai_rating(f, r_val)
                lbl = str(entry.get("label", f"Gen-{int(round(r_val))}"))
                s["ai_label"] = lbl
                s["username"] = f"AI [{lbl}]"
                self.active_ai_ratings[f] = r_val
            else:
                self.env.ai_controlled_factions.discard(f)
                uname = s["username"]
                self.last_seen_by_user[uname] = now
                u_prof = UserProfileManager.get_or_create(uname)
                u_r = float(u_prof.get("base_rating", 35.00))
                comp = float(np.clip(0.30 + (u_r / 100.00) * 0.75, 0.25, 1.35))
                self.env.set_faction_competence(f, comp)

        self.status = "playing"
        turn_order = self._get_turn_order()
        self.current_turn_faction = turn_order[0]
        self.env._prepare_faction_turn(faction=self.current_turn_faction, collect_economy=False)
        self._advance_ai_turns_if_needed()
        return True

    def _get_turn_order(self) -> list:
        return [2, 1, 3, 4] if self.max_players == 4 else [2, 1]

    def surrender_faction(self, faction: int, is_disconnect: bool = False):
        if self.status != "playing" or self.env is None:
            return
        f = int(faction)
        if not self.env.is_faction_alive(f):
            return

        uname = self.slots.get(f, {}).get("username") or f"Faction {f}"
        self.env._eliminate_faction(f)
        event_key = "player_disconnected_surrender" if is_disconnect else "player_surrendered"
        self.env.add_localized_event(event_key, player=uname, faction=f)

        winner_f = self.env.get_match_winner()
        if winner_f is not None:
            self._finalize_match(winner_f)
            return

        alive_humans = [
            af for af in self.active_factions
            if self.slots[af]["kind"] == "human" and self.env.is_faction_alive(af)
        ]
        if len(alive_humans) == 0:
            alive_any = self.env.get_alive_factions()
            if alive_any:
                self._finalize_match(alive_any[0])
            return

        if self.current_turn_faction == f:
            order = self._get_turn_order()
            self._step_to_next_faction(order)
            winner_f = self.env.get_match_winner()
            if winner_f is not None:
                self._finalize_match(winner_f)
                return
            self._advance_ai_turns_if_needed()

    def _advance_ai_turns_if_needed(self):
        if self.status != "playing" or self.env is None:
            return

        order = self._get_turn_order()
        safety_steps = 0
        cleared_traces_this_chain = False

        while self.status == "playing" and safety_steps < 20:
            safety_steps += 1
            winner_f = self.env.get_match_winner()
            if winner_f is not None:
                self._finalize_match(winner_f)
                return

            curr_f = self.current_turn_faction
            if not self.env.is_faction_alive(curr_f):
                self._step_to_next_faction(order)
                continue

            slot = self.slots[curr_f]
            if slot["kind"] == "human":
                return

            f_model = self.env.faction_models.get(curr_f)
            self.env.execute_full_ai_turn(
                faction=curr_f,
                model=f_model,
                deterministic=False,
                clear_traces=(not cleared_traces_this_chain),
                collect_economy=False
            )
            cleared_traces_this_chain = True

            winner_f = self.env.get_match_winner()
            if winner_f is not None:
                self._finalize_match(winner_f)
                return

            self._step_to_next_faction(order)

    def _step_to_next_faction(self, order: list):
        if self.env is None:
            return
        idx = order.index(self.current_turn_faction) if self.current_turn_faction in order else 0
        wrapped_round = False

        for _ in range(len(order)):
            idx = (idx + 1) % len(order)
            if idx == 0:
                wrapped_round = True
            cand_f = order[idx]
            if self.env.is_faction_alive(cand_f):
                self.current_turn_faction = cand_f
                break

        if wrapped_round:
            self.env.turn_count += 1
            self.env.collect_global_round_economy()
            self._apply_in_match_dda_Adjustments()
            self.env.record_turn_efficiency_snapshot()

        if self.env.is_faction_alive(self.current_turn_faction):
            self.env._prepare_faction_turn(faction=self.current_turn_faction, collect_economy=False)

    def _apply_in_match_dda_Adjustments(self):
        if self.env is None:
            return
        if self.env.turn_count <= 10 or self.env.turn_count % 3 == 0:
            human_factions = [f for f in self.active_factions if self.slots[f]["kind"] == "human" and self.env.is_faction_alive(f)]
            if not human_factions:
                return
            primary_human_f = human_factions[0]
            for f in self.active_factions:
                if self.slots[f]["kind"] == "ai" and self.env.is_faction_alive(f):
                    curr_r = float(self.active_ai_ratings.get(f, self.slots[f]["ai_rating"]))
                    new_r = self.env.compute_surge_dda_step(
                        current_rating=curr_r,
                        is_early_turn=(self.env.turn_count <= 10),
                        player_faction=primary_human_f,
                        ai_faction=f
                    )
                    self.active_ai_ratings[f] = new_r
                    entry = self.env.set_faction_ai_rating(f, new_r)
                    lbl = str(entry.get("label", f"Gen-{int(round(new_r))}"))
                    self.slots[f]["ai_label"] = lbl
                    self.slots[f]["username"] = f"AI [{lbl}]"

    def end_turn_for_faction(self, faction: int):
        if self.status != "playing" or self.env is None:
            return
        f = int(faction)
        if f != self.current_turn_faction:
            return

        self.env.heal_all_unmoved_units(faction=f)
        winner_f = self.env.get_match_winner()
        if winner_f is not None:
            self._finalize_match(winner_f)
            return

        order = self._get_turn_order()
        self._step_to_next_faction(order)
        winner_f = self.env.get_match_winner()
        if winner_f is not None:
            self._finalize_match(winner_f)
            return

        self._advance_ai_turns_if_needed()

    def _finalize_match(self, winning_faction: int):
        if self.status == "finished" or self.env is None:
            return
        self.status = "finished"
        wf = int(winning_faction)
        self.winner_faction = wf
        w_slot = self.slots.get(wf, {})
        self.winner_name = str(w_slot.get("username") or f"Faction {wf}")

        m_obs, m_acts, m_weights = self.env.collect_match_training_samples(
            winning_faction=wf,
            min_efficiency=MIN_TRAINING_EFFICIENCY
        )
        saved_cnt = self.env.export_high_efficiency_telemetry(
            winning_faction=wf,
            min_efficiency=MIN_TRAINING_EFFICIENCY
        )

        human_count = self.human_count()
        ai_ratings_list = [float(self.active_ai_ratings.get(f, self.slots[f]["ai_rating"])) for f in self.active_factions if self.slots[f]["kind"] == "ai"]
        mean_ai_active = float(np.mean(ai_ratings_list)) if ai_ratings_list else None

        for f in self.active_factions:
            s = self.slots[f]
            if s["kind"] == "human" and s["username"]:
                uname = s["username"]
                prof = UserProfileManager.get_or_create(uname)
                won = (f == wf)

                prof["matches_played"] = int(prof.get("matches_played", 0)) + 1
                wins = list(prof.get("recent_wins", []))
                wins.append(1.00 if won else 0.00)
                if len(wins) > 20:
                    wins.pop(0)
                prof["recent_wins"] = wins

                wr = float(np.mean(wins))
                fast_wr = float(np.mean(wins[-5:]))
                err_wr = wr - 0.50
                d_wr = wr - float(prof.get("prev_winrate", 0.50))
                prof["prev_winrate"] = round(wr, 4)

                pvp_bonus = (4.50 if won else -3.50) if human_count > 1 else 0.00
                delta_macro = round((8.00 * err_wr) + (32.00 * d_wr) + (12.00 * (fast_wr - 0.50)) + pvp_bonus, 2)
                max_pool_r = max(100.00, PoolRegistry.get_max_rating() + 15.00)

                old_base = float(prof.get("base_rating", 35.00))
                ref_active = mean_ai_active if mean_ai_active is not None else old_base
                blended = (0.55 * old_base) + (0.45 * ref_active) + delta_macro
                new_base = round(max(10.00, min(max_pool_r, blended)), 2)

                prof["base_rating"] = new_base
                UserProfileManager.update_profile(uname, prof)

                for af in self.active_factions:
                    if self.slots[af]["kind"] == "ai":
                        self.slots[af]["ai_rating"] = new_base

        evo_author = w_slot.get("username") if w_slot.get("kind") == "human" else self.host
        self.is_training_new_gen = True
        self.last_evolution_report = {
            "status": "training",
            "winner": self.winner_name,
            "winner_faction": wf,
            "samples": int(saved_cnt)
        }

        def _bg_evolve(obs_c, acts_c, w_c, uname, w_name, w_fac):
            try:
                new_entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(obs_c, acts_c, w_c, username=uname)
                self.last_evolution_report = {
                    "status": "completed",
                    "winner": w_name,
                    "winner_faction": w_fac,
                    "new_gen_label": str(new_entry["label"]),
                    "new_gen_rating": float(new_entry["rating"]),
                    "gauntlet_winrate": float(new_entry["w_gauntlet"]),
                    "doctrine_alignment": float(new_entry.get("doctrine_alignment", 100.00)),
                    "samples": int(saved_cnt)
                }
            finally:
                self.is_training_new_gen = False

        threading.Thread(
            target=_bg_evolve,
            args=(m_obs, m_acts, m_weights, evo_author, self.winner_name, wf),
            daemon=True
        ).start()

    def serialize_summary(self) -> dict:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "host": self.host,
            "max_players": self.max_players,
            "filled_slots": self.filled_count(),
            "human_count": self.human_count(),
            "ai_count": self.ai_count(),
            "is_private": self.is_private,
            "status": self.status,
            "is_full": self.is_full(),
            "slots": [
                {
                    "faction": f,
                    "kind": self.slots[f]["kind"],
                    "username": self.slots[f]["username"],
                    "ai_rating": round(float(self.slots[f].get("ai_rating", 35.00)), 2),
                    "ai_label": self.slots[f].get("ai_label", "")
                }
                for f in self.active_factions
            ]
        }

    def serialize_for_user(self, username: str, lang: str = "en") -> dict:
        self.touch_user(username)
        self.check_disconnected_players()

        norm_l = normalize_lang(lang)
        user_prof = UserProfileManager.get_or_create(username, norm_l)
        my_faction = self.get_user_faction(username)

        base_payload = {
            "screen": "game" if self.status in ("playing", "finished") else "room_lobby",
            "username": username,
            "is_registered": bool(user_prof.get("is_registered", False)),
            "lang": norm_l,
            "ui_text": get_ui_bundle(norm_l),
            "matches_played": int(user_prof.get("matches_played", 0)),
            "winrate": round(float(np.mean(user_prof.get("recent_wins", []))) * 100.00, 2) if user_prof.get("recent_wins") else 50.00,
            "base_rating": round(float(user_prof.get("base_rating", 35.00)), 2),
            "pool_size": len(PoolRegistry.get_all_generations()),
            "pool_max_rating": round(float(PoolRegistry.get_max_rating()), 2),
            "room": self.serialize_summary(),
            "num_players": self.max_players,
            "my_faction": my_faction,
            "is_host": (username == self.host),
            "can_start": (username == self.host and self.is_full() and self.status in ("waiting", "finished")),
            "is_training_new_gen": bool(self.is_training_new_gen),
            "last_evolution_report": self.last_evolution_report,
            "config": {
                "unit_stats": UNIT_STATS,
                "struct_stats": STRUCT_STATS,
                "disband_ratio": DISBAND_REFUND_RATIO,
                "demolish_ratio": DEMOLISH_REFUND_RATIO,
                "min_training_eff": MIN_TRAINING_EFFICIENCY
            }
        }

        if self.env is None:
            return clean_for_json(base_payload)

        env = self.env
        env.set_language(norm_l)

        factions_data = {}
        for f in self.active_factions:
            inc, upk, net, tiles, army = env.get_faction_economy_stats(f)
            tax = round(max(0, army - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT, 2)
            slot_info = self.slots[f]
            factions_data[str(f)] = {
                "faction": f,
                "name": get_faction_display_name(f, norm_l),
                "color": FACTION_META[f]["color"],
                "kind": slot_info["kind"],
                "controller": slot_info["username"] or "Empty",
                "alive": bool(env.is_faction_alive(f)),
                "gold": round(float(env.get_faction_gold(f)), 2),
                "inc": float(inc),
                "upk": float(upk),
                "tax": float(tax),
                "net": float(net),
                "tiles": int(tiles),
                "army": int(army),
                "castle_hp": round(float(env.get_castle_hp(f)), 2),
                "castle_pos": list(env.castle_positions[f])
            }

        my_alive = bool(my_faction is not None and env.is_faction_alive(my_faction))
        is_my_turn = (self.status == "playing" and my_alive and self.current_turn_faction == my_faction)
        sel_pos = self.selected_by_user.get(username)
        my_workers_cnt = sum(1 for u in env.units.values() if u['faction'] == my_faction and u['type'] == 'worker') if my_faction is not None else 0

        valid_moves = []
        valid_attacks = []
        inspector = {
            "title": get_text(norm_l, "nothing_selected"),
            "lines": [get_text(norm_l, "click_prompt")],
            "can_heal": False,
            "can_harvest": False,
            "can_build_tower": False,
            "can_build_barracks": False,
            "can_disband": False,
            "can_demolish": False,
            "can_spawn_worker": False,
            "can_spawn_warrior": False,
            "can_spawn_archer": False,
            "disband_refund_val": 0.00,
            "demolish_refund_val": 0.00
        }

        if sel_pos is not None:
            sr, sc = int(sel_pos[0]), int(sel_pos[1])
            base_v = float(env.tile_yields[sr, sc])
            eff_v = float(env.get_effective_tile_yield(sr, sc))
            terr_owner = int(env.territory[sr, sc])
            if terr_owner == 0:
                owner_str = get_text(norm_l, "owner_neutral")
            elif terr_owner == my_faction:
                owner_str = f"{get_faction_display_name(terr_owner, norm_l)} (You)"
            else:
                ctrl_name = self.slots.get(terr_owner, {}).get("username") or get_faction_display_name(terr_owner, norm_l)
                owner_str = f"{get_faction_display_name(terr_owner, norm_l)} [{ctrl_name}]"

            lines = []
            if env.is_tile_built_up(sr, sc):
                lines.append(get_text(norm_l, "land_built", base=base_v, owner=owner_str))
            else:
                lines.append(get_text(norm_l, "land_free", eff=eff_v, owner=owner_str))

            if (sr, sc) in env.units:
                u = env.units[(sr, sc)]
                uf = int(u['faction'])
                st = UNIT_STATS[u['type']]
                u_name = get_unit_name(u['type'], norm_l)
                f_str = f"{get_faction_display_name(uf, norm_l)} [{self.slots.get(uf, {}).get('username', '')}]"

                has_m = not u.get('moved', False)
                has_a = not u.get('attacked', False)
                if bool(u['ready']):
                    if has_m and has_a:
                        r_str = get_text(norm_l, "status_ready")
                    elif not has_m and has_a:
                        r_str = get_text(norm_l, "status_can_attack")
                    else:
                        r_str = get_text(norm_l, "status_ready")
                else:
                    r_str = get_text(norm_l, "status_moved")

                u_refund = round(float(st['cost']) * DISBAND_REFUND_RATIO, 2)
                u_refund_pct = round(DISBAND_REFUND_RATIO * 100.00, 2)
                inspector["disband_refund_val"] = u_refund

                lines.append(get_text(norm_l, "unit_header", name=u_name, faction=f_str, status=r_str))
                lines.append(get_text(norm_l, "hp_line", hp=u['hp'], max_hp=u['max_hp'], pct=(u['hp'] / u['max_hp']) * 100.00))
                lines.append(get_text(norm_l, "stats_line", move=st['move_range'], atk=st['atk_range'], dmg=st['dmg']))
                lines.append(get_text(norm_l, "upkeep_line", upk=st['upkeep'], eff=u['efficiency']))
                lines.append(get_text(norm_l, "unit_refund_line", refund=u_refund, cost=st['cost'], pct=u_refund_pct))

                if is_my_turn and uf == my_faction:
                    inspector["can_disband"] = True
                    if bool(u['ready']):
                        if has_m:
                            valid_moves = [[int(p[0]), int(p[1])] for p in env.get_valid_moves_for_unit((sr, sc))]
                        if has_a:
                            valid_attacks = [[int(p[0]), int(p[1])] for p in env.get_valid_attack_targets((sr, sc))]
                        inspector["can_heal"] = bool(has_m and has_a)
                        if u['type'] == 'worker' and has_a:
                            my_gold = env.get_faction_gold(my_faction)
                            inspector["can_harvest"] = bool(eff_v > 0.00)
                            inspector["can_build_tower"] = bool(
                                my_gold >= STRUCT_STATS[1]['cost']
                                and env._find_adjacent_tower_site((sr, sc), my_faction) is not None
                            )
                            inspector["can_build_barracks"] = bool(
                                my_gold >= STRUCT_STATS[2]['cost']
                                and env._find_barracks_build_site((sr, sc), my_faction, enforce_ai_distance=False) is not None
                            )
                            dem_site = env._find_demolishable_structure_site((sr, sc), my_faction)
                            if dem_site is not None:
                                inspector["can_demolish"] = True
                                dem_t = int(env.struct_type[dem_site[0], dem_site[1]])
                                inspector["demolish_refund_val"] = round(float(STRUCT_STATS[dem_t]['cost']) * DEMOLISH_REFUND_RATIO, 2)
            elif is_my_turn and int(env.struct_owner[sr, sc]) == my_faction and int(env.struct_type[sr, sc]) == 1 and bool(env.tower_ready[sr, sc]):
                valid_attacks = [[int(p[0]), int(p[1])] for p in env.get_valid_attack_targets((sr, sc), faction=my_faction, atk_range=2)]

            for cf in self.active_factions:
                if (sr, sc) == env.castle_positions[cf] and env.is_faction_alive(cf):
                    c_hp = env.get_castle_hp(cf)
                    if cf == my_faction:
                        lines.append(get_text(norm_l, "your_castle", hp=c_hp, max_hp=CASTLE_MAX_HP))
                        if is_my_turn:
                            if (sr, sc) not in env.units:
                                my_gold = env.get_faction_gold(my_faction)
                                inspector["can_spawn_worker"] = bool(my_gold >= UNIT_STATS['worker']['cost'] and my_workers_cnt < MAX_WORKERS_PER_FACTION)
                                inspector["can_spawn_warrior"] = bool(my_gold >= UNIT_STATS['warrior']['cost'])
                                inspector["can_spawn_archer"] = bool(my_gold >= UNIT_STATS['archer']['cost'])
                            else:
                                lines.append(get_text(norm_l, "castle_occupied"))
                    else:
                        lines.append(f"{get_faction_display_name(cf, norm_l)} Castle ({c_hp:.2f} / {CASTLE_MAX_HP:.2f} HP)")
                    break
            else:
                if int(env.struct_type[sr, sc]) != 0:
                    st_t = int(env.struct_type[sr, sc])
                    st_own = int(env.struct_owner[sr, sc])
                    st_info = STRUCT_STATS[st_t]
                    st_name = get_struct_name(st_t, norm_l)
                    s_refund = round(float(st_info['cost']) * DEMOLISH_REFUND_RATIO, 2)
                    s_refund_pct = round(DEMOLISH_REFUND_RATIO * 100.00, 2)
                    lines.append(get_text(norm_l, "struct_line", name=st_name, hp=float(env.struct_hp[sr, sc]), max_hp=st_info['max_hp']))
                    lines.append(get_text(norm_l, "struct_refund_line", refund=s_refund, cost=st_info['cost'], pct=s_refund_pct))
                    if is_my_turn and st_t == 2 and st_own == my_faction and (sr, sc) not in env.units:
                        my_gold = env.get_faction_gold(my_faction)
                        inspector["can_spawn_worker"] = bool(my_gold >= UNIT_STATS['worker']['cost'] and my_workers_cnt < MAX_WORKERS_PER_FACTION)
                        inspector["can_spawn_warrior"] = bool(my_gold >= UNIT_STATS['warrior']['cost'])
                        inspector["can_spawn_archer"] = bool(my_gold >= UNIT_STATS['archer']['cost'])

            inspector["lines"] = lines

        units_list = []
        for (ur, uc), u in env.units.items():
            units_list.append({
                "r": int(ur),
                "c": int(uc),
                "faction": int(u['faction']),
                "type": str(u['type']),
                "hp": round(float(u['hp']), 2),
                "max_hp": round(float(u['max_hp']), 2),
                "ready": bool(u['ready']),
                "moved": bool(u.get('moved', False)),
                "attacked": bool(u.get('attacked', False)),
                "efficiency": round(float(u.get('efficiency', 0.00)), 2)
            })

        structs_list = []
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if int(env.struct_type[r, c]) != 0:
                    structs_list.append({
                        "r": int(r),
                        "c": int(c),
                        "type": int(env.struct_type[r, c]),
                        "owner": int(env.struct_owner[r, c]),
                        "hp": round(float(env.struct_hp[r, c]), 2),
                        "ready": bool(env.tower_ready[r, c])
                    })

        ai_slots = [self.slots[f] for f in self.active_factions if self.slots[f]["kind"] == "ai"]
        if ai_slots:
            primary_ai_lbl = ", ".join(s["ai_label"] for s in ai_slots)
            primary_ai_r = round(float(np.mean([self.active_ai_ratings.get(s["faction"], s["ai_rating"]) for s in ai_slots])), 2)
        else:
            primary_ai_lbl = "PvP Multiplayer"
            primary_ai_r = round(float(user_prof.get("base_rating", 35.00)), 2)

        turn_slot = self.slots.get(self.current_turn_faction, {})
        turn_owner_name = turn_slot.get("username") or f"Faction {self.current_turn_faction}"

        my_f_key = str(my_faction if my_faction is not None else 2)
        opp_f_candidates = [f for f in self.active_factions if str(f) != my_f_key and env.is_faction_alive(f)]
        if not opp_f_candidates:
            opp_f_candidates = [f for f in self.active_factions if str(f) != my_f_key]
        opp_f_key = str(opp_f_candidates[0]) if opp_f_candidates else "1"

        p2_compat = factions_data.get(my_f_key, factions_data.get("2", factions_data["1"]))
        p1_compat = factions_data.get(opp_f_key, factions_data.get("1", p2_compat))

        base_payload.update({
            "turn": int(env.turn_count),
            "max_turns": int(MAX_STEPS),
            "current_turn_faction": int(self.current_turn_faction),
            "current_turn_user": str(turn_owner_name),
            "is_my_turn": bool(is_my_turn),
            "my_alive": bool(my_alive),
            "active_rating": float(primary_ai_r),
            "active_gen_label": str(primary_ai_lbl),
            "active_gen_origin": "multiplayer_room",
            "factions": factions_data,
            "p2": p2_compat,
            "p1": p1_compat,
            "terrain": env.terrain.tolist(),
            "territory": env.territory.tolist(),
            "eff_yields": np.round(env.get_effective_yield_map(), 2).tolist(),
            "units": units_list,
            "structs": structs_list,
            "selected_pos": [int(sel_pos[0]), int(sel_pos[1])] if sel_pos else None,
            "valid_moves": valid_moves,
            "valid_attacks": valid_attacks,
            "ai_traces": env.last_ai_turn_traces,
            "event_log": env.turn_event_log,
            "inspector": inspector,
            "stats": env.get_efficiency_statistics(),
            "match_finished": (self.status == "finished"),
            "winner_faction": self.winner_faction,
            "winner_name": self.winner_name
        })
        return clean_for_json(base_payload)

class RoomManager:
    @staticmethod
    def list_rooms() -> list:
        with ROOMS_LOCK:
            for r in list(ROOMS.values()):
                r.check_disconnected_players()
            return [r.serialize_summary() for r in sorted(ROOMS.values(), key=lambda x: x.created_at, reverse=True)]

    @staticmethod
    def get_user_room(username: str) -> Optional[GameRoom]:
        with ROOMS_LOCK:
            rid = USER_ACTIVE_ROOM.get(username)
            if rid and rid in ROOMS:
                room = ROOMS[rid]
                if room.get_user_faction(username) is not None or room.host == username:
                    return room
                USER_ACTIVE_ROOM.pop(username, None)
            return None

    @staticmethod
    def leave_current_room(username: str, is_disconnect: bool = False):
        with ROOMS_LOCK:
            room = RoomManager.get_user_room(username)
            USER_ACTIVE_ROOM.pop(username, None)
            if room is None:
                return
            uf = room.get_user_faction(username)
            if room.status == "waiting":
                if uf is not None:
                    room.clear_slot(uf)
                remaining_humans = [room.slots[f]["username"] for f in room.active_factions if room.slots[f]["kind"] == "human" and room.slots[f]["username"]]
                if not remaining_humans:
                    ROOMS.pop(room.room_id, None)
                elif room.host == username:
                    room.host = remaining_humans[0]
            elif room.status == "playing" and uf is not None and room.env is not None:
                room.surrender_faction(uf, is_disconnect=is_disconnect)
                remaining_alive_humans = [
                    room.slots[f]["username"] for f in room.active_factions
                    if room.slots[f]["kind"] == "human" and room.slots[f]["username"] != username and room.env.is_faction_alive(f)
                ]
                if not remaining_alive_humans and room.status == "finished":
                    ROOMS.pop(room.room_id, None)
                elif room.host == username and remaining_alive_humans:
                    room.host = remaining_alive_humans[0]
            elif room.status == "finished":
                if uf is not None:
                    room.clear_slot(uf)
                if room.human_count() == 0:
                    ROOMS.pop(room.room_id, None)

    @staticmethod
    def create_room(host: str, room_name: str, max_players: int, is_private: bool, password: str, fill_ai: bool = False) -> GameRoom:
        with ROOMS_LOCK:
            RoomManager.leave_current_room(host)
            rid = uuid.uuid4().hex[:6].upper()
            while rid in ROOMS:
                rid = uuid.uuid4().hex[:6].upper()

            room = GameRoom(
                room_id=rid,
                name=room_name or f"{host}'s Room",
                host=host,
                max_players=max_players,
                is_private=is_private,
                password=password
            )
            if fill_ai:
                for f in room.active_factions:
                    if room.slots[f]["kind"] == "empty":
                        room.add_ai_to_slot(f)

            ROOMS[rid] = room
            USER_ACTIVE_ROOM[host] = rid
            return room

class AuthRequest(BaseModel):
    mode: str
    username: Optional[str] = None
    password: Optional[str] = None
    lang: Optional[str] = "en"

class RoomActionRequest(BaseModel):
    username: str
    action: str
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    max_players: Optional[int] = 2
    is_private: Optional[bool] = False
    password: Optional[str] = ""
    slot_faction: Optional[int] = None
    ai_rating: Optional[float] = None
    fill_ai: Optional[bool] = False

class ActionRequest(BaseModel):
    username: str
    action_type: str
    r: Optional[int] = None
    c: Optional[int] = None
    param: Optional[str] = None

def _build_lobby_payload(username: str, lang: str = "en") -> dict:
    norm_l = normalize_lang(lang)
    prof = UserProfileManager.get_or_create(username, norm_l)
    room = RoomManager.get_user_room(username)
    if room is not None:
        return room.serialize_for_user(username, norm_l)

    wins = prof.get("recent_wins", [])
    wr = round(float(np.mean(wins)) * 100.00, 2) if wins else 50.00
    return clean_for_json({
        "screen": "lobby",
        "username": str(prof["username"]),
        "is_registered": bool(prof.get("is_registered", False)),
        "lang": norm_l,
        "ui_text": get_ui_bundle(norm_l),
        "matches_played": int(prof.get("matches_played", 0)),
        "winrate": float(wr),
        "base_rating": round(float(prof.get("base_rating", 35.00)), 2),
        "pool_size": len(PoolRegistry.get_all_generations()),
        "pool_max_rating": round(float(PoolRegistry.get_max_rating()), 2),
        "rooms": RoomManager.list_rooms(),
        "config": {
            "unit_stats": UNIT_STATS,
            "struct_stats": STRUCT_STATS,
            "disband_ratio": DISBAND_REFUND_RATIO,
            "demolish_ratio": DEMOLISH_REFUND_RATIO,
            "min_training_eff": MIN_TRAINING_EFFICIENCY
        }
    })

@app.get("/api/init_user")
def api_init_user(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    db = UserProfileManager.load_all()
    ip_map = db.get("__ip_map__", {})
    suggested_guest = ip_map.get(client_ip, "")
    lang = "en"
    if suggested_guest and suggested_guest in db:
        lang = normalize_lang(db[suggested_guest].get("lang", "en"))
    return {
        "suggested_guest": suggested_guest,
        "lang": lang,
        "ui_text": get_ui_bundle(lang)
    }

@app.post("/api/auth")
def api_auth(req: AuthRequest, request: Request):
    norm_l = normalize_lang(req.lang or "en")
    if req.mode == "guest":
        client_ip = request.client.host if request.client else "127.0.0.1"
        prof = UserProfileManager.create_guest_for_ip(client_ip, norm_l)
        return {"ok": True, "state": _build_lobby_payload(prof["username"], prof.get("lang", norm_l))}

    ok, msg, prof = UserProfileManager.authenticate_or_register(
        username=req.username or "",
        password=req.password or "",
        mode=req.mode,
        lang=norm_l
    )
    if not ok:
        return {"ok": False, "error": msg}
    return {"ok": True, "state": _build_lobby_payload(prof["username"], prof.get("lang", norm_l))}

@app.post("/api/leave_beacon")
async def api_leave_beacon(request: Request):
    try:
        body = await request.json()
        uname = str(body.get("username", "")).strip()
        if uname:
            RoomManager.leave_current_room(uname, is_disconnect=True)
    except Exception:
        pass
    return {"ok": True}

@app.post("/api/rooms")
def api_rooms_action(req: RoomActionRequest):
    uname = str(req.username).strip() or "Guest1"
    prof = UserProfileManager.get_or_create(uname)
    lang = normalize_lang(prof.get("lang", "en"))
    act = req.action

    with ROOMS_LOCK:
        if act == "list":
            return {"ok": True, "state": _build_lobby_payload(uname, lang)}

        if act == "create":
            room = RoomManager.create_room(
                host=uname,
                room_name=req.room_name or f"{uname}'s Battle",
                max_players=req.max_players or 2,
                is_private=bool(req.is_private),
                password=req.password or "",
                fill_ai=bool(req.fill_ai)
            )
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "quick_bot_match":
            room = RoomManager.create_room(
                host=uname,
                room_name=f"{uname} vs AI",
                max_players=req.max_players or 2,
                is_private=False,
                password="",
                fill_ai=True
            )
            room.start_match()
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "join":
            rid = str(req.room_id or "").strip().upper()
            room = ROOMS.get(rid)
            if room is None:
                err = "Кімнату не знайдено." if lang == "uk" else "Room not found."
                return {"ok": False, "error": err, "state": _build_lobby_payload(uname, lang)}

            if room.get_user_faction(uname) is None:
                if not room.verify_password(req.password or ""):
                    err = "Невірний пароль кімнати." if lang == "uk" else "Invalid room password."
                    return {"ok": False, "error": err, "state": _build_lobby_payload(uname, lang)}
                ok, _ = room.add_human_player(uname, preferred_faction=req.slot_faction)
                if not ok:
                    err = "Немає вільних слотів у кімнаті." if lang == "uk" else "No empty human slots available in this room."
                    return {"ok": False, "error": err, "state": _build_lobby_payload(uname, lang)}

            RoomManager.leave_current_room(uname)
            room.add_human_player(uname, preferred_faction=req.slot_faction)
            USER_ACTIVE_ROOM[uname] = room.room_id
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "leave":
            RoomManager.leave_current_room(uname, is_disconnect=False)
            return {"ok": True, "state": _build_lobby_payload(uname, lang)}

        room = RoomManager.get_user_room(uname)
        if room is None:
            return {"ok": False, "error": "Not in a room.", "state": _build_lobby_payload(uname, lang)}

        if act == "add_ai" and room.host == uname:
            if room.status == "finished":
                room.status = "waiting"
            room.add_ai_to_slot(slot_faction=req.slot_faction, ai_rating=req.ai_rating)
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "kick_slot" and room.host == uname and req.slot_faction is not None:
            if room.status == "finished":
                room.status = "waiting"
            sf = int(req.slot_faction)
            if sf in room.active_factions and room.slots[sf]["username"] != uname:
                kicked_user = room.clear_slot(sf)
                if kicked_user and USER_ACTIVE_ROOM.get(kicked_user) == room.room_id:
                    USER_ACTIVE_ROOM.pop(kicked_user, None)
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "switch_slot" and req.slot_faction is not None and room.status == "waiting":
            sf = int(req.slot_faction)
            curr_f = room.get_user_faction(uname)
            if sf in room.active_factions and room.slots[sf]["kind"] == "empty" and curr_f is not None:
                room.clear_slot(curr_f)
                room.add_human_player(uname, preferred_faction=sf)
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "start" and room.host == uname:
            if not room.is_full():
                err = "Заповніть усі слоти гравцями або ШІ перед стартом!" if lang == "uk" else "Fill all slots with players or AI before starting!"
                return {"ok": False, "error": err, "state": room.serialize_for_user(uname, lang)}
            room.start_match()
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        if act == "rematch" and room.host == uname and room.status == "finished":
            if room.is_full():
                room.start_match()
            else:
                room.status = "waiting"
            return {"ok": True, "state": room.serialize_for_user(uname, lang)}

        return {"ok": True, "state": room.serialize_for_user(uname, lang)}

@app.post("/api/state")
def api_get_state(req: ActionRequest):
    uname = str(req.username).strip() or "Guest1"
    if req.param in ("en", "uk", "ua"):
        prof = UserProfileManager.set_user_lang(uname, req.param)
    else:
        prof = UserProfileManager.get_or_create(uname)
    lang = normalize_lang(prof.get("lang", "en"))

    room = RoomManager.get_user_room(uname)
    if room is not None:
        return room.serialize_for_user(uname, lang)
    return _build_lobby_payload(uname, lang)

@app.post("/api/action")
def api_perform_action(req: ActionRequest):
    uname = str(req.username).strip() or "Guest1"
    atype = req.action_type

    if atype == "set_lang" and req.param:
        prof = UserProfileManager.set_user_lang(uname, req.param)
        lang = normalize_lang(prof.get("lang", "en"))
        room = RoomManager.get_user_room(uname)
        if room is not None:
            return room.serialize_for_user(uname, lang)
        return _build_lobby_payload(uname, lang)

    prof = UserProfileManager.get_or_create(uname)
    lang = normalize_lang(prof.get("lang", "en"))

    with ROOMS_LOCK:
        room = RoomManager.get_user_room(uname)
        if room is None or room.env is None:
            return _build_lobby_payload(uname, lang)

        room.touch_user(uname)
        env = room.env
        my_f = room.get_user_faction(uname)
        if my_f is None:
            return room.serialize_for_user(uname, lang)

        if atype == "surrender":
            room.surrender_faction(my_f, is_disconnect=False)
            return room.serialize_for_user(uname, lang)

        if atype == "select_tile" and req.r is not None and req.c is not None:
            room.selected_by_user[uname] = (int(req.r), int(req.c))
            return room.serialize_for_user(uname, lang)

        is_my_turn = (room.status == "playing" and env.is_faction_alive(my_f) and room.current_turn_faction == my_f)
        sel_pos = room.selected_by_user.get(uname)

        if atype in ("order_tile", "click_tile") and req.r is not None and req.c is not None:
            clicked = (int(req.r), int(req.c))
            if is_my_turn and sel_pos is not None:
                valid_m = []
                valid_a = []
                sr, sc = int(sel_pos[0]), int(sel_pos[1])
                if (sr, sc) in env.units and env.units[(sr, sc)]['faction'] == my_f and env.units[(sr, sc)]['ready']:
                    u_sel = env.units[(sr, sc)]
                    if not u_sel.get('moved', False):
                        valid_m = env.get_valid_moves_for_unit((sr, sc))
                    if not u_sel.get('attacked', False):
                        valid_a = env.get_valid_attack_targets((sr, sc))
                elif int(env.struct_owner[sr, sc]) == my_f and int(env.struct_type[sr, sc]) == 1 and bool(env.tower_ready[sr, sc]):
                    valid_a = env.get_valid_attack_targets((sr, sc), faction=my_f, atk_range=2)

                if clicked in valid_m:
                    dr, dc = clicked[0] - sr, clicked[1] - sc
                    if (dr, dc) in RELATIVE_OFFSETS:
                        env.record_entity_decision('unit', (sr, sc), my_f, RELATIVE_OFFSETS.index((dr, dc)))
                    u_type = env.units[(sr, sc)]['type']
                    if env.execute_unit_move((sr, sc), clicked):
                        env.add_localized_event("player_move", u_type=u_type, r=clicked[0], c=clicked[1])
                        room.selected_by_user[uname] = clicked
                    return room.serialize_for_user(uname, lang)

                if clicked in valid_a:
                    is_tower = ((sr, sc) not in env.units and int(env.struct_owner[sr, sc]) == my_f and int(env.struct_type[sr, sc]) == 1)
                    dr, dc = clicked[0] - sr, clicked[1] - sc
                    if (dr, dc) in RELATIVE_OFFSETS:
                        env.record_entity_decision('tower' if is_tower else 'unit', (sr, sc), my_f, 12 + RELATIVE_OFFSETS.index((dr, dc)))
                    dmg = env.execute_attack_on_target((sr, sc), clicked, is_tower=is_tower)
                    if dmg > 0.00:
                        env.add_localized_event("player_attack", dmg=dmg, r=clicked[0], c=clicked[1])
                        winner_f = env.get_match_winner()
                        if winner_f is not None:
                            room._finalize_match(winner_f)
                    return room.serialize_for_user(uname, lang)

            if atype == "click_tile":
                room.selected_by_user[uname] = clicked
            return room.serialize_for_user(uname, lang)

        if not is_my_turn:
            return room.serialize_for_user(uname, lang)

        if atype == "heal" and sel_pos in env.units:
            if env.units[sel_pos]['faction'] == my_f:
                env.record_entity_decision('unit', sel_pos, my_f, 24)
                gained = env.heal_unit_percentage(sel_pos)
                if gained > 0.00:
                    env.add_localized_event("player_heal", gained=gained)

        elif atype == "harvest" and sel_pos in env.units:
            if env.units[sel_pos]['faction'] == my_f:
                env.record_entity_decision('unit', sel_pos, my_f, 25)
                bonus = env.execute_worker_harvest(sel_pos)
                if bonus > 0.00:
                    env.add_localized_event("player_harvest", bonus=bonus)

        elif atype == "build" and sel_pos in env.units:
            if env.units[sel_pos]['faction'] == my_f:
                b_type = 1 if req.param == "tower" else 2
                env.record_entity_decision('unit', sel_pos, my_f, 26 if b_type == 1 else 27)
                if env.execute_worker_build(sel_pos, build_type=b_type):
                    env.add_localized_event("player_build", st_type=b_type)

        elif atype == "disband" and sel_pos in env.units:
            if env.units[sel_pos]['faction'] == my_f:
                u_type = env.units[sel_pos]['type']
                env.record_entity_decision('unit', sel_pos, my_f, 28)
                refund = env.execute_unit_disband(sel_pos)
                if refund > 0.00:
                    env.add_localized_event("player_disband", u_type=u_type, refund=refund)
                    room.selected_by_user[uname] = None

        elif atype == "demolish" and sel_pos in env.units:
            if env.units[sel_pos]['faction'] == my_f:
                env.record_entity_decision('unit', sel_pos, my_f, 29)
                refund, st_t = env.execute_worker_demolish(sel_pos)
                if refund > 0.00:
                    env.add_localized_event("player_demolish", st_type=st_t, refund=refund)

        elif atype == "spawn" and sel_pos is not None:
            u_type = req.param or "warrior"
            act_map = {'worker': 27, 'warrior': 28, 'archer': 29}
            env.record_entity_decision('spawn', sel_pos, my_f, act_map.get(u_type, 28))
            if env.execute_structure_spawn(sel_pos, faction=my_f, unit_type=u_type, enforce_ai_rules=False):
                env.add_localized_event("player_spawn", u_type=u_type)

        elif atype == "end_turn":
            room.end_turn_for_faction(my_f)

        return room.serialize_for_user(uname, lang)

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