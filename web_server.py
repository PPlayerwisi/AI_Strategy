import os
import json
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
    RELATIVE_OFFSETS, USERS_DB_FILE, POOL_DIR,
    MIN_TRAINING_EFFICIENCY, DISBAND_REFUND_RATIO, DEMOLISH_REFUND_RATIO
)
from microrts_env import (
    MicroRTSEnv, PoolRegistry, OnlineEvolutionEngine,
    get_text, get_unit_name, get_struct_name
)

app = FastAPI(title="Faction Warfare RTS — Multi-User Evolutionary Platform")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

USERS_LOCK = threading.Lock()
ACTIVE_SESSIONS = {}

SERVER_TEXT = {
    "en": {
        "nothing_selected": "Nothing selected",
        "click_prompt": "Click on any unit, tower, barracks, or castle on the map.",
        "owner_neutral": "Neutral (Gray)",
        "owner_ai": "Red Empire (AI)",
        "owner_player": "Blue Alliance (You)",
        "land_built": "Land: +0.00$/turn [Built-up, base: +{base:.2f}$] ({owner})",
        "land_free": "Land: +{eff:.2f}$/turn ({owner})",
        "unit_header": "Unit: {name} [{faction}] — {status}",
        "status_ready": "READY",
        "status_moved": "ACTED THIS TURN",
        "hp_line": "Health: {hp:.2f} / {max_hp:.2f} HP ({pct:.2f}%)",
        "stats_line": "Move: {move} tiles | Attack range: {atk} tiles | Dmg: {dmg:.2f}",
        "upkeep_line": "Upkeep: {upk:.2f}$/turn | Efficiency: {eff:.2f}",
        "your_castle": "Structure: YOUR CASTLE ({hp:.2f} / {max_hp:.2f} HP)",
        "castle_occupied": "Castle tile is occupied by a unit — move it out to recruit.",
        "ai_castle": "Structure: AI CASTLE ({hp:.2f} / {max_hp:.2f} HP)",
        "struct_line": "Structure: {name} ({hp:.2f} / {max_hp:.2f} HP)",
        "player_move": "You: {unit} moved to ({r},{c})",
        "player_attack": "You: dealt {dmg:.2f} dmg to target ({r},{c})",
        "player_heal": "You: unit restored +{gained:.2f} HP",
        "player_harvest": "You: Worker harvested +{bonus:.2f}$",
        "player_build": "You: Worker built {struct}",
        "player_spawn": "You: recruited {unit}",
        "player_disband": "You: disbanded {unit} (+{refund:.2f}$)",
        "player_demolish": "You: Worker demolished {struct} (+{refund:.2f}$)"
    },
    "uk": {
        "nothing_selected": "Нічого не обрано",
        "click_prompt": "Клікніть на будь-якого юніта, вежу, казарму або замок на карті.",
        "owner_neutral": "Нейтральна (Сіра)",
        "owner_ai": "Червона Імперія (ШІ)",
        "owner_player": "Синій Альянс (Ви)",
        "land_built": "Земля: +0.00$/хід [Забудовано, базова: +{base:.2f}$] ({owner})",
        "land_free": "Земля: +{eff:.2f}$/хід ({owner})",
        "unit_header": "Підрозділ: {name} [{faction}] — {status}",
        "status_ready": "ГОТОВИЙ ДО ДІЇ",
        "status_moved": "ВЖЕ ХОДИВ У ЦЬОМУ РАУНДІ",
        "hp_line": "Здоров'я: {hp:.2f} / {max_hp:.2f} HP ({pct:.2f}%)",
        "stats_line": "Рух: {move} кліт. | Дальність атаки: {atk} кліт. | Шкода: {dmg:.2f}",
        "upkeep_line": "Плата за службу: {upk:.2f}$/хід | Ефективність: {eff:.2f}",
        "your_castle": "Споруда: ВАШ ЗАМОК ({hp:.2f} / {max_hp:.2f} HP)",
        "castle_occupied": "Замок зайнятий юнітом — виведіть його для найму нового.",
        "ai_castle": "Споруда: ЗАМОК ШІ ({hp:.2f} / {max_hp:.2f} HP)",
        "struct_line": "Споруда: {name} ({hp:.2f} / {max_hp:.2f} HP)",
        "player_move": "Ви: {unit} перемістився на ({r},{c})",
        "player_attack": "Ви: завдано {dmg:.2f} шкоди по цілі ({r},{c})",
        "player_heal": "Ви: підрозділ відновив +{gained:.2f} HP",
        "player_harvest": "Ви: Робітник добув додатково +{bonus:.2f}$",
        "player_build": "Ви: Робітник звів {struct}",
        "player_spawn": "Ви: найнято підрозділ ({unit})",
        "player_disband": "Ви: розпущено {unit} (+{refund:.2f}$)",
        "player_demolish": "Ви: Робітник зніс {struct} (+{refund:.2f}$)"
    }
}

def srv_txt(lang: str, key: str, **kwargs) -> str:
    loc = SERVER_TEXT.get(lang, SERVER_TEXT["en"])
    tmpl = loc.get(key, SERVER_TEXT["en"].get(key, key))
    return tmpl.format(**kwargs) if kwargs else tmpl

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
    def load_all() -> dict:
        os.makedirs(POOL_DIR, exist_ok=True)
        with USERS_LOCK:
            if os.path.exists(USERS_DB_FILE):
                try:
                    with open(USERS_DB_FILE, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return {}

    @staticmethod
    def get_guest_name_for_ip(ip: str) -> str:
        db = UserProfileManager.load_all()
        with USERS_LOCK:
            ip_map = db.get("__ip_map__", {})
            if ip in ip_map:
                return ip_map[ip]

            counter = db.get("__guest_counter__", 0) + 1
            guest_name = f"Guest{counter}"

            ip_map[ip] = guest_name
            db["__ip_map__"] = ip_map
            db["__guest_counter__"] = counter

            if guest_name not in db:
                db[guest_name] = {
                    "username": guest_name,
                    "lang": "en",
                    "base_rating": 35.00,
                    "matches_played": 0,
                    "recent_wins": [],
                    "prev_winrate": 0.50,
                    "created_generations": []
                }

            with open(USERS_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2, ensure_ascii=False)

            return guest_name

    @staticmethod
    def get_or_create(username: str) -> dict:
        uname = username.strip() or "Guest1"
        db = UserProfileManager.load_all()
        with USERS_LOCK:
            if uname not in db:
                db[uname] = {
                    "username": uname,
                    "lang": "en",
                    "base_rating": 35.00,
                    "matches_played": 0,
                    "recent_wins": [],
                    "prev_winrate": 0.50,
                    "created_generations": []
                }
                with open(USERS_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(db, f, indent=2, ensure_ascii=False)
            elif "lang" not in db[uname]:
                db[uname]["lang"] = "en"
            return db[uname]

    @staticmethod
    def update_profile(username: str, profile_data: dict):
        db = UserProfileManager.load_all()
        with USERS_LOCK:
            db[username] = clean_for_json(profile_data)
            with open(USERS_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2, ensure_ascii=False)

class GameSession:
    def __init__(self, username: str):
        self.username = username
        self.profile = UserProfileManager.get_or_create(username)
        self.lang = self.profile.get("lang", "en")
        self.env = MicroRTSEnv(mode="manual_p2", record_telemetry=True, lang=self.lang)
        self.base_rating = float(self.profile.get("base_rating", 35.00))
        self.active_rating = self.base_rating
        self.env.set_dda_generation(self.active_rating)
        self.env._prepare_faction_turn(faction=2, collect_economy=False)
        self.selected_pos = None
        self.last_evolution_report = None
        self.is_training_new_gen = False

    def set_language(self, lang: str):
        self.lang = lang if lang in ("en", "uk") else "en"
        self.env.set_language(self.lang)
        self.profile["lang"] = self.lang
        UserProfileManager.update_profile(self.username, self.profile)

    def serialize_state(self) -> dict:
        inc1, upk1, net1, t1, u1 = self.env.get_faction_economy_stats(1)
        inc2, upk2, net2, t2, u2 = self.env.get_faction_economy_stats(2)
        tax1 = round(max(0, u1 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT, 2)
        tax2 = round(max(0, u2 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT, 2)

        valid_moves = []
        valid_attacks = []
        inspector = {
            "title": srv_txt(self.lang, "nothing_selected"),
            "lines": [srv_txt(self.lang, "click_prompt")],
            "can_heal": False,
            "can_harvest": False,
            "can_build_tower": False,
            "can_build_barracks": False,
            "can_disband": False,
            "can_demolish": False,
            "can_spawn_worker": False,
            "can_spawn_warrior": False,
            "can_spawn_archer": False
        }

        if self.selected_pos is not None:
            sr, sc = int(self.selected_pos[0]), int(self.selected_pos[1])
            base_v = float(self.env.tile_yields[sr, sc])
            eff_v = float(self.env.get_effective_tile_yield(sr, sc))
            owner_key = {0: "owner_neutral", 1: "owner_ai", 2: "owner_player"}[int(self.env.territory[sr, sc])]
            owner_str = srv_txt(self.lang, owner_key)
            lines = []
            if self.env.is_tile_built_up(sr, sc):
                lines.append(srv_txt(self.lang, "land_built", base=base_v, owner=owner_str))
            else:
                lines.append(srv_txt(self.lang, "land_free", eff=eff_v, owner=owner_str))

            if (sr, sc) in self.env.units:
                u = self.env.units[(sr, sc)]
                st = UNIT_STATS[u['type']]
                u_name = get_unit_name(u['type'], self.lang)
                f_str = srv_txt(self.lang, "owner_player") if int(u['faction']) == 2 else srv_txt(self.lang, "owner_ai")
                r_str = srv_txt(self.lang, "status_ready") if bool(u['ready']) else srv_txt(self.lang, "status_moved")
                lines.append(srv_txt(self.lang, "unit_header", name=u_name, faction=f_str, status=r_str))
                lines.append(srv_txt(self.lang, "hp_line", hp=u['hp'], max_hp=u['max_hp'], pct=(u['hp'] / u['max_hp']) * 100.00))
                lines.append(srv_txt(self.lang, "stats_line", move=st['move_range'], atk=st['atk_range'], dmg=st['dmg']))
                lines.append(srv_txt(self.lang, "upkeep_line", upk=st['upkeep'], eff=u['efficiency']))
                if int(u['faction']) == 2:
                    inspector["can_disband"] = True
                    if bool(u['ready']):
                        valid_moves = [[int(p[0]), int(p[1])] for p in self.env.get_valid_moves_for_unit((sr, sc))]
                        valid_attacks = [[int(p[0]), int(p[1])] for p in self.env.get_valid_attack_targets((sr, sc))]
                        inspector["can_heal"] = True
                        if u['type'] == 'worker':
                            inspector["can_harvest"] = bool(eff_v > 0.00)
                            inspector["can_build_tower"] = bool(
                                float(self.env.p2_gold) >= STRUCT_STATS[1]['cost']
                                and self.env._find_adjacent_tower_site((sr, sc), 2) is not None
                            )
                            inspector["can_build_barracks"] = bool(
                                float(self.env.p2_gold) >= STRUCT_STATS[2]['cost']
                                and self.env._find_barracks_build_site((sr, sc), 2, enforce_ai_distance=False) is not None
                            )
                            inspector["can_demolish"] = bool(
                                self.env._find_demolishable_structure_site((sr, sc), 2) is not None
                            )
            elif int(self.env.struct_owner[sr, sc]) == 2 and int(self.env.struct_type[sr, sc]) == 1 and bool(self.env.tower_ready[sr, sc]):
                valid_attacks = [[int(p[0]), int(p[1])] for p in self.env.get_valid_attack_targets((sr, sc), faction=2, atk_range=2)]

            if (sr, sc) == self.env.p2_castle_pos:
                lines.append(srv_txt(self.lang, "your_castle", hp=self.env.p2_castle_hp, max_hp=CASTLE_MAX_HP))
                if (sr, sc) not in self.env.units:
                    inspector["can_spawn_worker"] = bool(float(self.env.p2_gold) >= UNIT_STATS['worker']['cost'])
                    inspector["can_spawn_warrior"] = bool(float(self.env.p2_gold) >= UNIT_STATS['warrior']['cost'])
                    inspector["can_spawn_archer"] = bool(float(self.env.p2_gold) >= UNIT_STATS['archer']['cost'])
                else:
                    lines.append(srv_txt(self.lang, "castle_occupied"))
            elif (sr, sc) == self.env.p1_castle_pos:
                lines.append(srv_txt(self.lang, "ai_castle", hp=self.env.p1_castle_hp, max_hp=CASTLE_MAX_HP))
            elif int(self.env.struct_type[sr, sc]) != 0:
                st_t = int(self.env.struct_type[sr, sc])
                st_own = int(self.env.struct_owner[sr, sc])
                st_info = STRUCT_STATS[st_t]
                st_name = get_struct_name(st_t, self.lang)
                lines.append(srv_txt(self.lang, "struct_line", name=st_name, hp=float(self.env.struct_hp[sr, sc]), max_hp=st_info['max_hp']))
                if st_t == 2 and st_own == 2 and (sr, sc) not in self.env.units:
                    inspector["can_spawn_worker"] = bool(float(self.env.p2_gold) >= UNIT_STATS['worker']['cost'])
                    inspector["can_spawn_warrior"] = bool(float(self.env.p2_gold) >= UNIT_STATS['warrior']['cost'])
                    inspector["can_spawn_archer"] = bool(float(self.env.p2_gold) >= UNIT_STATS['archer']['cost'])

            inspector["lines"] = lines

        units_list = []
        for (ur, uc), u in self.env.units.items():
            units_list.append({
                "r": int(ur),
                "c": int(uc),
                "faction": int(u['faction']),
                "type": str(u['type']),
                "hp": round(float(u['hp']), 2),
                "max_hp": round(float(u['max_hp']), 2),
                "ready": bool(u['ready']),
                "efficiency": round(float(u.get('efficiency', 0.00)), 2)
            })

        structs_list = []
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                if int(self.env.struct_type[r, c]) != 0:
                    structs_list.append({
                        "r": int(r),
                        "c": int(c),
                        "type": int(self.env.struct_type[r, c]),
                        "owner": int(self.env.struct_owner[r, c]),
                        "hp": round(float(self.env.struct_hp[r, c]), 2),
                        "ready": bool(self.env.tower_ready[r, c])
                    })

        wins = self.profile.get("recent_wins", [])
        wr = round(float(np.mean(wins)) * 100.00, 2) if wins else 50.00

        raw_payload = {
            "username": str(self.username),
            "lang": str(self.lang),
            "matches_played": int(self.profile.get("matches_played", 0)),
            "turn": int(self.env.turn_count),
            "max_turns": int(MAX_STEPS),
            "winrate": float(wr),
            "base_rating": round(float(self.base_rating), 2),
            "active_rating": round(float(self.env.active_gen_idx), 2),
            "active_gen_label": str(self.env.active_gen_label),
            "active_gen_origin": str(self.env.active_gen_origin),
            "pool_size": len(PoolRegistry.get_all_generations()),
            "pool_max_rating": round(float(PoolRegistry.get_max_rating()), 2),
            "p2": {"gold": float(self.env.p2_gold), "inc": float(inc2), "upk": float(upk2), "tax": float(tax2), "net": float(net2), "tiles": int(t2), "army": int(u2), "castle_hp": float(self.env.p2_castle_hp)},
            "p1": {"gold": float(self.env.p1_gold), "inc": float(inc1), "upk": float(upk1), "tax": float(tax1), "net": float(net1), "tiles": int(t1), "army": int(u1), "castle_hp": float(self.env.p1_castle_hp)},
            "terrain": self.env.terrain.tolist(),
            "territory": self.env.territory.tolist(),
            "eff_yields": np.round(self.env.get_effective_yield_map(), 2).tolist(),
            "units": units_list,
            "structs": structs_list,
            "selected_pos": [int(self.selected_pos[0]), int(self.selected_pos[1])] if self.selected_pos else None,
            "valid_moves": valid_moves,
            "valid_attacks": valid_attacks,
            "ai_traces": self.env.last_ai_turn_traces,
            "event_log": self.env.turn_event_log,
            "inspector": inspector,
            "stats": self.env.get_efficiency_statistics(),
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
        return clean_for_json(raw_payload)

def get_session(username: str) -> GameSession:
    uname = username.strip() or "Guest1"
    if uname not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[uname] = GameSession(uname)
    return ACTIVE_SESSIONS[uname]

class ActionRequest(BaseModel):
    username: str
    action_type: str
    r: Optional[int] = None
    c: Optional[int] = None
    param: Optional[str] = None

@app.get("/api/init_user")
def api_init_user(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    guest_name = UserProfileManager.get_guest_name_for_ip(client_ip)
    session = get_session(guest_name)
    return {"username": guest_name, "lang": session.lang}

@app.post("/api/state")
def api_get_state(req: ActionRequest):
    session = get_session(req.username)
    if req.param in ("en", "uk"):
        session.set_language(req.param)
    return session.serialize_state()

@app.post("/api/action")
def api_perform_action(req: ActionRequest):
    session = get_session(req.username)
    env = session.env
    atype = req.action_type

    if atype == "set_lang" and req.param in ("en", "uk"):
        session.set_language(req.param)
        return session.serialize_state()

    if atype == "click_tile" and req.r is not None and req.c is not None:
        clicked = (int(req.r), int(req.c))
        if session.selected_pos is not None:
            valid_m = []
            valid_a = []
            sr, sc = int(session.selected_pos[0]), int(session.selected_pos[1])
            if (sr, sc) in env.units and env.units[(sr, sc)]['faction'] == 2 and env.units[(sr, sc)]['ready']:
                valid_m = env.get_valid_moves_for_unit((sr, sc))
                valid_a = env.get_valid_attack_targets((sr, sc))
            elif int(env.struct_owner[sr, sc]) == 2 and int(env.struct_type[sr, sc]) == 1 and bool(env.tower_ready[sr, sc]):
                valid_a = env.get_valid_attack_targets((sr, sc), faction=2, atk_range=2)

            if clicked in valid_m:
                dr, dc = clicked[0] - sr, clicked[1] - sc
                if (-dr, -dc) in RELATIVE_OFFSETS:
                    env.record_entity_decision('unit', (sr, sc), 2, RELATIVE_OFFSETS.index((-dr, -dc)))
                u_name = get_unit_name(env.units[(sr, sc)]['type'], session.lang)
                if env.execute_unit_move((sr, sc), clicked):
                    env.add_event_log(srv_txt(session.lang, "player_move", unit=u_name, r=clicked[0], c=clicked[1]))
                    session.selected_pos = clicked
                return session.serialize_state()

            if clicked in valid_a:
                is_tower = ((sr, sc) not in env.units and int(env.struct_owner[sr, sc]) == 2 and int(env.struct_type[sr, sc]) == 1)
                dr, dc = clicked[0] - sr, clicked[1] - sc
                if (-dr, -dc) in RELATIVE_OFFSETS:
                    env.record_entity_decision('tower' if is_tower else 'unit', (sr, sc), 2, 12 + RELATIVE_OFFSETS.index((-dr, -dc)))
                dmg = env.execute_attack_on_target((sr, sc), clicked, is_tower=is_tower)
                if dmg > 0.00:
                    env.add_event_log(srv_txt(session.lang, "player_attack", dmg=dmg, r=clicked[0], c=clicked[1]))
                    _check_and_handle_match_end(session)
                return session.serialize_state()

        session.selected_pos = clicked

    elif atype == "heal" and session.selected_pos in env.units:
        if env.units[session.selected_pos]['faction'] == 2:
            env.record_entity_decision('unit', session.selected_pos, 2, 24)
            gained = env.heal_unit_percentage(session.selected_pos)
            env.add_event_log(srv_txt(session.lang, "player_heal", gained=gained))

    elif atype == "harvest" and session.selected_pos in env.units:
        if env.units[session.selected_pos]['faction'] == 2:
            env.record_entity_decision('unit', session.selected_pos, 2, 25)
            bonus = env.execute_worker_harvest(session.selected_pos)
            if bonus > 0.00:
                env.add_event_log(srv_txt(session.lang, "player_harvest", bonus=bonus))

    elif atype == "build" and session.selected_pos in env.units:
        if env.units[session.selected_pos]['faction'] == 2:
            b_type = 1 if req.param == "tower" else 2
            env.record_entity_decision('unit', session.selected_pos, 2, 26 if b_type == 1 else 27)
            if env.execute_worker_build(session.selected_pos, build_type=b_type):
                b_name = get_struct_name(b_type, session.lang)
                env.add_event_log(srv_txt(session.lang, "player_build", struct=b_name))

    elif atype == "disband" and session.selected_pos in env.units:
        if env.units[session.selected_pos]['faction'] == 2:
            u_type = env.units[session.selected_pos]['type']
            u_name = get_unit_name(u_type, session.lang)
            env.record_entity_decision('unit', session.selected_pos, 2, 28)
            refund = env.execute_unit_disband(session.selected_pos)
            if refund > 0.00:
                env.add_event_log(srv_txt(session.lang, "player_disband", unit=u_name, refund=refund))
                session.selected_pos = None

    elif atype == "demolish" and session.selected_pos in env.units:
        if env.units[session.selected_pos]['faction'] == 2:
            env.record_entity_decision('unit', session.selected_pos, 2, 29)
            refund, st_t = env.execute_worker_demolish(session.selected_pos)
            if refund > 0.00:
                st_name = get_struct_name(st_t, session.lang)
                env.add_event_log(srv_txt(session.lang, "player_demolish", struct=st_name, refund=refund))

    elif atype == "spawn" and session.selected_pos is not None:
        u_type = req.param or "warrior"
        act_map = {'worker': 27, 'warrior': 28, 'archer': 29}
        env.record_entity_decision('spawn', session.selected_pos, 2, act_map.get(u_type, 28))
        if env.execute_structure_spawn(session.selected_pos, faction=2, unit_type=u_type):
            env.add_event_log(srv_txt(session.lang, "player_spawn", unit=get_unit_name(u_type, session.lang)))

    elif atype == "end_turn":
        env.heal_all_unmoved_units(faction=2)
        if not _check_and_handle_match_end(session):
            env.execute_full_ai_turn(faction=1, model=env.opponent_model, deterministic=False)
            if not _check_and_handle_match_end(session):
                env.turn_count += 1
                if env.turn_count <= 10 or env.turn_count % 3 == 0:
                    session.active_rating = env.compute_surge_dda_step(
                        current_rating=session.active_rating,
                        is_early_turn=(env.turn_count <= 10)
                    )
                    env.set_dda_generation(session.active_rating)
                env._prepare_faction_turn(faction=2, collect_economy=True)
                env.record_turn_efficiency_snapshot()
                _check_and_handle_match_end(session)

    return session.serialize_state()

def _check_and_handle_match_end(session: GameSession) -> bool:
    env = session.env
    done = False
    winner = None
    if env.p2_castle_hp <= 0.00:
        done = True
        winner = "p1"
    elif env.p1_castle_hp <= 0.00:
        done = True
        winner = "p2"
    elif env.turn_count > MAX_STEPS:
        done = True
        inc1, _, _, t1, u1 = env.get_faction_economy_stats(1)
        inc2, _, _, t2, u2 = env.get_faction_economy_stats(2)
        s1 = env.p1_castle_hp + t1 * 0.25 + inc1 * 4.00 + u1 * 1.50
        s2 = env.p2_castle_hp + t2 * 0.25 + inc2 * 4.00 + u2 * 1.50
        winner = "p1" if s1 >= s2 else "p2"

    if not done:
        return False

    human_won = (winner == "p2")
    winning_fac = 2 if human_won else 1

    m_obs, m_acts, m_weights = env.collect_match_training_samples(
        winning_faction=winning_fac,
        min_efficiency=MIN_TRAINING_EFFICIENCY
    )
    saved_cnt = env.export_high_efficiency_telemetry(
        winning_faction=winning_fac,
        min_efficiency=MIN_TRAINING_EFFICIENCY
    )

    prof = session.profile
    prof["matches_played"] = int(prof.get("matches_played", 0)) + 1
    wins = prof.get("recent_wins", [])
    wins.append(1.00 if human_won else 0.00)
    if len(wins) > 20:
        wins.pop(0)
    prof["recent_wins"] = wins

    wr = float(np.mean(wins))
    fast_wr = float(np.mean(wins[-5:]))
    err_wr = wr - 0.50
    d_wr = wr - float(prof.get("prev_winrate", 0.50))
    prof["prev_winrate"] = round(wr, 4)

    delta_macro = round((8.00 * err_wr) + (32.00 * d_wr) + (12.00 * (fast_wr - 0.50)), 2)
    max_pool_r = max(100.00, PoolRegistry.get_max_rating() + 15.00)
    blended = (0.55 * session.base_rating) + (0.45 * session.active_rating) + delta_macro
    session.base_rating = round(max(10.00, min(max_pool_r, blended)), 2)
    session.active_rating = session.base_rating
    prof["base_rating"] = session.base_rating
    UserProfileManager.update_profile(session.username, prof)

    winner_str = srv_txt(session.lang, "owner_player") if human_won else srv_txt(session.lang, "owner_ai")
    session.is_training_new_gen = True
    session.last_evolution_report = {
        "status": "training",
        "winner": winner_str,
        "samples": int(saved_cnt)
    }

    def _bg_evolve(obs_c, acts_c, w_c, uname, w_str):
        try:
            new_entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(obs_c, acts_c, w_c, username=uname)
            session.last_evolution_report = {
                "status": "completed",
                "winner": w_str,
                "new_gen_label": str(new_entry["label"]),
                "new_gen_rating": float(new_entry["rating"]),
                "gauntlet_winrate": float(new_entry["w_gauntlet"]),
                "doctrine_alignment": float(new_entry.get("doctrine_alignment", 100.00)),
                "samples": int(saved_cnt)
            }
        finally:
            session.is_training_new_gen = False
            session.env.set_dda_generation(session.active_rating)

    threading.Thread(target=_bg_evolve, args=(m_obs, m_acts, m_weights, session.username, winner_str), daemon=True).start()

    env.set_dda_generation(session.active_rating)
    env.reset()
    env._prepare_faction_turn(faction=2, collect_economy=False)
    session.selected_pos = None
    return True

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