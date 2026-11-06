import os
import json
import threading
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
import zipfile

from microrts_env import (
    MicroRTSEnv, PoolRegistry, OnlineEvolutionEngine,
    UNIT_STATS, STRUCT_STATS, CASTLE_MAX_HP, MAX_STEPS,
    MAX_GOLD, SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT,
    RELATIVE_OFFSETS, USERS_DB_FILE, POOL_DIR
)

app = FastAPI(title="Faction Warfare RTS — Multi-User Evolutionary Platform")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

USERS_LOCK = threading.Lock()
ACTIVE_SESSIONS = {}

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
        """Перевіряє IP і видає існуюче або нове ім'я Guest#"""
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
            
            # Одразу створюємо дефолтний профіль для нового гостя
            if guest_name not in db:
                db[guest_name] = {
                    "username": guest_name,
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
        uname = username.strip() or "Guest0"
        db = UserProfileManager.load_all()
        with USERS_LOCK:
            if uname not in db:
                db[uname] = {
                    "username": uname,
                    "base_rating": 35.00,
                    "matches_played": 0,
                    "recent_wins": [],
                    "prev_winrate": 0.50,
                    "created_generations": []
                }
                with open(USERS_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(db, f, indent=2, ensure_ascii=False)
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
        self.env = MicroRTSEnv(mode="manual_p2", record_telemetry=True)
        self.base_rating = float(self.profile.get("base_rating", 35.00))
        self.active_rating = self.base_rating
        self.env.set_dda_generation(self.active_rating)
        self.env._prepare_faction_turn(faction=2, collect_economy=False)
        self.selected_pos = None
        self.last_evolution_report = None
        self.is_training_new_gen = False

    def serialize_state(self) -> dict:
        inc1, upk1, net1, t1, u1 = self.env.get_faction_economy_stats(1)
        inc2, upk2, net2, t2, u2 = self.env.get_faction_economy_stats(2)
        tax1 = round(max(0, u1 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT, 2)
        tax2 = round(max(0, u2 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT, 2)

        valid_moves = []
        valid_attacks = []
        inspector = {
            "title": "Нічого не обрано",
            "lines": ["Клікніть на будь-якого юніта, вежу, казарму або замок на карті."],
            "can_heal": False,
            "can_harvest": False,
            "can_build_tower": False,
            "can_build_barracks": False,
            "can_spawn_worker": False,
            "can_spawn_warrior": False,
            "can_spawn_archer": False
        }

        if self.selected_pos is not None:
            sr, sc = int(self.selected_pos[0]), int(self.selected_pos[1])
            base_v = float(self.env.tile_yields[sr, sc])
            eff_v = float(self.env.get_effective_tile_yield(sr, sc))
            owner_str = {0: "Нейтральна (Сіра)", 1: "Червона Імперія (ШІ)", 2: "Синій Альянс (Ви)"}[int(self.env.territory[sr, sc])]
            lines = []
            if self.env.is_tile_built_up(sr, sc):
                lines.append(f"Земля: +0.00$/хід [Забудовано, базова: +{base_v:.2f}$] ({owner_str})")
            else:
                lines.append(f"Земля: +{eff_v:.2f}$/хід ({owner_str})")

            if (sr, sc) in self.env.units:
                u = self.env.units[(sr, sc)]
                st = UNIT_STATS[u['type']]
                f_str = "Синій Альянс (Ваш)" if int(u['faction']) == 2 else "Червона Імперія (ШІ)"
                r_str = "ГОТОВИЙ ДО ДІЇ" if bool(u['ready']) else "ВЖЕ ХОДИВ У ЦЬОМУ РАУНДІ"
                lines.append(f"Підрозділ: {st['name']} [{f_str}] — {r_str}")
                lines.append(f"Здоров'я: {u['hp']:.2f} / {u['max_hp']:.2f} HP ({(u['hp'] / u['max_hp']) * 100.00:.2f}%)")
                lines.append(f"Рух: {st['move_range']} кліт. | Дальність атаки: {st['atk_range']} кліт. | Шкода: {st['dmg']:.2f}")
                lines.append(f"Плата за службу: {st['upkeep']:.2f}$/хід | Ефективність: {u['efficiency']:.2f}")
                if int(u['faction']) == 2 and bool(u['ready']):
                    valid_moves = [[int(p[0]), int(p[1])] for p in self.env.get_valid_moves_for_unit((sr, sc))]
                    valid_attacks = [[int(p[0]), int(p[1])] for p in self.env.get_valid_attack_targets((sr, sc))]
                    inspector["can_heal"] = True
                    if u['type'] == 'worker':
                        inspector["can_harvest"] = bool(eff_v > 0.00)
                        inspector["can_build_tower"] = bool(
                            float(self.env.p2_gold) >= STRUCT_STATS[1]['cost']
                            and self.env._find_adjacent_tower_site((sr, sc), 2) is not None
                        )
                        can_b = bool(
                            (sr, sc) not in [self.env.p1_castle_pos, self.env.p2_castle_pos]
                            and int(self.env.struct_type[sr, sc]) == 0
                        )
                        inspector["can_build_barracks"] = bool(
                            float(self.env.p2_gold) >= STRUCT_STATS[2]['cost']
                            and (can_b or self.env._find_adjacent_tower_site((sr, sc), 2) is not None)
                        )
            elif int(self.env.struct_owner[sr, sc]) == 2 and int(self.env.struct_type[sr, sc]) == 1 and bool(self.env.tower_ready[sr, sc]):
                valid_attacks = [[int(p[0]), int(p[1])] for p in self.env.get_valid_attack_targets((sr, sc), faction=2, atk_range=2)]

            if (sr, sc) == self.env.p2_castle_pos:
                lines.append(f"Споруда: ВАШ ЗАМОК ({self.env.p2_castle_hp:.2f} / {CASTLE_MAX_HP:.2f} HP)")
                if (sr, sc) not in self.env.units:
                    inspector["can_spawn_worker"] = bool(float(self.env.p2_gold) >= UNIT_STATS['worker']['cost'])
                    inspector["can_spawn_warrior"] = bool(float(self.env.p2_gold) >= UNIT_STATS['warrior']['cost'])
                    inspector["can_spawn_archer"] = bool(float(self.env.p2_gold) >= UNIT_STATS['archer']['cost'])
                else:
                    lines.append("Замок зайнятий юнітом — виведіть його для найму нового.")
            elif (sr, sc) == self.env.p1_castle_pos:
                lines.append(f"Споруда: ЗАМОК ШІ ({self.env.p1_castle_hp:.2f} / {CASTLE_MAX_HP:.2f} HP)")
            elif int(self.env.struct_type[sr, sc]) != 0:
                st_t = int(self.env.struct_type[sr, sc])
                st_own = int(self.env.struct_owner[sr, sc])
                st_info = STRUCT_STATS[st_t]
                lines.append(f"Споруда: {st_info['name']} ({float(self.env.struct_hp[sr, sc]):.2f} / {st_info['max_hp']:.2f} HP)")
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
            "last_evolution_report": self.last_evolution_report
        }
        return clean_for_json(raw_payload)

def get_session(username: str) -> GameSession:
    uname = username.strip() or "Guest"
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
    """Визначає IP клієнта і повертає існуючий або новий Guest-акаунт"""
    client_ip = request.client.host
    guest_name = UserProfileManager.get_guest_name_for_ip(client_ip)
    return {"username": guest_name}

@app.post("/api/state")
def api_get_state(req: ActionRequest):
    session = get_session(req.username)
    return session.serialize_state()

@app.post("/api/action")
def api_perform_action(req: ActionRequest):
    session = get_session(req.username)
    env = session.env
    atype = req.action_type

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
                u_name = UNIT_STATS[env.units[(sr, sc)]['type']]['name']
                if env.execute_unit_move((sr, sc), clicked):
                    env.add_event_log(f"Ви: {u_name} перемістився на ({clicked[0]},{clicked[1]})")
                    session.selected_pos = clicked
                return session.serialize_state()

            if clicked in valid_a:
                is_tower = ((sr, sc) not in env.units and int(env.struct_owner[sr, sc]) == 2 and int(env.struct_type[sr, sc]) == 1)
                dr, dc = clicked[0] - sr, clicked[1] - sc
                if (-dr, -dc) in RELATIVE_OFFSETS:
                    env.record_entity_decision('tower' if is_tower else 'unit', (sr, sc), 2, 12 + RELATIVE_OFFSETS.index((-dr, -dc)))
                dmg = env.execute_attack_on_target((sr, sc), clicked, is_tower=is_tower)
                if dmg > 0.00:
                    env.add_event_log(f"Ви: завдано {dmg:.2f} шкоди по цілі ({clicked[0]},{clicked[1]})")
                    _check_and_handle_match_end(session)
                return session.serialize_state()

        session.selected_pos = clicked

    elif atype == "heal" and session.selected_pos in env.units:
        env.record_entity_decision('unit', session.selected_pos, 2, 24)
        gained = env.heal_unit_percentage(session.selected_pos)
        env.add_event_log(f"Ви: підрозділ відновив +{gained:.2f} HP")

    elif atype == "harvest" and session.selected_pos in env.units:
        env.record_entity_decision('unit', session.selected_pos, 2, 25)
        bonus = env.execute_worker_harvest(session.selected_pos)
        env.add_event_log(f"Ви: Робітник добув додатково +{bonus:.2f}$")

    elif atype == "build" and session.selected_pos in env.units:
        b_type = 1 if req.param == "tower" else 2
        env.record_entity_decision('unit', session.selected_pos, 2, 26 if b_type == 1 else 27)
        if env.execute_worker_build(session.selected_pos, build_type=b_type):
            b_name = "Оборонну Вежу" if b_type == 1 else "Передову Казарму"
            env.add_event_log(f"Ви: Робітник звів {b_name}")

    elif atype == "spawn" and session.selected_pos is not None:
        u_type = req.param or "warrior"
        act_map = {'worker': 27, 'warrior': 28, 'archer': 29}
        env.record_entity_decision('spawn', session.selected_pos, 2, act_map.get(u_type, 28))
        if env.execute_structure_spawn(session.selected_pos, faction=2, unit_type=u_type):
            env.add_event_log(f"Ви: найнято підрозділ ({UNIT_STATS[u_type]['name']})")

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

    m_obs, m_acts, m_weights = env.collect_match_training_samples(winning_faction=winning_fac, min_efficiency=2.50)
    saved_cnt = env.export_high_efficiency_telemetry(winning_faction=winning_fac, min_efficiency=2.50)

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

    session.is_training_new_gen = True
    session.last_evolution_report = {
        "status": "training",
        "winner": "Синій Альянс (Ви)" if human_won else "Червона Імперія (ШІ)",
        "samples": int(saved_cnt)
    }

    def _bg_evolve(obs_c, acts_c, w_c, uname):
        try:
            new_entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(obs_c, acts_c, w_c, username=uname)
            session.last_evolution_report = {
                "status": "completed",
                "winner": "Синій Альянс (Ви)" if human_won else "Червона Імперія (ШІ)",
                "new_gen_label": str(new_entry["label"]),
                "new_gen_rating": float(new_entry["rating"]),
                "gauntlet_winrate": float(new_entry["w_gauntlet"]),
                "samples": int(saved_cnt)
            }
        finally:
            session.is_training_new_gen = False
            session.env.set_dda_generation(session.active_rating)

    threading.Thread(target=_bg_evolve, args=(m_obs, m_acts, m_weights, session.username), daemon=True).start()

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
    print("[*] Багатокористувацький сервер Faction Warfare RTS запущено: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)