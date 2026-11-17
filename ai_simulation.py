import os
import copy
import time
import math
import random
import threading
from typing import Optional
import numpy as np

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
ROLE_IDS = getattr(cfg, "ROLE_IDS", {"worker": 0, "warrior": 1, "archer": 2, "tower": 3, "spawn": 4})

TERRAIN_MOUNTAIN = 0
TERRAIN_PLAIN = 1
TERRAIN_FERTILE = 2
TERRAIN_GOLD = 3

CASTLE_POSITIONS = {
    1: (GRID_SIZE - 1, 0),
    2: (0, GRID_SIZE - 1)
}

PoolRegistry = getattr(ai_pool, "PoolRegistry", None)
OnlineEvolutionEngine = getattr(ai_pool, "OnlineEvolutionEngine", None)
encode_state_from_live = getattr(ai_strategist, "encode_state_from_live", None)
compute_unit_efficiency = getattr(ai_strategist, "compute_unit_efficiency", lambda *a, **kw: 0.0)

SIM_LOCK = threading.Lock()


class SimUnit:
    def __init__(self, r: int, c: int, faction: int, u_type: str, ready: bool = True):
        self.r = int(r)
        self.c = int(c)
        self.faction = int(faction)
        self.u_type = str(u_type)
        stats = UNIT_STATS.get(self.u_type, {"max_hp": 4.0, "move_range": 2, "atk_range": 1, "dmg": 1.0, "cost": 3.0, "upkeep": 0.4})
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
        return float(compute_unit_efficiency(
            self.u_type,
            self.dmg_dealt,
            self.tiles_captured,
            self.gold_harvested,
            self.builds_done,
            self.turns_alive
        ))


class SimStruct:
    def __init__(self, r: int, c: int, owner: int, s_type: int, ready: bool = False):
        self.r = int(r)
        self.c = int(c)
        self.owner = int(owner)
        self.s_type = int(s_type)
        stats = STRUCT_STATS.get(self.s_type, {"max_hp": 8.0, "atk_range": 2, "dmg": 2.0, "cost": 4.0})
        self.max_hp = float(stats["max_hp"])
        self.hp = float(self.max_hp)
        self.atk_range = int(stats["atk_range"])
        self.dmg = float(stats["dmg"])
        self.cost = float(stats["cost"])
        self.ready = bool(ready)


class SimulationSession:
    def __init__(self, username: str):
        self.username = username
        self.paused = False
        self.skip_evo_wait = False
        self.step_delay = 0.50
        self.matches_played = 0
        self.f1_wins = 0
        self.f2_wins = 0
        self.f1_rating_anchor = 45.0
        self.f2_rating_anchor = 48.0
        self.f1_competence = 1.00
        self.f2_competence = 1.00
        self.is_training_new_gen = False
        self.last_evolution_report = None
        self.selected_pos = None
        self.history = []
        self.cursor = 0
        self._start_new_match()

    @property
    def f1_winrate(self) -> float:
        total = self.f1_wins + self.f2_wins
        if total <= 0:
            return 50.00
        return round((self.f1_wins / total) * 100.00, 2)

    @property
    def f2_winrate(self) -> float:
        total = self.f1_wins + self.f2_wins
        if total <= 0:
            return 50.00
        return round((self.f2_wins / total) * 100.00, 2)

    def _select_balanced_models(self):
        max_r = PoolRegistry.get_max_rating() if (PoolRegistry and hasattr(PoolRegistry, "get_max_rating")) else 100.0
        wr1 = self.f1_winrate
        total_m = self.f1_wins + self.f2_wins

        if total_m == 0:
            r1_target = min(max_r, 50.0)
            r2_target = min(max_r, 52.0)
            self.f1_competence = 1.00
            self.f2_competence = 1.00
        else:
            diff = wr1 - 50.00
            if wr1 > 60.00:
                r1_target = max(10.0, min(max_r, self.f1_rating_anchor - 4.0))
                r2_target = max(10.0, min(max_r, r1_target + 0.85 * diff))
            elif wr1 < 40.00:
                r2_target = max(10.0, min(max_r, self.f2_rating_anchor - 4.0))
                r1_target = max(10.0, min(max_r, r2_target + 0.85 * (50.00 - wr1)))
            else:
                base_r = min(max_r, 45.0 + (self.matches_played * 3.5) % max(20.0, max_r - 25.0))
                r1_target = base_r
                r2_target = max(10.0, min(max_r, base_r + 0.65 * diff))

            self.f1_competence = round(float(np.clip(1.00 - 1.20 * (diff / 100.00), 0.65, 1.35)), 2)
            self.f2_competence = round(float(np.clip(1.00 + 1.20 * (diff / 100.00), 0.65, 1.35)), 2)

        if PoolRegistry and hasattr(PoolRegistry, "select_closest_generation"):
            e1 = PoolRegistry.select_closest_generation(r1_target)
            e2 = PoolRegistry.select_closest_generation(r2_target)
        else:
            e1 = {"label": "Gen-50", "filename": "generation_50.zip", "rating": round(r1_target, 2)}
            e2 = {"label": "Gen-52", "filename": "generation_52.zip", "rating": round(r2_target, 2)}

        self.f1_rating_anchor = float(e1.get("rating", r1_target))
        self.f2_rating_anchor = float(e2.get("rating", r2_target))
        return e1, e2

    def _start_new_match(self):
        self.turn = 1
        self.max_turns = int(MAX_STEPS)
        self.match_finished = False
        self.winner_faction = None
        self.winner_name = ""
        self.event_log = []
        self.ai_traces = []
        self.units = {}
        self.structs = {}
        self.stats_turns = [1]
        self.stats_factions_eff = {"1": [0.0], "2": [0.0]}
        self.match_obs = {1: [], 2: []}
        self.match_acts = {1: [], 2: []}
        self.match_effs = {1: [], 2: []}
        self.match_roles = {1: [], 2: []}

        e1, e2 = self._select_balanced_models()
        self.f1_entry = e1
        self.f2_entry = e2
        self.active_gen_label = f"{e1.get('label', 'AI-1')} vs {e2.get('label', 'AI-2')}"
        self.active_rating = round((float(e1.get("rating", 50.0)) + float(e2.get("rating", 50.0))) / 2.0, 2)

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

        self.factions = {}
        for f, entry in [(1, e1), (2, e2)]:
            cr, cc = CASTLE_POSITIONS[f]
            lbl = f"AI [{entry.get('label', f'Gen-{f}')}]"
            self.factions[f] = {
                "faction": f,
                "controller": lbl,
                "kind": "ai",
                "ai_entry": entry,
                "gold": float(INITIAL_GOLD),
                "castle_hp": float(CASTLE_MAX_HP),
                "castle_pos": (cr, cc),
                "alive": True
            }
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
                2: [(1, 0, "worker"), (0, -1, "warrior"), (1, -1, "archer")]
            }
            for dr, dc, ut in spawn_offsets[f]:
                ur, uc = cr + dr, cc + dc
                self.units[(ur, uc)] = SimUnit(ur, uc, f, ut, ready=True)
                self.territory[ur, uc] = f

        self.current_turn_faction = 1
        self._log(f"Simulation Match #{self.matches_played + 1}: {self.active_gen_label}")
        self.history = [self._create_snapshot()]
        self.cursor = 0

    def _log(self, msg: str):
        self.event_log.append(str(msg))
        if len(self.event_log) > 25:
            self.event_log = self.event_log[-25:]

    def _effective_tile_yield(self, r: int, c: int) -> float:
        t_type = int(self.terrain[r, c])
        if t_type == TERRAIN_MOUNTAIN:
            return 0.0
        if (r, c) in self.structs:
            return 0.0
        for fd in self.factions.values():
            if fd["alive"] and fd["castle_pos"] == (r, c):
                return 0.0
        return float(TILE_VALUES.get(t_type, 0.05))

    def _compute_economy(self, faction: int):
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

    def _compute_total_eff(self, faction: int) -> float:
        f_units = [u for u in self.units.values() if u.faction == faction]
        u_eff = sum(u.efficiency for u in f_units)
        tiles_cnt, _, _, _, net = self._compute_economy(faction)
        return round(u_eff + tiles_cnt * 1.5 + max(0.0, net) * 8.0, 2)

    def _can_build_at(self, faction: int, r: int, c: int, b_type: int) -> bool:
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            return False
        if self.terrain[r, c] == TERRAIN_MOUNTAIN or (r, c) in self.structs:
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

    def _get_valid_build_tiles(self, u: SimUnit, b_type: int) -> list:
        cost = float(STRUCT_STATS.get(b_type, {"cost": 4.0})["cost"])
        if self.factions[u.faction]["gold"] + 1e-9 < cost:
            return []
        valid = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            br, bc = u.r + dr, u.c + dc
            if self._can_build_at(u.faction, br, bc, b_type):
                valid.append([br, bc])
        return valid

    def _path_capture_score(self, faction: int, path: list) -> float:
        score = 0.0
        for pr, pc in path:
            if self.territory[pr, pc] != faction:
                score += 1.0 + self._effective_tile_yield(pr, pc)
        return score

    def _get_valid_move_paths(self, u: SimUnit) -> dict:
        if not u or not u.ready or u.moved:
            return {}
        best_dist = {(u.r, u.c): 0}
        best_path = {(u.r, u.c): []}
        queue = [(u.r, u.c, 0, [])]
        valid_paths = {}

        while queue:
            cr, cc, dist, path = queue.pop(0)
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
                enemy_castle = any(fd["alive"] and fk != u.faction and fd["castle_pos"] == (nr, nc) for fk, fd in self.factions.items())
                if enemy_castle:
                    continue
                other_u = self.units.get((nr, nc))
                if other_u and other_u.faction != u.faction:
                    continue

                ndist = dist + 1
                npath = path + [(nr, nc)]
                prev_d = best_dist.get((nr, nc))
                if prev_d is None or ndist < prev_d:
                    best_dist[(nr, nc)] = ndist
                    best_path[(nr, nc)] = npath
                    queue.append((nr, nc, ndist, npath))
                    if other_u is None:
                        valid_paths[(nr, nc)] = npath
                elif ndist == prev_d:
                    if self._path_capture_score(u.faction, npath) > self._path_capture_score(u.faction, best_path[(nr, nc)]):
                        best_path[(nr, nc)] = npath
                        queue.append((nr, nc, ndist, npath))
                        if other_u is None:
                            valid_paths[(nr, nc)] = npath
        return valid_paths

    def _capture_along_path(self, u: SimUnit, start_pos: tuple, path: list):
        for sr, sc in [start_pos] + list(path):
            if self.territory[sr, sc] != u.faction:
                self.territory[sr, sc] = u.faction
                u.tiles_captured += 1

    def _get_valid_attacks_for_unit(self, u: SimUnit) -> list:
        if not u or not u.ready or u.attacked:
            return []
        targets = []
        for r in range(max(0, u.r - u.atk_range), min(GRID_SIZE, u.r + u.atk_range + 1)):
            for c in range(max(0, u.c - u.atk_range), min(GRID_SIZE, u.c + u.atk_range + 1)):
                if 1 <= abs(r - u.r) + abs(c - u.c) <= u.atk_range:
                    ou = self.units.get((r, c))
                    if ou and ou.faction != u.faction:
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

    def _get_valid_attacks_for_tower(self, st: SimStruct) -> list:
        if not st or st.s_type != 1 or not st.ready:
            return []
        targets = []
        for r in range(max(0, st.r - st.atk_range), min(GRID_SIZE, st.r + st.atk_range + 1)):
            for c in range(max(0, st.c - st.atk_range), min(GRID_SIZE, st.c + st.atk_range + 1)):
                if 1 <= abs(r - st.r) + abs(c - st.c) <= st.atk_range:
                    ou = self.units.get((r, c))
                    if ou and ou.faction != st.owner:
                        targets.append([r, c])
                        continue
                    ost = self.structs.get((r, c))
                    if ost and ost.owner != st.owner:
                        targets.append([r, c])
                        continue
        return targets

    def _record_sim_sample(self, faction: int, r: int, c: int, act_idx: int, role_key: str, weight: float):
        try:
            obs = np.zeros((13, GRID_SIZE, GRID_SIZE), dtype=np.float32)
            obs[0, r, c] = 1.0
            for tr_r in range(GRID_SIZE):
                for tr_c in range(GRID_SIZE):
                    t_val = int(self.terrain[tr_r, tr_c])
                    if 0 <= t_val <= 3:
                        obs[1 + t_val, tr_r, tr_c] = 1.0
                    owner = int(self.territory[tr_r, tr_c])
                    if owner == faction:
                        obs[5, tr_r, tr_c] = 1.0
                    elif owner != 0:
                        obs[6, tr_r, tr_c] = 1.0
            for fk, fd in self.factions.items():
                if fd["alive"]:
                    cr, cc = fd["castle_pos"]
                    obs[7 if fk == faction else 8, cr, cc] = fd["castle_hp"] / float(CASTLE_MAX_HP)
            for st in self.structs.values():
                obs[7 if st.owner == faction else 8, st.r, st.c] = st.hp / float(st.max_hp)
            for u in self.units.values():
                if u.u_type == "worker":
                    ch = 9
                elif u.u_type == "warrior":
                    ch = 10
                else:
                    ch = 11
                sign = 1.0 if u.faction == faction else -1.0
                obs[ch, u.r, u.c] = sign * (u.hp / float(u.max_hp))
            obs[12, :, :] = min(1.0, self.factions[faction]["gold"] / 50.0)

            self.match_obs[faction].append(obs)
            self.match_acts[faction].append(int(act_idx))
            self.match_effs[faction].append(float(max(MIN_TRAINING_EFFICIENCY, weight)))
            self.match_roles[faction].append(int(ROLE_IDS.get(role_key, 1)))
        except Exception:
            pass

    def _apply_damage_at(self, attacker_faction: int, tr_pos: int, tc_pos: int, dmg: float, u_attacker: Optional[SimUnit] = None) -> float:
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
                    fd["alive"] = False
                    self._finish_simulation_match(attacker_faction)
                return dealt
        return 0.0

    def _finish_simulation_match(self, winner_faction: int):
        if self.match_finished:
            return
        self.match_finished = True
        self.winner_faction = int(winner_faction)
        self.winner_name = self.factions[self.winner_faction]["controller"]
        self.matches_played += 1
        if self.winner_faction == 1:
            self.f1_wins += 1
        else:
            self.f2_wins += 1

        self._log(f"Match #{self.matches_played} Winner: F{self.winner_faction} ({self.winner_name}) | F1 WR: {self.f1_winrate:.2f}%")

        w_obs = list(self.match_obs.get(self.winner_faction, []))
        w_acts = list(self.match_acts.get(self.winner_faction, []))
        w_effs = list(self.match_effs.get(self.winner_faction, []))
        w_roles = list(self.match_roles.get(self.winner_faction, []))
        w_rating = float(self.f1_rating_anchor if self.winner_faction == 1 else self.f2_rating_anchor)

        if len(w_obs) >= 1 and OnlineEvolutionEngine is not None and not self.is_training_new_gen:
            self.is_training_new_gen = True
            self.last_evolution_report = {"status": "training", "samples": len(w_obs)}

            def _bg_sim_evolve(obs_l=w_obs, acts_l=w_acts, effs_l=w_effs, roles_l=w_roles, p_rating=w_rating):
                try:
                    try:
                        entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(
                            username="AI_Sim",
                            player_rating=p_rating,
                            match_obs=np.array(obs_l, dtype=np.float32),
                            match_actions=np.array(acts_l, dtype=np.int64),
                            match_efficiencies=np.array(effs_l, dtype=np.float32),
                            match_roles=roles_l
                        )
                    except TypeError:
                        entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(
                            match_obs=obs_l,
                            match_acts=acts_l,
                            match_weights=effs_l,
                            username="AI_Sim",
                            match_roles=roles_l
                        )
                    if entry:
                        self.last_evolution_report = {
                            "status": "completed",
                            "new_gen_label": entry.get("label", "Gen-Sim"),
                            "new_gen_rating": float(entry.get("rating", p_rating))
                        }
                except Exception as e:
                    print(f"[SimEvolution] Error: {e}")
                finally:
                    self.is_training_new_gen = False

            threading.Thread(target=_bg_sim_evolve, daemon=True).start()

    def _execute_single_faction_step(self, ai_faction: int):
        comp = self.f1_competence if ai_faction == 1 else self.f2_competence
        for u in self.units.values():
            if u.faction == ai_faction:
                u.moved = False
                u.attacked = False
                u.ready = True
                u.turns_alive += 1
        for st in self.structs.values():
            if st.owner == ai_faction:
                st.ready = True

        self.ai_traces = []
        moves_cnt = 0
        atks_cnt = 0

        for pos, st in list(self.structs.items()):
            if st.owner == ai_faction and st.s_type == 1 and st.ready:
                targets = self._get_valid_attacks_for_tower(st)
                if targets:
                    tr_pos, tc_pos = targets[0]
                    self._apply_damage_at(ai_faction, tr_pos, tc_pos, st.dmg)
                    st.ready = False
                    atks_cnt += 1
                    self.ai_traces.append({"from": [st.r, st.c], "to": [tr_pos, tc_pos], "kind": "attack"})
                    self._record_sim_sample(ai_faction, st.r, st.c, 12, "tower", 115.0)
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
                self._apply_damage_at(ai_faction, tr_pos, tc_pos, u.dmg, u_attacker=u)
                u.attacked = True
                u.ready = False
                atks_cnt += 1
                self.ai_traces.append({"from": [u.r, u.c], "to": [tr_pos, tc_pos], "kind": "attack"})
                self._record_sim_sample(ai_faction, u.r, u.c, 12, u.u_type, max(110.0, u.efficiency))
                continue

            if u.u_type == "worker":
                tile_y = self._effective_tile_yield(u.r, u.c)
                if tile_y >= 0.15 and random.random() < (0.50 * comp):
                    bonus = round(tile_y * 1.5, 4)
                    self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] + bonus, 4)
                    u.gold_harvested += bonus
                    u.moved = True
                    u.attacked = True
                    u.ready = False
                    self._record_sim_sample(ai_faction, u.r, u.c, 25, "worker", max(105.0, u.efficiency))
                    continue

                if self.factions[ai_faction]["gold"] >= 6.0 and random.random() < (0.32 * comp):
                    b_type = 2 if (self.factions[ai_faction]["gold"] >= 7.5 and random.random() < 0.48) else 1
                    b_tiles = self._get_valid_build_tiles(u, b_type)
                    if b_tiles:
                        br, bc = b_tiles[0]
                        cost = float(STRUCT_STATS.get(b_type, {"cost": 4.0})["cost"])
                        self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] - cost, 4)
                        self.structs[(br, bc)] = SimStruct(br, bc, ai_faction, b_type, ready=False)
                        self.territory[br, bc] = ai_faction
                        u.moved = True
                        u.attacked = True
                        u.ready = False
                        u.builds_done += 1
                        self._record_sim_sample(ai_faction, u.r, u.c, 26 if b_type == 1 else 27, "worker", max(112.0, u.efficiency))
                        continue

            move_paths = self._get_valid_move_paths(u)
            v_moves = [[pos[0], pos[1]] for pos in move_paths.keys()]
            if v_moves:
                enemy_f = 2 if ai_faction == 1 else 1
                target_goal = self.factions[enemy_f]["castle_pos"]
                noise_scale = max(0.05, 1.45 - comp)

                best_m = min(
                    v_moves,
                    key=lambda m: (
                        (abs(m[0] - target_goal[0]) + abs(m[1] - target_goal[1]))
                        - self._path_capture_score(ai_faction, move_paths.get((m[0], m[1]), [(m[0], m[1])])) * (2.2 * comp)
                        - self._effective_tile_yield(m[0], m[1]) * (4.2 * comp)
                        + random.uniform(0.0, noise_scale)
                    )
                )
                old_r, old_c = u.r, u.c
                nr, nc = best_m
                chosen_path = move_paths.get((nr, nc), [(nr, nc)])
                del self.units[(old_r, old_c)]
                self._capture_along_path(u, (old_r, old_c), chosen_path)
                u.r, u.c = nr, nc
                u.moved = True
                self.units[(nr, nc)] = u
                moves_cnt += 1
                self.ai_traces.append({"from": [old_r, old_c], "to": [nr, nc], "kind": "move"})
                self._record_sim_sample(ai_faction, old_r, old_c, 0, u.u_type, max(105.0, u.efficiency))

                post_atks = self._get_valid_attacks_for_unit(u)
                if post_atks:
                    tr_pos, tc_pos = post_atks[0]
                    self._apply_damage_at(ai_faction, tr_pos, tc_pos, u.dmg, u_attacker=u)
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
            elif gold >= UNIT_STATS["archer"]["cost"] and random.random() < (0.48 * comp):
                choice = "archer"
            elif gold >= UNIT_STATS["warrior"]["cost"]:
                choice = "warrior"
            elif w_cnt < MAX_WORKERS_PER_FACTION and gold >= UNIT_STATS["worker"]["cost"] and random.random() < 0.35:
                choice = "worker"
                w_cnt += 1

            if choice:
                c_val = float(UNIT_STATS[choice]["cost"])
                self.factions[ai_faction]["gold"] = round(self.factions[ai_faction]["gold"] - c_val, 4)
                self.units[(sp_r, sp_c)] = SimUnit(sp_r, sp_c, ai_faction, choice, ready=False)
                self.territory[sp_r, sp_c] = ai_faction
                sp_act = 27 if choice == "worker" else (28 if choice == "warrior" else 29)
                self._record_sim_sample(ai_faction, sp_r, sp_c, sp_act, "spawn", 110.0)

        self._log(f"R{self.turn} F{ai_faction}: {moves_cnt} moves, {atks_cnt} attacks")

        if not self.match_finished:
            if ai_faction == 1:
                self.current_turn_faction = 2
            else:
                self.current_turn_faction = 1
                self._process_round_end()

    def _process_round_end(self):
        self.turn += 1
        for f, fd in self.factions.items():
            if not fd["alive"]:
                continue
            _, _, _, _, net = self._compute_economy(f)
            fd["gold"] = round(fd["gold"] + net, 4)
            if fd["gold"] < -1e-5:
                fd["gold"] = 0.0
                to_del = [pos for pos, u in self.units.items() if u.faction == f]
                for pos in to_del:
                    del self.units[pos]

        self.stats_turns.append(int(self.turn))
        for fk in ("1", "2"):
            self.stats_factions_eff[fk].append(self._compute_total_eff(int(fk)))

        if self.turn >= self.max_turns and not self.match_finished:
            s1 = (self.factions[1]["castle_hp"], self._compute_total_eff(1))
            s2 = (self.factions[2]["castle_hp"], self._compute_total_eff(2))
            self._finish_simulation_match(1 if s1 >= s2 else 2)

    def _create_snapshot(self) -> dict:
        eff_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                eff_grid[r, c] = self._effective_tile_yield(r, c)

        factions_out = {}
        for fk, fd in self.factions.items():
            tiles_cnt, inc, army_cnt, upk, net = self._compute_economy(fk)
            factions_out[str(fk)] = {
                "controller": fd["controller"],
                "kind": "ai",
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

        factions_median = {
            fk: round(float(np.median(arr)), 2) if arr else 0.0
            for fk, arr in self.stats_factions_eff.items()
        }

        return {
            "turn": int(self.turn),
            "current_turn_faction": int(self.current_turn_faction),
            "current_turn_user": self.factions[self.current_turn_faction]["controller"],
            "match_finished": bool(self.match_finished),
            "winner_faction": self.winner_faction,
            "winner_name": self.winner_name,
            "factions": copy.deepcopy(factions_out),
            "terrain": self.terrain.tolist(),
            "territory": self.territory.tolist(),
            "eff_yields": np.round(eff_grid, 2).tolist(),
            "units": copy.deepcopy(units_out),
            "structs": copy.deepcopy(structs_out),
            "ai_traces": copy.deepcopy(self.ai_traces),
            "event_log": list(self.event_log),
            "stats": {
                "turns": list(self.stats_turns),
                "factions_eff": copy.deepcopy(self.stats_factions_eff),
                "factions_median": factions_median,
                "player_eff": list(self.stats_factions_eff.get("2", [0.0])),
                "ai_eff": list(self.stats_factions_eff.get("1", [0.0])),
                "player_median": factions_median.get("2", 0.0),
                "ai_median": factions_median.get("1", 0.0)
            }
        }

    def step_forward(self, is_auto: bool = False):
        if is_auto and self.paused:
            return
        if self.cursor < len(self.history) - 1:
            self.cursor += 1
            return
        if self.match_finished:
            if not self.skip_evo_wait and self.is_training_new_gen:
                return
            self._start_new_match()
            return
        self._execute_single_faction_step(self.current_turn_faction)
        self.history.append(self._create_snapshot())
        if len(self.history) > 240:
            self.history.pop(0)
        self.cursor = len(self.history) - 1

    def step_backward(self):
        self.paused = True
        if self.cursor > 0:
            self.cursor -= 1

    def toggle_pause(self):
        self.paused = not self.paused

    def build_state_payload(self, base_rating: float, ui_text: dict, lang: str) -> dict:
        snap = self.history[self.cursor]
        waiting_evo = bool(self.match_finished and not self.skip_evo_wait and self.is_training_new_gen)
        return {
            "screen": "game",
            "is_simulation": True,
            "sim_paused": self.paused,
            "sim_skip_evo_wait": self.skip_evo_wait,
            "sim_waiting_for_evo": waiting_evo,
            "sim_step_delay": self.step_delay,
            "sim_cursor": self.cursor,
            "sim_history_len": len(self.history),
            "sim_matches": self.matches_played,
            "sim_f1_wins": self.f1_wins,
            "sim_f2_wins": self.f2_wins,
            "sim_f1_winrate": self.f1_winrate,
            "sim_f2_winrate": self.f2_winrate,
            "sim_f1_comp": self.f1_competence,
            "sim_f2_comp": self.f2_competence,
            "username": self.username,
            "lang": lang,
            "base_rating": round(float(base_rating), 2),
            "winrate": self.f1_winrate,
            "turn": snap["turn"],
            "max_turns": self.max_turns,
            "my_faction": 0,
            "my_alive": True,
            "current_turn_faction": snap["current_turn_faction"],
            "current_turn_user": snap["current_turn_user"],
            "is_my_turn": False,
            "match_finished": snap["match_finished"],
            "winner_faction": snap["winner_faction"],
            "winner_name": snap["winner_name"],
            "active_gen_label": self.active_gen_label,
            "active_rating": self.active_rating,
            "is_training_new_gen": self.is_training_new_gen,
            "last_evolution_report": self.last_evolution_report,
            "factions": snap["factions"],
            "terrain": snap["terrain"],
            "territory": snap["territory"],
            "eff_yields": snap["eff_yields"],
            "units": snap["units"],
            "structs": snap["structs"],
            "selected_pos": self.selected_pos,
            "valid_moves": [],
            "valid_attacks": [],
            "ai_traces": snap["ai_traces"],
            "event_log": snap["event_log"],
            "inspector": {
                "lines": [],
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
            },
            "stats": snap["stats"],
            "config": {
                "min_training_eff": MIN_TRAINING_EFFICIENCY,
                "disband_ratio": DISBAND_REFUND_RATIO,
                "demolish_ratio": DEMOLISH_REFUND_RATIO,
                "unit_stats": UNIT_STATS,
                "struct_stats": {str(k): v for k, v in STRUCT_STATS.items()}
            },
            "ui_text": ui_text
        }


class SimulationManager:
    sessions: dict[str, SimulationSession] = {}

    @staticmethod
    def has_active(username: str) -> bool:
        with SIM_LOCK:
            return username in SimulationManager.sessions

    @staticmethod
    def start_for_user(username: str):
        with SIM_LOCK:
            SimulationManager.sessions[username] = SimulationSession(username)

    @staticmethod
    def stop_for_user(username: str):
        with SIM_LOCK:
            SimulationManager.sessions.pop(username, None)

    @staticmethod
    def handle_action(username: str, action: str, r: Optional[int] = None, c: Optional[int] = None, param: Optional[str] = None):
        with SIM_LOCK:
            if action == "start":
                SimulationManager.sessions[username] = SimulationSession(username)
                return
            if action == "stop":
                SimulationManager.sessions.pop(username, None)
                return
            sess = SimulationManager.sessions.get(username)
            if not sess:
                return
            if action == "auto_step":
                sess.step_forward(is_auto=True)
            elif action == "next":
                sess.step_forward(is_auto=False)
            elif action == "prev":
                sess.step_backward()
            elif action == "pause_toggle":
                sess.toggle_pause()
            elif action == "set_skip_evo_wait":
                sess.skip_evo_wait = (str(param).lower() in ("1", "true", "yes"))
            elif action == "set_delay":
                try:
                    val = float(param) if param is not None else 0.50
                    sess.step_delay = round(max(0.0, min(10.0, val)), 2)
                except Exception:
                    pass
            elif action == "select_tile":
                if r is not None and c is not None and 0 <= int(r) < GRID_SIZE and 0 <= int(c) < GRID_SIZE:
                    sess.selected_pos = [int(r), int(c)]
                else:
                    sess.selected_pos = None

    @staticmethod
    def get_user_state(username: str, base_rating: float, ui_text: dict, lang: str) -> Optional[dict]:
        with SIM_LOCK:
            sess = SimulationManager.sessions.get(username)
            if not sess:
                return None
            return sess.build_state_payload(base_rating=base_rating, ui_text=ui_text, lang=lang)