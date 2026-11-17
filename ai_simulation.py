import os
import copy
import random
import threading
from typing import Optional
import numpy as np

from game_config import (
    GRID_SIZE, MAX_STEPS, MIN_TRAINING_EFFICIENCY,
    MIN_COMBAT_TRAINING_EFFICIENCY, DISBAND_REFUND_RATIO,
    DEMOLISH_REFUND_RATIO, UNIT_STATS, STRUCT_STATS,
    POOL_DIR, ROLE_IDS
)
from microrts_env import MicroRTSEnv
from ai_pool import PoolRegistry, OnlineEvolutionEngine, GLOBAL_MODEL_CACHE

SIM_LOCK = threading.Lock()


class SimulationSession:
    def __init__(self, username: str):
        self.username = username
        self.paused = False
        self.skip_evo_wait = False
        self.step_delay = 0.50
        self.matches_played = 0
        self.f1_wins = 0
        self.f2_wins = 0
        self.f1_rating_anchor = 75.00
        self.f2_rating_anchor = 75.00
        self.f1_competence = 0.95
        self.f2_competence = 0.95
        self.is_training_new_gen = False
        self.last_evolution_report = None
        self.selected_pos = None
        self.history = []
        self.cursor = 0
        self.env = MicroRTSEnv(mode="manual_p2", record_telemetry=True, num_players=2)
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
        max_r = PoolRegistry.get_max_rating()
        wr1 = self.f1_winrate
        total_m = self.f1_wins + self.f2_wins

        if total_m == 0:
            r1_target = min(max_r, 75.00)
            r2_target = min(max_r, 75.00)
            self.f1_competence = 0.95
            self.f2_competence = 0.95
        else:
            diff = wr1 - 50.00
            if wr1 < 40.00:
                delta = (40.00 - wr1) * 1.25
                r1_target = min(max_r, max(30.00, self.f1_rating_anchor + delta))
                r2_target = max(15.00, min(max_r, self.f2_rating_anchor - delta))
            elif wr1 > 60.00:
                delta = (wr1 - 60.00) * 1.25
                r2_target = min(max_r, max(30.00, self.f2_rating_anchor + delta))
                r1_target = max(15.00, min(max_r, self.f1_rating_anchor - delta))
            else:
                center_r = random.uniform(35.00, max(65.00, max_r))
                r1_target = max(15.00, min(max_r, center_r - 0.45 * diff))
                r2_target = max(15.00, min(max_r, center_r + 0.45 * diff))

            self.f1_competence = round(float(np.clip(0.95 - 1.45 * (diff / 100.00), 0.45, 1.35)), 2)
            self.f2_competence = round(float(np.clip(0.95 + 1.45 * (diff / 100.00), 0.45, 1.35)), 2)

        e1 = PoolRegistry.select_closest_generation(r1_target, randomize_top_k=4)
        e2 = PoolRegistry.select_closest_generation(r2_target, randomize_top_k=4)
        if e1.get("gen_id") == e2.get("gen_id"):
            e2 = PoolRegistry.select_closest_generation(max(10.00, r2_target - 8.00), randomize_top_k=5)

        self.f1_rating_anchor = float(e1.get("rating", r1_target))
        self.f2_rating_anchor = float(e2.get("rating", r2_target))
        return e1, e2

    def _start_new_match(self):
        self.env.reset()
        self.env.ai_controlled_factions = {1, 2}

        e1, e2 = self._select_balanced_models()
        self.f1_entry = e1
        self.f2_entry = e2

        self.env.set_faction_ai_rating(1, float(e1.get("rating", 75.00)))
        self.env.set_faction_ai_rating(2, float(e2.get("rating", 75.00)))
        self.env.set_faction_competence(1, self.f1_competence)
        self.env.set_faction_competence(2, self.f2_competence)

        m1 = GLOBAL_MODEL_CACHE.get_model(os.path.join(POOL_DIR, e1.get("filename", "")))
        m2 = GLOBAL_MODEL_CACHE.get_model(os.path.join(POOL_DIR, e2.get("filename", "")))
        self.env.faction_models[1] = m1
        self.env.faction_models[2] = m2
        self.env.opponent_model = m1

        self.active_gen_label = f"{e1.get('label', 'AI-1')} vs {e2.get('label', 'AI-2')}"
        self.active_rating = round((float(e1.get("rating", 50.00)) + float(e2.get("rating", 50.00))) / 2.00, 2)

        self.match_finished = False
        self.winner_faction = None
        self.winner_name = ""

        self.turn_sequence = [1, 2] if (self.matches_played % 2 == 0) else [2, 1]
        self.seq_idx = 0
        self.current_turn_faction = self.turn_sequence[0]
        self.env._prepare_faction_turn(self.current_turn_faction, collect_economy=False)
        self.env.add_event_log(f"Sim #{self.matches_played + 1}: {self.active_gen_label}")

        self.history = [self._create_snapshot()]
        self.cursor = 0

    def _collect_samples_with_fallback(self, winning_faction: int):
        obs_l, acts_l, weights_l = self.env.collect_match_training_samples(
            winning_faction=winning_faction,
            min_efficiency=MIN_TRAINING_EFFICIENCY
        )
        if len(obs_l) < 8:
            obs_l, acts_l, weights_l = self.env.collect_match_training_samples(
                winning_faction=winning_faction,
                min_efficiency=MIN_COMBAT_TRAINING_EFFICIENCY
            )
        if len(obs_l) < 8:
            obs_l, acts_l, weights_l = self.env.collect_match_training_samples(
                winning_faction=winning_faction,
                min_efficiency=15.00
            )
        for item in self.env.building_action_records:
            b_obs, b_act, b_fac, b_rid = item
            if b_fac == winning_faction and np.asarray(b_obs).shape == (18, 18, 18):
                obs_l.append(b_obs)
                acts_l.append(int(b_act))
                weights_l.append(2.20)
        return obs_l, acts_l, weights_l

    def _finish_simulation_match(self, winner_faction: int):
        if self.match_finished:
            return
        self.match_finished = True
        self.winner_faction = int(winner_faction)
        w_entry = self.f1_entry if self.winner_faction == 1 else self.f2_entry
        self.winner_name = f"AI [{w_entry.get('label', f'F{self.winner_faction}')}]"
        self.matches_played += 1
        if self.winner_faction == 1:
            self.f1_wins += 1
        else:
            self.f2_wins += 1

        self.env.add_event_log(
            f"Winner: F{self.winner_faction} ({self.winner_name}) | F1 WR: {self.f1_winrate:.2f}%"
        )

        self.env.export_high_efficiency_telemetry(
            winning_faction=self.winner_faction,
            min_efficiency=MIN_COMBAT_TRAINING_EFFICIENCY
        )
        w_obs, w_acts, w_weights = self._collect_samples_with_fallback(self.winner_faction)
        w_rating = float(self.f1_rating_anchor if self.winner_faction == 1 else self.f2_rating_anchor)

        if len(w_obs) > 0 and not self.is_training_new_gen:
            self.is_training_new_gen = True
            self.last_evolution_report = {"status": "training", "samples": len(w_obs)}

            def _bg_sim_evolve(obs_arr=w_obs, acts_arr=w_acts, weights_arr=w_weights, p_rating=w_rating):
                try:
                    entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(
                        username="AI_Sim",
                        player_rating=p_rating,
                        match_obs=obs_arr,
                        match_actions=acts_arr,
                        match_efficiencies=weights_arr
                    )
                    if entry:
                        self.last_evolution_report = {
                            "status": "completed",
                            "new_gen_label": entry.get("label", "Gen-Sim"),
                            "new_gen_rating": float(entry.get("rating", p_rating))
                        }
                except Exception as e:
                    print(f"[SimEvolution] Error: {e}")
                    self.last_evolution_report = None
                finally:
                    self.is_training_new_gen = False

            threading.Thread(target=_bg_sim_evolve, daemon=True).start()

    def _execute_single_faction_step(self, ai_faction: int):
        f = int(ai_faction)
        model = self.env.faction_models.get(f)
        self.env.execute_full_ai_turn(
            faction=f,
            model=model,
            deterministic=False,
            clear_traces=True,
            collect_economy=False
        )

        winner = self.env.get_match_winner()
        if winner is not None:
            self._finish_simulation_match(winner)
            return

        self.seq_idx += 1
        if self.seq_idx >= len(self.turn_sequence):
            self.seq_idx = 0
            self.env.turn_count += 1
            self.env.collect_global_round_economy()
            self.env.record_turn_efficiency_snapshot()

            if self.env.turn_count % 4 == 0:
                adv_f2 = self.env.evaluate_player_advantage(player_faction=2, ai_faction=1)
                wr_bias = (50.00 - self.f1_winrate) / 100.00
                c1_adj = float(np.clip(self.f1_competence + (adv_f2 - 0.50) * 0.45 + wr_bias * 0.35, 0.40, 1.35))
                c2_adj = float(np.clip(self.f2_competence - (adv_f2 - 0.50) * 0.45 - wr_bias * 0.35, 0.40, 1.35))
                self.f1_competence = round(c1_adj, 2)
                self.f2_competence = round(c2_adj, 2)
                self.env.set_faction_competence(1, self.f1_competence)
                self.env.set_faction_competence(2, self.f2_competence)

            winner = self.env.get_match_winner()
            if winner is not None:
                self._finish_simulation_match(winner)
                return

        self.current_turn_faction = self.turn_sequence[self.seq_idx]
        self.env._prepare_faction_turn(self.current_turn_faction, collect_economy=False)

    def _create_snapshot(self) -> dict:
        eff_grid = self.env.get_effective_yield_map()
        factions_out = {}
        for fk, entry in [(1, self.f1_entry), (2, self.f2_entry)]:
            inc, upk, net, tiles_cnt, army_cnt = self.env.get_faction_economy_stats(fk)
            lbl = f"AI [{entry.get('label', f'Gen-{fk}')}]"
            factions_out[str(fk)] = {
                "controller": lbl,
                "kind": "ai",
                "gold": round(self.env.get_faction_gold(fk), 2),
                "tiles": int(tiles_cnt),
                "inc": round(float(inc), 2),
                "army": int(army_cnt),
                "upk": round(float(upk), 2),
                "net": round(float(net), 2),
                "castle_hp": round(self.env.get_castle_hp(fk), 2),
                "castle_pos": list(self.env.castle_positions[fk]),
                "alive": bool(self.env.is_faction_alive(fk))
            }

        units_out = [
            {
                "r": int(pos[0]),
                "c": int(pos[1]),
                "faction": int(u["faction"]),
                "type": str(u["type"]),
                "hp": round(float(u["hp"]), 2),
                "max_hp": round(float(u["max_hp"]), 2),
                "moved": bool(u.get("moved", False)),
                "attacked": bool(u.get("attacked", False)),
                "ready": bool(u.get("ready", False)),
                "efficiency": round(float(u.get("efficiency", 0.00)), 2)
            }
            for pos, u in self.env.units.items()
        ]

        structs_out = []
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                st_t = int(self.env.struct_type[r, c])
                if st_t in (1, 2):
                    structs_out.append({
                        "r": int(r),
                        "c": int(c),
                        "owner": int(self.env.struct_owner[r, c]),
                        "type": st_t,
                        "hp": round(float(self.env.struct_hp[r, c]), 2),
                        "max_hp": round(float(STRUCT_STATS[st_t]["max_hp"]), 2),
                        "ready": bool(self.env.tower_ready[r, c])
                    })

        st_stats = self.env.get_efficiency_statistics()
        curr_ctrl = factions_out[str(self.current_turn_faction)]["controller"]

        return {
            "turn": int(self.env.turn_count),
            "current_turn_faction": int(self.current_turn_faction),
            "current_turn_user": curr_ctrl,
            "match_finished": bool(self.match_finished),
            "winner_faction": self.winner_faction,
            "winner_name": self.winner_name,
            "factions": copy.deepcopy(factions_out),
            "terrain": self.env.terrain.tolist(),
            "territory": self.env.territory.tolist(),
            "eff_yields": np.round(eff_grid, 2).tolist(),
            "units": units_out,
            "structs": structs_out,
            "ai_traces": copy.deepcopy(self.env.last_ai_turn_traces),
            "event_log": list(self.env.turn_event_log),
            "stats": st_stats
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
        if len(self.history) > 200:
            self.history.pop(0)
        self.cursor = len(self.history) - 1

    def step_backward(self):
        self.paused = True
        if self.cursor > 0:
            self.cursor -= 1

    def toggle_pause(self):
        self.paused = not self.paused

    def build_state_payload(self, base_rating: float, ui_text: dict, lang: str) -> dict:
        self.env.set_language(lang)
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
            "max_turns": int(MAX_STEPS),
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