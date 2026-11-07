import os
import random
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import game_config
from game_config import (
    GRID_SIZE, MAX_STEPS, MAX_GOLD, INITIAL_GOLD, HEAL_PCT,
    SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT, CASTLE_MAX_HP,
    MAX_WORKERS_PER_FACTION, MIN_TRAINING_EFFICIENCY,
    DISBAND_REFUND_RATIO, DEMOLISH_REFUND_RATIO,
    MIN_BARRACKS_CASTLE_DIST, MAX_SLOW_UNIT_FRONTLINE_DIST,
    OSCILLATION_MEMORY_LEN, EFFICIENCY_REWARDS,
    POOL_DIR, HUMAN_TELEMETRY_FILE,
    POOL_REGISTRY_FILE, USERS_DB_FILE, TILE_VALUES,
    UNIT_STATS, STRUCT_STATS, RELATIVE_OFFSETS, ROLE_IDS
)
from localization import normalize_lang, get_text, get_unit_name, get_struct_name
from ai_pool import (
    REGISTRY_LOCK, ModelCache, GLOBAL_MODEL_CACHE,
    PoolRegistry, OnlineEvolutionEngine,
    transform_offset_for_faction, transform_action_for_faction,
    infer_role_id_from_obs, predict_masked_action
)
from ai_strategist import AIStrategistMixin

FACTION_CASTLE_POSITIONS = getattr(game_config, "FACTION_CASTLE_POSITIONS", {
    1: (GRID_SIZE - 1, 0),
    2: (0, GRID_SIZE - 1),
    3: (0, 0),
    4: (GRID_SIZE - 1, GRID_SIZE - 1)
})

class MicroRTSEnv(AIStrategistMixin, gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mode="selfplay", record_telemetry=True, lang="en", num_players=2):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.mode = mode
        self.record_telemetry = record_telemetry
        self.lang = normalize_lang(lang)
        self.num_players = 4 if int(num_players) == 4 else 2
        self.active_factions = [1, 2, 3, 4] if self.num_players == 4 else [1, 2]

        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(
            low=0.00, high=1.00,
            shape=(18, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        self.opponent_model = None
        self.faction_models = {1: None, 2: None, 3: None, 4: None}
        self.curriculum_tier = 5
        self.faction_tiers = {1: 5, 2: 5, 3: 5, 4: 5}
        self.base_gen_idx = 50.00
        self.active_gen_idx = 50.00
        self.active_gen_label = "Gen-50"
        self.active_gen_origin = "base_curriculum"
        self.f1_Target_competence = 0.65
        self.f2_target_competence = 0.65
        self.faction_competence = {1: 0.65, 2: 0.65, 3: 0.65, 4: 0.65}
        self.episodes_count = 0
        self.next_unit_id = 1

        grid_r, grid_c = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing='ij')
        self._grid_r = grid_r
        self._grid_c = grid_c

        self._fp_dirty = True
        self._cluster_dirty = True
        self._cached_fp_maps = {f: np.zeros((self.grid_size, self.grid_size), dtype=np.float32) for f in range(1, 5)}
        self._cached_cluster_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._cached_base_path_len = 34
        self._bfs_cache = {}
        self.unit_directives = {}

        self.castle_positions = dict(FACTION_CASTLE_POSITIONS)
        self.castle_hp = {1: float(CASTLE_MAX_HP), 2: float(CASTLE_MAX_HP), 3: 0.00, 4: 0.00}
        self.gold = {1: float(INITIAL_GOLD), 2: float(INITIAL_GOLD), 3: 0.00, 4: 0.00}
        self.eliminated_factions = set()

        self.efficiency_timeline = {
            "turns": [],
            "player_eff": [],
            "ai_eff": [],
            "factions_eff": {1: [], 2: [], 3: [], 4: []},
            "active_rating": [],
            "active_label": []
        }
        self.last_ai_turn_traces = []
        self.turn_event_records = []
        self.turn_event_log = []

        self.reset()

    @property
    def p1_castle_pos(self):
        return self.castle_positions[1]

    @p1_castle_pos.setter
    def p1_castle_pos(self, val):
        self.castle_positions[1] = val

    @property
    def p2_castle_pos(self):
        return self.castle_positions[2]

    @p2_castle_pos.setter
    def p2_castle_pos(self, val):
        self.castle_positions[2] = val

    @property
    def p1_castle_hp(self):
        return float(self.castle_hp.get(1, 0.00))

    @p1_castle_hp.setter
    def p1_castle_hp(self, val):
        self.castle_hp[1] = round(float(val), 2)

    @property
    def p2_castle_hp(self):
        return float(self.castle_hp.get(2, 0.00))

    @p2_castle_hp.setter
    def p2_castle_hp(self, val):
        self.castle_hp[2] = round(float(val), 2)

    @property
    def p1_gold(self):
        return float(self.gold.get(1, 0.00))

    @p1_gold.setter
    def p1_gold(self, val):
        self.gold[1] = round(float(val), 2)

    @property
    def p2_gold(self):
        return float(self.gold.get(2, 0.00))

    @p2_gold.setter
    def p2_gold(self, val):
        self.gold[2] = round(float(val), 2)

    def get_faction_gold(self, faction: int) -> float:
        return float(self.gold.get(int(faction), 0.00))

    def set_faction_gold(self, faction: int, val: float):
        self.gold[int(faction)] = round(float(val), 2)

    def get_castle_hp(self, faction: int) -> float:
        return float(self.castle_hp.get(int(faction), 0.00))

    def set_castle_hp(self, faction: int, val: float):
        self.castle_hp[int(faction)] = round(float(val), 2)

    def is_faction_alive(self, faction: int) -> bool:
        f = int(faction)
        return f in self.active_factions and f not in self.eliminated_factions and self.castle_hp.get(f, 0.00) > 0.00

    def get_alive_factions(self) -> list:
        return [f for f in self.active_factions if self.is_faction_alive(f)]

    def get_enemy_factions(self, faction: int) -> list:
        f = int(faction)
        return [ef for ef in self.active_factions if ef != f and self.is_faction_alive(ef)]

    def _get_primary_enemy_castle(self, faction: int, ref_pos: tuple = None) -> tuple:
        enemies = self.get_enemy_factions(faction)
        if not enemies:
            fallback = 2 if int(faction) == 1 else 1
            return self.castle_positions[fallback]
        if ref_pos is None:
            ref_pos = self.castle_positions[int(faction)]
        best_e = min(
            enemies,
            key=lambda ef: abs(self.castle_positions[ef][0] - ref_pos[0]) + abs(self.castle_positions[ef][1] - ref_pos[1])
        )
        return self.castle_positions[best_e]

    def set_language(self, lang: str):
        self.lang = normalize_lang(lang)
        self._rerender_event_log()

    def add_localized_event(self, key: str, **kwargs):
        self.turn_event_records.append({"key": key, "kwargs": kwargs})
        if len(self.turn_event_records) > 8:
            self.turn_event_records = self.turn_event_records[-8:]
        self._rerender_event_log()

    def _rerender_event_log(self):
        rendered = []
        for item in self.turn_event_records:
            if "raw" in item:
                rendered.append(str(item["raw"]))
            else:
                kw = dict(item.get("kwargs", {}))
                if "u_type" in kw:
                    kw["unit"] = get_unit_name(kw.pop("u_type"), self.lang)
                if "st_type" in kw:
                    kw["struct"] = get_struct_name(kw.pop("st_type"), self.lang)
                rendered.append(get_text(self.lang, item["key"], **kw))
        self.turn_event_log = rendered

    def add_event_log(self, msg: str):
        self.turn_event_records.append({"raw": str(msg)})
        if len(self.turn_event_records) > 8:
            self.turn_event_records = self.turn_event_records[-8:]
        self._rerender_event_log()

    def set_curriculum_tier(self, tier: int, faction: int = None):
        t = int(np.clip(tier, 1, 5))
        self.curriculum_tier = t
        if faction is not None:
            self.faction_tiers[int(faction)] = t
        else:
            for f in range(1, 5):
                self.faction_tiers[f] = t

    def set_faction_competence(self, faction: int, competence: float):
        val = float(np.clip(competence, 0.25, 1.35))
        f = int(faction)
        self.faction_competence[f] = val
        if f == 1:
            self.f1_Target_competence = val
        elif f == 2:
            self.f2_target_competence = val

    def set_faction_ai_rating(self, faction: int, gen_rating: float):
        f = int(faction)
        max_pool_r = PoolRegistry.get_max_rating()
        r_clamped = round(float(np.clip(gen_rating, 0.00, max(100.00, max_pool_r))), 2)
        comp = float(np.clip(0.30 + (r_clamped / 100.00) * 0.75, 0.25, 1.35))
        self.set_faction_competence(f, comp)

        if r_clamped <= 15.00:
            t = 1
        elif r_clamped <= 35.00:
            t = 2
        elif r_clamped <= 55.00:
            t = 3
        elif r_clamped <= 75.00:
            t = 4
        else:
            t = 5
        self.faction_tiers[f] = t

        entry = PoolRegistry.select_closest_generation(r_clamped)
        model_path = os.path.join(POOL_DIR, entry.get("filename", ""))
        loaded_m = GLOBAL_MODEL_CACHE.get_model(model_path)
        self.faction_models[f] = loaded_m

        if f == 1:
            self.active_gen_idx = r_clamped
            self.curriculum_tier = t
            self.active_gen_label = str(entry.get("label", f"Gen-{int(round(r_clamped))}"))
            self.active_gen_origin = str(entry.get("origin", "base_curriculum"))
            self.opponent_model = loaded_m
        return entry

    def set_dda_generation(self, gen_rating: float):
        self.set_faction_ai_rating(1, gen_rating)

    def _select_selfplay_opponent(self):
        if not os.path.exists(POOL_DIR):
            return
        checkpoints = [f for f in os.listdir(POOL_DIR) if f.endswith(".zip") and f != "latest_checkpoint.zip"]
        if checkpoints:
            chosen = random.choice(checkpoints)
            m = GLOBAL_MODEL_CACHE.get_model(os.path.join(POOL_DIR, chosen))
            self.opponent_model = m
            for f in range(2, 5):
                self.faction_models[f] = m

    def _shortest_path_length(self, start: tuple, goal: tuple, block_pos: tuple = None) -> int:
        queue = deque([(start[0], start[1], 0)])
        visited = {start}
        while queue:
            r, c, d = queue.popleft()
            if (r, c) == goal:
                return d
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) == block_pos:
                        continue
                    if self.terrain[nr, nc] != 0 and (nr, nc) not in visited:
                        if hasattr(self, 'struct_type') and self.struct_type[nr, nc] == 1:
                            continue
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))
        return 999

    def _get_bfs_distance_map(self, target_pos: tuple, faction: int) -> np.ndarray:
        tr, tc = int(target_pos[0]), int(target_pos[1])
        key = (tr, tc, int(faction))
        if key in self._bfs_cache:
            return self._bfs_cache[key]

        dist_map = np.full((self.grid_size, self.grid_size), 99.00, dtype=np.float32)
        dist_map[tr, tc] = 0.00
        queue = deque([(tr, tc, 0.00)])
        f = int(faction)

        while queue:
            r, c, d = queue.popleft()
            nd = d + 1.00
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if dist_map[nr, nc] > nd and int(self.terrain[nr, nc]) != 0:
                        if int(self.struct_type[nr, nc]) == 1 and (nr, nc) != (tr, tc):
                            continue
                        st_own = int(self.struct_owner[nr, nc])
                        if st_own != 0 and st_own != f and int(self.struct_type[nr, nc]) == 2 and (nr, nc) != (tr, tc):
                            continue
                        dist_map[nr, nc] = nd
                        queue.append((nr, nc, nd))

        self._bfs_cache[key] = dist_map
        return dist_map

    def _is_map_connected(self, terrain):
        start = self.castle_positions[1]
        goals = [self.castle_positions[f] for f in self.active_factions if f != 1]
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if terrain[nr, nc] != 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return all(g in visited for g in goals)

    def _generate_procedural_world(self):
        half = self.grid_size // 2
        for _ in range(60):
            terrain = np.ones((self.grid_size, self.grid_size), dtype=np.int8)
            if self.num_players == 4:
                for r in range(half):
                    for c in range(half):
                        roll = random.random()
                        if roll < 0.18:
                            t_type = 0
                        elif roll < 0.44:
                            t_type = 2
                        elif roll < 0.58:
                            t_type = 3
                        else:
                            t_type = 1
                        terrain[r, c] = t_type
                        terrain[r, self.grid_size - 1 - c] = t_type
                        terrain[self.grid_size - 1 - r, c] = t_type
                        terrain[self.grid_size - 1 - r, self.grid_size - 1 - c] = t_type
            else:
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if r + c <= self.grid_size - 1:
                            roll = random.random()
                            if roll < 0.18:
                                t_type = 0
                            elif roll < 0.44:
                                t_type = 2
                            elif roll < 0.58:
                                t_type = 3
                            else:
                                t_type = 1
                            terrain[r, c] = t_type
                            terrain[self.grid_size - 1 - r, self.grid_size - 1 - c] = t_type

            terrain[self.grid_size - 3:, :3] = 1
            terrain[:3, self.grid_size - 3:] = 1
            terrain[self.grid_size - 1, 2] = 3
            terrain[0, self.grid_size - 3] = 3
            terrain[self.grid_size - 3, 1] = 2
            terrain[2, self.grid_size - 2] = 2

            if self.num_players == 4:
                terrain[:3, :3] = 1
                terrain[self.grid_size - 3:, self.grid_size - 3:] = 1
                terrain[0, 2] = 3
                terrain[self.grid_size - 1, self.grid_size - 3] = 3
                terrain[2, 1] = 2
                terrain[self.grid_size - 3, self.grid_size - 2] = 2

            if self._is_map_connected(terrain):
                return terrain
        return np.ones((self.grid_size, self.grid_size), dtype=np.int8)

    def _compute_chokepoint_map(self) -> np.ndarray:
        choke = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.terrain[r, c] == 0:
                    continue
                passable_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.terrain[nr, nc] != 0:
                        passable_neighbors += 1
                if passable_neighbors <= 2:
                    choke[r, c] = 1.00
                elif passable_neighbors == 3:
                    choke[r, c] = 0.40
        return choke

    def _compute_resource_cluster_map(self) -> np.ndarray:
        if not self._cluster_dirty:
            return self._cached_cluster_map

        eff_map = self.get_effective_yield_map()
        cluster_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.terrain[r, c]) == 0:
                    continue
                r_min = max(0, r - 3)
                r_max = min(self.grid_size, r + 4)
                c_min = max(0, c - 3)
                c_max = min(self.grid_size, c + 4)

                sub_yields = eff_map[r_min:r_max, c_min:c_max]
                sub_r = self._grid_r[r_min:r_max, c_min:c_max]
                sub_c = self._grid_c[r_min:r_max, c_min:c_max]
                dists = np.abs(sub_r - r) + np.abs(sub_c - c)
                mask = (dists <= 3) & (sub_yields > 0.05)
                if np.any(mask):
                    weights = sub_yields[mask] / (1.00 + 0.35 * dists[mask].astype(np.float32))
                    cluster_map[r, c] = float(np.sum(weights))

        self._cached_cluster_map = cluster_map
        self._cluster_dirty = False
        return cluster_map

    def _create_unit_dict(self, faction: int, u_type: str, ready: bool = True) -> dict:
        max_hp = float(UNIT_STATS[u_type]['max_hp'])
        uid = int(self.next_unit_id)
        self.next_unit_id += 1
        return {
            'id': uid,
            'faction': int(faction),
            'type': str(u_type),
            'role_id': int(ROLE_IDS.get(str(u_type), 0)),
            'hp': max_hp,
            'max_hp': max_hp,
            'ready': bool(ready),
            'efficiency': 0.00,
            'history': [],
            'recent_positions': deque(maxlen=OSCILLATION_MEMORY_LEN)
        }

    def compute_faction_total_efficiency(self, faction: int) -> float:
        f = int(faction)
        alive_eff = sum(float(u.get('efficiency', 0.00)) for u in self.units.values() if u['faction'] == f)
        arch_eff = sum(float(u.get('efficiency', 0.00)) for u in self.archived_units if u['faction'] == f)
        _, _, net_inc, tile_cnt, _ = self.get_faction_economy_stats(f)
        total = alive_eff + arch_eff + (tile_cnt * 0.45) + (net_inc * 1.50)
        return round(max(0.00, float(total)), 2)

    def record_turn_efficiency_snapshot(self):
        p_eff = self.compute_faction_total_efficiency(faction=2)
        ai_eff = self.compute_faction_total_efficiency(faction=1)
        self.efficiency_timeline["turns"].append(int(self.turn_count))
        self.efficiency_timeline["player_eff"].append(float(p_eff))
        self.efficiency_timeline["ai_eff"].append(float(ai_eff))
        for f in range(1, 5):
            val = self.compute_faction_total_efficiency(f) if f in self.active_factions else 0.00
            self.efficiency_timeline["factions_eff"][f].append(float(val))
        self.efficiency_timeline["active_rating"].append(round(float(self.active_gen_idx), 2))
        self.efficiency_timeline["active_label"].append(str(self.active_gen_label))

    def get_efficiency_statistics(self) -> dict:
        p_arr = self.efficiency_timeline["player_eff"]
        ai_arr = self.efficiency_timeline["ai_eff"]
        p_med = round(float(np.median(p_arr)), 2) if len(p_arr) > 0 else 0.00
        ai_med = round(float(np.median(ai_arr)), 2) if len(ai_arr) > 0 else 0.00
        f_eff = {}
        f_med = {}
        for f in self.active_factions:
            arr = self.efficiency_timeline["factions_eff"].get(f, [])
            f_eff[str(f)] = [float(x) for x in arr]
            f_med[str(f)] = round(float(np.median(arr)), 2) if len(arr) > 0 else 0.00
        return {
            "turns": [int(x) for x in self.efficiency_timeline["turns"]],
            "player_eff": [float(x) for x in p_arr],
            "ai_eff": [float(x) for x in ai_arr],
            "player_median": float(p_med),
            "ai_median": float(ai_med),
            "player_current": float(p_arr[-1]) if p_arr else 0.00,
            "ai_current": float(ai_arr[-1]) if ai_arr else 0.00,
            "factions_eff": f_eff,
            "factions_median": f_med,
            "active_rating": [float(x) for x in self.efficiency_timeline["active_rating"]],
            "active_label": [str(x) for x in self.efficiency_timeline["active_label"]]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episodes_count += 1
        if self.mode == "selfplay" and self.episodes_count % 4 == 0:
            self._select_selfplay_opponent()

        self.active_factions = [1, 2, 3, 4] if self.num_players == 4 else [1, 2]
        self.eliminated_factions = set()
        self.castle_positions = dict(FACTION_CASTLE_POSITIONS)

        self.struct_type = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.terrain = self._generate_procedural_world()
        self.tile_yields = np.vectorize(lambda t: TILE_VALUES[int(t)])(self.terrain).astype(np.float32)
        self.chokepoint_map = self._compute_chokepoint_map()

        self.territory = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.struct_owner = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.struct_hp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.tower_ready = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        for f in range(1, 5):
            if f in self.active_factions:
                self.castle_hp[f] = float(CASTLE_MAX_HP)
                self.gold[f] = float(INITIAL_GOLD)
            else:
                self.castle_hp[f] = 0.00
                self.gold[f] = 0.00

        self.units = {}
        self.archived_units = []
        self.building_action_records = []
        self.unit_directives = {}
        self._bfs_cache = {}
        self._spawn_initial_forces()

        self._fp_dirty = True
        self._cluster_dirty = True
        self._compute_resource_cluster_map()
        self._cached_base_path_len = self._shortest_path_length(self.castle_positions[1], self.castle_positions[2])

        self.turn_count = 1
        self.step_count = 0
        self.p1_entity_queue = deque()
        self.last_ai_turn_traces = []
        self.turn_event_records = []
        self.add_localized_event("match_started")
        self._prepare_faction_turn(faction=1, collect_economy=False)

        self.efficiency_timeline = {
            "turns": [],
            "player_eff": [],
            "ai_eff": [],
            "factions_eff": {1: [], 2: [], 3: [], 4: []},
            "active_rating": [],
            "active_label": []
        }
        self.record_turn_efficiency_snapshot()
        return self._get_obs(), {}

    def _spawn_initial_forces(self):
        self.units[(self.grid_size - 1, 1)] = self._create_unit_dict(1, 'worker', ready=True)
        self.units[(self.grid_size - 2, 0)] = self._create_unit_dict(1, 'warrior', ready=True)
        self.units[(self.grid_size - 2, 1)] = self._create_unit_dict(1, 'archer', ready=True)
        for r, c in [self.castle_positions[1], (self.grid_size - 1, 1), (self.grid_size - 2, 0), (self.grid_size - 2, 1)]:
            self.territory[r, c] = 1

        self.units[(0, self.grid_size - 2)] = self._create_unit_dict(2, 'worker', ready=True)
        self.units[(1, self.grid_size - 1)] = self._create_unit_dict(2, 'warrior', ready=True)
        self.units[(1, self.grid_size - 2)] = self._create_unit_dict(2, 'archer', ready=True)
        for r, c in [self.castle_positions[2], (0, self.grid_size - 2), (1, self.grid_size - 1), (1, self.grid_size - 2)]:
            self.territory[r, c] = 2

        if self.num_players == 4:
            self.units[(0, 1)] = self._create_unit_dict(3, 'worker', ready=True)
            self.units[(1, 0)] = self._create_unit_dict(3, 'warrior', ready=True)
            self.units[(1, 1)] = self._create_unit_dict(3, 'archer', ready=True)
            for r, c in [self.castle_positions[3], (0, 1), (1, 0), (1, 1)]:
                self.territory[r, c] = 3

            self.units[(self.grid_size - 1, self.grid_size - 2)] = self._create_unit_dict(4, 'worker', ready=True)
            self.units[(self.grid_size - 2, self.grid_size - 1)] = self._create_unit_dict(4, 'warrior', ready=True)
            self.units[(self.grid_size - 2, self.grid_size - 2)] = self._create_unit_dict(4, 'archer', ready=True)
            for r, c in [self.castle_positions[4], (self.grid_size - 1, self.grid_size - 2), (self.grid_size - 2, self.grid_size - 1), (self.grid_size - 2, self.grid_size - 2)]:
                self.territory[r, c] = 4

    def is_castle_tile(self, r: int, c: int, active_only: bool = True) -> bool:
        factions_to_check = self.active_factions if active_only else [1, 2, 3, 4]
        for f in factions_to_check:
            if active_only and not self.is_faction_alive(f):
                continue
            if (r, c) == self.castle_positions[f]:
                return True
        return False

    def is_tile_built_up(self, r: int, c: int) -> bool:
        if self.is_castle_tile(r, c, active_only=True):
            return True
        if int(self.struct_type[r, c]) != 0:
            return True
        return False

    def get_effective_tile_yield(self, r: int, c: int) -> float:
        if self.is_tile_built_up(r, c):
            return 0.00
        return float(self.tile_yields[r, c])

    def get_effective_yield_map(self) -> np.ndarray:
        eff_map = self.tile_yields.copy()
        eff_map[self.struct_type != 0] = 0.00
        for f in self.active_factions:
            if self.is_faction_alive(f):
                cr, cc = self.castle_positions[f]
                eff_map[cr, cc] = 0.00
        return eff_map

    def _eliminate_faction(self, dead_faction: int):
        df = int(dead_faction)
        self.castle_hp[df] = 0.00
        self.eliminated_factions.add(df)
        for pos in [p for p, u in self.units.items() if u['faction'] == df]:
            u_dead = self.units.pop(pos)
            self.archived_units.append(u_dead)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.struct_owner[r, c]) == df:
                    self.struct_owner[r, c] = 0
                    self.struct_type[r, c] = 0
                    self.struct_hp[r, c] = 0.00
                    self.tower_ready[r, c] = False
                if int(self.territory[r, c]) == df:
                    self.territory[r, c] = 0
        self._fp_dirty = True
        self._cluster_dirty = True
        self._bfs_cache.clear()

    def record_entity_decision(self, entity_kind: str, pos: tuple, faction: int, action: int):
        if not self.record_telemetry:
            return
        obs = self._get_obs_for_entity(entity_kind, pos, faction)
        canonical_act = transform_action_for_faction(int(action), int(faction))
        if entity_kind == 'unit' and pos in self.units:
            u = self.units[pos]
            rid = int(ROLE_IDS.get(u['type'], 0))
            u['history'].append((obs, int(canonical_act), rid))
        else:
            rid = ROLE_IDS["tower"] if entity_kind == 'tower' else ROLE_IDS["spawn"]
            self.building_action_records.append((obs, int(canonical_act), int(faction), int(rid)))

    def collect_match_training_samples(self, winning_faction: int = None, min_efficiency: float = MIN_TRAINING_EFFICIENCY):
        eff_threshold = max(float(MIN_TRAINING_EFFICIENCY), float(min_efficiency))
        all_units = list(self.units.values()) + self.archived_units
        selected_obs = []
        selected_acts = []
        selected_weights = []

        for u in all_units:
            eff = float(u.get('efficiency', 0.00))
            u_type = u.get('type', 'worker')
            if eff >= eff_threshold and len(u.get('history', [])) > 0:
                outcome_bonus = 1.50 if (winning_faction is not None and u['faction'] == winning_faction) else 0.50
                role_mult = 1.35 if u_type != 'worker' else 0.85
                base_weight = min(6.00, round((1.00 + (eff / 60.00) + outcome_bonus) * role_mult, 2))
                for item in u['history']:
                    obs_arr, act_val = item[0], int(item[1])
                    rid = int(item[2]) if len(item) > 2 else int(ROLE_IDS.get(u_type, 0))
                    v_doc = self.evaluate_sample_doctrine_compliance(obs_arr, act_val, role_id=rid)
                    adj_weight = round(base_weight * v_doc, 2)
                    selected_obs.append(obs_arr)
                    selected_acts.append(act_val)
                    selected_weights.append(float(adj_weight))

        return selected_obs, selected_acts, selected_weights

    def export_high_efficiency_telemetry(self, winning_faction: int = None, min_efficiency: float = MIN_TRAINING_EFFICIENCY) -> int:
        if not self.record_telemetry:
            return 0
        os.makedirs(POOL_DIR, exist_ok=True)
        selected_obs, selected_acts, selected_weights = self.collect_match_training_samples(
            winning_faction=winning_faction,
            min_efficiency=min_efficiency
        )
        if not selected_obs:
            return 0

        selected_roles = [infer_role_id_from_obs(o, int(a)) for o, a in zip(selected_obs, selected_acts)]

        new_obs = np.array(selected_obs, dtype=np.float32)
        new_acts = np.array(selected_acts, dtype=np.int64)
        new_weights = np.array(selected_weights, dtype=np.float32)
        new_roles = np.array(selected_roles, dtype=np.int64)

        with REGISTRY_LOCK:
            if os.path.exists(HUMAN_TELEMETRY_FILE):
                try:
                    data = np.load(HUMAN_TELEMETRY_FILE)
                    old_obs = data['obs']
                    old_acts = data['acts']
                    old_weights = data['weights']
                    if 'roles' in data and len(data['roles']) == len(old_obs):
                        old_roles = data['roles']
                    else:
                        old_roles = np.array([infer_role_id_from_obs(o, int(a)) for o, a in zip(old_obs, old_acts)], dtype=np.int64)

                    new_obs = np.concatenate([old_obs, new_obs], axis=0)[-8000:]
                    new_acts = np.concatenate([old_acts, new_acts], axis=0)[-8000:]
                    new_weights = np.concatenate([old_weights, new_weights], axis=0)[-8000:]
                    new_roles = np.concatenate([old_roles, new_roles], axis=0)[-8000:]
                except Exception:
                    pass
            np.savez_compressed(HUMAN_TELEMETRY_FILE, obs=new_obs, acts=new_acts, weights=new_weights, roles=new_roles)
        return len(selected_obs)

    def get_faction_economy_stats(self, faction: int):
        f = int(faction)
        owned_mask = (self.territory == f)
        eff_yields = self.get_effective_yield_map()
        gross_inc = float(np.sum(eff_yields[owned_mask]))
        tile_cnt = int(np.sum(owned_mask))

        base_upkeep = 0.00
        unit_cnt = 0
        for u in self.units.values():
            if u['faction'] == f:
                base_upkeep += UNIT_STATS[u['type']]['upkeep']
                unit_cnt += 1

        supply_tax = max(0, unit_cnt - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT
        total_upkeep = base_upkeep + supply_tax

        net_inc = gross_inc - total_upkeep
        return round(gross_inc, 2), round(total_upkeep, 2), round(net_inc, 2), int(tile_cnt), int(unit_cnt)

    def _prepare_faction_turn(self, faction: int, collect_economy=True):
        f = int(faction)
        if not self.is_faction_alive(f):
            return

        if collect_economy:
            gross_inc, upkeep, net_inc, _, _ = self.get_faction_economy_stats(f)
            curr_gold = self.get_faction_gold(f)
            new_gold = curr_gold + net_inc

            if new_gold < 0.00:
                dead_coords = [pos for pos, u in self.units.items() if u['faction'] == f]
                for pos in dead_coords:
                    u_dead = self.units.pop(pos)
                    u_dead['efficiency'] = round(u_dead['efficiency'] - EFFICIENCY_REWARDS["early_death_penalty"], 2)
                    self.archived_units.append(u_dead)
                new_gold = 0.00
                self._fp_dirty = True
                msg_key = "bankruptcy_ai" if f == 1 else "bankruptcy_player"
                self.add_localized_event(msg_key)
            else:
                new_gold = min(MAX_GOLD, new_gold)

            self.set_faction_gold(f, new_gold)

        for pos, u in self.units.items():
            if u['faction'] == f:
                u['ready'] = True

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.struct_owner[r, c]) == f and int(self.struct_type[r, c]) == 1:
                    self.tower_ready[r, c] = True
                elif int(self.struct_owner[r, c]) == f:
                    self.tower_ready[r, c] = False

        if f == 1:
            self._rebuild_p1_entity_queue()

    def _rebuild_p1_entity_queue(self):
        self.p1_entity_queue.clear()
        if not self.is_faction_alive(1):
            return
        for pos, u in sorted(self.units.items()):
            if u['faction'] == 1 and u['ready']:
                self.p1_entity_queue.append(('unit', pos))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.struct_owner[r, c]) == 1 and int(self.struct_type[r, c]) == 1 and bool(self.tower_ready[r, c]):
                    if len(self.get_valid_attack_targets((r, c), faction=1, atk_range=2)) > 0:
                        self.p1_entity_queue.append(('tower', (r, c)))
        if self.get_faction_gold(1) >= UNIT_STATS['worker']['cost']:
            spawn_sites = [self.castle_positions[1]]
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if int(self.struct_owner[r, c]) == 1 and int(self.struct_type[r, c]) == 2:
                        spawn_sites.append((r, c))
            target_castle = self._get_primary_enemy_castle(1)
            spawn_sites.sort(key=lambda s: abs(s[0] - target_castle[0]) + abs(s[1] - target_castle[1]))
            for site in spawn_sites:
                if site not in self.units:
                    self.p1_entity_queue.append(('spawn', site))

    def _can_pass_tile(self, r: int, c: int, faction: int) -> bool:
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if int(self.terrain[r, c]) == 0:
            return False
        if int(self.struct_type[r, c]) == 1:
            return False
        f = int(faction)
        st_own = int(self.struct_owner[r, c])
        if st_own != 0 and st_own != f and int(self.struct_type[r, c]) == 2:
            return False
        for ef in self.get_enemy_factions(f):
            if (r, c) == self.castle_positions[ef]:
                return False
        if (r, c) in self.units and self.units[(r, c)]['faction'] != f:
            return False
        return True

    def get_valid_moves_for_unit(self, pos: tuple) -> list:
        if pos not in self.units:
            return []
        u = self.units[pos]
        if not u['ready']:
            return []
        faction = u['faction']
        max_dist = UNIT_STATS[u['type']]['move_range']

        sr, sc = pos
        queue = deque([(sr, sc, 0)])
        visited = {(sr, sc): 0}
        valid_destinations = []

        while queue:
            r, c, d = queue.popleft()
            if d > 0 and (r, c) not in self.units:
                valid_destinations.append((int(r), int(c)))
            if d < max_dist:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if self._can_pass_tile(nr, nc, faction):
                        if (nr, nc) not in visited or visited[(nr, nc)] > d + 1:
                            visited[(nr, nc)] = d + 1
                            queue.append((nr, nc, d + 1))
        return valid_destinations

    def get_valid_attack_targets(self, pos: tuple, faction: int = None, atk_range: int = None) -> list:
        if faction is None:
            if pos not in self.units:
                return []
            u = self.units[pos]
            if not u['ready']:
                return []
            faction = u['faction']
            atk_range = UNIT_STATS[u['type']]['atk_range']

        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        enemy_castles = {self.castle_positions[ef] for ef in enemies}
        r, c = pos
        targets = []

        for idx, (dr, dc) in enumerate(RELATIVE_OFFSETS):
            dist = abs(dr) + abs(dc)
            if dist <= atk_range:
                tr, tc = r + dr, c + dc
                if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
                    has_enemy_unit = ((tr, tc) in self.units and self.units[(tr, tc)]['faction'] in enemies)
                    has_enemy_struct = (int(self.struct_owner[tr, tc]) in enemies and int(self.struct_type[tr, tc]) != 0)
                    is_enemy_castle = ((tr, tc) in enemy_castles)
                    if has_enemy_unit or has_enemy_struct or is_enemy_castle:
                        targets.append((int(tr), int(tc)))
        return targets

    def _capture_tile(self, r: int, c: int, faction: int, u_type: str = 'worker') -> float:
        f = int(faction)
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size and int(self.terrain[r, c]) != 0:
            st_own = int(self.struct_owner[r, c])
            if st_own == 0 or st_own == f:
                old_owner = int(self.territory[r, c])
                self.territory[r, c] = f
                eff_y = self.get_effective_tile_yield(r, c)
                role_scale = 1.00 if u_type == 'worker' else float(EFFICIENCY_REWARDS.get("combat_capture_scale", 0.12))
                if old_owner == 0:
                    raw = EFFICIENCY_REWARDS["neutral_capture_base"] + eff_y * EFFICIENCY_REWARDS["neutral_capture_yield_mult"]
                    return round(raw * role_scale, 2)
                elif old_owner != f:
                    raw = EFFICIENCY_REWARDS["enemy_capture_base"] + eff_y * EFFICIENCY_REWARDS["enemy_capture_yield_mult"]
                    return round(raw * role_scale, 2)
        return 0.00

    def heal_unit_percentage(self, pos: tuple) -> float:
        if pos not in self.units:
            return 0.00
        u = self.units[pos]
        heal_amount = round(u['max_hp'] * HEAL_PCT, 2)
        old_hp = u['hp']
        u['hp'] = min(u['max_hp'], round(u['hp'] + heal_amount, 2))
        u['ready'] = False
        gained = round(u['hp'] - old_hp, 2)
        u['efficiency'] = round(u['efficiency'] + gained * EFFICIENCY_REWARDS["heal_mult"], 2)
        return float(gained)

    def heal_all_unmoved_units(self, faction: int):
        f = int(faction)
        for pos, u in list(self.units.items()):
            if u['faction'] == f and u['ready']:
                self.heal_unit_percentage(pos)

    def execute_unit_disband(self, pos: tuple) -> float:
        if pos not in self.units:
            return 0.00
        u = self.units.pop(pos)
        f = int(u['faction'])
        refund = round(float(UNIT_STATS[u['type']]['cost']) * DISBAND_REFUND_RATIO, 2)
        self.set_faction_gold(f, min(MAX_GOLD, self.get_faction_gold(f) + refund))
        self.archived_units.append(u)
        self._fp_dirty = True
        return float(refund)

    def _find_demolishable_structure_site(self, pos: tuple, faction: int):
        r, c = pos
        f = int(faction)
        check_coords = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        candidates = []
        for nr, nc in check_coords:
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if int(self.struct_owner[nr, nc]) == f and int(self.struct_type[nr, nc]) in (1, 2):
                    candidates.append((int(nr), int(nc)))
        if not candidates:
            return None
        my_castle = self.castle_positions[f]
        return min(candidates, key=lambda p: abs(p[0] - my_castle[0]) + abs(p[1] - my_castle[1]) - float(self.tile_yields[p[0], p[1]]) * 15.00)

    def execute_worker_demolish(self, pos: tuple) -> tuple:
        if pos not in self.units:
            return 0.00, 0
        u = self.units[pos]
        if u['type'] != 'worker' or not u['ready']:
            return 0.00, 0
        f = int(u['faction'])
        target_site = self._find_demolishable_structure_site(pos, f)
        if target_site is None:
            return 0.00, 0

        sr, sc = target_site
        st_type = int(self.struct_type[sr, sc])
        refund = round(float(STRUCT_STATS[st_type]['cost']) * DEMOLISH_REFUND_RATIO, 2)

        self.struct_owner[sr, sc] = 0
        self.struct_type[sr, sc] = 0
        self.struct_hp[sr, sc] = 0.00
        self.tower_ready[sr, sc] = False

        self.set_faction_gold(f, min(MAX_GOLD, self.get_faction_gold(f) + refund))

        u['ready'] = False
        self._fp_dirty = True
        self._cluster_dirty = True
        self._bfs_cache.clear()
        if st_type == 1:
            self._cached_base_path_len = self._shortest_path_length(self.castle_positions[1], self.castle_positions[2])
        return float(refund), int(st_type)

    def _find_best_intermediate_step(self, sr: int, sc: int, dr: int, dc: int, faction: int):
        f = int(faction)
        best_mid = None
        best_val = -1.00
        for mr, mc in [(sr + 1, sc), (sr - 1, sc), (sr, sc + 1), (sr, sc - 1)]:
            if abs(mr - dr) + abs(mc - dc) == 1 and self._can_pass_tile(mr, mc, f):
                unclaimed_bonus = 2.00 if int(self.territory[mr, mc]) != f else 0.00
                val = unclaimed_bonus + self.get_effective_tile_yield(mr, mc) * 10.00
                if val > best_val:
                    best_val = val
                    best_mid = (mr, mc)
        return best_mid

    def execute_unit_move(self, start_pos: tuple, dest_pos: tuple) -> bool:
        if start_pos not in self.units or dest_pos in self.units:
            return False
        u = self.units[start_pos]
        if not u['ready']:
            return False
        valid_moves = self.get_valid_moves_for_unit(start_pos)
        if dest_pos not in valid_moves:
            return False

        if 'recent_positions' not in u:
            u['recent_positions'] = deque(maxlen=OSCILLATION_MEMORY_LEN)
        u['recent_positions'].append((int(start_pos[0]), int(start_pos[1])))

        u_type = u['type']
        f = int(u['faction'])
        eff_gain = 0.00
        sr, sc = start_pos
        dr, dc = dest_pos

        if u_type != 'worker':
            enemies = set(self.get_enemy_factions(f))
            enemy_positions = [pos_e for pos_e, ue in self.units.items() if ue['faction'] in enemies]
            if enemy_positions:
                closest_e = min(enemy_positions, key=lambda p: abs(p[0] - sr) + abs(p[1] - sc))
                d_before = abs(closest_e[0] - sr) + abs(closest_e[1] - sc)
                d_after = abs(closest_e[0] - dr) + abs(closest_e[1] - dc)
                if d_before <= 4 and d_after > d_before:
                    eff_gain -= float(EFFICIENCY_REWARDS.get("retreat_penalty", 8.00))

        if abs(sr - dr) + abs(sc - dc) == 2:
            best_mid = self._find_best_intermediate_step(sr, sc, dr, dc, f)
            if best_mid is not None:
                eff_gain += self._capture_tile(best_mid[0], best_mid[1], f, u_type=u_type)

        del self.units[start_pos]
        u['ready'] = False
        eff_gain += self._capture_tile(dest_pos[0], dest_pos[1], f, u_type=u_type)
        u['efficiency'] = round(u['efficiency'] + eff_gain, 2)
        self.units[(int(dest_pos[0]), int(dest_pos[1]))] = u
        self._fp_dirty = True
        return True

    def execute_attack_on_target(self, attacker_pos: tuple, target_pos: tuple, is_tower: bool = False) -> float:
        if is_tower:
            r, c = attacker_pos
            faction = int(self.struct_owner[r, c])
            if faction == 0 or int(self.struct_type[r, c]) != 1 or not bool(self.tower_ready[r, c]):
                return 0.00
            valid_targets = self.get_valid_attack_targets(attacker_pos, faction=faction, atk_range=2)
            if target_pos not in valid_targets:
                return 0.00
            dmg = STRUCT_STATS[1]['dmg']
            self.tower_ready[r, c] = False
            attacker_u = None
        else:
            if attacker_pos not in self.units:
                return 0.00
            attacker_u = self.units[attacker_pos]
            if not attacker_u['ready']:
                return 0.00
            faction = attacker_u['faction']
            atk_range = UNIT_STATS[attacker_u['type']]['atk_range']
            valid_targets = self.get_valid_attack_targets(attacker_pos, faction=faction, atk_range=atk_range)
            if target_pos not in valid_targets:
                return 0.00
            dmg = UNIT_STATS[attacker_u['type']]['dmg']
            attacker_u['ready'] = False

        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        tr, tc = target_pos
        dealt = 0.00
        kill_bonus = 0.00

        if (tr, tc) in self.units and self.units[(tr, tc)]['faction'] in enemies:
            target_u = self.units[(tr, tc)]
            target_u['hp'] = round(max(0.00, target_u['hp'] - dmg), 2)
            dealt = dmg
            if target_u['hp'] <= 0.00:
                dead_u = self.units.pop((tr, tc))
                if dead_u['efficiency'] < 10.00:
                    dead_u['efficiency'] = round(dead_u['efficiency'] - EFFICIENCY_REWARDS["early_death_penalty"], 2)
                self.archived_units.append(dead_u)
                kill_bonus = EFFICIENCY_REWARDS["unit_kill_bonus"]
                self._fp_dirty = True
        elif int(self.struct_owner[tr, tc]) in enemies and int(self.struct_type[tr, tc]) != 0:
            self.struct_hp[tr, tc] = round(max(0.00, float(self.struct_hp[tr, tc]) - dmg), 2)
            dealt = dmg
            if self.struct_hp[tr, tc] <= 0.00:
                was_tower = (int(self.struct_type[tr, tc]) == 1)
                self.struct_owner[tr, tc] = 0
                self.struct_type[tr, tc] = 0
                kill_bonus = EFFICIENCY_REWARDS["struct_kill_bonus"]
                self._cluster_dirty = True
                self._bfs_cache.clear()
                if was_tower:
                    self._fp_dirty = True
                    self._cached_base_path_len = self._shortest_path_length(self.castle_positions[1], self.castle_positions[2])
        else:
            for ef in enemies:
                if (tr, tc) == self.castle_positions[ef]:
                    new_c_hp = round(max(0.00, self.get_castle_hp(ef) - dmg), 2)
                    self.set_castle_hp(ef, new_c_hp)
                    dealt = dmg
                    kill_bonus = EFFICIENCY_REWARDS["castle_hit_bonus"]
                    if new_c_hp <= 0.00:
                        self._eliminate_faction(ef)
                        kill_bonus += 15.00
                    break

        if attacker_u is not None and dealt > 0.00:
            combat_mult = 1.85 if attacker_u['type'] != 'worker' else 0.20
            attacker_u['efficiency'] = round(
                attacker_u['efficiency'] + (dealt * EFFICIENCY_REWARDS["damage_dealt_mult"] + kill_bonus) * combat_mult, 2
            )

        return float(dealt)

    def execute_worker_harvest(self, pos: tuple) -> float:
        if pos not in self.units:
            return 0.00
        u = self.units[pos]
        if u['type'] != 'worker' or not u['ready']:
            return 0.00
        f = int(u['faction'])
        r, c = pos
        eff_val = self.get_effective_tile_yield(r, c)
        if eff_val <= 0.00:
            return 0.00

        eff_cap = self._capture_tile(r, c, f, u_type='worker')
        bonus = round(eff_val * 1.50, 2)
        self.set_faction_gold(f, min(MAX_GOLD, self.get_faction_gold(f) + bonus))
        u['ready'] = False
        u['efficiency'] = round(u['efficiency'] + eff_cap + bonus * EFFICIENCY_REWARDS["harvest_bonus_mult"], 2)
        return float(bonus)

    def _is_logistically_safe_to_build(self, r: int, c: int, faction: int) -> bool:
        if float(self.chokepoint_map[r, c]) >= 1.00:
            return False

        f = int(faction)
        my_castle = self.castle_positions[f]
        if abs(r - my_castle[0]) + abs(c - my_castle[1]) <= 1:
            return False

        near_obstacle = False
        if r == 0 or r == self.grid_size - 1 or c == 0 or c == self.grid_size - 1:
            near_obstacle = True
        else:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if int(self.terrain[nr, nc]) == 0 or int(self.struct_type[nr, nc]) == 1:
                        near_obstacle = True
                        break
                if near_obstacle:
                    break

        if not near_obstacle:
            return True

        target_castle = self._get_primary_enemy_castle(f, my_castle)
        base_len = self._shortest_path_length(my_castle, target_castle)
        new_len = self._shortest_path_length(my_castle, target_castle, block_pos=(r, c))
        if new_len >= 999 or (new_len - base_len) > 4:
            return False
        return True

    def _find_adjacent_tower_site(self, pos: tuple, faction: int):
        r, c = pos
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if int(self.territory[nr, nc]) not in enemies and int(self.terrain[nr, nc]) != 0:
                    if not self.is_castle_tile(nr, nc, active_only=False):
                        if int(self.struct_type[nr, nc]) == 0 and (nr, nc) not in self.units:
                            if self._is_logistically_safe_to_build(nr, nc, f):
                                candidates.append((int(nr), int(nc)))
        if not candidates:
            return None
        enemy_castle = self._get_primary_enemy_castle(f, pos)
        return min(
            candidates,
            key=lambda p: (
                (abs(p[0] - enemy_castle[0]) + abs(p[1] - enemy_castle[1]))
                + float(self.tile_yields[p[0], p[1]]) * 12.00
                - (1.50 if float(self.chokepoint_map[p[0], p[1]]) == 0.40 else 0.00)
            )
        )

    def _find_barracks_build_site(self, pos: tuple, faction: int, enforce_ai_distance: bool = False):
        r, c = pos
        f = int(faction)
        my_castle = self.castle_positions[f]
        enemy_castle = self._get_primary_enemy_castle(f, pos)
        enemies = set(self.get_enemy_factions(f))

        check_sites = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        valid_sites = []
        for nr, nc in check_sites:
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                continue
            if self.is_castle_tile(nr, nc, active_only=False):
                continue
            if int(self.terrain[nr, nc]) == 0 or int(self.struct_type[nr, nc]) != 0:
                continue
            if int(self.territory[nr, nc]) in enemies:
                continue
            if (nr, nc) != (r, c) and (nr, nc) in self.units:
                continue
            dist_home = abs(nr - my_castle[0]) + abs(nc - my_castle[1])
            if enforce_ai_distance and dist_home < MIN_BARRACKS_CASTLE_DIST:
                continue
            if enforce_ai_distance and float(self.tile_yields[nr, nc]) > 0.05:
                continue
            valid_sites.append((int(nr), int(nc)))

        if not valid_sites:
            return None
        return min(
            valid_sites,
            key=lambda p: (abs(p[0] - enemy_castle[0]) + abs(p[1] - enemy_castle[1])) + float(self.tile_yields[p[0], p[1]]) * 15.00
        )

    def execute_worker_build(self, pos: tuple, build_type: int) -> bool:
        if pos not in self.units:
            return False
        u = self.units[pos]
        if u['type'] != 'worker' or not u['ready']:
            return False
        f = int(u['faction'])
        cost = STRUCT_STATS[build_type]['cost']
        gold = self.get_faction_gold(f)
        if gold < cost:
            return False

        is_ai = (f == 1 or self.mode == "selfplay")
        if build_type == 2:
            target_site = self._find_barracks_build_site(pos, f, enforce_ai_distance=is_ai)
        else:
            target_site = self._find_adjacent_tower_site(pos, f)

        if target_site is None:
            return False

        sr, sc = target_site
        self.set_faction_gold(f, gold - cost)

        self.territory[sr, sc] = f
        self.struct_owner[sr, sc] = f
        self.struct_type[sr, sc] = build_type
        self.struct_hp[sr, sc] = STRUCT_STATS[build_type]['max_hp']
        self.tower_ready[sr, sc] = False
        u['ready'] = False
        u['efficiency'] = round(u['efficiency'] + EFFICIENCY_REWARDS["build_struct_bonus"], 2)

        self._cluster_dirty = True
        self._bfs_cache.clear()
        if build_type == 1:
            self._fp_dirty = True
            self._cached_base_path_len = self._shortest_path_length(self.castle_positions[1], self.castle_positions[2])
        return True

    def _has_enemy_territory_within(self, site: tuple, faction: int, max_dist: int = MAX_SLOW_UNIT_FRONTLINE_DIST) -> bool:
        f = int(faction)
        enemy_mask = (self.territory != 0) & (self.territory != f)
        if not np.any(enemy_mask):
            return False
        sr, sc = int(site[0]), int(site[1])
        dists = np.abs(self._grid_r[enemy_mask] - sr) + np.abs(self._grid_c[enemy_mask] - sc)
        return bool(np.min(dists) <= max_dist)

    def execute_structure_spawn(self, struct_pos: tuple, faction: int, unit_type: str, enforce_ai_rules: bool = None) -> bool:
        if struct_pos in self.units:
            return False

        r, c = struct_pos
        f = int(faction)
        my_castle = self.castle_positions[f]
        is_my_castle = (struct_pos == my_castle)
        is_my_barracks = (int(self.struct_owner[r, c]) == f and int(self.struct_type[r, c]) == 2)
        if not (is_my_castle or is_my_barracks):
            return False

        if unit_type == 'worker':
            curr_workers = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'worker')
            if curr_workers >= MAX_WORKERS_PER_FACTION:
                return False

        is_ai = enforce_ai_rules if enforce_ai_rules is not None else (f == 1 or self.mode == "selfplay")
        if is_ai and UNIT_STATS[unit_type]['move_range'] <= 1:
            if not self._has_enemy_territory_within(struct_pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                return False

        cost = UNIT_STATS[unit_type]['cost']
        gold = self.get_faction_gold(f)
        if gold < cost:
            return False

        self.set_faction_gold(f, gold - cost)
        self.units[(int(r), int(c))] = self._create_unit_dict(f, unit_type, ready=False)
        self._capture_tile(r, c, f, u_type=unit_type)
        self._fp_dirty = True
        return True

    def get_valid_action_mask_for_entity(self, entity_kind: str, pos: tuple, faction: int) -> np.ndarray:
        mask = np.zeros(30, dtype=bool)
        r, c = pos
        f = int(faction)
        gold = self.get_faction_gold(f)
        is_ai = (f == 1 or self.mode == "selfplay")

        if entity_kind == 'unit':
            if pos not in self.units or not self.units[pos]['ready']:
                mask[24] = True
                return mask
            u = self.units[pos]
            valid_moves = set(self.get_valid_moves_for_unit(pos))
            valid_targets = set(self.get_valid_attack_targets(pos))

            for idx, (dr, dc) in enumerate(RELATIVE_OFFSETS):
                nr, nc = r + dr, c + dc
                if (nr, nc) in valid_moves:
                    mask[idx] = True
                if (nr, nc) in valid_targets:
                    mask[12 + idx] = True

            mask[24] = True
            mask[28] = True

            if u['type'] == 'worker':
                if int(self.terrain[r, c]) != 0 and self.get_effective_tile_yield(r, c) > 0.00:
                    mask[25] = True
                if gold >= STRUCT_STATS[1]['cost'] and self._find_adjacent_tower_site(pos, f) is not None:
                    mask[26] = True
                if gold >= STRUCT_STATS[2]['cost'] and self._find_barracks_build_site(pos, f, enforce_ai_distance=is_ai) is not None:
                    mask[27] = True
                if self._find_demolishable_structure_site(pos, f) is not None:
                    mask[29] = True

        elif entity_kind == 'tower':
            valid_targets = set(self.get_valid_attack_targets(pos, faction=f, atk_range=2))
            for idx, (dr, dc) in enumerate(RELATIVE_OFFSETS):
                nr, nc = r + dr, c + dc
                if (nr, nc) in valid_targets:
                    mask[12 + idx] = True
            if not np.any(mask):
                mask[24] = True

        elif entity_kind == 'spawn':
            mask[24] = True
            if pos not in self.units:
                curr_workers = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'worker')
                if gold >= UNIT_STATS['worker']['cost'] and curr_workers < MAX_WORKERS_PER_FACTION:
                    mask[27] = True
                if gold >= UNIT_STATS['warrior']['cost']:
                    mask[28] = True
                if gold >= UNIT_STATS['archer']['cost']:
                    if not is_ai or self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                        mask[29] = True

        return mask

    def apply_entity_action(self, entity_kind: str, pos: tuple, faction: int, action: int, precomputed_mask: np.ndarray = None) -> float:
        f = int(faction)
        mask = precomputed_mask if precomputed_mask is not None else self.get_valid_action_mask_for_entity(entity_kind, pos, f)
        if not mask[action]:
            valid_indices = np.where(mask)[0]
            action = int(valid_indices[0]) if len(valid_indices) > 0 else 24

        if self.record_telemetry:
            self.record_entity_decision(entity_kind, pos, f, action)

        r, c = pos
        enemies = set(self.get_enemy_factions(f))
        reward = 0.00

        if entity_kind == 'unit':
            if pos not in self.units:
                return 0.00
            u_type = self.units[pos]['type']
            if 0 <= action <= 11:
                dr, dc = RELATIVE_OFFSETS[action]
                dest = (r + dr, c + dc)
                old_owner = int(self.territory[dest[0], dest[1]])
                tile_val = self.get_effective_tile_yield(dest[0], dest[1])
                c_val = float(self._compute_resource_cluster_map()[dest[0], dest[1]])
                if self.execute_unit_move(pos, dest):
                    self.last_ai_turn_traces.append({"kind": "move", "faction": f, "from": [int(r), int(c)], "to": [int(dest[0]), int(dest[1])]})
                    cap_scale = 1.00 if u_type == 'worker' else 0.20
                    if old_owner == 0:
                        reward += (0.12 + tile_val * 0.85 + c_val * 0.15) * cap_scale
                    elif old_owner in enemies:
                        reward += (0.18 + tile_val * 0.95 + c_val * 0.15) * cap_scale
            elif 12 <= action <= 23:
                dr, dc = RELATIVE_OFFSETS[action - 12]
                target = (r + dr, c + dc)
                dmg = self.execute_attack_on_target(pos, target, is_tower=False)
                reward += dmg * (0.22 if u_type != 'worker' else 0.08)
                if dmg > 0.00:
                    self.last_ai_turn_traces.append({"kind": "attack", "faction": f, "from": [int(r), int(c)], "to": [int(target[0]), int(target[1])]})
                    self.add_localized_event("ai_attack", u_type=u_type, r=target[0], c=target[1], dmg=dmg)
            elif action == 24:
                healed = self.heal_unit_percentage(pos)
                reward += healed * 0.04
            elif action == 25:
                bonus = self.execute_worker_harvest(pos)
                reward += bonus * 0.20
                if bonus > 0.00:
                    self.add_localized_event("ai_harvest", bonus=bonus)
            elif action == 26:
                if self.execute_worker_build(pos, build_type=1):
                    reward += 0.22
                    self.add_localized_event("ai_build_tower")
            elif action == 27:
                if self.execute_worker_build(pos, build_type=2):
                    reward += 0.30
                    self.add_localized_event("ai_build_barracks")
            elif action == 28:
                refund = self.execute_unit_disband(pos)
                if refund > 0.00:
                    reward += 0.15 if (self.turn_count <= 2 and u_type == 'archer') else -0.05
                    self.add_localized_event("ai_disband", u_type=u_type, refund=refund)
            elif action == 29:
                refund, st_t = self.execute_worker_demolish(pos)
                if refund > 0.00:
                    reward += 0.10
                    self.add_localized_event("ai_demolish", refund=refund)

        elif entity_kind == 'tower':
            if 12 <= action <= 23:
                dr, dc = RELATIVE_OFFSETS[action - 12]
                target = (r + dr, c + dc)
                dmg = self.execute_attack_on_target(pos, target, is_tower=True)
                reward += dmg * 0.10
                if dmg > 0.00:
                    self.last_ai_turn_traces.append({"kind": "attack", "faction": f, "from": [int(r), int(c)], "to": [int(target[0]), int(target[1])]})
                    self.add_localized_event("ai_tower_shot", r=target[0], c=target[1], dmg=dmg)
            else:
                self.tower_ready[r, c] = False

        elif entity_kind == 'spawn':
            if action == 27:
                if self.execute_structure_spawn(pos, f, 'worker', enforce_ai_rules=True):
                    reward += 0.06
                    self.add_localized_event("ai_spawn", u_type="worker")
            elif action == 28:
                if self.execute_structure_spawn(pos, f, 'warrior', enforce_ai_rules=True):
                    reward += 0.16
                    self.add_localized_event("ai_spawn", u_type="warrior")
            elif action == 29:
                if self.execute_structure_spawn(pos, f, 'archer', enforce_ai_rules=True):
                    reward += 0.16
                    self.add_localized_event("ai_spawn", u_type="archer")

        return float(reward)

    def _get_obs_for_entity(self, entity_kind: str, pos: tuple, faction: int) -> np.ndarray:
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        obs = np.zeros((18, self.grid_size, self.grid_size), dtype=np.float32)

        obs[0, pos[0], pos[1]] = 1.00
        obs[1] = (self.terrain == 0).astype(np.float32)
        obs[2] = self.get_effective_yield_map()
        obs[3] = self.chokepoint_map
        obs[4] = (self.territory == f).astype(np.float32)
        obs[5] = ((self.territory != 0) & (self.territory != f)).astype(np.float32)
        obs[6] = (self.territory == 0).astype(np.float32)

        my_castle = self.castle_positions[f]
        my_c_hp = self.get_castle_hp(f)

        obs[7] = np.where(self.struct_owner == f, self.struct_hp / 10.00, 0.00)
        obs[7, my_castle[0], my_castle[1]] = my_c_hp / CASTLE_MAX_HP

        obs[8] = np.where((self.struct_owner != 0) & (self.struct_owner != f), self.struct_hp / 10.00, 0.00)
        for ef in enemies:
            ec_pos = self.castle_positions[ef]
            obs[8, ec_pos[0], ec_pos[1]] = self.get_castle_hp(ef) / CASTLE_MAX_HP

        type_to_ch = {'worker': 0, 'warrior': 1, 'archer': 2}
        for (ur, uc), u in self.units.items():
            ch_off = type_to_ch[u['type']]
            norm_hp = u['hp'] / UNIT_STATS[u['type']]['max_hp']
            if u['faction'] == f:
                obs[9 + ch_off, ur, uc] = norm_hp
            else:
                obs[12 + ch_off, ur, uc] = norm_hp

        ally_fp, enemy_fp = self._compute_firepower_maps(f)
        obs[15] = np.clip(ally_fp / 10.00, 0.00, 1.00)
        obs[16] = np.clip(enemy_fp / 10.00, 0.00, 1.00)

        my_gold = self.get_faction_gold(f)
        obs[17, :, :] = min(1.00, my_gold / MAX_GOLD)

        if f == 2:
            obs = obs[:, ::-1, ::-1].copy()
        elif f == 3:
            obs = obs[:, ::-1, :].copy()
        elif f == 4:
            obs = obs[:, :, ::-1].copy()
        return obs

    def _get_obs(self):
        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue[0]
            return self._get_obs_for_entity(kind, pos, faction=1)
        return self._get_obs_for_entity('spawn', self.castle_positions[1], faction=1)

    def get_current_p1_entity_and_mask(self):
        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue[0]
            return kind, pos, self.get_valid_action_mask_for_entity(kind, pos, faction=1)
        return 'spawn', self.castle_positions[1], self.get_valid_action_mask_for_entity('spawn', self.castle_positions[1], faction=1)

    def evaluate_player_advantage(self, player_faction: int = 2, ai_faction: int = 1) -> float:
        pf, af = int(player_faction), int(ai_faction)
        hp_sum = max(0.01, self.get_castle_hp(af) + self.get_castle_hp(pf))
        r_castle = self.get_castle_hp(pf) / hp_sum

        inc1, _, _, t1, _ = self.get_faction_economy_stats(af)
        inc2, _, _, t2, _ = self.get_faction_economy_stats(pf)
        t_sum = max(1.00, float(t1 + t2))
        r_terr = float(t2) / t_sum

        p1_army = self.get_faction_gold(af) + sum(u['hp'] * 1.50 for u in self.units.values() if u['faction'] == af) + inc1 * 4.00
        p2_army = self.get_faction_gold(pf) + sum(u['hp'] * 1.50 for u in self.units.values() if u['faction'] == pf) + inc2 * 4.00
        r_power = p2_army / max(0.01, p1_army + p2_army)

        adv = (0.35 * r_castle) + (0.30 * r_terr) + (0.35 * r_power)
        return float(np.clip(adv, 0.00, 1.00))

    def compute_surge_dda_step(self, current_rating: float, is_early_turn: bool = False, player_faction: int = 2, ai_faction: int = 1) -> float:
        adv = self.evaluate_player_advantage(player_faction=player_faction, ai_faction=ai_faction)
        e_adv = adv - 0.50
        if abs(e_adv) <= 0.03:
            return round(float(current_rating), 2)
        m_surge = 1.00 + (((max(0.00, abs(e_adv) - 0.04)) / 0.12) ** 2.00) * 3.20
        k_micro = 24.00 if is_early_turn else 18.00
        delta_r = round(e_adv * k_micro * m_surge, 2)
        max_pool_r = PoolRegistry.get_max_rating()
        new_r = max(10.00, min(max(100.00, max_pool_r), float(current_rating) + delta_r))
        return round(new_r, 2)

    def get_match_winner(self):
        alive = self.get_alive_factions()
        if len(alive) == 1:
            return alive[0]
        if len(alive) == 0:
            return 1
        if self.turn_count > MAX_STEPS:
            best_f = alive[0]
            best_score = -9999.00
            for f in alive:
                inc, _, _, tiles, units_cnt = self.get_faction_economy_stats(f)
                sc = self.get_castle_hp(f) + tiles * 0.25 + inc * 4.00 + units_cnt * 1.50
                if sc > best_score:
                    best_score = sc
                    best_f = f
            return best_f
        return None

    def step(self, action):
        self.step_count += 1
        reward = -0.002

        if not self.p1_entity_queue:
            self._prepare_faction_turn(faction=1, collect_economy=True)

        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue.popleft()
            reward += self.apply_entity_action(kind, pos, faction=1, action=int(action))

        winner_f = self.get_match_winner()
        if winner_f is not None:
            term_rew = 2.00 if winner_f == 1 else -2.00
            return self._get_obs(), float(reward + term_rew), True, False, {"winner": f"p{winner_f}", "winning_faction": winner_f}

        if not self.p1_entity_queue:
            self.heal_all_unmoved_units(faction=1)

            if self.mode != "manual_p2":
                for other_f in self.active_factions:
                    if other_f != 1 and self.is_faction_alive(other_f):
                        self.execute_full_ai_turn(faction=other_f, model=self.opponent_model, deterministic=False)

            winner_f = self.get_match_winner()
            if winner_f is not None:
                term_rew = 2.00 if winner_f == 1 else -2.00
                return self._get_obs(), float(reward + term_rew), True, False, {"winner": f"p{winner_f}", "winning_faction": winner_f}

            self.turn_count += 1
            self.record_turn_efficiency_snapshot()
            winner_f = self.get_match_winner()
            if winner_f is not None:
                term_rew = 1.00 if winner_f == 1 else -1.00
                return self._get_obs(), float(reward + term_rew), True, False, {"winner": f"p{winner_f}", "winning_faction": winner_f}

            self._prepare_faction_turn(faction=1, collect_economy=True)

        return self._get_obs(), float(reward), False, False, {}