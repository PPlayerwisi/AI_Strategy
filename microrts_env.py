import os
import json
import time
import copy
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque, OrderedDict
from stable_baselines3 import PPO

import game_config
from game_config import (
    GRID_SIZE, MAX_STEPS, MAX_GOLD, INITIAL_GOLD, HEAL_PCT,
    SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT, CASTLE_MAX_HP,
    MIN_TRAINING_EFFICIENCY, DISBAND_REFUND_RATIO, DEMOLISH_REFUND_RATIO,
    MIN_BARRACKS_CASTLE_DIST, MAX_SLOW_UNIT_FRONTLINE_DIST,
    OSCILLATION_MEMORY_LEN, OSCILLATION_PENALTY, EFFICIENCY_REWARDS,
    POOL_DIR, HUMAN_TELEMETRY_FILE,
    POOL_REGISTRY_FILE, USERS_DB_FILE, TILE_VALUES,
    UNIT_STATS, STRUCT_STATS, RELATIVE_OFFSETS
)
from localization import normalize_lang, get_text, get_unit_name, get_struct_name

FACTION_CASTLE_POSITIONS = getattr(game_config, "FACTION_CASTLE_POSITIONS", {
    1: (GRID_SIZE - 1, 0),
    2: (0, GRID_SIZE - 1),
    3: (0, 0),
    4: (GRID_SIZE - 1, GRID_SIZE - 1)
})

_orig_torch_load = torch.load
def _permissive_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _permissive_torch_load

REGISTRY_LOCK = threading.Lock()

class ModelCache:
    def __init__(self, max_size=8):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get_model(self, path):
        if not path or not os.path.exists(path):
            return None
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
            try:
                model = PPO.load(path, env=None, device="cpu")
                self.cache[path] = model
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                return model
            except Exception:
                return None

GLOBAL_MODEL_CACHE = ModelCache(max_size=8)

class PoolRegistry:
    @staticmethod
    def ensure_initialized():
        os.makedirs(POOL_DIR, exist_ok=True)
        with REGISTRY_LOCK:
            registry = {"generations": []}
            if os.path.exists(POOL_REGISTRY_FILE):
                try:
                    with open(POOL_REGISTRY_FILE, "r", encoding="utf-8") as f:
                        registry = json.load(f)
                except Exception:
                    registry = {"generations": []}

            existing_ids = {g["gen_id"] for g in registry.get("generations", [])}
            for i in range(0, 101):
                gid = f"gen_{i}"
                if gid not in existing_ids:
                    fname = f"generation_{i}.zip"
                    registry["generations"].append({
                        "gen_id": gid,
                        "label": f"Gen-{i}",
                        "filename": fname,
                        "rating": round(float(i), 2),
                        "origin": "base_curriculum",
                        "created_at": 0
                    })
            PoolRegistry._save_unlocked(registry)
            return registry

    @staticmethod
    def _save_unlocked(registry_data):
        with open(POOL_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_all_generations():
        reg = PoolRegistry.ensure_initialized()
        return reg.get("generations", [])

    @staticmethod
    def get_max_rating() -> float:
        gens = PoolRegistry.get_all_generations()
        if not gens:
            return 100.00
        return max(100.00, max(float(g.get("rating", 100.00)) for g in gens))

    @staticmethod
    def select_closest_generation(target_rating: float) -> dict:
        gens = PoolRegistry.get_all_generations()
        available_with_files = [
            g for g in gens
            if os.path.exists(os.path.join(POOL_DIR, g["filename"]))
            and (g["filename"] != "generation_0.zip" or target_rating <= 12.00)
        ]
        if available_with_files:
            best_file_entry = min(available_with_files, key=lambda g: abs(float(g["rating"]) - float(target_rating)))
            if abs(float(best_file_entry["rating"]) - float(target_rating)) <= 18.00 or str(best_file_entry.get("origin", "")).startswith("player:"):
                return dict(best_file_entry)

        best_entry = min(gens, key=lambda g: abs(float(g["rating"]) - float(target_rating)))
        return dict(best_entry)

    @staticmethod
    def enforce_pool_storage_limit(max_human_models: int = 25):
        PoolRegistry.ensure_initialized()
        with REGISTRY_LOCK:
            with open(POOL_REGISTRY_FILE, "r", encoding="utf-8") as f:
                reg = json.load(f)

            human_gens = [g for g in reg["generations"] if str(g.get("origin", "")).startswith("player:")]
            if len(human_gens) > max_human_models:
                sorted_by_priority = sorted(
                    human_gens,
                    key=lambda g: (float(g.get("rating", 0.0)), int(g.get("created_at", 0)))
                )
                to_remove = sorted_by_priority[0]
                f_path = os.path.join(POOL_DIR, to_remove["filename"])
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass
                reg["generations"] = [g for g in reg["generations"] if g["gen_id"] != to_remove["gen_id"]]
                PoolRegistry._save_unlocked(reg)

    @staticmethod
    def register_new_generation(label: str, filename: str, rating: float, origin: str, doctrine_alignment: float = 100.00) -> dict:
        PoolRegistry.ensure_initialized()
        with REGISTRY_LOCK:
            with open(POOL_REGISTRY_FILE, "r", encoding="utf-8") as f:
                reg = json.load(f)
            next_num = len(reg["generations"])
            entry = {
                "gen_id": f"gen_evo_{next_num}_{int(time.time())}",
                "label": label,
                "filename": filename,
                "rating": round(float(rating), 2),
                "origin": origin,
                "doctrine_alignment": round(float(doctrine_alignment), 2),
                "created_at": int(time.time())
            }
            reg["generations"].append(entry)
            PoolRegistry._save_unlocked(reg)
        PoolRegistry.enforce_pool_storage_limit(max_human_models=25)
        return entry

def transform_offset_for_faction(dr: int, dc: int, faction: int) -> tuple:
    f = int(faction)
    if f == 2:
        return (-int(dr), -int(dc))
    elif f == 3:
        return (-int(dr), int(dc))
    elif f == 4:
        return (int(dr), -int(dc))
    return (int(dr), int(dc))

def transform_action_for_faction(action: int, faction: int) -> int:
    act = int(action)
    f = int(faction)
    if f == 1:
        return act
    if 0 <= act <= 11:
        dr, dc = RELATIVE_OFFSETS[act]
        tdr, tdc = transform_offset_for_faction(dr, dc, f)
        if (tdr, tdc) in RELATIVE_OFFSETS:
            return int(RELATIVE_OFFSETS.index((tdr, tdc)))
    elif 12 <= act <= 23:
        dr, dc = RELATIVE_OFFSETS[act - 12]
        tdr, tdc = transform_offset_for_faction(dr, dc, f)
        if (tdr, tdc) in RELATIVE_OFFSETS:
            return int(12 + RELATIVE_OFFSETS.index((tdr, tdc)))
    return act

def predict_masked_action(
    model: PPO,
    obs: np.ndarray,
    valid_mask: np.ndarray,
    deterministic: bool = False,
    temperature: float = 1.00,
    logit_priors: np.ndarray = None
) -> int:
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return 24
    if len(valid_indices) == 1:
        return int(valid_indices[0])

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        dist = model.policy.get_distribution(obs_tensor)
        logits = dist.distribution.logits[0].clone()

        if logit_priors is not None:
            prior_t = torch.as_tensor(logit_priors, dtype=torch.float32, device=model.device)
            logits = logits + prior_t

        mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=model.device)
        logits[~mask_tensor] = -1e8

        if deterministic or temperature <= 0.12:
            return int(torch.argmax(logits).item())

        scaled_logits = logits / float(max(0.15, temperature))
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, 1).item()
        return int(sampled)

class MicroRTSEnv(gym.Env):
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

    def evaluate_sample_doctrine_compliance(self, obs_arr: np.ndarray, act_val: int) -> float:
        pos_plane = obs_arr[0]
        coords = np.argwhere(pos_plane > 0.50)
        if len(coords) == 0:
            return 0.50
        r, c = int(coords[0][0]), int(coords[0][1])

        yield_plane = obs_arr[2]
        ally_terr = obs_arr[4]
        enemy_terr = obs_arr[5]
        neutral_terr = obs_arr[6]
        ally_fp = obs_arr[15]
        enemy_fp = obs_arr[16]
        gold_norm = float(obs_arr[17, 0, 0])

        i_econ = 0.85
        i_cluster = 0.80
        i_pos = 0.85

        if 0 <= act_val <= 11:
            dr, dc = RELATIVE_OFFSETS[act_val]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if neutral_terr[nr, nc] > 0.50 or enemy_terr[nr, nc] > 0.50:
                    i_econ = 1.00 if yield_plane[nr, nc] >= 0.15 else 0.90
                    i_cluster = 1.00
                elif ally_terr[nr, nc] > 0.50:
                    r0, r1 = max(0, nr - 3), min(self.grid_size, nr + 4)
                    c0, c1 = max(0, nc - 3), min(self.grid_size, nc + 4)
                    local_rich = np.sum(yield_plane[r0:r1, c0:c1] * (1.00 - ally_terr[r0:r1, c0:c1]))
                    i_cluster = 0.85 if local_rich >= 0.30 else 0.25
                if enemy_fp[nr, nc] > ally_fp[nr, nc] + 0.25:
                    i_pos = 0.20
                else:
                    i_pos = 1.00
        elif 12 <= act_val <= 23:
            i_econ = 0.95
            i_cluster = 1.00
            i_pos = 1.00
        elif act_val == 25:
            i_econ = 1.00 if yield_plane[r, c] >= 0.15 else 0.45
            i_cluster = 0.90
        elif act_val in (26, 27):
            dist_home = abs(r - (self.grid_size - 1)) + abs(c - 0)
            if act_val == 27 and dist_home < MIN_BARRACKS_CASTLE_DIST:
                i_pos = 0.15
                i_econ = 0.30
            elif yield_plane[r, c] >= 0.15:
                i_econ = 0.20
            else:
                i_econ = 0.95
                i_pos = 0.95
        elif act_val in (28, 29):
            if gold_norm < 0.12:
                i_econ = 0.25
            else:
                i_econ = 0.90

        v_doc = (0.35 * i_econ) + (0.35 * i_cluster) + (0.30 * i_pos)
        return round(float(np.clip(v_doc, 0.10, 1.00)), 2)

    def record_entity_decision(self, entity_kind: str, pos: tuple, faction: int, action: int):
        if not self.record_telemetry:
            return
        obs = self._get_obs_for_entity(entity_kind, pos, faction)
        canonical_act = transform_action_for_faction(int(action), int(faction))
        if entity_kind == 'unit' and pos in self.units:
            self.units[pos]['history'].append((obs, int(canonical_act)))
        else:
            self.building_action_records.append((obs, int(canonical_act), int(faction)))

    def collect_match_training_samples(self, winning_faction: int = None, min_efficiency: float = MIN_TRAINING_EFFICIENCY):
        eff_threshold = max(float(MIN_TRAINING_EFFICIENCY), float(min_efficiency))
        all_units = list(self.units.values()) + self.archived_units
        selected_obs = []
        selected_acts = []
        selected_weights = []

        for u in all_units:
            eff = float(u.get('efficiency', 0.00))
            if eff >= eff_threshold and len(u.get('history', [])) > 0:
                outcome_bonus = 1.50 if (winning_faction is not None and u['faction'] == winning_faction) else 0.50
                base_weight = min(6.00, round(1.00 + (eff / 50.00) + outcome_bonus, 2))
                for obs_arr, act_val in u['history']:
                    v_doc = self.evaluate_sample_doctrine_compliance(obs_arr, int(act_val))
                    adj_weight = round(base_weight * v_doc, 2)
                    selected_obs.append(obs_arr)
                    selected_acts.append(int(act_val))
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

        new_obs = np.array(selected_obs, dtype=np.float32)
        new_acts = np.array(selected_acts, dtype=np.int64)
        new_weights = np.array(selected_weights, dtype=np.float32)

        with REGISTRY_LOCK:
            if os.path.exists(HUMAN_TELEMETRY_FILE):
                try:
                    data = np.load(HUMAN_TELEMETRY_FILE)
                    new_obs = np.concatenate([data['obs'], new_obs], axis=0)[-8000:]
                    new_acts = np.concatenate([data['acts'], new_acts], axis=0)[-8000:]
                    new_weights = np.concatenate([data['weights'], new_weights], axis=0)[-8000:]
                except Exception:
                    pass
            np.savez_compressed(HUMAN_TELEMETRY_FILE, obs=new_obs, acts=new_acts, weights=new_weights)
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

    def _capture_tile(self, r: int, c: int, faction: int) -> float:
        f = int(faction)
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size and int(self.terrain[r, c]) != 0:
            st_own = int(self.struct_owner[r, c])
            if st_own == 0 or st_own == f:
                old_owner = int(self.territory[r, c])
                self.territory[r, c] = f
                eff_y = self.get_effective_tile_yield(r, c)
                if old_owner == 0:
                    return round(EFFICIENCY_REWARDS["neutral_capture_base"] + eff_y * EFFICIENCY_REWARDS["neutral_capture_yield_mult"], 2)
                elif old_owner != f:
                    return round(EFFICIENCY_REWARDS["enemy_capture_base"] + eff_y * EFFICIENCY_REWARDS["enemy_capture_yield_mult"], 2)
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

        eff_gain = 0.00
        sr, sc = start_pos
        dr, dc = dest_pos
        if abs(sr - dr) + abs(sc - dc) == 2:
            best_mid = self._find_best_intermediate_step(sr, sc, dr, dc, u['faction'])
            if best_mid is not None:
                eff_gain += self._capture_tile(best_mid[0], best_mid[1], u['faction'])

        del self.units[start_pos]
        u['ready'] = False
        eff_gain += self._capture_tile(dest_pos[0], dest_pos[1], u['faction'])
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
                        kill_bonus += 12.00
                    break

        if attacker_u is not None and dealt > 0.00:
            attacker_u['efficiency'] = round(
                attacker_u['efficiency'] + dealt * EFFICIENCY_REWARDS["damage_dealt_mult"] + kill_bonus, 2
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

        eff_cap = self._capture_tile(r, c, f)
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
        self._capture_tile(r, c, f)
        self._fp_dirty = True
        return True

    def _compute_firepower_maps(self, faction: int):
        if self._fp_dirty:
            fp_maps = {f: np.zeros((self.grid_size, self.grid_size), dtype=np.float32) for f in range(1, 5)}

            for (ur, uc), u in self.units.items():
                uf = int(u['faction'])
                if uf not in fp_maps:
                    continue
                rng = UNIT_STATS[u['type']]['atk_range']
                dmg = UNIT_STATS[u['type']]['dmg']
                target_map = fp_maps[uf]
                for dr, dc in RELATIVE_OFFSETS:
                    if abs(dr) + abs(dc) <= rng:
                        nr, nc = ur + dr, uc + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            target_map[nr, nc] += dmg

            tower_coords = np.argwhere(self.struct_type == 1)
            for tr, tc in tower_coords:
                tr_i, tc_i = int(tr), int(tc)
                tf = int(self.struct_owner[tr_i, tc_i])
                if tf not in fp_maps:
                    continue
                target_map = fp_maps[tf]
                for dr, dc in RELATIVE_OFFSETS:
                    if abs(dr) + abs(dc) <= 2:
                        nr, nc = tr_i + dr, tc_i + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            target_map[nr, nc] += STRUCT_STATS[1]['dmg']

            self._cached_fp_maps = fp_maps
            self._fp_dirty = False

        f = int(faction)
        ally_fp = self._cached_fp_maps.get(f, np.zeros((self.grid_size, self.grid_size), dtype=np.float32))
        enemy_fp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for ef in self.get_enemy_factions(f):
            enemy_fp += self._cached_fp_maps[ef]
        return ally_fp, enemy_fp

    def _assess_strategic_situation(self, faction: int) -> dict:
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        my_castle = self.castle_positions[f]

        enemy_combat_positions = []
        my_combat_positions = []
        closest_castle_dist = 999

        for pos, u in self.units.items():
            if u['faction'] in enemies and u['type'] != 'worker':
                enemy_combat_positions.append(pos)
                d = abs(pos[0] - my_castle[0]) + abs(pos[1] - my_castle[1])
                if d < closest_castle_dist:
                    closest_castle_dist = d
            elif u['faction'] == f and u['type'] != 'worker':
                my_combat_positions.append(pos)

        min_inter_unit_dist = 999
        for mr, mc in my_combat_positions:
            for er, ec in enemy_combat_positions:
                dist = abs(mr - er) + abs(mc - ec)
                if dist < min_inter_unit_dist:
                    min_inter_unit_dist = dist

        first_contact = (min_inter_unit_dist <= 7 or closest_castle_dist <= 11)
        under_pressure = (closest_castle_dist <= 10 or (first_contact and len(enemy_combat_positions) > len(my_combat_positions)))
        enemy_retreating = (len(my_combat_positions) >= len(enemy_combat_positions) + 2 and closest_castle_dist > 14)

        return {
            'first_contact': bool(first_contact),
            'under_pressure': bool(under_pressure),
            'enemy_retreating': bool(enemy_retreating),
            'min_contact_dist': int(min_inter_unit_dist),
            'closest_enemy_dist': int(closest_castle_dist),
            'my_combat_count': len(my_combat_positions),
            'enemy_combat_count': len(enemy_combat_positions)
        }

    def _assign_tactical_directives(self, faction: int):
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        my_castle = self.castle_positions[f]
        enemy_castle = self._get_primary_enemy_castle(f, my_castle)

        cluster_map = self._compute_resource_cluster_map()
        eff_yields = self.get_effective_yield_map()
        sit = self._assess_strategic_situation(f)

        candidate_targets = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.terrain[r, c]) == 0 or int(self.struct_type[r, c]) == 1:
                    continue
                owner = int(self.territory[r, c])
                y_val = float(eff_yields[r, c])
                c_val = float(cluster_map[r, c])

                if owner == 0 and (y_val >= 0.15 or c_val >= 0.45):
                    base_p = (y_val * 45.00) + (c_val * 26.00)
                    candidate_targets.append({
                        'pos': (r, c),
                        'role': 'capture_neutral_cluster',
                        'priority': base_p,
                        'max_assignees': 1
                    })
                elif owner in enemies and (y_val >= 0.15 or c_val >= 0.40):
                    raid_mult = 1.25 if sit['enemy_retreating'] else 1.00
                    base_p = ((y_val * 50.00) + (c_val * 24.00) + 6.00) * raid_mult
                    candidate_targets.append({
                        'pos': (r, c),
                        'role': 'raid_enemy_resource',
                        'priority': base_p,
                        'max_assignees': 1
                    })
                elif owner == 0 and y_val > 0.00:
                    candidate_targets.append({
                        'pos': (r, c),
                        'role': 'expand_territory',
                        'priority': (y_val * 25.00) + (c_val * 14.00),
                        'max_assignees': 1
                    })

        for e_pos, eu in self.units.items():
            if eu['faction'] in enemies:
                dist_home = abs(e_pos[0] - my_castle[0]) + abs(e_pos[1] - my_castle[1])
                urgency = 28.00 if dist_home <= 10 else (18.00 if sit['first_contact'] else 10.00)
                if sit['under_pressure']:
                    urgency *= 1.45
                candidate_targets.append({
                    'pos': e_pos,
                    'role': 'eliminate_enemy_unit',
                    'priority': urgency + float(UNIT_STATS[eu['type']]['cost']) * 2.20,
                    'max_assignees': 2
                })

        if sit['enemy_retreating'] or sit['my_combat_count'] >= 4:
            for ef in enemies:
                candidate_targets.append({
                    'pos': self.castle_positions[ef],
                    'role': 'siege_enemy_castle',
                    'priority': 34.00,
                    'max_assignees': 3
                })

        assigned_counts = {}
        my_units = [(pos, u) for pos, u in sorted(self.units.items()) if u['faction'] == f and u['ready']]
        for pos, u in my_units:
            u_type = u['type']
            best_target = enemy_castle
            best_role = 'expand_territory'
            best_utility = -9999.00

            for cand in candidate_targets:
                t_pos = cand['pos']
                role = cand['role']
                used = assigned_counts.get(t_pos, 0)
                if used >= cand['max_assignees']:
                    continue

                if u_type == 'worker' and role in ('eliminate_enemy_unit', 'siege_enemy_castle'):
                    continue

                manhattan_d = abs(pos[0] - t_pos[0]) + abs(pos[1] - t_pos[1])
                role_bonus = 0.00
                if u_type == 'worker' and role in ('capture_neutral_cluster', 'raid_enemy_resource'):
                    role_bonus = 14.00
                elif u_type != 'worker' and role in ('eliminate_enemy_unit', 'raid_enemy_resource', 'siege_enemy_castle'):
                    role_bonus = 12.00

                crowd_penalty = used * 18.00
                utility = cand['priority'] + role_bonus - (manhattan_d * 2.10) - crowd_penalty
                if utility > best_utility:
                    best_utility = utility
                    best_target = t_pos
                    best_role = role

            assigned_counts[best_target] = assigned_counts.get(best_target, 0) + 1
            self.unit_directives[u['id']] = {
                'role': best_role,
                'target': best_target
            }

    def _compute_fertility_attraction(self, nr: int, nc: int, faction: int) -> float:
        f = int(faction)
        eff_yields = self.get_effective_yield_map()
        cluster_map = self._compute_resource_cluster_map()
        rich_mask = (self.territory != f) & ((eff_yields >= 0.15) | (cluster_map >= 0.45))
        if not np.any(rich_mask):
            rich_mask = (self.territory != f) & (self.terrain != 0)
            if not np.any(rich_mask):
                return 0.00

        dists = np.abs(self._grid_r[rich_mask] - nr) + np.abs(self._grid_c[rich_mask] - nc)
        close_mask = (dists <= 11)
        if not np.any(close_mask):
            dists_min = float(np.min(dists))
            return 0.60 / max(1.00, dists_min)

        vals = eff_yields[rich_mask][close_mask]
        c_vals = cluster_map[rich_mask][close_mask]
        d_close = np.maximum(1.00, dists[close_mask].astype(np.float32))
        weights = np.where(vals >= 0.30, vals * 3.50, vals * 2.00) + (c_vals * 1.60)
        return float(np.sum(weights / d_close))

    def _compute_allied_clump_penalty(self, current_pos: tuple, nr: int, nc: int, faction: int) -> float:
        f = int(faction)
        clump = 0.00
        for pos, u in self.units.items():
            if u['faction'] == f and pos != current_pos:
                dist = abs(pos[0] - nr) + abs(pos[1] - nc)
                if dist <= 4:
                    clump += (5.00 - float(dist)) / 4.00
        return clump

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
                if gold >= UNIT_STATS['worker']['cost']:
                    mask[27] = True
                if gold >= UNIT_STATS['warrior']['cost']:
                    mask[28] = True
                if gold >= UNIT_STATS['archer']['cost']:
                    if not is_ai or self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                        mask[29] = True

        return mask

    def _is_recruitment_safe(self, faction: int, new_upkeep: float, unit_cost: float, first_contact: bool) -> bool:
        f = int(faction)
        gold = self.get_faction_gold(f)
        remaining_gold = gold - unit_cost
        if remaining_gold < 0.00:
            return False

        gross_inc, current_upkeep, net_inc, _, unit_cnt = self.get_faction_economy_stats(f)
        extra_tax = SUPPLY_TAX_PER_UNIT if unit_cnt >= SUPPLY_LIMIT else 0.00
        projected_net = net_inc - new_upkeep - extra_tax

        if not first_contact:
            if projected_net >= 0.60 and remaining_gold >= 5.00:
                return True
            return False

        if projected_net >= -0.10:
            return True
        return (remaining_gold + projected_net * 4.00) >= 3.00

    def _evaluate_unit_move_score(self, pos: tuple, action_idx: int, faction: int) -> float:
        r, c = pos
        dr, dc = RELATIVE_OFFSETS[action_idx]
        nr, nc = r + dr, c + dc
        step_dist = abs(dr) + abs(dc)
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))

        u = self.units[pos]
        u_type = u['type']
        ally_fp, enemy_fp = self._compute_firepower_maps(f)
        cluster_map = self._compute_resource_cluster_map()
        sit = self._assess_strategic_situation(f)

        v_double = 0.00
        tiles_captured = 0.00
        if int(self.territory[nr, nc]) != f:
            v_double += self.get_effective_tile_yield(nr, nc)
            tiles_captured += (1.40 if int(self.territory[nr, nc]) in enemies else 1.00)

        if step_dist == 2:
            mid = self._find_best_intermediate_step(r, c, nr, nc, f)
            if mid is not None and int(self.territory[mid[0], mid[1]]) != f:
                v_double += self.get_effective_tile_yield(mid[0], mid[1])
                tiles_captured += (1.40 if int(self.territory[mid[0], mid[1]]) in enemies else 1.00)

        p_fertility = self._compute_fertility_attraction(nr, nc, f)
        c_density = float(cluster_map[nr, nc]) if int(self.territory[nr, nc]) != f else float(cluster_map[nr, nc]) * 0.25
        d_clump = self._compute_allied_clump_penalty(pos, nr, nc, f) if not sit['first_contact'] else 0.00

        directive_info = self.unit_directives.get(u['id'])
        delta_bfs = 0.00
        if directive_info is not None and directive_info.get('target') is not None:
            t_pos = directive_info['target']
            bfs_map = self._get_bfs_distance_map(t_pos, f)
            curr_d = float(bfs_map[r, c])
            next_d = float(bfs_map[nr, nc])
            if curr_d < 90.00 and next_d < 90.00:
                delta_bfs = curr_d - next_d

        osc_penalty = 0.00
        recent_pos = u.get('recent_positions', ())
        if (nr, nc) in recent_pos and int(self.territory[nr, nc]) == f and tiles_captured == 0.00:
            osc_penalty = OSCILLATION_PENALTY

        score = (
            (v_double * 35.00)
            + (c_density * 18.00)
            + (p_fertility * 16.00)
            + (delta_bfs * 14.00)
            + (tiles_captured * 5.00)
            - (d_clump * 5.50)
            - osc_penalty
        )

        if u_type != 'worker' and sit['first_contact']:
            enemy_positions = [pos_e for pos_e, ue in self.units.items() if ue['faction'] in enemies]
            if enemy_positions:
                closest_e = min(enemy_positions, key=lambda p: abs(p[0] - nr) + abs(p[1] - nc))
                dist_e = abs(closest_e[0] - nr) + abs(closest_e[1] - nc)
                if u_type == 'warrior':
                    score += max(0.00, 18.00 - dist_e * 1.40)
                elif u_type == 'archer':
                    score += 13.00 if dist_e == 2 else max(0.00, 10.00 - abs(dist_e - 2) * 1.60)
            score += (ally_fp[nr, nc] - enemy_fp[nr, nc]) * 1.60
        elif u_type == 'worker':
            score -= enemy_fp[nr, nc] * 14.00

        if self.is_tile_built_up(r, c):
            score += 14.00

        return float(score)

    def _compute_strategic_logit_prior(self, entity_kind: str, pos: tuple, faction: int, mask: np.ndarray) -> np.ndarray:
        priors = np.zeros(30, dtype=np.float32)
        valid_acts = np.where(mask)[0]
        if len(valid_acts) <= 1:
            return priors

        f = int(faction)
        if entity_kind == 'unit' and pos in self.units:
            u = self.units[pos]
            if self.turn_count <= 2 and u['type'] == 'archer' and mask[28]:
                enemy_targets = self.get_valid_attack_targets(pos)
                if not enemy_targets:
                    priors[28] = 50.00
                    return priors

            priors[28] = -25.00
            if mask[29]:
                dem_site = self._find_demolishable_structure_site(pos, f)
                my_castle = self.castle_positions[f]
                if dem_site is not None and (
                    (abs(dem_site[0] - my_castle[0]) + abs(dem_site[1] - my_castle[1]) < MIN_BARRACKS_CASTLE_DIST and int(self.struct_type[dem_site[0], dem_site[1]]) == 2)
                    or float(self.tile_yields[dem_site[0], dem_site[1]]) >= 0.35
                ):
                    priors[29] = 18.00
                else:
                    priors[29] = -20.00

            for a in valid_acts:
                if 0 <= a <= 11:
                    m_score = self._evaluate_unit_move_score(pos, a, f)
                    priors[a] = np.clip(m_score * 0.12, -8.00, 10.00)
                elif 12 <= a <= 23:
                    priors[a] = 9.00
        elif entity_kind == 'spawn':
            sit = self._assess_strategic_situation(f)
            _, _, net_inc, _, _ = self.get_faction_economy_stats(f)
            if not sit['first_contact'] and net_inc < 0.60:
                priors[24] = 20.00
            if mask[29] and not self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                priors[29] = -50.00

        return priors

    def get_strategic_teacher_action(self, entity_kind: str, pos: tuple, faction: int, tier: int = 5, precomputed_mask: np.ndarray = None) -> int:
        f = int(faction)
        mask = precomputed_mask if precomputed_mask is not None else self.get_valid_action_mask_for_entity(entity_kind, pos, f)
        valid_acts = np.where(mask)[0]
        if len(valid_acts) == 1:
            return int(valid_acts[0])

        if entity_kind == 'unit' and pos in self.units:
            u = self.units[pos]
            if self.turn_count <= 2 and u['type'] == 'archer' and mask[28]:
                if len(self.get_valid_attack_targets(pos)) == 0:
                    return 28

        comp = self.faction_competence.get(f, 0.65)
        if comp < 1.00 and random.random() > comp:
            move_or_atk = [a for a in valid_acts if a < 24]
            if move_or_atk:
                return int(random.choice(move_or_atk))

        enemies = set(self.get_enemy_factions(f))
        my_castle = self.castle_positions[f]
        enemy_castle = self._get_primary_enemy_castle(f, pos)
        gold = self.get_faction_gold(f)
        r, c = pos

        if tier == 1:
            if entity_kind == 'spawn':
                return 24
            atk_acts = [a for a in valid_acts if 12 <= a <= 23]
            if atk_acts and random.random() < 0.55:
                return int(random.choice(atk_acts))
            move_acts = [a for a in valid_acts if 0 <= a <= 11]
            if move_acts:
                return int(random.choice(move_acts))
            return 24

        ally_fp, enemy_fp = self._compute_firepower_maps(f)
        sit = self._assess_strategic_situation(f)
        gross_inc, upkeep_inc, net_inc, my_tiles_cnt, my_unit_cnt = self.get_faction_economy_stats(f)

        if entity_kind == 'tower':
            best_act = valid_acts[0]
            best_score = -999.00
            for a in valid_acts:
                if 12 <= a <= 23:
                    dr, dc = RELATIVE_OFFSETS[a - 12]
                    tr, tc = r + dr, c + dc
                    score = 10.00
                    if (tr, tc) in self.units:
                        tu = self.units[(tr, tc)]
                        if tu['hp'] <= STRUCT_STATS[1]['dmg']:
                            score += 25.00
                        score += UNIT_STATS[tu['type']]['dmg'] * 3.00
                    if score > best_score:
                        best_score = score
                        best_act = a
            return int(best_act)

        if entity_kind == 'spawn':
            if not sit['first_contact'] and net_inc < 0.60:
                return 24

            my_workers = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'worker')
            my_warriors = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'warrior')
            my_archers = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'archer')
            my_combat = my_warriors + my_archers
            my_barracks = int(np.sum((self.struct_owner == f) & (self.struct_type == 2)))

            max_combat_cap = max(3, int(round(comp * 16.00)))
            if my_combat >= max_combat_cap:
                return 24

            desired_workers = 3 if comp >= 1.00 else 2
            if my_workers < desired_workers and net_inc >= 0.60 and mask[27]:
                if self._is_recruitment_safe(f, UNIT_STATS['worker']['upkeep'], UNIT_STATS['worker']['cost'], sit['first_contact']):
                    return 27

            if not sit['first_contact']:
                if net_inc >= 3.00 and gold >= 16.00 and mask[28]:
                    if self._is_recruitment_safe(f, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], False):
                        return 28
                return 24

            if tier >= 3 and my_barracks == 0 and my_workers >= 1 and net_inc >= 2.20 and not sit['under_pressure']:
                if gold < (STRUCT_STATS[2]['cost'] + UNIT_STATS['warrior']['cost']):
                    return 24

            can_spawn_archer_here = mask[29] and self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST)
            want_more_troops = (my_combat <= sit['enemy_combat_count'] + int(round(comp * 5.00))) or sit['under_pressure'] or (net_inc >= 3.50 and gold >= 18.00)
            if want_more_troops:
                if my_warriors > my_archers and can_spawn_archer_here and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], True):
                    return 29
                if mask[28] and self._is_recruitment_safe(f, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], True):
                    return 28
                if can_spawn_archer_here and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], True):
                    return 29

            return 24

        u = self.units[pos]
        u_type = u['type']
        if u['id'] not in self.unit_directives:
            self._assign_tactical_directives(f)

        atk_acts = [a for a in valid_acts if 12 <= a <= 23]
        if atk_acts:
            best_atk = atk_acts[0]
            best_atk_score = -999.00
            enemy_castles = {self.castle_positions[ef] for ef in enemies}
            for a in atk_acts:
                dr, dc = RELATIVE_OFFSETS[a - 12]
                tr, tc = r + dr, c + dc
                score = 15.00
                if (tr, tc) in enemy_castles:
                    score += 45.00
                elif (tr, tc) in self.units:
                    tu = self.units[(tr, tc)]
                    if tu['hp'] <= UNIT_STATS[u_type]['dmg']:
                        score += 35.00
                    score += UNIT_STATS[tu['type']]['cost'] * 2.50
                elif int(self.struct_owner[tr, tc]) in enemies:
                    score += 28.00 if int(self.struct_type[tr, tc]) == 2 else 22.00
                if score > best_atk_score:
                    best_atk_score = score
                    best_atk = a
            return int(best_atk)

        if u_type == 'worker':
            if mask[29]:
                dem_site = self._find_demolishable_structure_site(pos, f)
                if dem_site is not None:
                    ds_r, ds_c = dem_site
                    dist_dem_home = abs(ds_r - my_castle[0]) + abs(ds_c - my_castle[1])
                    if (int(self.struct_type[ds_r, ds_c]) == 2 and dist_dem_home < MIN_BARRACKS_CASTLE_DIST) or float(self.tile_yields[ds_r, ds_c]) >= 0.35:
                        return 29

            my_barracks = int(np.sum((self.struct_owner == f) & (self.struct_type == 2)))
            my_towers = int(np.sum((self.struct_owner == f) & (self.struct_type == 1)))
            dist_to_home = abs(r - my_castle[0]) + abs(c - my_castle[1])
            dist_to_enemy = abs(r - enemy_castle[0]) + abs(c - enemy_castle[1])

            can_afford_buildings = (sit['first_contact'] and net_inc >= 1.50 and gold >= 10.00) or (net_inc >= 3.80 and gold >= 16.00)
            if can_afford_buildings and tier >= 3:
                max_barracks_target = 1 if comp < 0.65 else (2 if comp < 0.95 else 3)
                if mask[27] and my_barracks < max_barracks_target and dist_to_home >= MIN_BARRACKS_CASTLE_DIST and dist_to_enemy <= 20:
                    return 27
                if mask[26] and my_towers < 2 and sit['under_pressure'] and float(self.chokepoint_map[r, c]) >= 0.40:
                    if float(self.tile_yields[r, c]) <= 0.05:
                        return 26

        best_move_act = 24
        best_move_score = -999.00

        for a in valid_acts:
            if 0 <= a <= 11:
                score = self._evaluate_unit_move_score(pos, a, f)
                if score > best_move_score:
                    best_move_score = score
                    best_move_act = a

        if u_type == 'worker' and mask[25]:
            eff_here = self.get_effective_tile_yield(r, c)
            if eff_here >= 0.35 and best_move_score < 22.00 and enemy_fp[r, c] == 0.00:
                return 25

        if best_move_score > -50.00:
            return int(best_move_act)
        return 25 if (u_type == 'worker' and mask[25]) else 24

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
                    if old_owner == 0:
                        reward += 0.12 + tile_val * 0.85 + c_val * 0.15
                    elif old_owner in enemies:
                        reward += 0.18 + tile_val * 0.95 + c_val * 0.15
            elif 12 <= action <= 23:
                dr, dc = RELATIVE_OFFSETS[action - 12]
                target = (r + dr, c + dc)
                dmg = self.execute_attack_on_target(pos, target, is_tower=False)
                reward += dmg * 0.12
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
                    reward += 0.08
                    self.add_localized_event("ai_spawn", u_type="worker")
            elif action == 28:
                if self.execute_structure_spawn(pos, f, 'warrior', enforce_ai_rules=True):
                    reward += 0.12
                    self.add_localized_event("ai_spawn", u_type="warrior")
            elif action == 29:
                if self.execute_structure_spawn(pos, f, 'archer', enforce_ai_rules=True):
                    reward += 0.14
                    self.add_localized_event("ai_spawn", u_type="archer")

        return float(reward)

    def _predict_faction_model_action(self, model: PPO, kind: str, pos: tuple, faction: int, world_mask: np.ndarray, deterministic: bool, temp: float) -> int:
        f = int(faction)
        obs = self._get_obs_for_entity(kind, pos, f)
        world_priors = self._compute_strategic_logit_prior(kind, pos, f, world_mask)

        if f == 1:
            return predict_masked_action(model, obs, world_mask, deterministic=deterministic, temperature=temp, logit_priors=world_priors)

        canonical_mask = np.zeros(30, dtype=bool)
        canonical_priors = np.zeros(30, dtype=np.float32)
        for w_act in range(30):
            c_act = transform_action_for_faction(w_act, f)
            if world_mask[w_act]:
                canonical_mask[c_act] = True
            canonical_priors[c_act] = world_priors[w_act]

        c_chosen = predict_masked_action(model, obs, canonical_mask, deterministic=deterministic, temperature=temp, logit_priors=canonical_priors)
        w_chosen = transform_action_for_faction(c_chosen, f)
        return int(w_chosen)

    def execute_full_ai_turn(self, faction: int, model: PPO = None, deterministic: bool = False, clear_traces: bool = True):
        f = int(faction)
        if not self.is_faction_alive(f):
            return

        if clear_traces:
            self.last_ai_turn_traces = []

        self._prepare_faction_turn(faction=f, collect_economy=True)
        if not self.is_faction_alive(f):
            return

        active_model = model if model is not None else self.faction_models.get(f, self.opponent_model if f == 1 else None)
        f_tier = self.faction_tiers.get(f, self.curriculum_tier)
        temp = max(0.15, 1.80 - (self.active_gen_idx / 100.00) * 1.60) if f == 1 else 0.65

        if self.turn_count <= 2:
            for pos, u in list(self.units.items()):
                if u['faction'] == f and u['type'] == 'archer' and u['ready']:
                    if len(self.get_valid_attack_targets(pos)) == 0:
                        mask = self.get_valid_action_mask_for_entity('unit', pos, f)
                        self.apply_entity_action('unit', pos, f, 28, precomputed_mask=mask)

        self._assign_tactical_directives(f)

        unit_positions = [pos for pos, u in sorted(self.units.items()) if u['faction'] == f and u['ready']]
        for pos in unit_positions:
            if pos not in self.units or not self.units[pos]['ready']:
                continue
            mask = self.get_valid_action_mask_for_entity('unit', pos, f)
            if active_model is not None:
                act = self._predict_faction_model_action(active_model, 'unit', pos, f, mask, deterministic, temp)
                if act == 24 and self.units[pos]['hp'] >= self.units[pos]['max_hp'] and np.any(mask[:24]):
                    act = self.get_strategic_teacher_action('unit', pos, f, tier=f_tier, precomputed_mask=mask)
            else:
                act = self.get_strategic_teacher_action('unit', pos, f, tier=f_tier, precomputed_mask=mask)
            self.apply_entity_action('unit', pos, f, act, precomputed_mask=mask)

        tower_coords = np.argwhere((self.struct_owner == f) & (self.struct_type == 1) & self.tower_ready)
        for tr, tc in tower_coords:
            t_pos = (int(tr), int(tc))
            if len(self.get_valid_attack_targets(t_pos, faction=f, atk_range=2)) > 0:
                mask = self.get_valid_action_mask_for_entity('tower', t_pos, f)
                if active_model is not None:
                    act = self._predict_faction_model_action(active_model, 'tower', t_pos, f, mask, deterministic, temp)
                else:
                    act = self.get_strategic_teacher_action('tower', t_pos, f, tier=f_tier, precomputed_mask=mask)
                self.apply_entity_action('tower', t_pos, f, act, precomputed_mask=mask)

        gold = self.get_faction_gold(f)
        if gold >= UNIT_STATS['worker']['cost']:
            my_castle = self.castle_positions[f]
            enemy_castle = self._get_primary_enemy_castle(f, my_castle)
            spawn_sites = [my_castle]
            barracks_coords = np.argwhere((self.struct_owner == f) & (self.struct_type == 2))
            for br, bc in barracks_coords:
                spawn_sites.append((int(br), int(bc)))
            spawn_sites.sort(key=lambda s: abs(s[0] - enemy_castle[0]) + abs(s[1] - enemy_castle[1]))

            for site in spawn_sites:
                curr_gold = self.get_faction_gold(f)
                if site not in self.units and curr_gold >= UNIT_STATS['worker']['cost']:
                    mask = self.get_valid_action_mask_for_entity('spawn', site, f)
                    if active_model is not None:
                        act = self._predict_faction_model_action(active_model, 'spawn', site, f, mask, deterministic, temp)
                    else:
                        act = self.get_strategic_teacher_action('spawn', site, f, tier=f_tier, precomputed_mask=mask)
                    self.apply_entity_action('spawn', site, f, act, precomputed_mask=mask)

        self.heal_all_unmoved_units(f)
        moves_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "move" and t.get("faction") == f)
        atks_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "attack" and t.get("faction") == f)
        if f == 1 or moves_cnt > 0 or atks_cnt > 0:
            self.add_localized_event("ai_turn_summary", turn=self.turn_count, moves=moves_cnt, atks=atks_cnt)

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

class OnlineEvolutionEngine:
    @staticmethod
    def _evaluate_candidate_vs_gen100(candidate_model: PPO, gen100_model: PPO) -> float:
        eval_env = MicroRTSEnv(mode="manual_p2", record_telemetry=False, num_players=2)
        scores = []
        for _ in range(2):
            eval_env.reset()
            for _ in range(26):
                eval_env.set_faction_competence(1, 1.05)
                eval_env.set_curriculum_tier(5)
                eval_env.execute_full_ai_turn(faction=1, model=candidate_model, deterministic=False)
                if eval_env.p2_castle_hp <= 0.00:
                    break
                eval_env.set_faction_competence(2, 1.05)
                eval_env.set_curriculum_tier(5)
                eval_env.execute_full_ai_turn(faction=2, model=gen100_model, deterministic=False)
                if eval_env.p1_castle_hp <= 0.00:
                    break
                eval_env.turn_count += 1

            inc1, _, _, t1, u1 = eval_env.get_faction_economy_stats(1)
            inc2, _, _, t2, u2 = eval_env.get_faction_economy_stats(2)
            s_cand = (eval_env.p1_castle_hp * 0.45) + (t1 * 0.30) + (inc1 * 4.50) + (u1 * 1.80)
            s_g100 = (eval_env.p2_castle_hp * 0.45) + (t2 * 0.30) + (inc2 * 4.50) + (u2 * 1.80)
            scores.append(s_cand / max(0.01, s_cand + s_g100))
        return float(np.mean(scores))

    @staticmethod
    def evolve_and_benchmark_from_match(
        match_obs: list,
        match_acts: list,
        match_weights: list,
        username: str = "Player"
    ) -> dict:
        os.makedirs(POOL_DIR, exist_ok=True)
        PoolRegistry.ensure_initialized()

        dummy_env = MicroRTSEnv(mode="selfplay", record_telemetry=False, num_players=2)
        latest_path = os.path.join(POOL_DIR, "latest_checkpoint.zip")
        gen100_path = os.path.join(POOL_DIR, "generation_100.zip")

        gen100_model = GLOBAL_MODEL_CACHE.get_model(gen100_path)
        model = None
        if os.path.exists(latest_path):
            try:
                model = PPO.load(latest_path, env=dummy_env, device="cpu", verbose=0)
            except Exception:
                model = None

        if model is None and os.path.exists(gen100_path):
            try:
                model = PPO.load(gen100_path, env=dummy_env, device="cpu", verbose=0)
            except Exception:
                model = None

        if model is None:
            gens = PoolRegistry.get_all_generations()
            existing = [g for g in gens if os.path.exists(os.path.join(POOL_DIR, g["filename"]))]
            if existing:
                top_g = max(existing, key=lambda x: float(x["rating"]))
                try:
                    model = PPO.load(os.path.join(POOL_DIR, top_g["filename"]), env=dummy_env, device="cpu", verbose=0)
                except Exception:
                    model = None

        if model is None:
            model = PPO("MlpPolicy", dummy_env, verbose=0, learning_rate=0.0005, n_steps=512, batch_size=128, ent_coef=0.02, device="cpu")

        obs_list = list(match_obs)
        acts_list = list(match_acts)
        weights_list = list(match_weights)

        if os.path.exists(HUMAN_TELEMETRY_FILE):
            try:
                data = np.load(HUMAN_TELEMETRY_FILE)
                if len(data['obs']) > 0:
                    take_n = min(256, len(data['obs']))
                    idx = np.random.choice(len(data['obs']), size=take_n, replace=False)
                    obs_list.extend(data['obs'][idx])
                    acts_list.extend(data['acts'][idx])
                    weights_list.extend(data['weights'][idx])
            except Exception:
                pass

        doctrine_scores = [dummy_env.evaluate_sample_doctrine_compliance(o, int(a)) for o, a in zip(obs_list, acts_list)] if len(obs_list) > 0 else [1.00]
        mean_doctrine_alignment = round(float(np.mean(doctrine_scores)) * 100.00, 2)

        if len(obs_list) > 0:
            obs_t = torch.as_tensor(np.array(obs_list, dtype=np.float32), device=model.device)
            acts_t = torch.as_tensor(np.array(acts_list, dtype=np.int64), device=model.device)
            weights_t = torch.as_tensor(np.array(weights_list, dtype=np.float32), device=model.device)
            doc_t = torch.as_tensor(np.array(doctrine_scores, dtype=np.float32), device=model.device)

            if gen100_model is not None:
                with torch.no_grad():
                    g100_dist = gen100_model.policy.get_distribution(obs_t)
                    g100_acts_t = torch.argmax(g100_dist.distribution.logits, dim=-1)
            else:
                g100_acts_t = acts_t.clone()

            base_state_dict = copy.deepcopy(model.policy.state_dict())
            ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
            batch_sz = min(128, len(obs_t))

            optimizer_pure = optim.Adam(model.policy.parameters(), lr=0.0008)
            for _ in range(5):
                perm = torch.randperm(len(obs_t))
                for start_i in range(0, len(obs_t), batch_sz):
                    b_idx = perm[start_i:start_i + batch_sz]
                    dist = model.policy.get_distribution(obs_t[b_idx])
                    logits = dist.distribution.logits
                    ce = ce_loss_fn(logits, acts_t[b_idx])
                    entropy = dist.entropy().mean()
                    loss = torch.mean(ce * weights_t[b_idx]) - (0.02 * entropy)
                    optimizer_pure.zero_grad()
                    loss.backward()
                    optimizer_pure.step()
            pure_state_dict = copy.deepcopy(model.policy.state_dict())
            score_pure = OnlineEvolutionEngine._evaluate_candidate_vs_gen100(model, gen100_model)

            model.policy.load_state_dict(base_state_dict)
            optimizer_hybrid = optim.Adam(model.policy.parameters(), lr=0.0008)
            for _ in range(6):
                perm = torch.randperm(len(obs_t))
                for start_i in range(0, len(obs_t), batch_sz):
                    b_idx = perm[start_i:start_i + batch_sz]
                    dist = model.policy.get_distribution(obs_t[b_idx])
                    logits = dist.distribution.logits
                    ce_human = ce_loss_fn(logits, acts_t[b_idx])
                    ce_g100 = ce_loss_fn(logits, g100_acts_t[b_idx])
                    entropy = dist.entropy().mean()
                    hybrid_term = (doc_t[b_idx] * weights_t[b_idx] * ce_human) + ((1.00 - doc_t[b_idx]) * 2.50 * ce_g100)
                    loss = torch.mean(hybrid_term) - (0.02 * entropy)
                    optimizer_hybrid.zero_grad()
                    loss.backward()
                    optimizer_hybrid.step()
            score_hybrid = OnlineEvolutionEngine._evaluate_candidate_vs_gen100(model, gen100_model)

            if score_pure > score_hybrid + 0.02:
                model.policy.load_state_dict(pure_state_dict)

        all_gens = PoolRegistry.get_all_generations()
        max_curr_rating = max(100.00, max(float(g["rating"]) for g in all_gens))
        test_ratings = [20.00, 45.00, 70.00, 90.00, 100.00, max_curr_rating]

        sim_env = MicroRTSEnv(mode="manual_p2", record_telemetry=False, num_players=2)
        domination_scores = []
        top_wins = 0.00
        top_count = 0.00

        for opp_r in test_ratings:
            sim_env.reset()
            opp_entry = PoolRegistry.select_closest_generation(opp_r)
            opp_model = GLOBAL_MODEL_CACHE.get_model(os.path.join(POOL_DIR, opp_entry.get("filename", "")))
            opp_tier = max(1, min(5, int(round(opp_r / 20.00))))
            opp_comp = float(np.clip(0.30 + (opp_r / 100.00) * 0.75, 0.25, 1.35))

            for _ in range(28):
                sim_env.set_faction_competence(1, 1.05)
                sim_env.set_curriculum_tier(5)
                sim_env.execute_full_ai_turn(faction=1, model=model, deterministic=False)
                if sim_env.p2_castle_hp <= 0.00:
                    break

                sim_env.set_faction_competence(2, opp_comp)
                sim_env.set_curriculum_tier(opp_tier)
                sim_env.execute_full_ai_turn(faction=2, model=opp_model, deterministic=False)
                if sim_env.p1_castle_hp <= 0.00:
                    break
                sim_env.turn_count += 1

            inc1, _, _, t1, u1 = sim_env.get_faction_economy_stats(1)
            inc2, _, _, t2, u2 = sim_env.get_faction_economy_stats(2)
            s_new = (sim_env.p1_castle_hp * 0.45) + (t1 * 0.30) + (inc1 * 4.50) + (u1 * 1.80)
            s_opp = (sim_env.p2_castle_hp * 0.45) + (t2 * 0.30) + (inc2 * 4.50) + (u2 * 1.80)
            p_win = s_new / max(0.01, s_new + s_opp)
            domination_scores.append(p_win)

            if opp_r >= 90.00:
                top_count += 1.00
                if p_win >= 0.52:
                    top_wins += 1.00

        w_gauntlet = float(np.mean(domination_scores))
        r_opp_mean = float(np.mean(test_ratings))
        b_over100 = (top_wins / max(1.00, top_count)) if w_gauntlet > 0.52 else 0.00

        r_assigned = max(10.00, round(r_opp_mean + ((w_gauntlet - 0.50) * 85.00) + (b_over100 * 14.00), 2))

        gen_count = len(all_gens)
        label = f"Gen-{gen_count}-H ({username})"
        filename = f"generation_human_{gen_count}_{int(round(r_assigned))}.zip"
        save_path = os.path.join(POOL_DIR, filename[:-4])

        model.save(save_path)
        model.save(latest_path[:-4])

        entry = PoolRegistry.register_new_generation(
            label=label,
            filename=filename,
            rating=r_assigned,
            origin=f"player:{username}",
            doctrine_alignment=mean_doctrine_alignment
        )
        entry["w_gauntlet"] = round(w_gauntlet * 100.00, 2)
        return entry