import os
import json
import time
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

_orig_torch_load = torch.load
def _permissive_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _permissive_torch_load

GRID_SIZE = 18
MAX_STEPS = 300
MAX_GOLD = 100.00
HEAL_PCT = 0.25
SUPPLY_LIMIT = 8
SUPPLY_TAX_PER_UNIT = 0.15
POOL_DIR = "./selfplay_pool/"
HUMAN_TELEMETRY_FILE = os.path.join(POOL_DIR, "human_telemetry.npz")
POOL_REGISTRY_FILE = os.path.join(POOL_DIR, "pool_registry.json")
USERS_DB_FILE = os.path.join(POOL_DIR, "users_db.json")

TILE_VALUES = {
    0: 0.00,
    1: 0.05,
    2: 0.15,
    3: 0.35
}

UNIT_STATS = {
    'worker':  {'name': 'Робітник',   'cost': 3.00, 'upkeep': 0.40, 'max_hp': 4.00, 'move_range': 2, 'atk_range': 1, 'dmg': 1.00},
    'warrior': {'name': 'Піхотинець', 'cost': 6.00, 'upkeep': 1.10, 'max_hp': 8.00, 'move_range': 2, 'atk_range': 1, 'dmg': 3.00},
    'archer':  {'name': 'Стрілець',   'cost': 8.50, 'upkeep': 1.60, 'max_hp': 4.50, 'move_range': 1, 'atk_range': 2, 'dmg': 2.25}
}

STRUCT_STATS = {
    1: {'name': 'Оборонна Вежа',    'cost': 4.00, 'max_hp': 8.00,  'atk_range': 2, 'dmg': 2.00},
    2: {'name': 'Передова Казарма', 'cost': 5.00, 'max_hp': 10.00, 'atk_range': 0, 'dmg': 0.00}
}

CASTLE_MAX_HP = 25.00

RELATIVE_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

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
    def register_new_generation(label: str, filename: str, rating: float, origin: str) -> dict:
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
                "created_at": int(time.time())
            }
            reg["generations"].append(entry)
            PoolRegistry._save_unlocked(reg)
            return entry

def predict_masked_action(
    model: PPO,
    obs: np.ndarray,
    valid_mask: np.ndarray,
    deterministic: bool = False,
    temperature: float = 1.00
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

    def __init__(self, mode="selfplay", record_telemetry=True):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.mode = mode
        self.record_telemetry = record_telemetry
        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(
            low=0.00, high=1.00,
            shape=(18, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        self.opponent_model = None
        self.curriculum_tier = 5
        self.base_gen_idx = 50.00
        self.active_gen_idx = 50.00
        self.active_gen_label = "Gen-50"
        self.active_gen_origin = "base_curriculum"
        self.f1_Target_competence = 0.65
        self.f2_target_competence = 0.65
        self.episodes_count = 0
        self.next_unit_id = 1

        grid_r, grid_c = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing='ij')
        self._grid_r = grid_r
        self._grid_c = grid_c

        self._fp_dirty = True
        self._cached_p1_fp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._cached_p2_fp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._cached_base_path_len = 34

        self.efficiency_timeline = {
            "turns": [],
            "player_eff": [],
            "ai_eff": [],
            "active_rating": [],
            "active_label": []
        }
        self.last_ai_turn_traces = []
        self.turn_event_log = []

        self.reset()

    def set_curriculum_tier(self, tier: int):
        self.curriculum_tier = int(np.clip(tier, 1, 5))

    def set_faction_competence(self, faction: int, competence: float):
        val = float(np.clip(competence, 0.25, 1.35))
        if faction == 1:
            self.f1_Target_competence = val
        else:
            self.f2_target_competence = val

    def set_dda_generation(self, gen_rating: float):
        max_pool_r = PoolRegistry.get_max_rating()
        self.active_gen_idx = round(float(np.clip(gen_rating, 0.00, max(100.00, max_pool_r))), 2)
        self.f1_Target_competence = float(np.clip(0.30 + (self.active_gen_idx / 100.00) * 0.75, 0.25, 1.35))

        if self.active_gen_idx <= 15.00:
            self.curriculum_tier = 1
        elif self.active_gen_idx <= 35.00:
            self.curriculum_tier = 2
        elif self.active_gen_idx <= 55.00:
            self.curriculum_tier = 3
        elif self.active_gen_idx <= 75.00:
            self.curriculum_tier = 4
        else:
            self.curriculum_tier = 5

        entry = PoolRegistry.select_closest_generation(self.active_gen_idx)
        self.active_gen_label = str(entry.get("label", f"Gen-{int(round(self.active_gen_idx))}"))
        self.active_gen_origin = str(entry.get("origin", "base_curriculum"))
        model_path = os.path.join(POOL_DIR, entry.get("filename", ""))
        self.opponent_model = GLOBAL_MODEL_CACHE.get_model(model_path)

    def _select_selfplay_opponent(self):
        if not os.path.exists(POOL_DIR):
            return
        checkpoints = [f for f in os.listdir(POOL_DIR) if f.endswith(".zip") and f != "latest_checkpoint.zip"]
        if checkpoints:
            chosen = random.choice(checkpoints)
            self.opponent_model = GLOBAL_MODEL_CACHE.get_model(os.path.join(POOL_DIR, chosen))

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

    def _is_map_connected(self, terrain):
        start = (self.grid_size - 1, 0)
        goal = (0, self.grid_size - 1)
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if terrain[nr, nc] != 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def _generate_procedural_world(self):
        for _ in range(60):
            terrain = np.ones((self.grid_size, self.grid_size), dtype=np.int8)
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
            'history': []
        }

    def add_event_log(self, msg: str):
        self.turn_event_log.append(str(msg))
        if len(self.turn_event_log) > 8:
            self.turn_event_log = self.turn_event_log[-8:]

    def compute_faction_total_efficiency(self, faction: int) -> float:
        alive_eff = sum(float(u.get('efficiency', 0.00)) for u in self.units.values() if u['faction'] == faction)
        arch_eff = sum(float(u.get('efficiency', 0.00)) for u in self.archived_units if u['faction'] == faction)
        _, _, net_inc, tile_cnt, _ = self.get_faction_economy_stats(faction)
        total = alive_eff + arch_eff + (tile_cnt * 0.45) + (net_inc * 1.50)
        return round(max(0.00, float(total)), 2)

    def record_turn_efficiency_snapshot(self):
        p_eff = self.compute_faction_total_efficiency(faction=2)
        ai_eff = self.compute_faction_total_efficiency(faction=1)
        self.efficiency_timeline["turns"].append(int(self.turn_count))
        self.efficiency_timeline["player_eff"].append(float(p_eff))
        self.efficiency_timeline["ai_eff"].append(float(ai_eff))
        self.efficiency_timeline["active_rating"].append(round(float(self.active_gen_idx), 2))
        self.efficiency_timeline["active_label"].append(str(self.active_gen_label))

    def get_efficiency_statistics(self) -> dict:
        p_arr = self.efficiency_timeline["player_eff"]
        ai_arr = self.efficiency_timeline["ai_eff"]
        p_med = round(float(np.median(p_arr)), 2) if len(p_arr) > 0 else 0.00
        ai_med = round(float(np.median(ai_arr)), 2) if len(ai_arr) > 0 else 0.00
        return {
            "turns": [int(x) for x in self.efficiency_timeline["turns"]],
            "player_eff": [float(x) for x in p_arr],
            "ai_eff": [float(x) for x in ai_arr],
            "player_median": float(p_med),
            "ai_median": float(ai_med),
            "player_current": float(p_arr[-1]) if p_arr else 0.00,
            "ai_current": float(ai_arr[-1]) if ai_arr else 0.00,
            "active_rating": [float(x) for x in self.efficiency_timeline["active_rating"]],
            "active_label": [str(x) for x in self.efficiency_timeline["active_label"]]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episodes_count += 1
        if self.mode == "selfplay" and self.episodes_count % 4 == 0:
            self._select_selfplay_opponent()

        self.struct_type = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.terrain = self._generate_procedural_world()
        self.tile_yields = np.vectorize(lambda t: TILE_VALUES[int(t)])(self.terrain).astype(np.float32)
        self.chokepoint_map = self._compute_chokepoint_map()

        self.territory = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.struct_owner = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.struct_hp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.tower_ready = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.p1_castle_pos = (self.grid_size - 1, 0)
        self.p2_castle_pos = (0, self.grid_size - 1)
        self.p1_castle_hp = float(CASTLE_MAX_HP)
        self.p2_castle_hp = float(CASTLE_MAX_HP)

        self.p1_gold = 12.00
        self.p2_gold = 12.00

        self.units = {}
        self.archived_units = []
        self.building_action_records = []
        self._spawn_initial_forces()

        self._fp_dirty = True
        self._cached_base_path_len = self._shortest_path_length(self.p1_castle_pos, self.p2_castle_pos)

        self.turn_count = 1
        self.step_count = 0
        self.p1_entity_queue = deque()
        self.last_ai_turn_traces = []
        self.turn_event_log = ["Партію розпочато. Оберіть синій підрозділ або замок."]
        self._prepare_faction_turn(faction=1, collect_economy=False)

        self.efficiency_timeline = {
            "turns": [],
            "player_eff": [],
            "ai_eff": [],
            "active_rating": [],
            "active_label": []
        }
        self.record_turn_efficiency_snapshot()
        return self._get_obs(), {}

    def _spawn_initial_forces(self):
        self.units[(self.grid_size - 1, 1)] = self._create_unit_dict(1, 'worker', ready=True)
        self.units[(self.grid_size - 2, 0)] = self._create_unit_dict(1, 'warrior', ready=True)
        self.units[(self.grid_size - 2, 1)] = self._create_unit_dict(1, 'archer', ready=True)
        for r, c in [self.p1_castle_pos, (self.grid_size - 1, 1), (self.grid_size - 2, 0), (self.grid_size - 2, 1)]:
            self.territory[r, c] = 1

        self.units[(0, self.grid_size - 2)] = self._create_unit_dict(2, 'worker', ready=True)
        self.units[(1, self.grid_size - 1)] = self._create_unit_dict(2, 'warrior', ready=True)
        self.units[(1, self.grid_size - 2)] = self._create_unit_dict(2, 'archer', ready=True)
        for r, c in [self.p2_castle_pos, (0, self.grid_size - 2), (1, self.grid_size - 1), (1, self.grid_size - 2)]:
            self.territory[r, c] = 2

    def is_tile_built_up(self, r: int, c: int) -> bool:
        if (r, c) == self.p1_castle_pos or (r, c) == self.p2_castle_pos:
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
        eff_map[self.p1_castle_pos[0], self.p1_castle_pos[1]] = 0.00
        eff_map[self.p2_castle_pos[0], self.p2_castle_pos[1]] = 0.00
        return eff_map

    def record_entity_decision(self, entity_kind: str, pos: tuple, faction: int, action: int):
        if not self.record_telemetry:
            return
        obs = self._get_obs_for_entity(entity_kind, pos, faction)
        if entity_kind == 'unit' and pos in self.units:
            self.units[pos]['history'].append((obs, int(action)))
        else:
            self.building_action_records.append((obs, int(action), int(faction)))

    def collect_match_training_samples(self, winning_faction: int = None, min_efficiency: float = 2.50):
        all_units = list(self.units.values()) + self.archived_units
        selected_obs = []
        selected_acts = []
        selected_weights = []

        for u in all_units:
            eff = float(u.get('efficiency', 0.00))
            if eff >= min_efficiency and len(u.get('history', [])) > 0:
                outcome_bonus = 1.50 if (winning_faction is not None and u['faction'] == winning_faction) else 0.35
                weight = min(6.00, round(1.00 + (eff / 3.50) + outcome_bonus, 2))
                for obs_arr, act_val in u['history']:
                    selected_obs.append(obs_arr)
                    selected_acts.append(int(act_val))
                    selected_weights.append(float(weight))

        if winning_faction is not None:
            for obs_arr, act_val, fac in self.building_action_records:
                if fac == winning_faction and act_val != 24:
                    selected_obs.append(obs_arr)
                    selected_acts.append(int(act_val))
                    selected_weights.append(2.00)

        return selected_obs, selected_acts, selected_weights

    def export_high_efficiency_telemetry(self, winning_faction: int = None, min_efficiency: float = 2.50) -> int:
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
        owned_mask = (self.territory == faction)
        eff_yields = self.get_effective_yield_map()
        gross_inc = float(np.sum(eff_yields[owned_mask]))
        tile_cnt = int(np.sum(owned_mask))

        base_upkeep = 0.00
        unit_cnt = 0
        for u in self.units.values():
            if u['faction'] == faction:
                base_upkeep += UNIT_STATS[u['type']]['upkeep']
                unit_cnt += 1

        supply_tax = max(0, unit_cnt - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT
        total_upkeep = base_upkeep + supply_tax

        net_inc = gross_inc - total_upkeep
        return round(gross_inc, 2), round(total_upkeep, 2), round(net_inc, 2), int(tile_cnt), int(unit_cnt)

    def _prepare_faction_turn(self, faction: int, collect_economy=True):
        if collect_economy:
            gross_inc, upkeep, net_inc, _, _ = self.get_faction_economy_stats(faction)
            curr_gold = self.p1_gold if faction == 1 else self.p2_gold
            new_gold = curr_gold + net_inc

            if new_gold < 0.00:
                dead_coords = [pos for pos, u in self.units.items() if u['faction'] == faction]
                for pos in dead_coords:
                    u_dead = self.units.pop(pos)
                    u_dead['efficiency'] = round(u_dead['efficiency'] - 2.50, 2)
                    self.archived_units.append(u_dead)
                new_gold = 0.00
                self._fp_dirty = True
                f_name = "Червона Імперія (ШІ)" if faction == 1 else "Синій Альянс (Ви)"
                self.add_event_log(f"БАНКРУТСТВО: {f_name} втрачає армію через дефіцит казни!")
            else:
                new_gold = min(MAX_GOLD, new_gold)

            if faction == 1:
                self.p1_gold = round(float(new_gold), 2)
            else:
                self.p2_gold = round(float(new_gold), 2)

        for pos, u in self.units.items():
            if u['faction'] == faction:
                u['ready'] = True

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.struct_owner[r, c]) == faction and int(self.struct_type[r, c]) == 1:
                    self.tower_ready[r, c] = True
                elif int(self.struct_owner[r, c]) == faction:
                    self.tower_ready[r, c] = False

        if faction == 1:
            self._rebuild_p1_entity_queue()

    def _rebuild_p1_entity_queue(self):
        self.p1_entity_queue.clear()
        for pos, u in sorted(self.units.items()):
            if u['faction'] == 1 and u['ready']:
                self.p1_entity_queue.append(('unit', pos))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.struct_owner[r, c]) == 1 and int(self.struct_type[r, c]) == 1 and bool(self.tower_ready[r, c]):
                    if len(self.get_valid_attack_targets((r, c), faction=1, atk_range=2)) > 0:
                        self.p1_entity_queue.append(('tower', (r, c)))
        if self.p1_gold >= UNIT_STATS['worker']['cost']:
            spawn_sites = [self.p1_castle_pos]
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if int(self.struct_owner[r, c]) == 1 and int(self.struct_type[r, c]) == 2:
                        spawn_sites.append((r, c))
            spawn_sites.sort(key=lambda s: abs(s[0] - self.p2_castle_pos[0]) + abs(s[1] - self.p2_castle_pos[1]))
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
        enemy_faction = 2 if faction == 1 else 1
        if int(self.struct_owner[r, c]) == enemy_faction and int(self.struct_type[r, c]) == 2:
            return False
        enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        if (r, c) == enemy_castle:
            return False
        if (r, c) in self.units and self.units[(r, c)]['faction'] == enemy_faction:
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

        enemy_faction = 2 if faction == 1 else 1
        enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        r, c = pos
        targets = []

        for idx, (dr, dc) in enumerate(RELATIVE_OFFSETS):
            dist = abs(dr) + abs(dc)
            if dist <= atk_range:
                tr, tc = r + dr, c + dc
                if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
                    has_enemy_unit = ((tr, tc) in self.units and self.units[(tr, tc)]['faction'] == enemy_faction)
                    has_enemy_struct = (int(self.struct_owner[tr, tc]) == enemy_faction and int(self.struct_type[tr, tc]) != 0)
                    is_enemy_castle = ((tr, tc) == enemy_castle)
                    if has_enemy_unit or has_enemy_struct or is_enemy_castle:
                        targets.append((int(tr), int(tc)))
        return targets

    def _capture_tile(self, r: int, c: int, faction: int) -> float:
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size and int(self.terrain[r, c]) != 0:
            enemy_faction = 2 if faction == 1 else 1
            if int(self.struct_owner[r, c]) != enemy_faction:
                old_owner = int(self.territory[r, c])
                self.territory[r, c] = faction
                tile_bonus = self.get_effective_tile_yield(r, c) * 4.00
                if old_owner == 0:
                    return round(1.20 + tile_bonus, 2)
                elif old_owner == enemy_faction:
                    return round(2.00 + tile_bonus, 2)
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
        u['efficiency'] = round(u['efficiency'] + gained * 0.40, 2)
        return float(gained)

    def heal_all_unmoved_units(self, faction: int):
        for pos, u in list(self.units.items()):
            if u['faction'] == faction and u['ready']:
                self.heal_unit_percentage(pos)

    def _find_best_intermediate_step(self, sr: int, sc: int, dr: int, dc: int, faction: int):
        best_mid = None
        best_val = -1.00
        for mr, mc in [(sr + 1, sc), (sr - 1, sc), (sr, sc + 1), (sr, sc - 1)]:
            if abs(mr - dr) + abs(mc - dc) == 1 and self._can_pass_tile(mr, mc, faction):
                unclaimed_bonus = 2.00 if int(self.territory[mr, mc]) != faction else 0.00
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

        enemy_faction = 2 if faction == 1 else 1
        enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        tr, tc = target_pos
        dealt = 0.00
        kill_bonus = 0.00

        if (tr, tc) in self.units and self.units[(tr, tc)]['faction'] == enemy_faction:
            target_u = self.units[(tr, tc)]
            target_u['hp'] = round(max(0.00, target_u['hp'] - dmg), 2)
            dealt = dmg
            if target_u['hp'] <= 0.00:
                dead_u = self.units.pop((tr, tc))
                if dead_u['efficiency'] < 2.00:
                    dead_u['efficiency'] = round(dead_u['efficiency'] - 2.50, 2)
                self.archived_units.append(dead_u)
                kill_bonus = 2.50
                self._fp_dirty = True
        elif int(self.struct_owner[tr, tc]) == enemy_faction and int(self.struct_type[tr, tc]) != 0:
            self.struct_hp[tr, tc] = round(max(0.00, float(self.struct_hp[tr, tc]) - dmg), 2)
            dealt = dmg
            if self.struct_hp[tr, tc] <= 0.00:
                was_tower = (int(self.struct_type[tr, tc]) == 1)
                self.struct_owner[tr, tc] = 0
                self.struct_type[tr, tc] = 0
                kill_bonus = 2.00
                if was_tower:
                    self._fp_dirty = True
                    self._cached_base_path_len = self._shortest_path_length(self.p1_castle_pos, self.p2_castle_pos)
        elif (tr, tc) == enemy_castle:
            if faction == 1:
                self.p2_castle_hp = round(max(0.00, self.p2_castle_hp - dmg), 2)
            else:
                self.p1_castle_hp = round(max(0.00, self.p1_castle_hp - dmg), 2)
            dealt = dmg
            kill_bonus = 1.50

        if attacker_u is not None and dealt > 0.00:
            attacker_u['efficiency'] = round(attacker_u['efficiency'] + dealt * 1.15 + kill_bonus, 2)

        return float(dealt)

    def execute_worker_harvest(self, pos: tuple) -> float:
        if pos not in self.units:
            return 0.00
        u = self.units[pos]
        if u['type'] != 'worker' or not u['ready']:
            return 0.00
        faction = u['faction']
        r, c = pos
        eff_val = self.get_effective_tile_yield(r, c)
        if eff_val <= 0.00:
            return 0.00

        eff_cap = self._capture_tile(r, c, faction)
        bonus = round(eff_val * 1.50, 2)
        if faction == 1:
            self.p1_gold = round(min(MAX_GOLD, self.p1_gold + bonus), 2)
        else:
            self.p2_gold = round(min(MAX_GOLD, self.p2_gold + bonus), 2)
        u['ready'] = False
        u['efficiency'] = round(u['efficiency'] + eff_cap + bonus * 2.50, 2)
        return float(bonus)

    def _is_logistically_safe_to_build(self, r: int, c: int, faction: int) -> bool:
        if float(self.chokepoint_map[r, c]) >= 1.00:
            return False

        my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos
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

        new_len = self._shortest_path_length(self.p1_castle_pos, self.p2_castle_pos, block_pos=(r, c))
        if new_len >= 999 or (new_len - self._cached_base_path_len) > 4:
            return False
        return True

    def _find_adjacent_tower_site(self, pos: tuple, faction: int):
        r, c = pos
        enemy_faction = 2 if faction == 1 else 1
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if int(self.territory[nr, nc]) != enemy_faction and int(self.terrain[nr, nc]) != 0:
                    if (nr, nc) not in [self.p1_castle_pos, self.p2_castle_pos]:
                        if int(self.struct_type[nr, nc]) == 0 and (nr, nc) not in self.units:
                            if self._is_logistically_safe_to_build(nr, nc, faction):
                                candidates.append((int(nr), int(nc)))
        if not candidates:
            return None
        enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        return min(
            candidates,
            key=lambda p: (
                (abs(p[0] - enemy_castle[0]) + abs(p[1] - enemy_castle[1]))
                + float(self.tile_yields[p[0], p[1]]) * 12.00
                - (1.50 if float(self.chokepoint_map[p[0], p[1]]) == 0.40 else 0.00)
            )
        )

    def execute_worker_build(self, pos: tuple, build_type: int) -> bool:
        if pos not in self.units:
            return False
        u = self.units[pos]
        if u['type'] != 'worker' or not u['ready']:
            return False
        faction = u['faction']
        cost = STRUCT_STATS[build_type]['cost']
        gold = self.p1_gold if faction == 1 else self.p2_gold
        if gold < cost:
            return False

        r, c = pos
        target_site = None
        if build_type == 2:
            if (r, c) not in [self.p1_castle_pos, self.p2_castle_pos] and int(self.struct_type[r, c]) == 0:
                target_site = (r, c)
            else:
                target_site = self._find_adjacent_tower_site(pos, faction)
        else:
            target_site = self._find_adjacent_tower_site(pos, faction)

        if target_site is None:
            return False

        sr, sc = target_site
        if faction == 1:
            self.p1_gold = round(self.p1_gold - cost, 2)
        else:
            self.p2_gold = round(self.p2_gold - cost, 2)

        self.territory[sr, sc] = faction
        self.struct_owner[sr, sc] = faction
        self.struct_type[sr, sc] = build_type
        self.struct_hp[sr, sc] = STRUCT_STATS[build_type]['max_hp']
        self.tower_ready[sr, sc] = False
        u['ready'] = False
        u['efficiency'] = round(u['efficiency'] + 3.00, 2)

        if build_type == 1:
            self._fp_dirty = True
            self._cached_base_path_len = self._shortest_path_length(self.p1_castle_pos, self.p2_castle_pos)
        return True

    def execute_structure_spawn(self, struct_pos: tuple, faction: int, unit_type: str) -> bool:
        if struct_pos in self.units:
            return False

        r, c = struct_pos
        my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos
        is_my_castle = (struct_pos == my_castle)
        is_my_barracks = (int(self.struct_owner[r, c]) == faction and int(self.struct_type[r, c]) == 2)
        if not (is_my_castle or is_my_barracks):
            return False

        cost = UNIT_STATS[unit_type]['cost']
        gold = self.p1_gold if faction == 1 else self.p2_gold
        if gold < cost:
            return False

        if faction == 1:
            self.p1_gold = round(self.p1_gold - cost, 2)
        else:
            self.p2_gold = round(self.p2_gold - cost, 2)

        self.units[(int(r), int(c))] = self._create_unit_dict(faction, unit_type, ready=False)
        self._capture_tile(r, c, faction)
        self._fp_dirty = True
        return True

    def _compute_firepower_maps(self, faction: int):
        if self._fp_dirty:
            p1_fp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            p2_fp = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

            for (ur, uc), u in self.units.items():
                rng = UNIT_STATS[u['type']]['atk_range']
                dmg = UNIT_STATS[u['type']]['dmg']
                target_map = p1_fp if u['faction'] == 1 else p2_fp
                for dr, dc in RELATIVE_OFFSETS:
                    if abs(dr) + abs(dc) <= rng:
                        nr, nc = ur + dr, uc + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            target_map[nr, nc] += dmg

            tower_coords = np.argwhere(self.struct_type == 1)
            for tr, tc in tower_coords:
                tr_i, tc_i = int(tr), int(tc)
                target_map = p1_fp if int(self.struct_owner[tr_i, tc_i]) == 1 else p2_fp
                for dr, dc in RELATIVE_OFFSETS:
                    if abs(dr) + abs(dc) <= 2:
                        nr, nc = tr_i + dr, tc_i + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            target_map[nr, nc] += STRUCT_STATS[1]['dmg']

            self._cached_p1_fp = p1_fp
            self._cached_p2_fp = p2_fp
            self._fp_dirty = False

        if faction == 1:
            return self._cached_p1_fp, self._cached_p2_fp
        return self._cached_p2_fp, self._cached_p1_fp

    def _assess_strategic_situation(self, faction: int) -> dict:
        enemy_faction = 2 if faction == 1 else 1
        my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos

        enemy_combat_positions = []
        my_combat_positions = []
        closest_castle_dist = 999

        for pos, u in self.units.items():
            if u['faction'] == enemy_faction and u['type'] != 'worker':
                enemy_combat_positions.append(pos)
                d = abs(pos[0] - my_castle[0]) + abs(pos[1] - my_castle[1])
                if d < closest_castle_dist:
                    closest_castle_dist = d
            elif u['faction'] == faction and u['type'] != 'worker':
                my_combat_positions.append(pos)

        min_inter_unit_dist = 999
        for mr, mc in my_combat_positions:
            for er, ec in enemy_combat_positions:
                dist = abs(mr - er) + abs(mc - ec)
                if dist < min_inter_unit_dist:
                    min_inter_unit_dist = dist

        first_contact = (min_inter_unit_dist <= 7 or closest_castle_dist <= 11)
        under_pressure = (closest_castle_dist <= 10 or (first_contact and len(enemy_combat_positions) > len(my_combat_positions)))

        return {
            'first_contact': bool(first_contact),
            'under_pressure': bool(under_pressure),
            'min_contact_dist': int(min_inter_unit_dist),
            'closest_enemy_dist': int(closest_castle_dist),
            'my_combat_count': len(my_combat_positions),
            'enemy_combat_count': len(enemy_combat_positions)
        }

    def _compute_fertility_attraction(self, nr: int, nc: int, faction: int) -> float:
        eff_yields = self.get_effective_yield_map()
        rich_mask = (self.territory != faction) & (eff_yields >= 0.15)
        if not np.any(rich_mask):
            rich_mask = (self.territory != faction) & (self.terrain != 0)
            if not np.any(rich_mask):
                return 0.00

        dists = np.abs(self._grid_r[rich_mask] - nr) + np.abs(self._grid_c[rich_mask] - nc)
        close_mask = (dists <= 9)
        if not np.any(close_mask):
            dists_min = float(np.min(dists))
            return 0.50 / max(1.00, dists_min)

        vals = eff_yields[rich_mask][close_mask]
        d_close = np.maximum(1.00, dists[close_mask].astype(np.float32))
        weights = np.where(vals >= 0.30, vals * 3.20, vals * 1.80)
        return float(np.sum(weights / d_close))

    def _compute_allied_clump_penalty(self, current_pos: tuple, nr: int, nc: int, faction: int) -> float:
        clump = 0.00
        for pos, u in self.units.items():
            if u['faction'] == faction and pos != current_pos:
                dist = abs(pos[0] - nr) + abs(pos[1] - nc)
                if dist <= 4:
                    clump += (5.00 - float(dist)) / 4.00
        return clump

    def get_valid_action_mask_for_entity(self, entity_kind: str, pos: tuple, faction: int) -> np.ndarray:
        mask = np.zeros(30, dtype=bool)
        r, c = pos
        gold = self.p1_gold if faction == 1 else self.p2_gold

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

            if u['type'] == 'worker':
                if int(self.terrain[r, c]) != 0 and self.get_effective_tile_yield(r, c) > 0.00:
                    mask[25] = True
                if gold >= STRUCT_STATS[1]['cost'] and self._find_adjacent_tower_site(pos, faction) is not None:
                    mask[26] = True
                can_barracks_here = (
                    pos not in [self.p1_castle_pos, self.p2_castle_pos]
                    and int(self.struct_type[r, c]) == 0
                )
                if gold >= STRUCT_STATS[2]['cost'] and (can_barracks_here or self._find_adjacent_tower_site(pos, faction) is not None):
                    mask[27] = True

        elif entity_kind == 'tower':
            valid_targets = set(self.get_valid_attack_targets(pos, faction=faction, atk_range=2))
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
                    mask[29] = True

        return mask

    def _is_recruitment_safe(self, faction: int, new_upkeep: float, unit_cost: float, first_contact: bool) -> bool:
        gold = self.p1_gold if faction == 1 else self.p2_gold
        remaining_gold = gold - unit_cost
        if remaining_gold < 0.00:
            return False

        gross_inc, current_upkeep, net_inc, _, unit_cnt = self.get_faction_economy_stats(faction)
        extra_tax = SUPPLY_TAX_PER_UNIT if unit_cnt >= SUPPLY_LIMIT else 0.00
        projected_net = net_inc - new_upkeep - extra_tax

        if not first_contact:
            if projected_net >= 0.60 and remaining_gold >= 5.00:
                return True
            return False

        if projected_net >= -0.10:
            return True
        return (remaining_gold + projected_net * 4.00) >= 3.00

    def get_strategic_teacher_action(self, entity_kind: str, pos: tuple, faction: int, tier: int = 5, precomputed_mask: np.ndarray = None) -> int:
        mask = precomputed_mask if precomputed_mask is not None else self.get_valid_action_mask_for_entity(entity_kind, pos, faction)
        valid_acts = np.where(mask)[0]
        if len(valid_acts) == 1:
            return int(valid_acts[0])

        comp = self.f1_Target_competence if faction == 1 else self.f2_target_competence
        if comp < 1.00 and random.random() > comp:
            move_or_atk = [a for a in valid_acts if a < 24]
            if move_or_atk:
                return int(random.choice(move_or_atk))

        enemy_faction = 2 if faction == 1 else 1
        my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos
        enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        gold = self.p1_gold if faction == 1 else self.p2_gold
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

        ally_fp, enemy_fp = self._compute_firepower_maps(faction)
        sit = self._assess_strategic_situation(faction)
        gross_inc, upkeep_inc, net_inc, my_tiles_cnt, my_unit_cnt = self.get_faction_economy_stats(faction)

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

            my_workers = sum(1 for u in self.units.values() if u['faction'] == faction and u['type'] == 'worker')
            my_warriors = sum(1 for u in self.units.values() if u['faction'] == faction and u['type'] == 'warrior')
            my_archers = sum(1 for u in self.units.values() if u['faction'] == faction and u['type'] == 'archer')
            my_combat = my_warriors + my_archers
            my_barracks = int(np.sum((self.struct_owner == faction) & (self.struct_type == 2)))

            max_combat_cap = max(3, int(round(comp * 16.00)))
            if my_combat >= max_combat_cap:
                return 24

            desired_workers = 3 if comp >= 1.00 else 2
            if my_workers < desired_workers and net_inc >= 0.60 and mask[27]:
                if self._is_recruitment_safe(faction, UNIT_STATS['worker']['upkeep'], UNIT_STATS['worker']['cost'], sit['first_contact']):
                    return 27

            if not sit['first_contact']:
                if net_inc >= 3.00 and gold >= 16.00 and mask[28]:
                    if self._is_recruitment_safe(faction, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], False):
                        return 28
                return 24

            if tier >= 3 and my_barracks == 0 and my_workers >= 1 and net_inc >= 2.20 and not sit['under_pressure']:
                if gold < (STRUCT_STATS[2]['cost'] + UNIT_STATS['warrior']['cost']):
                    return 24

            want_more_troops = (my_combat <= sit['enemy_combat_count'] + int(round(comp * 5.00))) or sit['under_pressure'] or (net_inc >= 3.50 and gold >= 18.00)
            if want_more_troops:
                if my_warriors > my_archers and mask[29] and self._is_recruitment_safe(faction, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], True):
                    return 29
                if mask[28] and self._is_recruitment_safe(faction, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], True):
                    return 28
                if mask[29] and self._is_recruitment_safe(faction, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], True):
                    return 29

            return 24

        u = self.units[pos]
        u_type = u['type']

        atk_acts = [a for a in valid_acts if 12 <= a <= 23]
        if atk_acts:
            best_atk = atk_acts[0]
            best_atk_score = -999.00
            for a in atk_acts:
                dr, dc = RELATIVE_OFFSETS[a - 12]
                tr, tc = r + dr, c + dc
                score = 15.00
                if (tr, tc) == enemy_castle:
                    score += 45.00
                elif (tr, tc) in self.units:
                    tu = self.units[(tr, tc)]
                    if tu['hp'] <= UNIT_STATS[u_type]['dmg']:
                        score += 35.00
                    score += UNIT_STATS[tu['type']]['cost'] * 2.50
                elif int(self.struct_owner[tr, tc]) == enemy_faction:
                    score += 28.00 if int(self.struct_type[tr, tc]) == 2 else 22.00
                if score > best_atk_score:
                    best_atk_score = score
                    best_atk = a
            return int(best_atk)

        if u_type == 'worker':
            my_barracks = int(np.sum((self.struct_owner == faction) & (self.struct_type == 2)))
            my_towers = int(np.sum((self.struct_owner == faction) & (self.struct_type == 1)))
            dist_to_home = abs(r - my_castle[0]) + abs(c - my_castle[1])
            dist_to_enemy = abs(r - enemy_castle[0]) + abs(c - enemy_castle[1])

            can_afford_buildings = (sit['first_contact'] and net_inc >= 1.50 and gold >= 10.00) or (net_inc >= 3.80 and gold >= 16.00)
            if can_afford_buildings and tier >= 3:
                max_barracks_target = 1 if comp < 0.65 else (2 if comp < 0.95 else 3)
                if mask[27] and my_barracks < max_barracks_target and dist_to_home >= 5 and dist_to_enemy <= 22:
                    if float(self.tile_yields[r, c]) <= 0.05:
                        return 27
                if mask[26] and my_towers < 2 and sit['under_pressure'] and float(self.chokepoint_map[r, c]) >= 0.40:
                    if float(self.tile_yields[r, c]) <= 0.05:
                        return 26

        best_move_act = 24
        best_move_score = -999.00

        for a in valid_acts:
            if 0 <= a <= 11:
                dr, dc = RELATIVE_OFFSETS[a]
                nr, nc = r + dr, c + dc
                step_dist = abs(dr) + abs(dc)

                v_double = 0.00
                tiles_captured = 0.00
                if int(self.territory[nr, nc]) != faction:
                    v_double += self.get_effective_tile_yield(nr, nc)
                    tiles_captured += (1.35 if int(self.territory[nr, nc]) == enemy_faction else 1.00)

                if step_dist == 2:
                    mid = self._find_best_intermediate_step(r, c, nr, nc, faction)
                    if mid is not None and int(self.territory[mid[0], mid[1]]) != faction:
                        v_double += self.get_effective_tile_yield(mid[0], mid[1])
                        tiles_captured += (1.35 if int(self.territory[mid[0], mid[1]]) == enemy_faction else 1.00)

                p_fertility = self._compute_fertility_attraction(nr, nc, faction)
                d_clump = self._compute_allied_clump_penalty(pos, nr, nc, faction) if not sit['first_contact'] else 0.00

                score = (v_double * 32.00) + (p_fertility * 22.00) + (tiles_captured * 4.50) - (d_clump * 6.00)

                if u_type != 'worker' and sit['first_contact']:
                    enemy_positions = [pos_e for pos_e, ue in self.units.items() if ue['faction'] == enemy_faction]
                    if enemy_positions:
                        closest_e = min(enemy_positions, key=lambda p: abs(p[0] - nr) + abs(p[1] - nc))
                        dist_e = abs(closest_e[0] - nr) + abs(closest_e[1] - nc)
                        if u_type == 'warrior':
                            score += max(0.00, 18.00 - dist_e * 1.40)
                        elif u_type == 'archer':
                            score += 13.00 if dist_e == 2 else max(0.00, 10.00 - abs(dist_e - 2) * 1.60)
                    score += (ally_fp[nr, nc] - enemy_fp[nr, nc]) * 1.60
                elif u_type == 'worker':
                    score -= enemy_fp[nr, nc] * 12.00

                if self.is_tile_built_up(r, c):
                    score += 12.00

                if score > best_move_score:
                    best_move_score = score
                    best_move_act = a

        if u_type == 'worker' and mask[25]:
            eff_here = self.get_effective_tile_yield(r, c)
            if eff_here >= 0.35 and best_move_score < 18.00 and enemy_fp[r, c] == 0.00:
                return 25

        if best_move_score > -50.00:
            return int(best_move_act)
        return 25 if (u_type == 'worker' and mask[25]) else 24

    def apply_entity_action(self, entity_kind: str, pos: tuple, faction: int, action: int, precomputed_mask: np.ndarray = None) -> float:
        mask = precomputed_mask if precomputed_mask is not None else self.get_valid_action_mask_for_entity(entity_kind, pos, faction)
        if not mask[action]:
            valid_indices = np.where(mask)[0]
            action = int(valid_indices[0]) if len(valid_indices) > 0 else 24

        if self.record_telemetry:
            self.record_entity_decision(entity_kind, pos, faction, action)

        r, c = pos
        enemy_faction = 2 if faction == 1 else 1
        reward = 0.00

        if entity_kind == 'unit':
            if pos not in self.units:
                return 0.00
            u_name = UNIT_STATS[self.units[pos]['type']]['name']
            if 0 <= action <= 11:
                dr, dc = RELATIVE_OFFSETS[action]
                dest = (r + dr, c + dc)
                old_owner = int(self.territory[dest[0], dest[1]])
                tile_val = self.get_effective_tile_yield(dest[0], dest[1])
                if self.execute_unit_move(pos, dest):
                    if faction == 1:
                        self.last_ai_turn_traces.append({"kind": "move", "from": [int(r), int(c)], "to": [int(dest[0]), int(dest[1])]})
                    if old_owner == 0:
                        reward += 0.10 + tile_val * 0.80
                    elif old_owner == enemy_faction:
                        reward += 0.16 + tile_val * 0.90
            elif 12 <= action <= 23:
                dr, dc = RELATIVE_OFFSETS[action - 12]
                target = (r + dr, c + dc)
                dmg = self.execute_attack_on_target(pos, target, is_tower=False)
                reward += dmg * 0.12
                if dmg > 0.00 and faction == 1:
                    self.last_ai_turn_traces.append({"kind": "attack", "from": [int(r), int(c)], "to": [int(target[0]), int(target[1])]})
                    self.add_event_log(f"ШІ: {u_name} атакував ціль ({target[0]},{target[1]}) на -{dmg:.2f} HP")
            elif action == 24:
                healed = self.heal_unit_percentage(pos)
                reward += healed * 0.04
            elif action == 25:
                bonus = self.execute_worker_harvest(pos)
                reward += bonus * 0.20
                if bonus > 0.00 and faction == 1:
                    self.add_event_log(f"ШІ: Робітник обробив родовище (+{bonus:.2f}$)")
            elif action == 26:
                if self.execute_worker_build(pos, build_type=1):
                    reward += 0.22
                    if faction == 1:
                        self.add_event_log("ШІ звів Оборонну Вежу")
            elif action == 27:
                if self.execute_worker_build(pos, build_type=2):
                    reward += 0.30
                    if faction == 1:
                        self.add_event_log("ШІ звів Передову Казарму")

        elif entity_kind == 'tower':
            if 12 <= action <= 23:
                dr, dc = RELATIVE_OFFSETS[action - 12]
                target = (r + dr, c + dc)
                dmg = self.execute_attack_on_target(pos, target, is_tower=True)
                reward += dmg * 0.10
                if dmg > 0.00 and faction == 1:
                    self.last_ai_turn_traces.append({"kind": "attack", "from": [int(r), int(c)], "to": [int(target[0]), int(target[1])]})
                    self.add_event_log(f"ШІ: Вежа обстріляла ({target[0]},{target[1]}) на -{dmg:.2f} HP")
            else:
                self.tower_ready[r, c] = False

        elif entity_kind == 'spawn':
            if action == 27:
                if self.execute_structure_spawn(pos, faction, 'worker'):
                    reward += 0.08
                    if faction == 1:
                        self.add_event_log("ШІ найняв Робітника")
            elif action == 28:
                if self.execute_structure_spawn(pos, faction, 'warrior'):
                    reward += 0.12
                    if faction == 1:
                        self.add_event_log("ШІ найняв Піхотинця")
            elif action == 29:
                if self.execute_structure_spawn(pos, faction, 'archer'):
                    reward += 0.14
                    if faction == 1:
                        self.add_event_log("ШІ найняв Стрільця")

        return float(reward)

    def execute_full_ai_turn(self, faction: int, model: PPO = None, deterministic: bool = False):
        if faction == 1:
            self.last_ai_turn_traces = []
        self._prepare_faction_turn(faction=faction, collect_economy=True)
        temp = max(0.15, 1.80 - (self.active_gen_idx / 100.00) * 1.60) if faction == 1 else 1.00

        unit_positions = [pos for pos, u in sorted(self.units.items()) if u['faction'] == faction and u['ready']]
        for pos in unit_positions:
            if pos not in self.units or not self.units[pos]['ready']:
                continue
            mask = self.get_valid_action_mask_for_entity('unit', pos, faction)
            if model is not None:
                obs = self._get_obs_for_entity('unit', pos, faction)
                act = predict_masked_action(model, obs, mask, deterministic=deterministic, temperature=temp)
                # Якщо сира модель пропонує пасивно стояти (24), коли юніт повністю здоровий і є куди рухатись — підстраховуємо вчителем
                if act == 24 and self.units[pos]['hp'] >= self.units[pos]['max_hp'] and np.any(mask[:24]):
                    act = self.get_strategic_teacher_action('unit', pos, faction, tier=self.curriculum_tier, precomputed_mask=mask)
            else:
                act = self.get_strategic_teacher_action('unit', pos, faction, tier=self.curriculum_tier, precomputed_mask=mask)
            self.apply_entity_action('unit', pos, faction, act, precomputed_mask=mask)

        tower_coords = np.argwhere((self.struct_owner == faction) & (self.struct_type == 1) & self.tower_ready)
        for tr, tc in tower_coords:
            t_pos = (int(tr), int(tc))
            if len(self.get_valid_attack_targets(t_pos, faction=faction, atk_range=2)) > 0:
                mask = self.get_valid_action_mask_for_entity('tower', t_pos, faction)
                if model is not None:
                    obs = self._get_obs_for_entity('tower', t_pos, faction)
                    act = predict_masked_action(model, obs, mask, deterministic=deterministic, temperature=temp)
                else:
                    act = self.get_strategic_teacher_action('tower', t_pos, faction, tier=self.curriculum_tier, precomputed_mask=mask)
                self.apply_entity_action('tower', t_pos, faction, act, precomputed_mask=mask)

        gold = self.p1_gold if faction == 1 else self.p2_gold
        if gold >= UNIT_STATS['worker']['cost']:
            my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos
            enemy_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
            spawn_sites = [my_castle]
            barracks_coords = np.argwhere((self.struct_owner == faction) & (self.struct_type == 2))
            for br, bc in barracks_coords:
                spawn_sites.append((int(br), int(bc)))
            spawn_sites.sort(key=lambda s: abs(s[0] - enemy_castle[0]) + abs(s[1] - enemy_castle[1]))

            for site in spawn_sites:
                curr_gold = self.p1_gold if faction == 1 else self.p2_gold
                if site not in self.units and curr_gold >= UNIT_STATS['worker']['cost']:
                    mask = self.get_valid_action_mask_for_entity('spawn', site, faction)
                    if model is not None:
                        obs = self._get_obs_for_entity('spawn', site, faction)
                        act = predict_masked_action(model, obs, mask, deterministic=deterministic, temperature=temp)
                    else:
                        act = self.get_strategic_teacher_action('spawn', site, faction, tier=self.curriculum_tier, precomputed_mask=mask)
                    self.apply_entity_action('spawn', site, faction, act, precomputed_mask=mask)

        self.heal_all_unmoved_units(faction)
        if faction == 1:
            moves_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "move")
            atks_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "attack")
            self.add_event_log(f"Хід {self.turn_count}: ШІ виконав {moves_cnt} переміщень та {atks_cnt} атак.")

    def _get_obs_for_entity(self, entity_kind: str, pos: tuple, faction: int) -> np.ndarray:
        enemy_faction = 2 if faction == 1 else 1
        obs = np.zeros((18, self.grid_size, self.grid_size), dtype=np.float32)

        obs[0, pos[0], pos[1]] = 1.00
        obs[1] = (self.terrain == 0).astype(np.float32)
        obs[2] = self.get_effective_yield_map()
        obs[3] = self.chokepoint_map
        obs[4] = (self.territory == faction).astype(np.float32)
        obs[5] = (self.territory == enemy_faction).astype(np.float32)
        obs[6] = (self.territory == 0).astype(np.float32)

        my_castle = self.p1_castle_pos if faction == 1 else self.p2_castle_pos
        opp_castle = self.p2_castle_pos if faction == 1 else self.p1_castle_pos
        my_c_hp = self.p1_castle_hp if faction == 1 else self.p2_castle_hp
        opp_c_hp = self.p2_castle_hp if faction == 1 else self.p1_castle_hp

        obs[7] = np.where(self.struct_owner == faction, self.struct_hp / 10.00, 0.00)
        obs[7, my_castle[0], my_castle[1]] = my_c_hp / CASTLE_MAX_HP

        obs[8] = np.where(self.struct_owner == enemy_faction, self.struct_hp / 10.00, 0.00)
        obs[8, opp_castle[0], opp_castle[1]] = opp_c_hp / CASTLE_MAX_HP

        type_to_ch = {'worker': 0, 'warrior': 1, 'archer': 2}
        for (ur, uc), u in self.units.items():
            ch_off = type_to_ch[u['type']]
            norm_hp = u['hp'] / UNIT_STATS[u['type']]['max_hp']
            if u['faction'] == faction:
                obs[9 + ch_off, ur, uc] = norm_hp
            else:
                obs[12 + ch_off, ur, uc] = norm_hp

        ally_fp, enemy_fp = self._compute_firepower_maps(faction)
        obs[15] = np.clip(ally_fp / 10.00, 0.00, 1.00)
        obs[16] = np.clip(enemy_fp / 10.00, 0.00, 1.00)

        my_gold = self.p1_gold if faction == 1 else self.p2_gold
        obs[17, :, :] = min(1.00, my_gold / MAX_GOLD)

        if faction == 2:
            obs = obs[:, ::-1, ::-1].copy()
        return obs

    def _get_obs(self):
        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue[0]
            return self._get_obs_for_entity(kind, pos, faction=1)
        return self._get_obs_for_entity('spawn', self.p1_castle_pos, faction=1)

    def get_current_p1_entity_and_mask(self):
        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue[0]
            return kind, pos, self.get_valid_action_mask_for_entity(kind, pos, faction=1)
        return 'spawn', self.p1_castle_pos, self.get_valid_action_mask_for_entity('spawn', self.p1_castle_pos, faction=1)

    def evaluate_player_advantage(self) -> float:
        hp_sum = max(0.01, self.p1_castle_hp + self.p2_castle_hp)
        r_castle = self.p2_castle_hp / hp_sum

        inc1, _, _, t1, _ = self.get_faction_economy_stats(1)
        inc2, _, _, t2, _ = self.get_faction_economy_stats(2)
        t_sum = max(1.00, float(t1 + t2))
        r_terr = float(t2) / t_sum

        p1_army = self.p1_gold + sum(u['hp'] * 1.50 for u in self.units.values() if u['faction'] == 1) + inc1 * 4.00
        p2_army = self.p2_gold + sum(u['hp'] * 1.50 for u in self.units.values() if u['faction'] == 2) + inc2 * 4.00
        r_power = p2_army / max(0.01, p1_army + p2_army)

        adv = (0.35 * r_castle) + (0.30 * r_terr) + (0.35 * r_power)
        return float(np.clip(adv, 0.00, 1.00))

    def compute_surge_dda_step(self, current_rating: float, is_early_turn: bool = False) -> float:
        adv = self.evaluate_player_advantage()
        e_adv = adv - 0.50
        if abs(e_adv) <= 0.03:
            return round(float(current_rating), 2)
        m_surge = 1.00 + (((max(0.00, abs(e_adv) - 0.04)) / 0.12) ** 2.00) * 3.20
        k_micro = 24.00 if is_early_turn else 18.00
        delta_r = round(e_adv * k_micro * m_surge, 2)
        max_pool_r = PoolRegistry.get_max_rating()
        new_r = max(10.00, min(max(100.00, max_pool_r), float(current_rating) + delta_r))
        return round(new_r, 2)

    def step(self, action):
        self.step_count += 1
        reward = -0.002

        if not self.p1_entity_queue:
            self._prepare_faction_turn(faction=1, collect_economy=True)

        if self.p1_entity_queue:
            kind, pos = self.p1_entity_queue.popleft()
            reward += self.apply_entity_action(kind, pos, faction=1, action=int(action))

        if self.p2_castle_hp <= 0.00:
            return self._get_obs(), float(reward + 2.00), True, False, {"winner": "p1"}

        if not self.p1_entity_queue:
            self.heal_all_unmoved_units(faction=1)

            if self.mode != "manual_p2":
                self.execute_full_ai_turn(faction=2, model=self.opponent_model, deterministic=False)

            if self.p1_castle_hp <= 0.00:
                return self._get_obs(), float(reward - 2.00), True, False, {"winner": "p2"}

            self.turn_count += 1
            self.record_turn_efficiency_snapshot()
            if self.turn_count > MAX_STEPS:
                inc1, _, _, t1, u1 = self.get_faction_economy_stats(1)
                inc2, _, _, t2, u2 = self.get_faction_economy_stats(2)
                s1 = self.p1_castle_hp + t1 * 0.25 + inc1 * 4.00 + u1 * 1.50
                s2 = self.p2_castle_hp + t2 * 0.25 + inc2 * 4.00 + u2 * 1.50
                winner = "p1" if s1 >= s2 else "p2"
                term_rew = 1.00 if winner == "p1" else -1.00
                return self._get_obs(), float(reward + term_rew), True, False, {"winner": winner}

            self._prepare_faction_turn(faction=1, collect_economy=True)

        return self._get_obs(), float(reward), False, False, {}

class OnlineEvolutionEngine:
    @staticmethod
    def evolve_and_benchmark_from_match(
        match_obs: list,
        match_acts: list,
        match_weights: list,
        username: str = "Player"
    ) -> dict:
        os.makedirs(POOL_DIR, exist_ok=True)
        PoolRegistry.ensure_initialized()

        dummy_env = MicroRTSEnv(mode="selfplay", record_telemetry=False)
        latest_path = os.path.join(POOL_DIR, "latest_checkpoint.zip")

        model = None
        if os.path.exists(latest_path):
            try:
                model = PPO.load(latest_path, env=dummy_env, device="cpu", verbose=0)
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

        if len(obs_list) > 0:
            obs_t = torch.as_tensor(np.array(obs_list, dtype=np.float32), device=model.device)
            acts_t = torch.as_tensor(np.array(acts_list, dtype=np.int64), device=model.device)
            weights_t = torch.as_tensor(np.array(weights_list, dtype=np.float32), device=model.device)

            optimizer = optim.Adam(model.policy.parameters(), lr=0.0008)
            ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

            epochs = 6
            batch_sz = min(128, len(obs_t))
            for _ in range(epochs):
                perm = torch.randperm(len(obs_t))
                for start_i in range(0, len(obs_t), batch_sz):
                    b_idx = perm[start_i:start_i + batch_sz]
                    dist = model.policy.get_distribution(obs_t[b_idx])
                    logits = dist.distribution.logits
                    ce = ce_loss_fn(logits, acts_t[b_idx])
                    entropy = dist.entropy().mean()
                    loss = torch.mean(ce * weights_t[b_idx]) - (0.02 * entropy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        all_gens = PoolRegistry.get_all_generations()
        max_curr_rating = max(100.00, max(float(g["rating"]) for g in all_gens))
        test_ratings = [20.00, 45.00, 70.00, 90.00, 100.00, max_curr_rating]

        sim_env = MicroRTSEnv(mode="manual_p2", record_telemetry=False)
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
            origin=f"player:{username}"
        )
        entry["w_gauntlet"] = round(w_gauntlet * 100.00, 2)
        return entry

@staticmethod
def enforce_pool_storage_limit(max_human_models: int = 25):
    """Видаляє найслабші та найстаріші моделі при досягненні ліміту."""
    PoolRegistry.ensure_initialized()
    with REGISTRY_LOCK:
        with open(POOL_REGISTRY_FILE, "r", encoding="utf-8") as f:
            reg = json.load(f)
        
        human_gens = [g for g in reg["generations"] if str(g.get("origin", "")).startswith("player:")]
        if len(human_gens) > max_human_models:
            # Сортуємо: кандидати на видалення — з низьким рейтингом і найстаріші
            sorted_by_priority = sorted(
                human_gens, 
                key=lambda g: (float(g.get("rating", 0.0)), int(g.get("created_at", 0)))
            )
            to_remove = sorted_by_priority[0]
            
            # Фізично видаляємо файл з диска
            f_path = os.path.join(POOL_DIR, to_remove["filename"])
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception:
                    pass
            
            # Видаляємо запис із реєстру
            reg["generations"] = [g for g in reg["generations"] if g["gen_id"] != to_remove["gen_id"]]
            PoolRegistry._save_unlocked(reg)