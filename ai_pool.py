import os
import json
import time
import copy
import threading
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

from game_config import (
    POOL_DIR, HUMAN_TELEMETRY_FILE, POOL_REGISTRY_FILE,
    RELATIVE_OFFSETS, ROLE_IDS
)

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

def infer_role_id_from_obs(obs_arr: np.ndarray, act_val: int) -> int:
    pos_plane = obs_arr[0]
    coords = np.argwhere(pos_plane > 0.50)
    if len(coords) == 0:
        return ROLE_IDS["worker"]
    r, c = int(coords[0][0]), int(coords[0][1])
    if obs_arr[10, r, c] > 0.01:
        return ROLE_IDS["warrior"]
    if obs_arr[11, r, c] > 0.01:
        return ROLE_IDS["archer"]
    if obs_arr[9, r, c] > 0.01:
        return ROLE_IDS["worker"]
    if obs_arr[7, r, c] > 0.01 and act_val in (27, 28, 29, 24):
        return ROLE_IDS["spawn"]
    return ROLE_IDS["tower"]

def is_action_valid_for_role(role_id: int, act_val: int) -> bool:
    if role_id in (ROLE_IDS["warrior"], ROLE_IDS["archer"]):
        return 0 <= act_val <= 24 or act_val == 28
    if role_id == ROLE_IDS["worker"]:
        return 0 <= act_val <= 29
    if role_id == ROLE_IDS["tower"]:
        return 12 <= act_val <= 24
    if role_id == ROLE_IDS["spawn"]:
        return act_val in (24, 27, 28, 29)
    return True

def build_role_action_mask_tensor(roles_t: torch.Tensor, device) -> torch.Tensor:
    bsz = roles_t.shape[0]
    mask = torch.ones((bsz, 30), dtype=torch.bool, device=device)
    for i in range(bsz):
        rid = int(roles_t[i].item())
        if rid in (ROLE_IDS["warrior"], ROLE_IDS["archer"]):
            mask[i, 25:28] = False
            mask[i, 29] = False
        elif rid == ROLE_IDS["tower"]:
            mask[i, 0:12] = False
            mask[i, 25:30] = False
        elif rid == ROLE_IDS["spawn"]:
            mask[i, 0:24] = False
            mask[i, 25:27] = False
    return mask

def balance_samples_by_role(obs_list: list, acts_list: list, weights_list: list, roles_list: list, max_per_role: int = 160):
    buckets = {}
    for o, a, w, r in zip(obs_list, acts_list, weights_list, roles_list):
        rid = int(r)
        if not is_action_valid_for_role(rid, int(a)):
            continue
        if rid not in buckets:
            buckets[rid] = []
        buckets[rid].append((o, int(a), float(w), rid))

    b_obs, b_acts, b_weights, b_roles = [], [], [], []
    for rid, items in buckets.items():
        items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
        selected = items_sorted[:max_per_role]
        role_boost = 1.45 if rid in (ROLE_IDS["warrior"], ROLE_IDS["archer"]) else (0.75 if rid == ROLE_IDS["worker"] else 1.15)
        for o, a, w, r_val in selected:
            b_obs.append(o)
            b_acts.append(a)
            b_weights.append(round(w * role_boost, 2))
            b_roles.append(r_val)

    return b_obs, b_acts, b_weights, b_roles

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

class OnlineEvolutionEngine:
    @staticmethod
    def _evaluate_candidate_vs_gen100(candidate_model: PPO, gen100_model: PPO) -> float:
        from microrts_env import MicroRTSEnv
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
        username: str = "Player",
        match_roles: list = None
    ) -> dict:
        from microrts_env import MicroRTSEnv
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
        if match_roles is not None and len(match_roles) == len(obs_list):
            roles_list = [int(r) for r in match_roles]
        else:
            roles_list = [infer_role_id_from_obs(o, int(a)) for o, a in zip(obs_list, acts_list)]

        if os.path.exists(HUMAN_TELEMETRY_FILE):
            try:
                data = np.load(HUMAN_TELEMETRY_FILE)
                if len(data['obs']) > 0:
                    take_n = min(256, len(data['obs']))
                    idx = np.random.choice(len(data['obs']), size=take_n, replace=False)
                    f_obs = data['obs'][idx]
                    f_acts = data['acts'][idx]
                    f_weights = data['weights'][idx]
                    if 'roles' in data and len(data['roles']) == len(data['obs']):
                        f_roles = data['roles'][idx]
                    else:
                        f_roles = [infer_role_id_from_obs(o, int(a)) for o, a in zip(f_obs, f_acts)]
                    obs_list.extend(f_obs)
                    acts_list.extend(f_acts)
                    weights_list.extend(f_weights)
                    roles_list.extend(f_roles)
            except Exception:
                pass

        obs_list, acts_list, weights_list, roles_list = balance_samples_by_role(
            obs_list, acts_list, weights_list, roles_list, max_per_role=160
        )

        doctrine_scores = [
            dummy_env.evaluate_sample_doctrine_compliance(o, int(a), role_id=int(r))
            for o, a, r in zip(obs_list, acts_list, roles_list)
        ] if len(obs_list) > 0 else [1.00]
        mean_doctrine_alignment = round(float(np.mean(doctrine_scores)) * 100.00, 2)

        if len(obs_list) > 0:
            obs_t = torch.as_tensor(np.array(obs_list, dtype=np.float32), device=model.device)
            acts_t = torch.as_tensor(np.array(acts_list, dtype=np.int64), device=model.device)
            weights_t = torch.as_tensor(np.array(weights_list, dtype=np.float32), device=model.device)
            roles_t = torch.as_tensor(np.array(roles_list, dtype=np.int64), device=model.device)
            doc_t = torch.as_tensor(np.array(doctrine_scores, dtype=np.float32), device=model.device)
            role_masks_t = build_role_action_mask_tensor(roles_t, model.device)

            if gen100_model is not None:
                with torch.no_grad():
                    g100_dist = gen100_model.policy.get_distribution(obs_t)
                    g100_logits = g100_dist.distribution.logits.clone()
                    g100_logits[~role_masks_t] = -1e8
                    g100_acts_t = torch.argmax(g100_logits, dim=-1)
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
                    logits = dist.distribution.logits.clone()
                    logits[~role_masks_t[b_idx]] = -1e8
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
                    logits = dist.distribution.logits.clone()
                    logits[~role_masks_t[b_idx]] = -1e8
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

        entry = PoolRegistry.register_new_generation(
            label=label,
            filename=filename,
            rating=r_assigned,
            origin=f"player:{username}",
            doctrine_alignment=mean_doctrine_alignment
        )
        entry["w_gauntlet"] = round(w_gauntlet * 100.00, 2)

        from cloud_backup import CloudBackupManager
        CloudBackupManager.backup_new_generation_async(entry, os.path.join(POOL_DIR, filename))

        return entry