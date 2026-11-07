import os
import json
import signal
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

from game_config import POOL_DIR, HUMAN_TELEMETRY_FILE, MIN_TRAINING_EFFICIENCY
from microrts_env import MicroRTSEnv, GLOBAL_MODEL_CACHE, predict_masked_action

STATE_FILE = os.path.join(POOL_DIR, "training_state.json")
LATEST_MODEL_PATH = os.path.join(POOL_DIR, "latest_checkpoint.zip")

TOTAL_GENERATIONS = 100
STEPS_PER_GEN = 20000
TOTAL_PIPELINE_STEPS = TOTAL_GENERATIONS * STEPS_PER_GEN

def get_tier_for_generation(gen_idx: int) -> tuple:
    if gen_idx <= 15:
        return 1, "Tier 1 (Gen 0-15): Basic movement and combat fundamentals"
    elif gen_idx <= 35:
        return 2, "Tier 2 (Gen 16-35): Early 18x18 resource cluster rush and expansion"
    elif gen_idx <= 55:
        return 3, "Tier 3 (Gen 36-55): Chokepoint control, forward barracks, and non-blocking towers"
    elif gen_idx <= 75:
        return 4, "Tier 4 (Gen 56-75): Firepower radar, treasury protection, and coordinated roles"
    else:
        return 5, "Tier 5 (Gen 76-100): Grandmaster tactics + Validated Elite Human Telemetry (>=100 eff)"

class TrainingController:
    def __init__(self):
        self.stop_requested = False
        self.status_text = "Initializing evolutionary training..."
        self.tier_text = ""
        self.progress_pct = 0.00

    def request_stop(self):
        self.stop_requested = True
        self.status_text = "Stop signal received! Safely saving neural network weights..."

    @staticmethod
    def load_state():
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"current_gen": 0, "steps_in_gen": 0, "total_steps_done": 0, "last_loss": 0.00}

    @staticmethod
    def save_state(current_gen, steps_in_gen, total_steps_done, last_loss=0.00):
        os.makedirs(POOL_DIR, exist_ok=True)
        data = {
            "current_gen": int(current_gen),
            "steps_in_gen": int(steps_in_gen),
            "total_steps_done": int(total_steps_done),
            "last_loss": round(float(last_loss), 4)
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

def train_on_human_telemetry(model: PPO, optimizer: optim.Optimizer, env: MicroRTSEnv) -> float:
    if not os.path.exists(HUMAN_TELEMETRY_FILE):
        return 0.00
    try:
        data = np.load(HUMAN_TELEMETRY_FILE)
        obs_arr = data['obs']
        acts_arr = data['acts']
        weights_arr = data['weights']
        if len(obs_arr) < 16:
            return 0.00

        batch_size = min(128, len(obs_arr))
        indices = np.random.choice(len(obs_arr), size=batch_size, replace=False)

        b_obs = obs_arr[indices]
        b_acts = acts_arr[indices]
        b_weights = weights_arr[indices]
        doc_scores = [env.evaluate_sample_doctrine_compliance(o, int(a)) for o, a in zip(b_obs, b_acts)]

        obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=model.device)
        acts_t = torch.as_tensor(b_acts, dtype=torch.long, device=model.device)
        weights_t = torch.as_tensor(b_weights, dtype=torch.float32, device=model.device)
        doc_t = torch.as_tensor(doc_scores, dtype=torch.float32, device=model.device)

        gen100_path = os.path.join(POOL_DIR, "generation_100.zip")
        gen100_model = GLOBAL_MODEL_CACHE.get_model(gen100_path)
        if gen100_model is not None:
            with torch.no_grad():
                g100_dist = gen100_model.policy.get_distribution(obs_t)
                g100_acts_t = torch.argmax(g100_dist.distribution.logits, dim=-1)
        else:
            g100_acts_t = acts_t

        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits
        ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
        ce_human = ce_loss_fn(logits, acts_t)
        ce_g100 = ce_loss_fn(logits, g100_acts_t)
        entropy = dist.entropy().mean()

        hybrid_loss = torch.mean(
            (doc_t * weights_t * ce_human) + ((1.00 - doc_t) * 2.50 * ce_g100)
        ) - (0.02 * entropy)

        optimizer.zero_grad()
        hybrid_loss.backward()
        optimizer.step()
        return float(hybrid_loss.item())
    except Exception:
        return 0.00

def run_evolutionary_training(controller: TrainingController):
    os.makedirs(POOL_DIR, exist_ok=True)
    env = MicroRTSEnv(mode="selfplay", lang="en")
    state = TrainingController.load_state()

    if os.path.exists(LATEST_MODEL_PATH):
        try:
            print(f"[*] Resuming neural network from checkpoint {LATEST_MODEL_PATH}...")
            model = PPO.load(LATEST_MODEL_PATH, env=env, verbose=0)
        except Exception:
            print("[*] Creating new PPO neural network for 18x18 map (18 channels, 30 actions)...")
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=256, ent_coef=0.02)
            state = {"current_gen": 0, "steps_in_gen": 0, "total_steps_done": 0, "last_loss": 0.00}
    else:
        print("[*] Creating new PPO neural network for 18x18 map...")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=256, ent_coef=0.02)

    optimizer = optim.Adam(model.policy.parameters(), lr=0.0006)
    loss_fn = nn.CrossEntropyLoss()

    start_gen = state["current_gen"]
    running_loss = state.get("last_loss", 0.00)
    total_done = state.get("total_steps_done", 0)

    gen0_file = os.path.join(POOL_DIR, "generation_0.zip")
    if not os.path.exists(gen0_file) and start_gen == 0:
        model.save(os.path.join(POOL_DIR, "generation_0"))

    obs, _ = env.reset()

    for gen_idx in range(start_gen, TOTAL_GENERATIONS + 1):
        tier, tier_desc = get_tier_for_generation(gen_idx)
        env.set_curriculum_tier(tier)
        controller.tier_text = tier_desc

        step_start = state["steps_in_gen"] if gen_idx == start_gen else 0
        guided_cutoff = int(STEPS_PER_GEN * (0.75 if tier < 5 else 0.45))

        for s in range(step_start + 1, STEPS_PER_GEN + 1):
            if gen_idx >= 16 and s % 50 == 0:
                train_on_human_telemetry(model, optimizer, env)

            kind, pos, mask = env.get_current_p1_entity_and_mask()

            if s <= guided_cutoff:
                target_act = env.get_strategic_teacher_action(kind, pos, faction=1, tier=tier, precomputed_mask=mask)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                target_tensor = torch.tensor([target_act], dtype=torch.long, device=model.device)

                dist = model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits
                loss = loss_fn(logits, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = 0.99 * running_loss + 0.01 * float(loss.item())

                priors = env._compute_strategic_logit_prior(kind, pos, 1, mask)
                chosen_act = predict_masked_action(model, obs, mask, deterministic=False, logit_priors=priors)
                obs, _, done, _, info = env.step(chosen_act)
                if done:
                    w_fac = 1 if info.get("winner") == "p1" else 2
                    env.export_high_efficiency_telemetry(winning_faction=w_fac, min_efficiency=MIN_TRAINING_EFFICIENCY)
                    obs, _ = env.reset()
                mode_label = f"Guided Doctrine Training (Tier {tier})"
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits[0]
                priors = env._compute_strategic_logit_prior(kind, pos, 1, mask)
                prior_t = torch.as_tensor(priors, dtype=torch.float32, device=model.device)
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
                masked_logits = logits.clone() + prior_t
                masked_logits[~mask_tensor] = -1e8
                probs = torch.softmax(masked_logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                act_t = m.sample()
                log_prob = m.log_prob(act_t)

                next_obs, rew, done, _, info = env.step(int(act_t.item()))
                advantage = float(rew) + (0.05 if not done else 0.00)
                rl_loss = -log_prob * advantage
                optimizer.zero_grad()
                rl_loss.backward()
                optimizer.step()

                obs = next_obs
                if done:
                    w_fac = 1 if info.get("winner") == "p1" else 2
                    env.export_high_efficiency_telemetry(winning_faction=w_fac, min_efficiency=MIN_TRAINING_EFFICIENCY)
                    obs, _ = env.reset()
                mode_label = "Autonomous RL + Validated Human Pool"

            total_done += 1
            controller.progress_pct = (total_done / float(TOTAL_PIPELINE_STEPS)) * 100.00
            controller.status_text = (
                f"Generation: {gen_idx}/{TOTAL_GENERATIONS} | Step: {s}/{STEPS_PER_GEN}\n"
                f"Mode: {mode_label} | Progress: {controller.progress_pct:.2f}% | Loss: {running_loss:.2f}"
            )

            if s % 2500 == 0:
                model.save(LATEST_MODEL_PATH)
                TrainingController.save_state(gen_idx, s, total_done, running_loss)
                print(f"[Gen {gen_idx:3d}] Step {s:5d}/{STEPS_PER_GEN} ({controller.progress_pct:.2f}%) | {mode_label}")

            if controller.stop_requested:
                model.save(LATEST_MODEL_PATH)
                model.save(os.path.join(POOL_DIR, f"generation_{gen_idx}"))
                TrainingController.save_state(gen_idx, s, total_done, running_loss)
                print(f"\n[!] Training safely saved at generation {gen_idx} (step {s}).")
                return

        gen_path = os.path.join(POOL_DIR, f"generation_{gen_idx}")
        model.save(gen_path)
        model.save(LATEST_MODEL_PATH)
        TrainingController.save_state(gen_idx + 1, 0, total_done, running_loss)
        print(f"[*] Saved Generation {gen_idx} -> generation_{gen_idx}.zip ({tier_desc})")

    controller.status_text = "All 100 neural network generations successfully trained!"

def start_gui_and_training():
    controller = TrainingController()
    signal.signal(signal.SIGINT, lambda s, f: controller.request_stop())

    train_thread = threading.Thread(target=run_evolutionary_training, args=(controller,), daemon=True)
    train_thread.start()

    root = tk.Tk()
    root.title("Faction RTS AI (18x18) — Evolutionary Training Manager (Gen 0..100)")
    root.geometry("660x270")
    root.resizable(False, False)

    tk.Label(root, text="Evolutionary Neural Network Training (18x18) + Validated Human Pool", font=("Arial", 12, "bold")).pack(pady=10)
    lbl_tier = tk.Label(root, text="", font=("Arial", 9, "bold"), fg="#2980b9", wraplength=620)
    lbl_tier.pack(pady=4)

    lbl_status = tk.Label(root, text=controller.status_text, font=("Arial", 10), wraplength=620, justify="center")
    lbl_status.pack(pady=6)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=580, mode="determinate", maximum=100.0)
    progress_bar.pack(pady=8)

    def on_save_and_stop():
        btn_stop.config(state="disabled", text="Saving model weights... Please wait")
        controller.request_stop()

    btn_stop = tk.Button(
        root, text="SAVE PROGRESS AND SAFELY EXIT",
        font=("Arial", 11, "bold"), bg="#d9534f", fg="white",
        padx=15, pady=8, command=on_save_and_stop
    )
    btn_stop.pack(pady=10)

    def update_ui():
        lbl_tier.config(text=controller.tier_text)
        lbl_status.config(text=controller.status_text)
        progress_bar['value'] = controller.progress_pct
        if not train_thread.is_alive():
            if controller.stop_requested:
                root.destroy()
                return
            btn_stop.config(text="TRAINING COMPLETED (CLOSE)", bg="#5cb85c", state="normal", command=root.destroy)
        else:
            root.after(200, update_ui)

    root.protocol("WM_DELETE_WINDOW", on_save_and_stop)
    root.after(200, update_ui)
    root.mainloop()
    train_thread.join(timeout=10.0)

if __name__ == "__main__":
    start_gui_and_training()