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
from microrts_env import MicroRTSEnv, POOL_DIR, HUMAN_TELEMETRY_FILE, predict_masked_action

STATE_FILE = os.path.join(POOL_DIR, "training_state.json")
LATEST_MODEL_PATH = os.path.join(POOL_DIR, "latest_checkpoint.zip")

TOTAL_GENERATIONS = 100
STEPS_PER_GEN = 20000                      # 20 000 кроків на кожне покоління (разом 2 000 000 кроків)
TOTAL_PIPELINE_STEPS = TOTAL_GENERATIONS * STEPS_PER_GEN

def get_tier_for_generation(gen_idx: int) -> tuple:
    """
    Визначає еволюційний ступінь (Tier 1..5) та опис навичок для конкретного покоління (0..100).
    """
    if gen_idx <= 15:
        return 1, "Ступінь 1 (Покоління 0-15): Базовий рух та простий бій (рівень для новачків)"
    elif gen_idx <= 35:
        return 2, "Ступінь 2 (Покоління 16-35): Рання експансія 18x18, зачистка сірих клітинок, тилові робітники"
    elif gen_idx <= 55:
        return 3, "Ступінь 3 (Покоління 36-55): Контроль перешийків (2 проти 1), переслідування, вежі без блокування"
    elif gen_idx <= 75:
        return 4, "Ступінь 4 (Покоління 56-75): Глибинний радар загроз, захист казни, зв'язки Піхота+Стрільці"
    else:
        return 5, "Ступінь 5 (Покоління 76-100): Заманювання під вежі vs Розрахунок жертви + Досвід реальних гравців"

class TrainingController:
    def __init__(self):
        self.stop_requested = False
        self.status_text = "Ініціалізація еволюційного тренування..."
        self.tier_text = ""
        self.progress_pct = 0.00

    def request_stop(self):
        self.stop_requested = True
        self.status_text = "Отримано сигнал зупинки! Безпечно зберігаємо ваги нейромережі..."

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

def train_on_human_telemetry(model: PPO, optimizer: optim.Optimizer) -> float:
    """
    Навчає нейромережу на збережених ходах найефективніших юнітів із реальних партій гравця.
    Кожен хід зважується на індивідуальний показник ефективності юніта (weights).
    """
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

        obs_t = torch.as_tensor(obs_arr[indices], dtype=torch.float32, device=model.device)
        acts_t = torch.as_tensor(acts_arr[indices], dtype=torch.long, device=model.device)
        weights_t = torch.as_tensor(weights_arr[indices], dtype=torch.float32, device=model.device)

        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, acts_t)
        weighted_loss = torch.mean(ce_loss * weights_t)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        return float(weighted_loss.item())
    except Exception:
        return 0.00

def run_evolutionary_training(controller: TrainingController):
    os.makedirs(POOL_DIR, exist_ok=True)
    env = MicroRTSEnv(mode="selfplay")
    state = TrainingController.load_state()

    if os.path.exists(LATEST_MODEL_PATH):
        try:
            print(f"[*] Відновлення нейромережі з чекпоїнта {LATEST_MODEL_PATH}...")
            model = PPO.load(LATEST_MODEL_PATH, env=env, verbose=0)
        except Exception:
            print("[*] Створення нової архітектури нейромережі PPO під мапу 18x18 (18 каналів, 30 дій)...")
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=256, ent_coef=0.02)
            state = {"current_gen": 0, "steps_in_gen": 0, "total_steps_done": 0, "last_loss": 0.00}
    else:
        print("[*] Створення нової нейромережі PPO під мапу 18x18...")
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
            # Кожні 50 кроків підмішуємо досвід високоефективних юнітів із телеметрії реальних ігор
            if gen_idx >= 16 and s % 50 == 0:
                train_on_human_telemetry(model, optimizer)

            kind, pos, mask = env.get_current_p1_entity_and_mask()

            if s <= guided_cutoff:
                target_act = env.get_strategic_teacher_action(kind, pos, faction=1, tier=tier)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                target_tensor = torch.tensor([target_act], dtype=torch.long, device=model.device)

                dist = model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits
                loss = loss_fn(logits, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = 0.99 * running_loss + 0.01 * float(loss.item())

                chosen_act = predict_masked_action(model, obs, mask, deterministic=False)
                obs, _, done, _, info = env.step(chosen_act)
                if done:
                    # Зберігаємо ходи високоефективних юнітів із зіграної партії
                    w_fac = 1 if info.get("winner") == "p1" else 2
                    env.export_high_efficiency_telemetry(winning_faction=w_fac, min_efficiency=5.50)
                    obs, _ = env.reset()
                mode_label = f"Навчання навичок Ступеня {tier}"
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits[0]
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
                masked_logits = logits.clone()
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
                    env.export_high_efficiency_telemetry(winning_faction=w_fac, min_efficiency=5.50)
                    obs, _ = env.reset()
                mode_label = "Вільне плавання (Autonomous RL + Human Pool)"

            total_done += 1
            controller.progress_pct = (total_done / float(TOTAL_PIPELINE_STEPS)) * 100.00
            controller.status_text = (
                f"Покоління: {gen_idx}/{TOTAL_GENERATIONS} | Крок у поколінні: {s}/{STEPS_PER_GEN}\n"
                f"Режим: {mode_label} | Загальний прогрес: {controller.progress_pct:.2f}% | Loss: {running_loss:.2f}"
            )

            if s % 2500 == 0:
                model.save(LATEST_MODEL_PATH)
                TrainingController.save_state(gen_idx, s, total_done, running_loss)
                print(f"[Покоління {gen_idx:3d}] Крок {s:5d}/{STEPS_PER_GEN} ({controller.progress_pct:.2f}%) | {mode_label}")

            if controller.stop_requested:
                model.save(LATEST_MODEL_PATH)
                model.save(os.path.join(POOL_DIR, f"generation_{gen_idx}"))
                TrainingController.save_state(gen_idx, s, total_done, running_loss)
                print(f"\n[!] Навчання безпечно збережено на поколінні {gen_idx} (крок {s}). Можна вимикати ПК.")
                return

        gen_path = os.path.join(POOL_DIR, f"generation_{gen_idx}")
        model.save(gen_path)
        model.save(LATEST_MODEL_PATH)
        TrainingController.save_state(gen_idx + 1, 0, total_done, running_loss)
        print(f"[*] Сформовано та збережено Покоління {gen_idx} -> generation_{gen_idx}.zip ({tier_desc})")

    controller.status_text = "Усі 100 поколінь нейромережі успішно сформовані!"

def start_gui_and_training():
    controller = TrainingController()
    signal.signal(signal.SIGINT, lambda s, f: controller.request_stop())

    train_thread = threading.Thread(target=run_evolutionary_training, args=(controller,), daemon=True)
    train_thread.start()

    root = tk.Tk()
    root.title("Faction RTS AI (18x18) — Еволюційний Менеджер Навчання (Покоління 0..100)")
    root.geometry("660x270")
    root.resizable(False, False)

    tk.Label(root, text="Поетапна Еволюція Нейромережі (18x18) + Досвід Юнітів Гравця", font=("Arial", 12, "bold")).pack(pady=10)
    lbl_tier = tk.Label(root, text="", font=("Arial", 9, "bold"), fg="#2980b9", wraplength=620)
    lbl_tier.pack(pady=4)

    lbl_status = tk.Label(root, text=controller.status_text, font=("Arial", 10), wraplength=620, justify="center")
    lbl_status.pack(pady=6)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=580, mode="determinate", maximum=100.0)
    progress_bar.pack(pady=8)

    def on_save_and_stop():
        btn_stop.config(state="disabled", text="Збереження ваг моделі... Зачекайте")
        controller.request_stop()

    btn_stop = tk.Button(
        root, text="ЗБЕРЕГТИ ПРОГРЕС І БЕЗПЕЧНО ВИЙТИ",
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
            btn_stop.config(text="НАВЧАННЯ ЗАВЕРШЕНО (ЗАКРИТИ)", bg="#5cb85c", state="normal", command=root.destroy)
        else:
            root.after(200, update_ui)

    root.protocol("WM_DELETE_WINDOW", on_save_and_stop)
    root.after(200, update_ui)
    root.mainloop()
    train_thread.join(timeout=10.0)

if __name__ == "__main__":
    start_gui_and_training()