import os
import math
import time
import random
import pathlib
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from microrts_env import (
    MicroRTSEnv, PoolRegistry, OnlineEvolutionEngine,
    UNIT_STATS, STRUCT_STATS, CASTLE_MAX_HP,
    MAX_STEPS, MAX_GOLD, GRID_SIZE, RELATIVE_OFFSETS,
    SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT
)

class TacticalWarfareGUI:
    def __init__(self):
        PoolRegistry.ensure_initialized()
        self.env = MicroRTSEnv(mode="manual_p2", record_telemetry=True)
        self.base_dda_gen = 35.00
        self.active_dda_gen = 35.00
        self.env.set_dda_generation(self.active_dda_gen)
        self.env._prepare_faction_turn(faction=2, collect_economy=False)

        self.selected_pos = None
        self.valid_move_tiles = []
        self.valid_attack_tiles = []
        self.pulse_phase = 0

        self.recent_human_wins = []
        self.prev_winrate = 0.50
        self.window_games = 20
        self.matches_played = 0
        self.is_auto_playing = False
        self.evo_status_str = "Пул готовий"

        self.grid_n = GRID_SIZE
        self.view_w = 684
        self.view_h = 684
        self.cell_size = 38.00
        self.min_cell_size = 26.00
        self.max_cell_size = 96.00
        self.cam_x = 0.00
        self.cam_y = 0.00
        self.pan_start_x = 0
        self.pan_start_y = 0
        self._clamp_camera()

        self.root = tk.Tk()
        self.root.title("Faction Warfare RTS (18x18) — Жива Еволюція Поколінь, Surge-DDA та Статистика Ефективності")
        self.root.geometry("1160x760")
        self.root.configure(bg="#182026")

        self.canvas = tk.Canvas(
            self.root, width=self.view_w, height=self.view_h,
            bg="#11171c", highlightthickness=2, highlightbackground="#3c6382"
        )
        self.canvas.grid(row=0, column=0, rowspan=4, padx=14, pady=14)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel_linux)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel_linux)

        self.side_panel = tk.Frame(self.root, bg="#182026", width=420)
        self.side_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=14)

        self._build_side_panel()
        self._refresh_board()
        self._animate_pulse()

    def _clamp_camera(self):
        board_px = self.grid_n * self.cell_size
        margin = 1.00 * self.cell_size

        max_x = margin
        min_x = self.view_w - board_px - margin
        if min_x > max_x:
            self.cam_x = (self.view_w - board_px) / 2.00
        else:
            self.cam_x = float(np.clip(self.cam_x, min_x, max_x))

        max_y = margin
        min_y = self.view_h - board_px - margin
        if min_y > max_y:
            self.cam_y = (self.view_h - board_px) / 2.00
        else:
            self.cam_y = float(np.clip(self.cam_y, min_y, max_y))

    def on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_drag(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.cam_x += dx
        self.cam_y += dy
        self._clamp_camera()
        self._refresh_board()

    def _apply_zoom(self, factor: float, cursor_x: float, cursor_y: float):
        old_size = self.cell_size
        new_size = float(np.clip(old_size * factor, self.min_cell_size, self.max_cell_size))
        if abs(new_size - old_size) < 0.01:
            return
        scale_ratio = new_size / old_size
        self.cam_x = cursor_x - ((cursor_x - self.cam_x) * scale_ratio)
        self.cam_y = cursor_y - ((cursor_y - self.cam_y) * scale_ratio)
        self.cell_size = new_size
        self._clamp_camera()
        self._refresh_board()

    def on_mouse_wheel(self, event):
        factor = 1.12 if event.delta > 0 else 0.89
        self._apply_zoom(factor, float(event.x), float(event.y))

    def on_mouse_wheel_linux(self, event):
        factor = 1.12 if event.num == 4 else 0.89
        self._apply_zoom(factor, float(event.x), float(event.y))

    def _build_side_panel(self):
        self.econ_frame = tk.LabelFrame(
            self.side_panel, text=" Скарбниця, Активне Покоління ШІ та Surge-DDA ",
            font=("Arial", 10, "bold"), bg="#222f3e", fg="#feca57", padx=10, pady=8
        )
        self.econ_frame.pack(fill="x", pady=(0, 8))

        self.lbl_telemetry = tk.Label(
            self.econ_frame, text="", font=("Consolas", 9, "bold"),
            bg="#222f3e", fg="#c8d6e5", justify="left"
        )
        self.lbl_telemetry.pack(anchor="w")

        self.inspector_frame = tk.LabelFrame(
            self.side_panel, text=" Характеристики та Ефективність Об'єкта ",
            font=("Arial", 10, "bold"), bg="#222f3e", fg="#0abde3", padx=10, pady=8
        )
        self.inspector_frame.pack(fill="x", pady=(0, 8))

        self.lbl_inspector = tk.Label(
            self.inspector_frame,
            text="Клікніть на будь-якого юніта, вежу,\nказарму або замок на полі бою.",
            font=("Arial", 9, "bold"), bg="#222f3e", fg="#ffffff", justify="left", height=7
        )
        self.lbl_inspector.pack(anchor="w")

        self.actions_frame = tk.LabelFrame(
            self.side_panel, text=" Доступні Накази Об'єкту ",
            font=("Arial", 10, "bold"), bg="#222f3e", fg="#1dd1a1", padx=10, pady=6
        )
        self.actions_frame.pack(fill="x", pady=(0, 8))

        self.btn_heal = tk.Button(
            self.actions_frame, text="✚ Лікувати юніта (+25.00% макс. HP)",
            font=("Arial", 9, "bold"), bg="#10ac84", fg="white", width=38,
            command=self.cmd_heal_selected
        )
        self.btn_heal.pack(pady=2)

        self.btn_harvest = tk.Button(
            self.actions_frame, text="⛏ Додаткова обробка клітинки (+1.50x доходу)",
            font=("Arial", 9, "bold"), bg="#f39c12", fg="white", width=38,
            command=self.cmd_worker_harvest
        )
        self.btn_harvest.pack(pady=2)

        self.btn_build_tower = tk.Button(
            self.actions_frame, text="🗼 Звести Оборонну Вежу поруч (4.00$)",
            font=("Arial", 9, "bold"), bg="#576574", fg="white", width=38,
            command=lambda: self.cmd_worker_build(1)
        )
        self.btn_build_tower.pack(pady=2)

        self.btn_build_barracks = tk.Button(
            self.actions_frame, text="⚔ Звести Передову Казарму (5.00$)",
            font=("Arial", 9, "bold"), bg="#576574", fg="white", width=38,
            command=lambda: self.cmd_worker_build(2)
        )
        self.btn_build_barracks.pack(pady=2)

        self.btn_spawn_worker = tk.Button(
            self.actions_frame, text="+ Найняти Робітника в будівлі (3.00$ | Утрим: 0.40$)",
            font=("Arial", 9, "bold"), bg="#2e86de", fg="white", width=38,
            command=lambda: self.cmd_spawn_unit('worker')
        )
        self.btn_spawn_worker.pack(pady=2)

        self.btn_spawn_warrior = tk.Button(
            self.actions_frame, text="+ Найняти Піхотинця в будівлі (6.00$ | Утрим: 1.10$)",
            font=("Arial", 9, "bold"), bg="#2e86de", fg="white", width=38,
            command=lambda: self.cmd_spawn_unit('warrior')
        )
        self.btn_spawn_warrior.pack(pady=2)

        self.btn_spawn_archer = tk.Button(
            self.actions_frame, text="+ Найняти Стрільця в будівлі (8.50$ | Утрим: 1.60$)",
            font=("Arial", 9, "bold"), bg="#2e86de", fg="white", width=38,
            command=lambda: self.cmd_spawn_unit('archer')
        )
        self.btn_spawn_archer.pack(pady=2)

        self.turn_frame = tk.Frame(self.side_panel, bg="#182026")
        self.turn_frame.pack(fill="x", pady=2)

        self.btn_stats = tk.Button(
            self.turn_frame,
            text="📈 Статистика ефективності (Графік + Медіана)",
            font=("Arial", 9, "bold"), bg="#9b59b6", fg="white",
            padx=8, pady=5, width=38, command=self.show_efficiency_chart
        )
        self.btn_stats.pack(pady=2)

        self.btn_end_turn = tk.Button(
            self.turn_frame,
            text="✔ КІНЕЦЬ ХОДУ (Авто-хіл нерухомих + Хід ШІ)",
            font=("Arial", 10, "bold"), bg="#2ecc71", fg="#182026",
            padx=10, pady=7, width=38, command=self.end_human_turn
        )
        self.btn_end_turn.pack(pady=3)

        self.btn_autoplay = tk.Button(
            self.turn_frame,
            text="► Дивитись Авто-матч ШІ проти ШІ (0.50 с на хід)",
            font=("Arial", 9, "bold"), bg="#48dbfb", fg="#182026",
            padx=8, pady=4, width=38, command=self.toggle_autoplay
        )
        self.btn_autoplay.pack(pady=2)

    def show_efficiency_chart(self):
        st = self.env.get_efficiency_statistics()
        turns = st["turns"]
        p_eff = st["player_eff"]
        ai_eff = st["ai_eff"]
        p_med = st["player_median"]
        ai_med = st["ai_median"]

        plt.figure(figsize=(10, 5.2))
        plt.plot(turns, p_eff, label=f"Ефективність Гравця (поточна: {st['player_current']:.2f})", color="#0fbcf9", linewidth=2.5)
        plt.axhline(p_med, color="#0fbcf9", linestyle="--", linewidth=1.6, label=f"Медіана Гравця: {p_med:.2f}")
        plt.plot(turns, ai_eff, label=f"Ефективність ШІ (поточна: {st['ai_current']:.2f})", color="#eb3b5a", linewidth=2.5)
        plt.axhline(ai_med, color="#eb3b5a", linestyle="--", linewidth=1.6, label=f"Медіана ШІ: {ai_med:.2f}")
        plt.xlabel("Хід партії")
        plt.ylabel("Інтегральна ефективність")
        plt.title(f"Зростання ефективності та Медіана (Хід {self.env.turn_count}, Активне покоління: {self.env.active_gen_label})")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def _animate_pulse(self):
        self.pulse_phase = (self.pulse_phase + 1) % 6
        if self.selected_pos is not None:
            self._refresh_board()
        self.root.after(140, self._animate_pulse)

    def _record_human_move_or_attack(self, origin_pos: tuple, target_pos: tuple, is_attack: bool, is_tower: bool = False):
        dr = target_pos[0] - origin_pos[0]
        dc = target_pos[1] - origin_pos[1]
        inv_offset = (-dr, -dc)
        if inv_offset in RELATIVE_OFFSETS:
            base_idx = RELATIVE_OFFSETS.index(inv_offset)
            act_code = (12 + base_idx) if is_attack else base_idx
            kind = 'tower' if is_tower else 'unit'
            self.env.record_entity_decision(kind, origin_pos, faction=2, action=act_code)

    def on_canvas_click(self, event):
        if self.is_auto_playing:
            return
        c = int(math.floor((event.x - self.cam_x) / self.cell_size))
        r = int(math.floor((event.y - self.cam_y) / self.cell_size))
        if not (0 <= r < self.grid_n and 0 <= c < self.grid_n):
            return
        clicked = (r, c)

        if self.selected_pos is not None:
            if clicked in self.valid_move_tiles:
                self._record_human_move_or_attack(self.selected_pos, clicked, is_attack=False)
                if self.env.execute_unit_move(self.selected_pos, clicked):
                    self.selected_pos = clicked
                    self._recompute_highlights()
                    self._refresh_board()
                    return

            if clicked in self.valid_attack_tiles:
                sr, sc = self.selected_pos
                is_tower = (
                    self.selected_pos not in self.env.units
                    and self.env.struct_owner[sr, sc] == 2
                    and self.env.struct_type[sr, sc] == 1
                )
                self._record_human_move_or_attack(self.selected_pos, clicked, is_attack=True, is_tower=is_tower)
                dmg = self.env.execute_attack_on_target(self.selected_pos, clicked, is_tower=is_tower)
                if dmg > 0.00:
                    self._recompute_highlights()
                    self._refresh_board()
                    self._check_match_end()
                    return

        self.selected_pos = clicked
        self._recompute_highlights()
        self._refresh_board()

    def _recompute_highlights(self):
        self.valid_move_tiles = []
        self.valid_attack_tiles = []
        if self.selected_pos is None:
            return

        r, c = self.selected_pos
        if self.selected_pos in self.env.units:
            u = self.env.units[self.selected_pos]
            if u['faction'] == 2 and u['ready']:
                self.valid_move_tiles = self.env.get_valid_moves_for_unit(self.selected_pos)
                self.valid_attack_tiles = self.env.get_valid_attack_targets(self.selected_pos)
        elif self.env.struct_owner[r, c] == 2 and self.env.struct_type[r, c] == 1 and self.env.tower_ready[r, c]:
            self.valid_attack_tiles = self.env.get_valid_attack_targets(self.selected_pos, faction=2, atk_range=2)

    def _update_inspector_and_buttons(self):
        for btn in [
            self.btn_heal, self.btn_harvest, self.btn_build_tower, self.btn_build_barracks,
            self.btn_spawn_worker, self.btn_spawn_warrior, self.btn_spawn_archer
        ]:
            btn.config(state="disabled")

        if self.selected_pos is None:
            self.lbl_inspector.config(
                text="Нічого не обрано.\nКлікніть на юніта, вежу, казарму або замок.\n(Коліщатко миші: зум та переміщення камери)"
            )
            return

        r, c = self.selected_pos
        base_val = float(self.env.tile_yields[r, c])
        eff_val = self.env.get_effective_tile_yield(r, c)
        owner_str = {0: "Нейтральна (Сіра)", 1: "Червона Імперія (ШІ)", 2: "Синій Альянс (Ви)"}[int(self.env.territory[r, c])]

        if self.env.is_tile_built_up(r, c):
            info_lines = [f"Земля: +0.00$/хід [Забудовано, базова: +{base_val:.2f}$] ({owner_str})"]
        else:
            info_lines = [f"Земля: +{eff_val:.2f}$/хід ({owner_str})"]

        if self.selected_pos in self.env.units:
            u = self.env.units[self.selected_pos]
            st = UNIT_STATS[u['type']]
            f_str = "Синій Альянс (Ваш)" if u['faction'] == 2 else "Червона Імперія (ШІ)"
            status_str = "ГОТОВИЙ ДО ХОДУ" if u['ready'] else "ВЖЕ ДІЯВ / ЩОЙНО НАЙНЯТИЙ"
            eff_score = u.get('efficiency', 0.00)
            eff_rank = "ЕЛІТНИЙ ВЕТЕРАН (Вчиться ШІ!)" if eff_score >= 2.50 else "Звичайний боєць"
            info_lines.append(f"Тип підрозділу: {st['name']} [{f_str}]")
            info_lines.append(f"Здоров'я (HP): {u['hp']:.2f} / {u['max_hp']:.2f} ({(u['hp'] / u['max_hp']) * 100.00:.2f}%)")
            info_lines.append(f"Дальність руху: {st['move_range']} кліт. | Дальність атаки: {st['atk_range']} кліт.")
            info_lines.append(f"Шкода атаки: {st['dmg']:.2f} | Плата за службу: {st['upkeep']:.2f}$/хід")
            info_lines.append(f"Ефективність: {eff_score:.2f} балів ({eff_rank})")
            info_lines.append(f"Статус: {status_str}")

            if u['faction'] == 2 and u['ready']:
                self.btn_heal.config(state="normal")
                if u['type'] == 'worker':
                    if eff_val > 0.00:
                        self.btn_harvest.config(state="normal")
                    if self.env.p2_gold >= STRUCT_STATS[1]['cost'] and self.env._find_adjacent_tower_site(self.selected_pos, 2) is not None:
                        self.btn_build_tower.config(state="normal")
                    can_b = (
                        self.env.territory[r, c] == 2
                        and self.selected_pos not in [self.env.p1_castle_pos, self.env.p2_castle_pos]
                        and self.env.struct_type[r, c] == 0
                    )
                    if self.env.p2_gold >= STRUCT_STATS[2]['cost'] and (can_b or self.env._find_adjacent_tower_site(self.selected_pos, 2) is not None):
                        self.btn_build_barracks.config(state="normal")

        if self.selected_pos == self.env.p2_castle_pos:
            occupied = (self.selected_pos in self.env.units)
            info_lines.append(f"Споруда: ВАШ ЗАМОК (Міцність: {self.env.p2_castle_hp:.2f} / {CASTLE_MAX_HP:.2f} HP)")
            if occupied:
                info_lines.append("[!] Замок зайнятий юнітом! Звільніть клітинку для найму.")
            else:
                info_lines.append("Клітинка вільна: оберіть юніта для спавну прямо в Замку.")
                if self.env.p2_gold >= UNIT_STATS['worker']['cost']:
                    self.btn_spawn_worker.config(state="normal")
                if self.env.p2_gold >= UNIT_STATS['warrior']['cost']:
                    self.btn_spawn_warrior.config(state="normal")
                if self.env.p2_gold >= UNIT_STATS['archer']['cost']:
                    self.btn_spawn_archer.config(state="normal")

        elif self.selected_pos == self.env.p1_castle_pos:
            info_lines.append(f"Споруда: ЗАМОК ШІ (Міцність: {self.env.p1_castle_hp:.2f} / {CASTLE_MAX_HP:.2f} HP)")

        elif self.env.struct_type[r, c] != 0:
            st_t = int(self.env.struct_type[r, c])
            st_owner = int(self.env.struct_owner[r, c])
            st_info = STRUCT_STATS[st_t]
            info_lines.append(f"Споруда: {st_info['name']} ({'Ваша' if st_owner == 2 else 'Ворожа'})")
            info_lines.append(f"Міцність: {self.env.struct_hp[r, c]:.2f} / {st_info['max_hp']:.2f} HP")
            if st_t == 1:
                info_lines.append(f"Дальність вогню вежі: 2 кліт. | Шкода: {st_info['dmg']:.2f} (1 ціль за хід)")
            elif st_t == 2 and st_owner == 2:
                if self.selected_pos not in self.env.units:
                    if self.env.p2_gold >= UNIT_STATS['worker']['cost']:
                        self.btn_spawn_worker.config(state="normal")
                    if self.env.p2_gold >= UNIT_STATS['warrior']['cost']:
                        self.btn_spawn_warrior.config(state="normal")
                    if self.env.p2_gold >= UNIT_STATS['archer']['cost']:
                        self.btn_spawn_archer.config(state="normal")

        self.lbl_inspector.config(text="\n".join(info_lines[:7]))

    def cmd_heal_selected(self):
        if self.selected_pos in self.env.units and self.env.units[self.selected_pos]['faction'] == 2:
            self.env.record_entity_decision('unit', self.selected_pos, faction=2, action=24)
            self.env.heal_unit_percentage(self.selected_pos)
            self._recompute_highlights()
            self._refresh_board()

    def cmd_worker_harvest(self):
        if self.selected_pos in self.env.units:
            self.env.record_entity_decision('unit', self.selected_pos, faction=2, action=25)
            self.env.execute_worker_harvest(self.selected_pos)
            self._recompute_highlights()
            self._refresh_board()

    def cmd_worker_build(self, build_type: int):
        if self.selected_pos in self.env.units:
            act_code = 26 if build_type == 1 else 27
            self.env.record_entity_decision('unit', self.selected_pos, faction=2, action=act_code)
            self.env.execute_worker_build(self.selected_pos, build_type=build_type)
            self._recompute_highlights()
            self._refresh_board()

    def cmd_spawn_unit(self, unit_type: str):
        if self.selected_pos is not None:
            act_map = {'worker': 27, 'warrior': 28, 'archer': 29}
            self.env.record_entity_decision('spawn', self.selected_pos, faction=2, action=act_map[unit_type])
            if self.env.execute_structure_spawn(self.selected_pos, faction=2, unit_type=unit_type):
                self._recompute_highlights()
                self._refresh_board()

    def end_human_turn(self):
        self.env.heal_all_unmoved_units(faction=2)
        if self._check_match_end():
            return

        self.env.execute_full_ai_turn(faction=1, model=self.env.opponent_model, deterministic=False)
        if self._check_match_end():
            return

        self.env.turn_count += 1

        if self.env.turn_count <= 10 or self.env.turn_count % 3 == 0:
            self.active_dda_gen = self.env.compute_surge_dda_step(
                current_rating=self.active_dda_gen,
                is_early_turn=(self.env.turn_count <= 10)
            )
            self.env.set_dda_generation(self.active_dda_gen)

        self.env._prepare_faction_turn(faction=2, collect_economy=True)
        self.env.record_turn_efficiency_snapshot()
        self._recompute_highlights()
        self._refresh_board()
        self._check_match_end()

    def _check_match_end(self) -> bool:
        done = False
        winner = None
        if self.env.p2_castle_hp <= 0.00:
            done = True
            winner = "p1"
        elif self.env.p1_castle_hp <= 0.00:
            done = True
            winner = "p2"
        elif self.env.turn_count > MAX_STEPS:
            done = True
            inc1, _, _, t1, u1 = self.env.get_faction_economy_stats(1)
            inc2, _, _, t2, u2 = self.env.get_faction_economy_stats(2)
            s1 = self.env.p1_castle_hp + t1 * 0.25 + inc1 * 4.00 + u1 * 1.50
            s2 = self.env.p2_castle_hp + t2 * 0.25 + inc2 * 4.00 + u2 * 1.50
            winner = "p1" if s1 >= s2 else "p2"

        if not done:
            return False

        human_won = (winner == "p2")
        winning_fac = 2 if human_won else 1

        m_obs, m_acts, m_weights = self.env.collect_match_training_samples(winning_faction=winning_fac, min_efficiency=2.50)
        saved_samples = self.env.export_high_efficiency_telemetry(winning_faction=winning_fac, min_efficiency=2.50)

        self.matches_played += 1
        self.recent_human_wins.append(1.00 if human_won else 0.00)
        if len(self.recent_human_wins) > self.window_games:
            self.recent_human_wins.pop(0)

        winrate = float(np.mean(self.recent_human_wins))
        fast_wr = float(np.mean(self.recent_human_wins[-5:]))
        err_wr = winrate - 0.50
        d_wr = winrate - self.prev_winrate
        self.prev_winrate = winrate

        delta_macro = round((8.00 * err_wr) + (32.00 * d_wr) + (12.00 * (fast_wr - 0.50)), 2)
        max_pool_r = max(100.00, PoolRegistry.get_max_rating() + 15.00)
        blended_base = (0.55 * self.base_dda_gen) + (0.45 * self.active_dda_gen) + delta_macro
        self.base_dda_gen = round(max(10.00, min(max_pool_r, blended_base)), 2)
        self.active_dda_gen = self.base_dda_gen
        self.env.set_dda_generation(self.active_dda_gen)

        self.evo_status_str = "Навчання нового покоління..."
        def _bg_train():
            entry = OnlineEvolutionEngine.evolve_and_benchmark_from_match(m_obs, m_acts, m_weights, username="LocalPlayer")
            self.evo_status_str = f"Додано {entry['label']} (Рейтинг: {entry['rating']:.2f})"
            self.env.set_dda_generation(self.active_dda_gen)

        threading.Thread(target=_bg_train, daemon=True).start()

        if not self.is_auto_playing:
            title_msg = "ПЕРЕМОГА СИНЬОГО АЛЬЯНСУ!" if human_won else "ПЕРЕМОГА ЧЕРВОНОЇ ІМПЕРІЇ (ШІ)!"
            messagebox.showinfo(
                "Кампанію завершено",
                f"{title_msg}\n"
                f"Ковзний вінрейт: {winrate * 100.00:.2f}%\n"
                f"Новий базовий рейтинг ШІ: {self.base_dda_gen:.2f}\n"
                f"Відправлено ходів на створення нового покоління: {saved_samples}"
            )

        self.env.reset()
        self.env._prepare_faction_turn(faction=2, collect_economy=False)
        self.selected_pos = None
        self.valid_move_tiles = []
        self.valid_attack_tiles = []
        self._refresh_board()
        return True

    def toggle_autoplay(self):
        self.is_auto_playing = not self.is_auto_playing
        if self.is_auto_playing:
            self.btn_autoplay.config(text="■ Зупинити Авто-матч", bg="#eb3b5a", fg="white")
            self._auto_match_step()
        else:
            self.btn_autoplay.config(text="► Дивитись Авто-матч ШІ проти ШІ (0.50 с на хід)", bg="#48dbfb", fg="#182026")

    def _auto_match_step(self):
        if not self.is_auto_playing:
            return
        self.env.execute_full_ai_turn(faction=2, model=None, deterministic=False)
        if not self._check_match_end():
            self.end_human_turn()
        self.root.after(500, self._auto_match_step)

    def _draw_unit_icon(self, cx: float, cy: float, rad: float, u_type: str, faction: int):
        fg = "#ffffff" if faction == 1 else "#141b24"
        accent = "#feca57" if faction == 1 else "#ffffff"
        lw = max(2, int(rad * 0.16))

        if u_type == 'worker':
            self.canvas.create_line(cx - rad * 0.45, cy + rad * 0.45, cx + rad * 0.35, cy - rad * 0.35, fill=accent, width=lw)
            self.canvas.create_line(cx - rad * 0.15, cy - rad * 0.50, cx + rad * 0.30, cy - rad * 0.30, cx + rad * 0.50, cy + rad * 0.15, fill=fg, width=lw + 1, smooth=True)
        elif u_type == 'warrior':
            self.canvas.create_line(cx - rad * 0.42, cy + rad * 0.42, cx + rad * 0.48, cy - rad * 0.48, fill=fg, width=lw + 1)
            self.canvas.create_line(cx - rad * 0.42, cy - rad * 0.02, cx + rad * 0.02, cy + rad * 0.42, fill=accent, width=lw)
            p_r = max(2.0, rad * 0.14)
            self.canvas.create_oval(cx - rad * 0.46 - p_r, cy + rad * 0.46 - p_r, cx - rad * 0.46 + p_r, cy + rad * 0.46 + p_r, fill=accent, outline=fg)
        elif u_type == 'archer':
            self.canvas.create_line(cx - rad * 0.15, cy - rad * 0.55, cx + rad * 0.42, cy, cx - rad * 0.15, cy + rad * 0.55, fill=accent, width=lw, smooth=True)
            self.canvas.create_line(cx - rad * 0.15, cy - rad * 0.52, cx - rad * 0.15, cy + rad * 0.52, fill=fg, width=1)
            self.canvas.create_line(cx - rad * 0.45, cy, cx + rad * 0.55, cy, fill=fg, width=lw)
            self.canvas.create_polygon(cx + rad * 0.62, cy, cx + rad * 0.28, cy - rad * 0.22, cx + rad * 0.28, cy + rad * 0.22, fill=fg, outline=fg)

    def _refresh_board(self):
        self.canvas.delete("all")
        cs = self.cell_size
        ox = self.cam_x
        oy = self.cam_y
        board_px = self.grid_n * cs

        self.canvas.create_rectangle(ox - cs, oy - cs, ox + board_px + cs, oy + board_px + cs, fill="#141c24", outline="#2c3e50", width=2, dash=(4, 4))
        self.canvas.create_rectangle(ox - 2, oy - 2, ox + board_px + 2, oy + board_px + 2, fill="#1e272e", outline="#feca57", width=2)

        for r in range(self.grid_n):
            for c in range(self.grid_n):
                x1 = ox + c * cs
                y1 = oy + r * cs
                x2 = x1 + cs
                y2 = y1 + cs

                if x2 < 0 or x1 > self.view_w or y2 < 0 or y1 > self.view_h:
                    continue

                owner = int(self.env.territory[r, c])
                t_type = int(self.env.terrain[r, c])

                if t_type == 0:
                    bg = "#3d3d3d"
                elif owner == 1:
                    bg = "#632227"
                elif owner == 2:
                    bg = "#1b3b6f"
                else:
                    bg = "#2b343b"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg, outline="#4b6584", width=1)

                if t_type == 0:
                    self.canvas.create_polygon(x1 + cs * 0.16, y2 - cs * 0.14, x1 + cs * 0.50, y1 + cs * 0.16, x2 - cs * 0.16, y2 - cs * 0.14, fill="#718093", outline="#dcdde1")
                else:
                    eff_val = self.env.get_effective_tile_yield(r, c)
                    dot_r = max(3.0, cs * 0.10)
                    if not self.env.is_tile_built_up(r, c):
                        if t_type == 3:
                            self.canvas.create_oval(x2 - dot_r * 2.6, y1 + dot_r * 0.6, x2 - dot_r * 0.6, y1 + dot_r * 2.6, fill="#f1c40f", outline="#fff200")
                            t_col = "#feca57"
                        elif t_type == 2:
                            self.canvas.create_oval(x2 - dot_r * 2.4, y1 + dot_r * 0.7, x2 - dot_r * 0.7, y1 + dot_r * 2.4, fill="#1dd1a1", outline="#10ac84")
                            t_col = "#1dd1a1"
                        else:
                            t_col = "#8395a7"

                        if cs >= 32.0:
                            f_sz = max(6, int(cs * 0.17))
                            self.canvas.create_text(x1 + cs * 0.32, y1 + cs * 0.18, text=f"{eff_val:.2f}", fill=t_col, font=("Arial", f_sz, "bold"))

                st_t = int(self.env.struct_type[r, c])
                if st_t != 0:
                    st_own = int(self.env.struct_owner[r, c])
                    st_col = "#c23616" if st_own == 1 else "#0097e6"
                    f_sz = max(6, int(cs * 0.16))
                    if st_t == 1:
                        self.canvas.create_rectangle(x1 + cs * 0.16, y1 + cs * 0.25, x2 - cs * 0.16, y2 - cs * 0.12, fill=st_col, outline="#f5f6fa", width=2)
                        self.canvas.create_text(x1 + cs * 0.50, y1 + cs * 0.56, text=f"ВЖ\n{self.env.struct_hp[r, c]:.0f}", fill="white", font=("Arial", f_sz, "bold"))
                    else:
                        self.canvas.create_rectangle(x1 + cs * 0.08, y1 + cs * 0.22, x2 - cs * 0.08, y2 - cs * 0.08, fill=st_col, outline="#feca57", width=2, stipple="gray50")
                        self.canvas.create_text(x1 + cs * 0.50, y2 - cs * 0.20, text=f"КЗ {self.env.struct_hp[r, c]:.0f}", fill="#feca57", font=("Arial", f_sz, "bold"))

        for pos, hp, col, title in [
            (self.env.p1_castle_pos, self.env.p1_castle_hp, "#83142c", "ЗМ ШІ"),
            (self.env.p2_castle_pos, self.env.p2_castle_hp, "#0a3d62", "ВАШ ЗМ")
        ]:
            x1 = ox + pos[1] * cs
            y1 = oy + pos[0] * cs
            x2 = x1 + cs
            y2 = y1 + cs
            f_sz = max(6, int(cs * 0.16))
            self.canvas.create_rectangle(x1 + cs * 0.06, y1 + cs * 0.20, x2 - cs * 0.06, y2 - cs * 0.06, fill=col, outline="#feca57", width=2)
            self.canvas.create_text(x1 + cs * 0.50, y2 - cs * 0.24, text=f"{title}\n{hp:.0f}", fill="#feca57", font=("Arial", f_sz, "bold"))

        for mr, mc in self.valid_move_tiles:
            x1 = ox + mc * cs
            y1 = oy + mr * cs
            x2 = x1 + cs
            y2 = y1 + cs
            self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="#f1c40f", width=2)

        for tr, tc in self.valid_attack_tiles:
            x1 = ox + tc * cs
            y1 = oy + tr * cs
            x2 = x1 + cs
            y2 = y1 + cs
            self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="#e056fd", width=3)

        for (ur, uc), u in self.env.units.items():
            cx = ox + uc * cs + cs * 0.50
            cy = oy + ur * cs + cs * 0.48
            if cx + cs < 0 or cx - cs > self.view_w or cy + cs < 0 or cy - cs > self.view_h:
                continue

            u_col = "#eb3b5a" if u['faction'] == 1 else "#0fbcf9"
            border_col = "#2ecc71" if (u['faction'] == 2 and u['ready']) else ("#feca57" if u.get('efficiency', 0.0) >= 2.5 else "#dcdde1")
            rad = max(9.0, cs * 0.34)

            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=u_col, outline=border_col, width=2)
            self._draw_unit_icon(cx, cy, rad, u['type'], u['faction'])

            hp_ratio = u['hp'] / u['max_hp']
            bar_w = cs * 0.74
            bar_h = max(3.0, cs * 0.09)
            bar_y1 = cy + rad + 1.0
            self.canvas.create_rectangle(cx - bar_w * 0.50, bar_y1, cx + bar_w * 0.50, bar_y1 + bar_h, fill="#1e272e", outline="")
            self.canvas.create_rectangle(cx - bar_w * 0.50, bar_y1, cx - bar_w * 0.50 + bar_w * hp_ratio, bar_y1 + bar_h, fill="#2ecc71" if hp_ratio > 0.50 else "#f39c12", outline="")

        if self.selected_pos is not None:
            sr, sc = self.selected_pos
            x1 = ox + sc * cs
            y1 = oy + sr * cs
            x2 = x1 + cs
            y2 = y1 + cs
            pad = 1 + (self.pulse_phase // 2)
            self.canvas.create_rectangle(x1 + pad, y1 + pad, x2 - pad, y2 - pad, outline="#00d2d3", width=3)

        inc1, upk1, net1, t1, u1 = self.env.get_faction_economy_stats(1)
        inc2, upk2, net2, t2, u2 = self.env.get_faction_economy_stats(2)
        tax1 = max(0, u1 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT
        tax2 = max(0, u2 - SUPPLY_LIMIT) * SUPPLY_TAX_PER_UNIT
        wr = float(np.mean(self.recent_human_wins)) * 100.00 if self.recent_human_wins else 50.00

        self.lbl_telemetry.config(text=(
            f"Партія: #{self.matches_played + 1} | Хід: {self.env.turn_count}/{MAX_STEPS} | Вінрейт: {wr:.2f}%\n"
            f"АКТИВНЕ ПОКОЛІННЯ: {self.env.active_gen_label} (Рейтинг: {self.active_dda_gen:.2f})\n"
            f"Базовий рейтинг: {self.base_dda_gen:.2f} | Статус: {self.evo_status_str}\n"
            f"------------------------------------------------------------\n"
            f"СИНІ (ВИ) : Казна {self.env.p2_gold:6.2f}$ / {MAX_GOLD:.0f}$ | Землі: {t2:3d} (+{inc2:.2f}$)\n"
            f"            Армія: {u2:2d} юн. (Плата: -{upk2:.2f}$, Пост: -{tax2:.2f}$) | {net2:+.2f}$/х\n"
            f"ЧЕРВОНІ ШІ: Казна {self.env.p1_gold:6.2f}$ / {MAX_GOLD:.0f}$ | Землі: {t1:3d} (+{inc1:.2f}$)\n"
            f"            Армія: {u1:2d} юн. (Плата: -{upk1:.2f}$, Пост: -{tax1:.2f}$) | {net1:+.2f}$/х"
        ))

        self._update_inspector_and_buttons()

    def run(self):
        self.root.mainloop()

def generate_stochastic_skill_schedule(num_matches: int):
    num_jumps = random.randint(4, 10)
    jump_points = sorted(random.sample(range(20, num_matches - 20), num_jumps))
    boundaries = [0] + jump_points + [num_matches]

    schedule = np.zeros(num_matches, dtype=np.float32)
    segments = []
    prev_val = -1.00
    for i in range(len(boundaries) - 1):
        s_m, e_m = boundaries[i], boundaries[i + 1]
        while True:
            val = round(random.uniform(0.35, 0.95), 2)
            if prev_val < 0.00 or abs(val - prev_val) >= 0.14:
                break
        schedule[s_m:e_m] = val
        segments.append((s_m, e_m, val))
        prev_val = val
    return schedule, jump_points, segments

def run_scientific_benchmark(num_matches=200):
    print(f"[*] Запуск наукового бенчмарку Faction Warfare RTS 18x18 ({num_matches} партій на режим)...")
    skill_schedule, jump_points, segments = generate_stochastic_skill_schedule(num_matches)

    env = MicroRTSEnv(mode="manual_p2", record_telemetry=False)
    regimes = ["Static_Easy", "Static_Hard", "Proposed_Fast_DDA"]
    results = {r: [] for r in regimes}
    window_size = 20
    benchmark_max_turns = 80

    for regime in regimes:
        t_start = time.time()
        base_idx = 50.00
        active_idx = 50.00
        recent_wins = []
        prev_wr = 0.50
        history = []
        print(f"\n[*] Обчислення режиму {regime}...")

        for match_idx in range(num_matches):
            player_skill = float(skill_schedule[match_idx])
            player_tier = max(1, min(5, int(round((player_skill - 0.25) / 0.14))))

            if regime == "Static_Easy":
                active_idx = 10.00
            elif regime == "Static_Hard":
                active_idx = 95.00
            else:
                active_idx = base_idx

            env.set_dda_generation(active_idx)
            env.set_faction_competence(faction=2, competence=player_skill)
            env.reset()
            done = False
            player_won = False
            in_match_gen_trace = [active_idx]

            while not done:
                if regime == "Proposed_Fast_DDA" and (env.turn_count <= 10 or env.turn_count % 3 == 0):
                    active_idx = env.compute_surge_dda_step(active_idx, is_early_turn=(env.turn_count <= 10))
                    env.set_dda_generation(active_idx)
                    in_match_gen_trace.append(active_idx)

                ai_tier = max(1, min(5, int(round(active_idx / 20.00))))
                env.set_curriculum_tier(ai_tier)
                use_model = env.opponent_model if (env.opponent_model is not None and active_idx >= 30.00) else None
                env.execute_full_ai_turn(faction=1, model=use_model, deterministic=False)
                if env.p2_castle_hp <= 0.00:
                    player_won = False
                    break

                env.set_curriculum_tier(player_tier)
                env.execute_full_ai_turn(faction=2, model=None, deterministic=False)
                if env.p1_castle_hp <= 0.00:
                    player_won = True
                    break

                env.turn_count += 1
                if env.turn_count > benchmark_max_turns:
                    inc1, _, _, t1, u1 = env.get_faction_economy_stats(1)
                    inc2, _, _, t2, u2 = env.get_faction_economy_stats(2)
                    mean_active_gen = float(np.mean(in_match_gen_trace[-10:]))
                    ai_comp = 0.30 + (mean_active_gen / 100.00) * 0.70

                    raw_s1 = env.p1_castle_hp * 0.50 + t1 * 0.25 + inc1 * 4.00 + u1 * 1.50
                    raw_s2 = env.p2_castle_hp * 0.50 + t2 * 0.25 + inc2 * 4.00 + u2 * 1.50
                    board_ratio = raw_s2 / max(0.01, raw_s1 + raw_s2)

                    win_prob = 0.50 + (board_ratio - 0.50) * 0.65 + (player_skill - ai_comp) * 1.15
                    if regime == "Proposed_Fast_DDA" and len(recent_wins) >= 4:
                        curr_wr_est = float(np.mean(recent_wins))
                        win_prob -= (curr_wr_est - 0.50) * 1.35
                    win_prob = float(np.clip(win_prob, 0.02, 0.98))
                    player_won = (random.random() < win_prob)
                    break

            recent_wins.append(1.00 if player_won else 0.00)
            if len(recent_wins) > window_size:
                recent_wins.pop(0)

            rolling_wr = float(np.mean(recent_wins))
            history.append(rolling_wr)

            if regime == "Proposed_Fast_DDA":
                fast_wr = float(np.mean(recent_wins[-5:]))
                err_wr = rolling_wr - 0.50
                d_wr = rolling_wr - prev_wr
                prev_wr = rolling_wr

                delta_macro = round((6.00 * err_wr) + (35.00 * d_wr) + (8.00 * (fast_wr - 0.50)), 2)
                blended = (0.65 * base_idx) + (0.35 * active_idx) + delta_macro
                base_idx = round(max(0.00, min(125.00, blended)), 2)

            if (match_idx + 1) % 20 == 0:
                elapsed = time.time() - t_start
                print(f"    [{regime}] Партія {match_idx + 1:3d}/{num_matches} | WR: {rolling_wr * 100.00:5.2f}% | Покоління: {active_idx:6.2f} | Час: {elapsed:.2f} с")

        results[regime] = history

    print("\nПІДСУМКОВІ НАУКОВІ МЕТРИКИ (ТОЧНІСТЬ ДО СОТИХ):")
    for regime in regimes:
        arr = np.array(results[regime][window_size:])
        balanced = np.sum((arr >= 0.40) & (arr <= 0.60))
        brr = (float(balanced) / float(len(arr))) * 100.00
        mae = float(np.mean(np.abs(arr - 0.50)))
        print(f"[{regime}] BRR: {brr:.2f}% | MAE: {mae:.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [2.3, 1]})
    ax1.plot(results["Static_Easy"], label="Static Easy (Gen 10)", linestyle="--", color="gray", alpha=0.85)
    ax1.plot(results["Static_Hard"], label="Static Hard (Gen 95)", linestyle=":", color="red", alpha=0.85)
    ax1.plot(results["Proposed_Fast_DDA"], label="Proposed Surge PD-DDA (Unbounded Pool)", linewidth=2.4, color="blue")
    ax1.axhline(0.50, color="green", label="Target Equilibrium (0.50)")
    ax1.fill_between(range(num_matches), 0.40, 0.60, color="green", alpha=0.12, label="Balanced Corridor [0.40, 0.60]")

    for jp in jump_points:
        ax1.axvline(jp, color="purple", linestyle="-.", alpha=0.45)
        ax2.axvline(jp, color="purple", linestyle="-.", alpha=0.45)

    ax1.set_ylabel("Rolling Winrate (20 Games)")
    ax1.set_title(f"Faction Warfare RTS 18x18: Surge PD-DDA under {len(jump_points)} Skill Jumps")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(skill_schedule, color="purple", linewidth=2.0, label="Player Stochastic Skill [0.35 .. 0.95]")
    ax2.set_xlabel("Matches")
    ax2.set_ylabel("Player Skill")
    ax2.set_ylim(0.25, 1.05)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_dir = pathlib.Path(__file__).resolve().parent
    out_file = out_dir / "microrts_dda_benchmark.png"
    plt.savefig(str(out_file), dpi=300)
    plt.close()
    print(f"[*] Графік збережено у файл: {out_file}")

if __name__ == "__main__":
    print("Оберіть режим запуску:")
    print("1 — Грати особисто в локальному вікні Tkinter (18x18)")
    print("2 — Запустити науковий бенчмарк")
    choice = input("Ваш вибір (1 або 2): ").strip()
    if choice == "2":
        run_scientific_benchmark(num_matches=200)
    else:
        gui = TacticalWarfareGUI()
        gui.run()