import random
import numpy as np

from game_config import (
    MAX_GOLD, SUPPLY_LIMIT, SUPPLY_TAX_PER_UNIT,
    MIN_WORKERS_PER_FACTION, MAX_WORKERS_PER_FACTION,
    AI_BUILD_LOCK_TURNS, AUTO_SPAWN_NET_INCOME_THRESHOLD,
    MIN_GOLD_RESERVE_FOR_DEFICIT_SPAWN,
    DEFICIT_RUNWAY_TURNS, EMERGENCY_RECOVERY_TARGET_UNITS,
    MIN_BARRACKS_CASTLE_DIST, MAX_SLOW_UNIT_FRONTLINE_DIST,
    OSCILLATION_PENALTY, UNIT_STATS, STRUCT_STATS,
    RELATIVE_OFFSETS, ROLE_IDS
)
from ai_pool import (
    transform_action_for_faction, infer_role_id_from_obs,
    predict_masked_action
)

class AIStrategistMixin:
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

    def _distance_to_nearest_non_allied_tile(self, pos: tuple, faction: int) -> int:
        f = int(faction)
        r, c = int(pos[0]), int(pos[1])
        mask = (self.terrain != 0) & (self.territory != f)
        if not np.any(mask):
            enemy_castle = self._get_primary_enemy_castle(f, pos)
            return abs(r - enemy_castle[0]) + abs(c - enemy_castle[1])
        dists = np.abs(self._grid_r[mask] - r) + np.abs(self._grid_c[mask] - c)
        return int(np.min(dists))

    def _prevent_bankruptcy_by_disbanding(self, faction: int):
        f = int(faction)
        if not self.is_faction_alive(f):
            return

        safety = 0
        while safety < 30:
            safety += 1
            curr_gold = self.get_faction_gold(f)
            _, _, net_inc, _, _ = self.get_faction_economy_stats(f)
            if curr_gold + net_inc >= 0.00:
                break

            combat_candidates = [
                (pos, u) for pos, u in self.units.items()
                if u['faction'] == f and u['type'] in ('archer', 'warrior')
            ]
            if not combat_candidates:
                worker_candidates = [
                    (pos, u) for pos, u in self.units.items()
                    if u['faction'] == f and u['type'] == 'worker'
                ]
                if len(worker_candidates) <= MIN_WORKERS_PER_FACTION:
                    break
                combat_candidates = worker_candidates

            def _disband_sort_key(item):
                pos, u = item
                type_prio = 0 if u['type'] == 'archer' else (1 if u['type'] == 'warrior' else 2)
                hp_ratio = round(float(u['hp']) / max(0.01, float(u['max_hp'])), 2)
                dist_front = self._distance_to_nearest_non_allied_tile(pos, f)
                return (type_prio, hp_ratio, -dist_front, float(u.get('efficiency', 0.00)))

            worst_pos, worst_u = min(combat_candidates, key=_disband_sort_key)
            u_type = worst_u['type']
            refund = self.execute_unit_disband(worst_pos)
            if refund > 0.00:
                self.add_localized_event("ai_disband", u_type=u_type, refund=refund)
            else:
                break

    def _assess_strategic_situation(self, faction: int) -> dict:
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        my_castle = self.castle_positions[f]

        enemy_combat_positions = []
        enemy_all_positions = []
        my_combat_positions = []
        my_worker_positions = []
        closest_castle_dist = 999

        for pos, u in self.units.items():
            if u['faction'] in enemies:
                enemy_all_positions.append(pos)
                d = abs(pos[0] - my_castle[0]) + abs(pos[1] - my_castle[1])
                if d < closest_castle_dist:
                    closest_castle_dist = d
                if u['type'] != 'worker':
                    enemy_combat_positions.append(pos)
            elif u['faction'] == f:
                if u['type'] == 'worker':
                    my_worker_positions.append(pos)
                else:
                    my_combat_positions.append(pos)

        min_inter_unit_dist = 999
        for mr, mc in my_combat_positions:
            for er, ec in enemy_all_positions:
                dist = abs(mr - er) + abs(mc - ec)
                if dist < min_inter_unit_dist:
                    min_inter_unit_dist = dist

        first_contact = (min_inter_unit_dist <= 8 or closest_castle_dist <= 11)
        under_pressure = (
            closest_castle_dist <= 8
            or (first_contact and len(enemy_combat_positions) > len(my_combat_positions) + 1)
            or (len(my_combat_positions) == 0 and closest_castle_dist <= 12)
        )
        enemy_retreating = (len(my_combat_positions) >= len(enemy_combat_positions) + 2 and closest_castle_dist > 14)

        return {
            'first_contact': bool(first_contact),
            'under_pressure': bool(under_pressure),
            'enemy_retreating': bool(enemy_retreating),
            'min_contact_dist': int(min_inter_unit_dist),
            'closest_enemy_dist': int(closest_castle_dist),
            'my_combat_count': len(my_combat_positions),
            'my_worker_count': len(my_worker_positions),
            'my_total_units': len(my_combat_positions) + len(my_worker_positions),
            'enemy_combat_count': len(enemy_combat_positions),
            'enemy_total_units': len(enemy_all_positions)
        }

    def _can_recover_economy_from_zero(self, faction: int, spawn_site: tuple = None) -> tuple:
        f = int(faction)
        my_castle = spawn_site if spawn_site is not None else self.castle_positions[f]
        sit = self._assess_strategic_situation(f)
        gold = self.get_faction_gold(f)

        if sit['closest_enemy_dist'] <= 6:
            return False, 0.00

        eff_yields = self.get_effective_yield_map()
        r0, c0 = int(my_castle[0]), int(my_castle[1])
        dists = np.abs(self._grid_r - r0) + np.abs(self._grid_c - c0)
        reachable_mask = (dists <= 7) & (self.terrain != 0) & (self.territory != f)

        avail_yields = np.sort(eff_yields[reachable_mask])[::-1]
        potential_gain = float(np.sum(avail_yields[:8]))
        _, _, current_net, _, _ = self.get_faction_economy_stats(f)

        two_workers_upkeep = 2.00 * float(UNIT_STATS['worker']['upkeep'])
        projected_recovery_net = round(current_net + potential_gain - two_workers_upkeep, 2)
        min_bundle_cost = (2.00 * float(UNIT_STATS['worker']['cost'])) + float(UNIT_STATS['warrior']['cost'])

        can_recover = (
            potential_gain >= two_workers_upkeep
            and (gold >= min_bundle_cost or projected_recovery_net >= 0.05)
        )
        return bool(can_recover), float(projected_recovery_net)

    def _is_recruitment_safe(self, faction: int, new_upkeep: float, unit_cost: float, first_contact: bool, is_combat_unit: bool = True) -> bool:
        f = int(faction)
        gold = self.get_faction_gold(f)
        remaining_gold = round(gold - unit_cost, 2)
        if remaining_gold < 0.00:
            return False

        _, _, net_inc, _, unit_cnt = self.get_faction_economy_stats(f)
        if net_inc >= AUTO_SPAWN_NET_INCOME_THRESHOLD:
            return True

        extra_tax = SUPPLY_TAX_PER_UNIT if unit_cnt >= SUPPLY_LIMIT else 0.00
        projected_net = round(net_inc - new_upkeep - extra_tax, 2)
        gold_after_runway = round(remaining_gold + (float(DEFICIT_RUNWAY_TURNS) * projected_net), 2)

        if remaining_gold >= MIN_GOLD_RESERVE_FOR_DEFICIT_SPAWN and gold_after_runway >= 5.00:
            return True

        if projected_net >= 0.15:
            return True

        if is_combat_unit and (first_contact or unit_cnt <= 3):
            if projected_net >= -0.35 and (remaining_gold + projected_net >= 0.50) and gold_after_runway >= 2.00:
                return True

        return False

    def _assign_tactical_directives(self, faction: int):
        f = int(faction)
        enemies = set(self.get_enemy_factions(f))
        my_castle = self.castle_positions[f]
        enemy_castle = self._get_primary_enemy_castle(f, my_castle)

        cluster_map = self._compute_resource_cluster_map()
        eff_yields = self.get_effective_yield_map()
        sit = self._assess_strategic_situation(f)

        worker_targets = []
        combat_targets = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if int(self.terrain[r, c]) == 0 or int(self.struct_type[r, c]) == 1:
                    continue
                owner = int(self.territory[r, c])
                y_val = float(eff_yields[r, c])
                c_val = float(cluster_map[r, c])

                if owner == 0 and (y_val >= 0.15 or c_val >= 0.45):
                    worker_targets.append({
                        'pos': (r, c),
                        'role': 'capture_neutral_cluster',
                        'priority': (y_val * 48.00) + (c_val * 28.00),
                        'max_assignees': 1
                    })
                elif owner in enemies and (y_val >= 0.15 or c_val >= 0.40):
                    worker_targets.append({
                        'pos': (r, c),
                        'role': 'raid_enemy_resource',
                        'priority': (y_val * 42.00) + (c_val * 22.00),
                        'max_assignees': 1
                    })
                    combat_targets.append({
                        'pos': (r, c),
                        'role': 'raid_enemy_resource',
                        'priority': (y_val * 24.00) + (c_val * 14.00) + 12.00,
                        'max_assignees': 1
                    })
                elif owner == 0 and y_val > 0.00:
                    worker_targets.append({
                        'pos': (r, c),
                        'role': 'expand_territory',
                        'priority': (y_val * 26.00) + (c_val * 15.00),
                        'max_assignees': 1
                    })

        for e_pos, eu in self.units.items():
            if eu['faction'] in enemies:
                dist_home = abs(e_pos[0] - my_castle[0]) + abs(e_pos[1] - my_castle[1])
                urgency = 55.00 if dist_home <= 10 else 42.00
                if sit['under_pressure']:
                    urgency += 20.00
                combat_targets.append({
                    'pos': e_pos,
                    'role': 'eliminate_enemy_unit',
                    'priority': urgency + float(UNIT_STATS[eu['type']]['cost']) * 3.00 + (10.00 - min(9.00, eu['hp'])),
                    'max_assignees': 3
                })

        enemy_struct_coords = np.argwhere(np.isin(self.struct_owner, list(enemies)) & (self.struct_type != 0))
        for sr, sc in enemy_struct_coords:
            s_pos = (int(sr), int(sc))
            combat_targets.append({
                'pos': s_pos,
                'role': 'destroy_enemy_struct',
                'priority': 40.00,
                'max_assignees': 2
            })

        for ef in enemies:
            combat_targets.append({
                'pos': self.castle_positions[ef],
                'role': 'siege_enemy_castle',
                'priority': 46.00 if (sit['enemy_retreating'] or sit['my_combat_count'] >= 3) else 32.00,
                'max_assignees': 4
            })

        assigned_counts = {}
        my_units = [(pos, u) for pos, u in sorted(self.units.items()) if u['faction'] == f and u['ready']]
        for pos, u in my_units:
            u_type = u['type']
            pool = worker_targets if u_type == 'worker' else combat_targets
            if not pool:
                pool = combat_targets if combat_targets else worker_targets

            best_target = enemy_castle
            best_role = 'expand_territory' if u_type == 'worker' else 'siege_enemy_castle'
            best_utility = -9999.00

            for cand in pool:
                t_pos = cand['pos']
                role = cand['role']
                used = assigned_counts.get((role, t_pos), 0)
                if used >= cand['max_assignees']:
                    continue

                manhattan_d = abs(pos[0] - t_pos[0]) + abs(pos[1] - t_pos[1])
                crowd_penalty = used * (18.00 if u_type == 'worker' else 8.00)
                utility = cand['priority'] - (manhattan_d * 2.20) - crowd_penalty
                if utility > best_utility:
                    best_utility = utility
                    best_target = t_pos
                    best_role = role

            assigned_counts[(best_role, best_target)] = assigned_counts.get((best_role, best_target), 0) + 1
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
                if dist <= 3:
                    clump += (4.00 - float(dist)) / 3.00
        return clump

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

        if u_type == 'worker':
            p_fertility = self._compute_fertility_attraction(nr, nc, f)
            c_density = float(cluster_map[nr, nc]) if int(self.territory[nr, nc]) != f else float(cluster_map[nr, nc]) * 0.20
            d_clump = self._compute_allied_clump_penalty(pos, nr, nc, f)
            score = (
                (v_double * 38.00)
                + (c_density * 20.00)
                + (p_fertility * 18.00)
                + (delta_bfs * 14.00)
                + (tiles_captured * 5.50)
                - (d_clump * 6.00)
                - (enemy_fp[nr, nc] * 16.00)
                - osc_penalty
            )
            if self.is_tile_built_up(r, c):
                score += 14.00
            return float(score)

        enemy_positions = [pos_e for pos_e, ue in self.units.items() if ue['faction'] in enemies]
        enemy_castle = self._get_primary_enemy_castle(f, pos)
        target_combat_pos = min(
            enemy_positions,
            key=lambda p: abs(p[0] - r) + abs(p[1] - c)
        ) if enemy_positions else enemy_castle

        bfs_enemy = self._get_bfs_distance_map(target_combat_pos, f)
        curr_e_dist = float(bfs_enemy[r, c])
        next_e_dist = float(bfs_enemy[nr, nc])
        if curr_e_dist < 90.00 and next_e_dist < 90.00:
            delta_enemy_dist = curr_e_dist - next_e_dist
        else:
            curr_m = abs(target_combat_pos[0] - r) + abs(target_combat_pos[1] - c)
            next_m = abs(target_combat_pos[0] - nr) + abs(target_combat_pos[1] - nc)
            delta_enemy_dist = float(curr_m - next_m)

        retreat_pen = 0.00
        if delta_enemy_dist < 0.00:
            if u_type == 'warrior':
                retreat_pen = 30.00
            elif u_type == 'archer' and curr_e_dist > 2.00:
                retreat_pen = 24.00

        range_engage_bonus = 0.00
        direct_dist = abs(target_combat_pos[0] - nr) + abs(target_combat_pos[1] - nc)
        if u_type == 'warrior' and direct_dist <= 1:
            range_engage_bonus = 24.00
        elif u_type == 'warrior' and direct_dist == 2:
            range_engage_bonus = 14.00
        elif u_type == 'archer' and direct_dist == 2:
            range_engage_bonus = 22.00

        fp_diff = float(ally_fp[nr, nc]) - (float(enemy_fp[nr, nc]) * 0.35)
        score = (
            (delta_enemy_dist * 24.00)
            + (delta_bfs * 16.00)
            + range_engage_bonus
            + (v_double * 6.00)
            + (tiles_captured * 2.00)
            + fp_diff
            - retreat_pen
            - osc_penalty
        )

        if self.is_tile_built_up(r, c):
            score += 18.00

        return float(score)

    def _compute_strategic_logit_prior(self, entity_kind: str, pos: tuple, faction: int, mask: np.ndarray) -> np.ndarray:
        priors = np.zeros(30, dtype=np.float32)
        valid_acts = np.where(mask)[0]
        if len(valid_acts) <= 1:
            return priors

        f = int(faction)
        sit = self._assess_strategic_situation(f)
        _, _, net_inc, _, _ = self.get_faction_economy_stats(f)

        if entity_kind == 'unit' and pos in self.units:
            u = self.units[pos]
            u_type = u['type']
            if self.turn_count <= 2 and u_type == 'archer' and mask[28]:
                if not self.get_valid_attack_targets(pos):
                    priors[28] = 50.00
                    return priors

            priors[28] = -35.00
            if self.turn_count <= AI_BUILD_LOCK_TURNS:
                priors[26] = -100.00
                priors[27] = -100.00
            else:
                my_barracks = int(np.sum((self.struct_owner == f) & (self.struct_type == 2)))
                my_towers = int(np.sum((self.struct_owner == f) & (self.struct_type == 1)))
                if mask[27] and my_barracks < 2 and (net_inc >= 1.20 or self.get_faction_gold(f) >= 14.00):
                    priors[27] = 24.00
                if mask[26] and my_towers < 3 and (sit['under_pressure'] or self.get_faction_gold(f) >= 16.00):
                    priors[26] = 20.00

            if mask[29]:
                dem_site = self._find_demolishable_structure_site(pos, f)
                if dem_site is not None and not self._is_valid_barracks_distance(dem_site[0], dem_site[1]) and int(self.struct_type[dem_site[0], dem_site[1]]) == 2:
                    priors[29] = 18.00
                else:
                    priors[29] = -25.00

            for a in valid_acts:
                if 0 <= a <= 11:
                    m_score = self._evaluate_unit_move_score(pos, a, f)
                    scale = 0.35 if u_type != 'worker' else 0.15
                    priors[a] = float(np.clip(m_score * scale, -18.00, 18.00))
                elif 12 <= a <= 23:
                    priors[a] = 22.00 if u_type != 'worker' else 10.00

            if u_type != 'worker' and np.any(mask[:24]):
                priors[24] = -15.00

        elif entity_kind == 'spawn':
            my_workers = sit['my_worker_count']
            my_combat = sit['my_combat_count']
            my_total = sit['my_total_units']

            if my_workers < MIN_WORKERS_PER_FACTION and mask[27]:
                priors[27] = 45.00
                priors[24] = -35.00
                return priors

            if my_workers >= MAX_WORKERS_PER_FACTION or (sit['under_pressure'] and my_combat <= sit['enemy_combat_count'] + 1 and my_workers >= MIN_WORKERS_PER_FACTION):
                priors[27] = -100.00

            if my_total == 0:
                can_rec, _ = self._can_recover_economy_from_zero(f, pos)
                if can_rec and mask[27]:
                    priors[27] = 35.00
                    priors[24] = -30.00
                elif mask[28]:
                    priors[28] = 40.00
                    priors[27] = -50.00
                    priors[24] = -30.00
                return priors

            if net_inc >= AUTO_SPAWN_NET_INCOME_THRESHOLD:
                priors[24] = -40.00
                if my_workers < 2 and mask[27] and not sit['under_pressure']:
                    priors[27] = 26.00
                if mask[28]:
                    priors[28] = 28.00
                if mask[29] and self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                    priors[29] = 25.00
            elif sit['under_pressure'] or my_combat < sit['enemy_combat_count'] + 2 or self.get_faction_gold(f) >= MIN_GOLD_RESERVE_FOR_DEFICIT_SPAWN:
                if mask[28] and self._is_recruitment_safe(f, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], sit['first_contact'], True):
                    priors[28] = 22.00
                    priors[24] = -15.00
                if mask[29] and self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST) and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], sit['first_contact'], True):
                    priors[29] = 18.00
                    priors[24] = -15.00

            if mask[29] and not self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST):
                priors[29] = -60.00

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
        enemies = set(self.get_enemy_factions(f))
        gold = self.get_faction_gold(f)
        r, c = pos

        ally_fp, enemy_fp = self._compute_firepower_maps(f)
        sit = self._assess_strategic_situation(f)
        _, _, net_inc, _, _ = self.get_faction_economy_stats(f)

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
            my_workers = sit['my_worker_count']
            my_combat = sit['my_combat_count']
            my_total = sit['my_total_units']
            my_warriors = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'warrior')
            my_archers = sum(1 for u in self.units.values() if u['faction'] == f and u['type'] == 'archer')

            if my_total < EMERGENCY_RECOVERY_TARGET_UNITS and (my_total == 0 or (my_combat == 0 and my_workers < 2)):
                can_rec, _ = self._can_recover_economy_from_zero(f, pos)
                if can_rec:
                    if my_workers < 2 and mask[27] and not sit['under_pressure']:
                        return 27
                    if mask[28]:
                        return 28
                    if mask[27]:
                        return 27
                else:
                    if my_combat == 0 and mask[28]:
                        return 28
                    if my_combat == 0 and my_workers == 0 and mask[27]:
                        return 27
                    return 24

            if my_workers < MIN_WORKERS_PER_FACTION and mask[27]:
                if not sit['under_pressure'] or my_combat >= 1 or gold < UNIT_STATS['warrior']['cost']:
                    return 27

            can_spawn_archer_here = mask[29] and self._has_enemy_territory_within(pos, f, MAX_SLOW_UNIT_FRONTLINE_DIST)

            if net_inc >= AUTO_SPAWN_NET_INCOME_THRESHOLD:
                if my_workers < 2 and mask[27] and not sit['under_pressure']:
                    return 27
                if my_workers < MAX_WORKERS_PER_FACTION and my_combat >= my_workers * 2 and mask[27] and not sit['under_pressure']:
                    return 27
                if my_warriors >= my_archers + 2 and can_spawn_archer_here:
                    return 29
                if mask[28]:
                    return 28
                if can_spawn_archer_here:
                    return 29
                if my_workers < MAX_WORKERS_PER_FACTION and mask[27]:
                    return 27

            if sit['under_pressure'] or my_combat < sit['enemy_combat_count']:
                if mask[28] and self._is_recruitment_safe(f, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], True, True):
                    return 28
                if can_spawn_archer_here and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], True, True):
                    return 29

            desired_workers = min(MAX_WORKERS_PER_FACTION, 2 if self.turn_count <= 6 else (3 if comp < 0.85 else 4))
            if (
                my_workers < desired_workers
                and mask[27]
                and not sit['under_pressure']
                and (my_combat >= 1 or self.turn_count <= 5)
            ):
                if self._is_recruitment_safe(f, UNIT_STATS['worker']['upkeep'], UNIT_STATS['worker']['cost'], sit['first_contact'], False):
                    return 27

            max_combat_cap = max(4, int(round(comp * 18.00)))
            if my_combat >= max_combat_cap and net_inc < AUTO_SPAWN_NET_INCOME_THRESHOLD:
                return 24

            want_more_troops = (
                my_combat <= sit['enemy_combat_count'] + int(round(comp * 5.00))
                or sit['under_pressure']
                or gold >= MIN_GOLD_RESERVE_FOR_DEFICIT_SPAWN + UNIT_STATS['warrior']['cost']
                or net_inc >= 1.80
            )
            if want_more_troops:
                if my_warriors >= my_archers + 2 and can_spawn_archer_here and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], sit['first_contact'], True):
                    return 29
                if mask[28] and self._is_recruitment_safe(f, UNIT_STATS['warrior']['upkeep'], UNIT_STATS['warrior']['cost'], sit['first_contact'], True):
                    return 28
                if can_spawn_archer_here and self._is_recruitment_safe(f, UNIT_STATS['archer']['upkeep'], UNIT_STATS['archer']['cost'], sit['first_contact'], True):
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
                score = 18.00
                if (tr, tc) in enemy_castles:
                    score += 48.00
                elif (tr, tc) in self.units:
                    tu = self.units[(tr, tc)]
                    if tu['hp'] <= UNIT_STATS[u_type]['dmg']:
                        score += 40.00
                    score += UNIT_STATS[tu['type']]['cost'] * 3.00
                elif int(self.struct_owner[tr, tc]) in enemies:
                    score += 30.00 if int(self.struct_type[tr, tc]) == 2 else 24.00
                if score > best_atk_score:
                    best_atk_score = score
                    best_atk = a
            return int(best_atk)

        if u_type == 'worker' and self.turn_count > AI_BUILD_LOCK_TURNS:
            if mask[29]:
                dem_site = self._find_demolishable_structure_site(pos, f)
                if dem_site is not None:
                    ds_r, ds_c = dem_site
                    if int(self.struct_type[ds_r, ds_c]) == 2 and not self._is_valid_barracks_distance(ds_r, ds_c):
                        return 29

            my_barracks = int(np.sum((self.struct_owner == f) & (self.struct_type == 2)))
            my_towers = int(np.sum((self.struct_owner == f) & (self.struct_type == 1)))

            can_afford_buildings = (net_inc >= 1.00 and gold >= 8.00) or (gold >= 16.00)
            if can_afford_buildings:
                max_barracks_target = 1 if comp < 0.60 else (2 if comp < 0.90 else 3)
                if mask[27] and my_barracks < max_barracks_target:
                    return 27
                if mask[26] and my_towers < 3 and (sit['under_pressure'] or sit['first_contact'] or gold >= 14.00):
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

    def _predict_faction_model_action(self, model, kind: str, pos: tuple, faction: int, world_mask: np.ndarray, deterministic: bool, temp: float) -> int:
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

    def execute_full_ai_turn(self, faction: int, model=None, deterministic: bool = False, clear_traces: bool = True):
        f = int(faction)
        if not self.is_faction_alive(f):
            return

        self.ai_controlled_factions.add(f)

        if clear_traces:
            self.last_ai_turn_traces = []

        self._prevent_bankruptcy_by_disbanding(f)
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
            u_type = self.units[pos]['type']
            atk_indices = [a for a in range(12, 24) if mask[a]]

            if u_type != 'worker' and len(atk_indices) > 0:
                act = self.get_strategic_teacher_action('unit', pos, f, tier=f_tier, precomputed_mask=mask)
            elif u_type == 'worker' and self.turn_count > AI_BUILD_LOCK_TURNS and (mask[26] or mask[27]):
                teacher_w_act = self.get_strategic_teacher_action('unit', pos, f, tier=f_tier, precomputed_mask=mask)
                if teacher_w_act in (26, 27):
                    act = teacher_w_act
                elif active_model is not None:
                    act = self._predict_faction_model_action(active_model, 'unit', pos, f, mask, deterministic, temp)
                else:
                    act = teacher_w_act
            elif active_model is not None:
                act = self._predict_faction_model_action(active_model, 'unit', pos, f, mask, deterministic, temp)
                if act == 24 and (self.units[pos]['hp'] >= self.units[pos]['max_hp'] or u_type != 'worker') and np.any(mask[:24]):
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
                if site in self.units or curr_gold < UNIT_STATS['worker']['cost']:
                    continue

                _, _, live_net_inc, _, _ = self.get_faction_economy_stats(f)
                mask = self.get_valid_action_mask_for_entity('spawn', site, f)
                teacher_spawn = self.get_strategic_teacher_action('spawn', site, f, tier=f_tier, precomputed_mask=mask)

                if live_net_inc >= AUTO_SPAWN_NET_INCOME_THRESHOLD:
                    act = teacher_spawn
                elif active_model is not None:
                    act = self._predict_faction_model_action(active_model, 'spawn', site, f, mask, deterministic, temp)
                    if act == 24 and teacher_spawn in (27, 28, 29) and mask[teacher_spawn]:
                        act = teacher_spawn
                    elif act == 27 and teacher_spawn in (28, 29) and mask[teacher_spawn]:
                        act = teacher_spawn
                else:
                    act = teacher_spawn

                if act in (27, 28, 29):
                    u_key = {27: 'worker', 28: 'warrior', 29: 'archer'}[act]
                    u_cost = float(UNIT_STATS[u_key]['cost'])
                    u_upk = float(UNIT_STATS[u_key]['upkeep'])
                    if (curr_gold - u_cost) + (live_net_inc - u_upk) < 0.00:
                        act = 24

                self.apply_entity_action('spawn', site, f, act, precomputed_mask=mask)

        self._prevent_bankruptcy_by_disbanding(f)
        self.heal_all_unmoved_units(f)
        moves_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "move" and t.get("faction") == f)
        atks_cnt = sum(1 for t in self.last_ai_turn_traces if t["kind"] == "attack" and t.get("faction") == f)
        if f == 1 or moves_cnt > 0 or atks_cnt > 0:
            self.add_localized_event("ai_turn_summary", turn=self.turn_count, moves=moves_cnt, atks=atks_cnt)

    def evaluate_sample_doctrine_compliance(self, obs_arr: np.ndarray, act_val: int, role_id: int = None) -> float:
        pos_plane = obs_arr[0]
        coords = np.argwhere(pos_plane > 0.50)
        if len(coords) == 0:
            return 0.50
        r, c = int(coords[0][0]), int(coords[0][1])

        rid = int(role_id) if role_id is not None else infer_role_id_from_obs(obs_arr, act_val)

        yield_plane = obs_arr[2]
        ally_terr = obs_arr[4]
        enemy_terr = obs_arr[5]
        neutral_terr = obs_arr[6]
        ally_fp = obs_arr[15]
        enemy_fp = obs_arr[16]
        gold_norm = float(obs_arr[17, 0, 0])

        enemy_units_plane = obs_arr[12] + obs_arr[13] + obs_arr[14]
        enemy_coords = np.argwhere(enemy_units_plane > 0.01)

        i_econ = 0.85
        i_cluster = 0.80
        i_pos = 0.85

        if rid in (ROLE_IDS["warrior"], ROLE_IDS["archer"]):
            if 12 <= act_val <= 23:
                return 1.00
            if 0 <= act_val <= 11:
                dr, dc = RELATIVE_OFFSETS[act_val]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if len(enemy_coords) > 0:
                        d_curr = float(np.min(np.abs(enemy_coords[:, 0] - r) + np.abs(enemy_coords[:, 1] - c)))
                        d_next = float(np.min(np.abs(enemy_coords[:, 0] - nr) + np.abs(enemy_coords[:, 1] - nc)))
                        if d_next < d_curr:
                            i_pos = 1.00
                            i_cluster = 1.00
                        elif d_next > d_curr and d_curr <= 8.00:
                            i_pos = 0.15
                            i_cluster = 0.20
                    elif enemy_terr[nr, nc] > 0.50:
                        i_cluster = 0.95
            v_doc = (0.20 * i_econ) + (0.35 * i_cluster) + (0.45 * i_pos)
            return round(float(np.clip(v_doc, 0.10, 1.00)), 2)

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
                if enemy_fp[nr, nc] > ally_fp[nr, nc] + 0.15:
                    i_pos = 0.20
                else:
                    i_pos = 1.00
        elif act_val == 25:
            i_econ = 1.00 if yield_plane[r, c] >= 0.15 else 0.45
            i_cluster = 0.90
        elif act_val in (26, 27) and rid == ROLE_IDS["worker"]:
            if act_val == 27 and not self._is_valid_barracks_distance(r, c):
                i_pos = 0.15
                i_econ = 0.30
            else:
                i_econ = 0.95
                i_pos = 0.95
        elif rid == ROLE_IDS["spawn"]:
            worker_count_est = int(np.sum(obs_arr[9] > 0.01))
            if act_val == 27 and worker_count_est >= MAX_WORKERS_PER_FACTION:
                return 0.10
            if act_val in (28, 29) and gold_norm >= 0.20:
                return 1.00

        v_doc = (0.35 * i_econ) + (0.35 * i_cluster) + (0.30 * i_pos)
        return round(float(np.clip(v_doc, 0.10, 1.00)), 2)