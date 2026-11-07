# File for easy access to everything, that need to be fast cofigurated in a process of developing, improving game.

import os

GRID_SIZE = 18
MAX_STEPS = 300
MAX_GOLD = 100.00
HEAL_PCT = 0.25
SUPPLY_LIMIT = 8
SUPPLY_TAX_PER_UNIT = 0.15
CASTLE_MAX_HP = 25.00
START_GOLD = 12.00

MIN_TRAINING_EFFICIENCY = 100.00
MIN_BARRACKS_CASTLE_DIST = 6
MAX_SLOW_UNIT_FRONTLINE_DIST = 10
DISBAND_REFUND_RATIO = 0.40
DEMOLISH_REFUND_RATIO = 0.50

EFF_CAP_NEUTRAL_BASE = 4.50
EFF_CAP_ENEMY_BASE = 7.50
EFF_CAP_YIELD_MULT = 18.00
EFF_DMG_MULT = 4.50
EFF_KILL_UNIT_BONUS = 14.00
EFF_KILL_STRUCT_BONUS = 12.00
EFF_CASTLE_HIT_BONUS = 6.00
EFF_HARVEST_MULT = 12.00
EFF_BUILD_BONUS = 10.00
EFF_HEAL_MULT = 1.50

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
    'worker':  {
        'name': 'Worker',
        'name_uk': 'Робітник',
        'cost': 3.00,
        'upkeep': 0.40,
        'max_hp': 4.00,
        'move_range': 2,
        'atk_range': 1,
        'dmg': 1.00
    },
    'warrior': {
        'name': 'Warrior',
        'name_uk': 'Піхотинець',
        'cost': 6.00,
        'upkeep': 1.10,
        'max_hp': 8.00,
        'move_range': 2,
        'atk_range': 1,
        'dmg': 3.00
    },
    'archer':  {
        'name': 'Archer',
        'name_uk': 'Стрілець',
        'cost': 8.50,
        'upkeep': 1.60,
        'max_hp': 4.50,
        'move_range': 1,
        'atk_range': 2,
        'dmg': 2.25
    }
}

STRUCT_STATS = {
    1: {
        'name': 'Defense Tower',
        'name_uk': 'Оборонна Вежа',
        'cost': 4.00,
        'max_hp': 8.00,
        'atk_range': 2,
        'dmg': 2.00
    },
    2: {
        'name': 'Forward Barracks',
        'name_uk': 'Передова Казарма',
        'cost': 5.00,
        'max_hp': 10.00,
        'atk_range': 0,
        'dmg': 0.00
    }
}

RELATIVE_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]