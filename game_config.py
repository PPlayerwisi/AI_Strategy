import os

GRID_SIZE = 18
MAX_STEPS = 300
MAX_GOLD = 100.00
INITIAL_GOLD = 12.00
HEAL_PCT = 0.25
SUPPLY_LIMIT = 8
SUPPLY_TAX_PER_UNIT = 0.15
CASTLE_MAX_HP = 25.00

MIN_TRAINING_EFFICIENCY = 100.00
DISBAND_REFUND_RATIO = 0.40
DEMOLISH_REFUND_RATIO = 0.50
MIN_BARRACKS_CASTLE_DIST = 7
MAX_SLOW_UNIT_FRONTLINE_DIST = 10
OSCILLATION_MEMORY_LEN = 6
OSCILLATION_PENALTY = 40.00

FACTION_CASTLE_POSITIONS = {
    1: (GRID_SIZE - 1, 0),
    2: (0, GRID_SIZE - 1),
    3: (0, 0),
    4: (GRID_SIZE - 1, GRID_SIZE - 1)
}

FACTION_COLORS = {
    1: {"bg": "#632227", "unit": "#eb3b5a", "struct": "#c23616", "castle": "#83142c"},
    2: {"bg": "#1b3b6f", "unit": "#0fbcf9", "struct": "#0097e6", "castle": "#0a3d62"},
    3: {"bg": "#1e5128", "unit": "#2ecc71", "struct": "#27ae60", "castle": "#14361b"},
    4: {"bg": "#6e511b", "unit": "#f1c40f", "struct": "#f39c12", "castle": "#4d3812"}
}

EFFICIENCY_REWARDS = {
    "neutral_capture_base": 3.50,
    "neutral_capture_yield_mult": 12.00,
    "enemy_capture_base": 5.50,
    "enemy_capture_yield_mult": 14.00,
    "damage_dealt_mult": 3.20,
    "unit_kill_bonus": 8.00,
    "struct_kill_bonus": 7.00,
    "castle_hit_bonus": 4.50,
    "harvest_bonus_mult": 6.50,
    "build_struct_bonus": 8.00,
    "heal_mult": 0.80,
    "early_death_penalty": 5.00
}

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
    "worker": {
        "cost": 3.00,
        "upkeep": 0.40,
        "max_hp": 4.00,
        "move_range": 2,
        "atk_range": 1,
        "dmg": 1.00
    },
    "warrior": {
        "cost": 6.00,
        "upkeep": 1.10,
        "max_hp": 8.00,
        "move_range": 2,
        "atk_range": 1,
        "dmg": 3.00
    },
    "archer": {
        "cost": 8.50,
        "upkeep": 1.60,
        "max_hp": 4.50,
        "move_range": 1,
        "atk_range": 2,
        "dmg": 2.25
    }
}

STRUCT_STATS = {
    1: {
        "cost": 4.00,
        "max_hp": 8.00,
        "atk_range": 2,
        "dmg": 2.00
    },
    2: {
        "cost": 5.00,
        "max_hp": 10.00,
        "atk_range": 0,
        "dmg": 0.00
    }
}

RELATIVE_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]