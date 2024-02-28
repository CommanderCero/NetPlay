import numpy as np

import nle.nethack as nh
from netplay.nethack_utils import monster as MON
from netplay.nethack_utils import screen_symbols as SS

# --- FEATURES ---
STAIRCASES = [SS.S_upstair, SS.S_dnstair]
LADDERS = [SS.S_upladder, SS.S_dnladder]
ALTARS = [SS.S_altar]
SINKS = [SS.S_sink]
FOUNTAINS = [SS.S_fountain]
TREES = [SS.S_tree]
TRAPS = frozenset({
    SS.S_arrow_trap, SS.S_dart_trap, SS.S_falling_rock_trap, SS.S_squeaky_board, SS.S_bear_trap,
    SS.S_land_mine, SS.S_rolling_boulder_trap, SS.S_sleeping_gas_trap, SS.S_rust_trap,
    SS.S_fire_trap, SS.S_pit, SS.S_spiked_pit, SS.S_hole, SS.S_trap_door, SS.S_teleportation_trap,
    SS.S_level_teleporter, SS.S_magic_portal, SS.S_web, SS.S_statue_trap, SS.S_magic_trap,
    SS.S_anti_magic_trap, SS.S_polymorph_trap
})
OPENED_DOORS = [SS.S_vodoor, SS.S_hodoor]
CLOSED_DOORS = [SS.S_vcdoor, SS.S_hcdoor]
DOORS = [*OPENED_DOORS, *CLOSED_DOORS]
DOORWAYS = [SS.S_ndoor]
WALLS = [
    SS.S_vwall, SS.S_hwall, SS.S_tlcorn, SS.S_trcorn, SS.S_blcorn, SS.S_brcorn,
    SS.S_crwall, SS.S_tuwall, SS.S_tdwall, SS.S_tlwall, SS.S_trwall
]
OPENED_DRAWBRIDGES = [SS.S_vodbridge, SS.S_hodbridge]
CLOSED_DRAWBRIDGES = [SS.S_vcdbridge, SS.S_hcdbridge]
DRAWBRIDGES = [*OPENED_DRAWBRIDGES, *CLOSED_DRAWBRIDGES]
IRON_BARS = [SS.S_bars]
FLOORS = [SS.S_room, SS.S_darkroom]
CORRIDORS = [SS.S_corr, SS.S_litcorr]
THRONES = [SS.S_throne]
GRAVES = [SS.S_grave]
WATER = [SS.S_water]
ICE = [SS.S_ice]
LAVA = [SS.S_lava]
CLOUD = [SS.S_cloud]
AIR = [SS.S_air]
ROCKS = [SS.S_stone]
POOLS = [SS.S_pool]

FEATURES = [
    *STAIRCASES,
    *LADDERS,
    *ALTARS,
    *SINKS,
    *FOUNTAINS,
    *TREES,
    *TRAPS,
    *DOORS,
    *DOORWAYS,
    *WALLS,
    *DRAWBRIDGES,
    *IRON_BARS,
    *FLOORS,
    *CORRIDORS,
    *THRONES,
    *GRAVES,
    *WATER,
    *ICE,
    *LAVA,
    *CLOUD,
    *AIR,
    *ROCKS
]

WALKABLE_FEATURES = [
    *STAIRCASES,
    *FLOORS,
    *CORRIDORS,
    *OPENED_DOORS,
    *OPENED_DRAWBRIDGES,
    *ALTARS,
    *FOUNTAINS,
    *SINKS,
    *DOORWAYS
]

# --- MONSTERS ---
MONSTERS = [nh.GLYPH_MON_OFF + i for i in range(nh.NUMMONS)]
PETS = [nh.GLYPH_PET_OFF + i for i in range(nh.NUMMONS)]

# --- ITEMS / OBJECTS ---
OBJECTS = [nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS)]
POTIONS = [nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.POTION_CLASS]
FOODS = [nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.FOOD_CLASS]
BOULDERS = [nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.ROCK_CLASS]

BODIES = [nh.GLYPH_BODY_OFF + i for i in range(nh.NUMMONS)]



def is_dungeon_feature(glyphs: np.array) -> np.array:
    return np.isin(glyphs, FEATURES)

def is_monster(glyphs: np.array) -> np.array:
    return nh.glyph_is_monster(glyphs)

def is_object(glyphs: np.array) -> np.array:
    return nh.glyph_is_object(glyphs)