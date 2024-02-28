import nle.nethack as nh

import warnings
from collections import namedtuple

CMAP_LOOKUP = [
    "dark area",
    "vertical wall",
    "horizontal wall",
    "northwest room corner",
    "northeast room corner",
    "southwest corner",
    "southeast corner",
    "cross wall",
    "t up wall",
    "t down wall",
    "t west wall",
    "t east wall",
    "doorway",
    "vertical open door",
    "horizontal open door",
    "vertical closed door",
    "horizontal closed door",
    "bars",
    "tree",
    "room floor",
    "dark room floor",
    "corridor floor",
    "lit corridor floor",
    "stairs up",
    "stairs down",
    "ladder up",
    "ladder down",
    "alter",
    "grave",
    "throne",
    "sink",
    "fountain",
    "pool",
    "ice",
    "lava",
    "vertical lowered drawbridge",
    "horizontal lowered drawbridge",
    "vertical raised drawbridge",
    "horizontal raised drawbridge",
    "air floor",
    "cloud floor",
    "water floor",
    "arrow trap",
    "dart trap",
    "falling rock trap",
    "squeaky board",
    "bear trap",
    "land mine",
    "rolling boulder trap",
    "sleeping gas trap",
    "rust trap",
    "fire trap",
    "pit",
    "spiked pit",
    "hole",
    "trap door",
    "teleportation trap",
    "level teleporter",
    "magic portal",
    "web",
    "statue trap",
    "magic trap",
    "anti magic trap",
    "polymorph trap",
    "vibrating square",
    "vertical beam",
    "horizontal beam",
    "left slant beam",
    "right slant beam",
    "dig beam",
    "flash beam",
    "boom left",
    "boom right",
    "shield 1",
    "shield 2",
    "shield 3",
    "shield 4",
    "poison cloud",
    "valid position",
    "swallow top left",
    "swallow top center",
    "swallow top right",
    "swallow middle left",
    "swallow middle right",
    "swallow bottom left",
    "swallow bottom center",
    "swallow bottom right",
    "explosion top left",
    "explosion top center",
    "explosion top right",
    "explosion middle left",
    "explosion middle center",
    "explosion middle right",
    "explosion bottom left",
    "explosion bottom center",
    "explosion bottom right",
    "MAXPCHARS",
]

def offset_to_compass(x, y):
    if x == 0 and y == 0:
        return "under you"
    
    if x == 0:
        if y > 0:
            return "south"
        else:
            return "north"
    if y == 0:
        if x > 0:
            return "east"
        else:
            return "west"
    if x > 0:
        if y > 0:
            return "south-east"
        else:
            return "north-east"
    if y > 0:
        return "south-west"
    else:
        return "north-west"

def describe_object_glyph(glyph: int):
    obj_id = nh.glyph_to_obj(glyph)
    obj_class = ord(nh.objclass(obj_id).oc_class) # Returns an char, but it is a enum
    obj_name = nh.objdescr.from_idx(obj_id).oc_name
    obj_description = nh.objdescr.from_idx(obj_id).oc_descr
    name_or_description = obj_name if obj_description is None else obj_description
    
    if obj_class == nh.ILLOBJ_CLASS:
        return obj_name
    if obj_class == nh.WEAPON_CLASS:
        return obj_name
    if obj_class == nh.ARMOR_CLASS:
        return name_or_description
    if obj_class == nh.RING_CLASS:
        return f"{obj_description} ring"
    if obj_class == nh.AMULET_CLASS:
        if "Amulet" in obj_description:
            return obj_description # Amulet of Yendor special case
        else:
            return f"{obj_description} amulet"
    if obj_class == nh.TOOL_CLASS:
        return name_or_description
    if obj_class == nh.FOOD_CLASS:
        return obj_name
    if obj_class == nh.POTION_CLASS:
        return f"{obj_description} potion"
    if obj_class == nh.SCROLL_CLASS:
        return f"scroll labeled {obj_description}"
    if obj_class == nh.SPBOOK_CLASS:
        return f"{obj_description} spellbook"
    if obj_class == nh.WAND_CLASS:
        return f"{obj_description} wand"
    if obj_class == nh.COIN_CLASS:
        return obj_name
    if obj_class == nh.GEM_CLASS:
        if obj_description is None or obj_description in obj_name:
            return obj_name
        else:
            return f"{obj_description} {obj_name}"
    if obj_class == nh.ROCK_CLASS:
        return obj_name
    if obj_class == nh.BALL_CLASS:
        return obj_name
    if obj_class == nh.CHAIN_CLASS:
        return obj_name
    if obj_class == nh.VENOM_CLASS:
        return f"splash of {obj_name}"
    
    warnings.warn("Tried describing unknown object {glyph}")
    return ""

def describe_glyph(glyph: int):
    if nh.glyph_is_statue(glyph):
        monster = nh.permonst(nh.glyph_to_mon(glyph))
        return f"{monster.mname} statue"
    if nh.glyph_is_warning(glyph):
        warning_id = nh.glyph_to_warning(glyph)
        # Taken from def_warnsyms in nle
        warning_descriptions = [
            "unknown creature causing you worry",
            "unknown creature causing you concern", 
            "unknown creature causing you anxiety", 
            "unknown creature causing you disquiet",
            "unknown creature causing you alarm",
            "unknown creature causing you dread"
        ]
        return warning_descriptions[warning_id]
    if nh.glyph_is_swallow(glyph):
        swallow_id = nh.glyph_to_swallow(glyph)
        # Taken from build_fullscreen_view_glyph_map in nle-language-wrapper
        swallow_descriptions = [
          "swallow top left",      "swallow top center",
          "swallow top right",     "swallow middle left",
          "swallow middle right",  "swallow bottom left",
          "swallow bottom center", "swallow bottom right",
        ]
        return swallow_descriptions[swallow_id]
    # There is no nh.glyph_is_zap or nh.glyph_to_zap
    # So this code does the same as build_fullscreen_view_glyph_map in nle-language-wrapper
    if glyph >= nh.GLYPH_ZAP_OFF: 
        id = (glyph - nh.GLYPH_ZAP_OFF) % nh.NUM_ZAP
        zap_descriptions = [
          "horizontal zap beam", "vertical zap beam",
          "left slant zap beam", "right slant zap beam"
        ]
        return zap_descriptions[id]
    # There is no nh.glyph_is_explode or nh.glyph_to_explode
    # So this code does the same as build_fullscreen_view_glyph_map in nle-language-wrapper
    if glyph >= nh.GLYPH_EXPLODE_OFF: 
        id = (glyph - nh.GLYPH_EXPLODE_OFF) % nh.EXPL_MAX
        explode_descriptions = [
          "explosion top left",      "explosion top center",
          "explosion top right",     "explosion middle left",
          "explosion middle center", "explosion middle right",
          "explosion bottom left",   "explosion bottom center",
          "explosion bottom right",
        ]
        return explode_descriptions[id]
    if nh.glyph_is_cmap(glyph):
        return CMAP_LOOKUP[nh.glyph_to_cmap(glyph)]
    if nh.glyph_is_normal_object(glyph):
        return describe_object_glyph(glyph)
    if nh.glyph_is_ridden_monster(glyph):
        monster = nh.permonst(nh.glyph_to_mon(glyph))
        return f"ridden {monster.mname}"
    if nh.glyph_is_body(glyph):
        # There is no method for converting a body glyph to the corresponding monster
        # So we do the same as build_fullscreen_view_glyph_map in nle-language-wrapper
        monster = nh.permonst(nh.glyph_to_mon(glyph - nh.GLYPH_BODY_OFF))
        return f"{monster.mname} corpse"
    if nh.glyph_is_detected_monster(glyph):
        monster = nh.permonst(nh.glyph_to_mon(glyph))
        return f"detected {monster.mname}"
    if nh.glyph_is_invisible(glyph):
        return "invisible creature"
    if nh.glyph_is_pet(glyph):
        monster = nh.permonst(nh.glyph_to_mon(glyph))
        return f"tame {monster.mname}"
    if nh.glyph_is_normal_monster(glyph):
        monster = nh.permonst(nh.glyph_to_mon(glyph))
        return monster.mname
    
    warnings.warn("No description for unidentified glyph {glyph}")
    return None