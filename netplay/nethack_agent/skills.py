import netplay.nethack_utils.glyphs as G
import netplay.nethack_agent.describe as describe
from netplay.nethack_agent.agent import NetHackAgent
from netplay.core.skill import skill, SkillParameter, Step
from netplay.nethack_utils.nle_wrapper import RawKeyPress

from nle.env import NLE
from nle.nethack import actions, glyph_is_pet

import numpy as np

from enum import IntEnum
from functools import wraps

def fail_on_popup(skill_fn):
    @wraps(skill_fn)
    def wrapper(agent: NetHackAgent, *args, **kwargs):
        if agent.waiting_for_popup():
            yield Step.failed("Cannot handle the popup.")
            return
        
        for step in skill_fn(agent, *args, **kwargs):
            yield step
            if agent.waiting_for_popup():
                yield Step.failed("Cannot handle the popup.")
                return
    return wrapper

def direction_to_action(dir_string):
    action = {
        'n': actions.CompassDirection.N,
        'north': actions.CompassDirection.N,
        's': actions.CompassDirection.S,
        'south': actions.CompassDirection.S,
        'e': actions.CompassDirection.E,
        'east': actions.CompassDirection.E,
        'w': actions.CompassDirection.W,
        'west': actions.CompassDirection.W,
        'ne': actions.CompassDirection.NE,
        'northeast': actions.CompassDirection.NE,
        'se': actions.CompassDirection.SE,
        'southeast': actions.CompassDirection.SE,
        'nw': actions.CompassDirection.NW,
        'northwest': actions.CompassDirection.NW,
        'sw': actions.CompassDirection.SW,
        'southwest': actions.CompassDirection.SW,
        '.': actions.MiscDirection.WAIT,
        'self': actions.MiscDirection.WAIT,
    }.get(dir_string.lower().strip())

    return action

def get_move_action(from_x, from_y, to_x, to_y):
    ret = ''
    if to_y == from_y + 1: ret += 's'
    if to_y == from_y - 1: ret += 'n'
    if to_x == from_x + 1: ret += 'e'
    if to_x == from_x - 1: ret += 'w'
    if ret == '': ret = '.'

    return direction_to_action(ret)

def get_move_towards_action(agent: NetHackAgent, x: int, y: int, bump=False, avoid_monsters=False):
    if (agent.blstats.x == x and agent.blstats.y == y):
        return actions.MiscDirection.WAIT
    
    path = agent.get_path_to(x, y, bump_into_unwalkables=bump, avoid_monsters=avoid_monsters)
    if path is None:
        return None
    
    next_x, next_y = path[1] # Skip first position since its our own position
    return get_move_action(agent.blstats.x, agent.blstats.y, next_x, next_y)

@skill(
    name="set_avoid_monster_flag",
    description="If set to true skills will try to avoid monsters.",
    parameters=[
        SkillParameter.bool("value"),
    ]
)
def set_avoid_monster_flag(agent: NetHackAgent, value: bool):
    agent.avoid_monsters = value
    yield Step.completed(f"Set avoid_monsters flag to {value}")

@skill(
    name="move_to",
    description="Move to the specified position using pathfinding.",
    parameters=[
        SkillParameter.integer("x"),
        SkillParameter.integer("y")
    ]
)
@fail_on_popup
def move_to(agent: NetHackAgent, x: int, y: int, avoid_monsters=False):
    last_move_count = 0
    while True:
        if (agent.blstats.x == x and agent.blstats.y == y):
            yield Step.completed(f"Reached position {x, y}.")
            return

        path = agent.get_path_to(x, y, avoid_monsters=avoid_monsters)
        if path is None:
            yield Step.failed(f"No valid path found to reach position {x, y}.")
            return
        
        # We are executing the last action, which might lead into an unmovable object like a door or a wall
        # We track this so we only try moving into it once
        if last_move_count == 1:
            yield Step.completed(f"Tile {x, y} is blocked, stopping adjacent to it.")
            return
        if len(path) == 2:
            last_move_count += 1

        next_x, next_y = path[1] # Skip first position since its our own position
        yield agent.step(get_move_action(agent.blstats.x, agent.blstats.y, next_x, next_y))

@skill(
    name="explore",
    description="Explores the given room or corridor to reveal undiscovered tiles.",
    parameters=[
        SkillParameter.integer("room_id")
    ]
)
@fail_on_popup
def explore(agent: NetHackAgent, room_id: int):
    while True:
        room = agent.current_level.graph.get_room_data(room_id)
        if room is None:
            yield Step.failed(f"The room with id {room_id} does not exist anymore.")
            return

        # Determine which tiles we still want to explore
        explore_mask = room.get_adjacent_mask()
        explore_mask[agent.current_level.has_seen] = False
        if not np.any(explore_mask):
            yield Step.completed(f"The {room.type} with id {room_id} is fully explored.")

        # Find best tile to go to next
        target_tiles = []
        for (x, y) in room.get_tiles():
            if agent.get_path_to(x, y, bump_into_unwalkables=False) is None:
                continue

            unexplored_neighbor_count = 0
            for (nx, ny) in agent.current_level.get_neighbors(x, y):
                if explore_mask[ny, nx]:
                    unexplored_neighbor_count += 1
            if unexplored_neighbor_count == 0:
                continue

            target_tiles.append((x,y))

        if len(target_tiles) == 0:
            yield Step.failed(f"The {room.type} with id {room_id} cannot be further explored as all paths are blocked.")
            return
        
        # Only execute one action and then re-check the room
        for x, y in sorted(target_tiles, key=lambda pos: agent.distance_to(*pos)):
            step = next(move_to(agent, x, y))
            if not step.is_done():
                yield step
                break
        else:
            yield Step.error("Unable to reach the unexplored tiles.")

@skill(
    name="go_to",
    description="Moves to the specified room or corridor using the shortest path possible.",
    parameters=[
        SkillParameter.integer("room_id")
    ]
)
@fail_on_popup
def go_to(agent: NetHackAgent, room_id: int):
    while True:
        room = agent.current_level.graph.get_room_data(room_id)
        if room is None:
            yield Step.failed(f"The room with id {room_id} does not exist anymore.")
            return
        if room.is_inside(agent.blstats.x, agent.blstats.y):
            # Exits belong to multiple rooms, tell the agent that this room is most important
            agent.set_current_room(room_id)
            yield Step.completed(f"Reached the {room.type} with id {room_id}")
            return
        tiles = room.get_tiles()
        reachable_tiles = [(x, y) for (x, y) in tiles if agent.get_path_to(x, y, bump_into_unwalkables=False) is not None]
        if len(reachable_tiles) == 0:
            yield Step.failed(f"No path found that leads to the {room.type} with id {room_id}.")

        # Find best room tile to go to
        # Only execute one action and then re-check the room
        for x, y in sorted(reachable_tiles, key=lambda x: len(agent.get_path_to(*x, bump_into_unwalkables=False))):
            step = next(move_to(agent, x, y))
            if not step.is_done():
                yield step
                break
        else:
            yield Step.error("All attempts to reach the room have failed.")

@skill(
    name="search_room",
    description="Searches the perimeter of a room to find hidden passages.",
    parameters=[
        SkillParameter.integer("room_id")
    ]
)
@fail_on_popup
def search_room(agent: NetHackAgent, room_id: int, max_search_count=3):
    while True:
        room = agent.current_level.graph.get_room_data(room_id)

        # Check what tiles we still need to search
        search_mask = room.get_adjacent_mask()
        wall_mask = np.isin(agent.current_level.features, room.get_wall_glyphs())
        search_mask &= wall_mask # Only search walls
        search_mask &= agent.current_level.search_count < max_search_count # Filter out already searched walls
        if not np.any(search_mask):
            yield Step.completed(f"No tiles left to search.")
        
        target_tiles = []
        for (x, y) in room.get_interior_tiles():
            if agent.get_walkable_mask()[y,x] and agent.get_path_to(x, y) is None:
                continue

            searchable_neighbor_count = 0
            for (nx, ny) in agent.current_level.get_neighbors(x, y):
                if search_mask[ny, nx]:
                    searchable_neighbor_count += 1
            if searchable_neighbor_count == 0:
                continue

            # Use average search count per step to evaluate each position
            search_score = searchable_neighbor_count / (1 + agent.distance_to(x,y))
            target_tiles.append(((x,y), search_score))
        if len(target_tiles) == 0:
            yield Step.failed(f"Unable to reach any searchable tile.")

        (x, y), score = sorted(target_tiles, key=lambda x: x[1], reverse=True)[0]
        failed = False
        for step in move_to(agent, x, y):
            failed = failed or step.has_failed()
            if step.is_done():
                break
            yield step
        
        if failed:
            continue

        yield agent.step(actions.Command.SEARCH)

@skill(
    name="press_key",
    description="Presses the given letter. For special keys only ESC, SPACE, and ENTER are supported.",
    parameters=[
        SkillParameter.string("key")
    ]
)
def press_key(agent: NetHackAgent, key: str):
    try:
        yield agent.step(RawKeyPress.parse(key), thoughts=f"Pressing key '{key}'.")
        yield Step.completed()
    except ValueError:
        yield Step.error(f"Unable to press the given key {key}.")

@skill(
    name="type_text",
    description="Types the text by pressing the keys in order.",
    parameters=[
        SkillParameter.string("text")
    ]
)
def type_text(agent: NetHackAgent, text: str):
    try:
        for i, key in enumerate(text):
            yield agent.step(RawKeyPress.parse(key))
        yield Step.completed()
    except ValueError:
        yield Step.error(f"Failed to press the {i}th key '{key}'.")

def translate(data, dx, dy, constant=0):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

@skill(
    name="open_door",
    description="Tries to open the door at the given position.",
    parameters=[
        SkillParameter.integer("x"),
        SkillParameter.integer("y")
    ]
)
@fail_on_popup
def open_door(agent: NetHackAgent, x: int, y: int):
    assert agent.current_level.features[y, x] in G.CLOSED_DOORS
    yield agent.step(get_move_action(agent.blstats.x, agent.blstats.y, x, y))
    agent.current_level.door_open_attempts[y, x] += 1
    if agent.current_level.features[y, x] not in G.CLOSED_DOORS:
        yield Step.completed(f"Successfully opened the door at ({x}, {y}).")
    else:
        yield Step.failed(f"Failed to open the door at ({x}, {y}).")

@skill(
    name="open_neighbor_doors",
    description="Attempts to open any neighboring door.",
    parameters=[]
)
@fail_on_popup
def open_neighbor_doors(agent: NetHackAgent, door_open_count=4):
    for px, py in agent.current_level.get_neighbors(agent.blstats.x, agent.blstats.y, include_diagonal=False):
        remaining_attempts = max(0, door_open_count - agent.current_level.door_open_attempts[py,px])
        for _ in range(remaining_attempts):
            if agent.current_level.features[py, px] not in G.CLOSED_DOORS:
                break

            failed = False
            for step in open_door(agent, px, py):
                failed = failed or step.has_failed()
                if step.is_done():
                    break
                yield step
            if not failed:
                break
    yield Step.completed()

def compute_visit_mask(agent: NetHackAgent, door_open_count=4):
    level = agent.current_level
    stone = ~level.has_seen & np.isin(level.features, G.ROCKS)
    doors = np.isin(level.features, G.CLOSED_DOORS) & (level.door_open_attempts < door_open_count)
    if not stone.any() and not doors.any():
        return stone

    to_visit = np.zeros(level.shape, dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy != 0 or dx != 0:
                to_visit |= translate(stone, dy, dx)
                if dx == 0 or dy == 0:
                    to_visit |= translate(doors, dy, dx)
    return to_visit

@skill(
    name="melee_attack",
    description="Pursues and attacks a given target using melee attacks until it is dead.",
    parameters=[
        SkillParameter.integer("x"),
        SkillParameter.integer("y")
    ]
)
@fail_on_popup
def melee_attack(agent: NetHackAgent, x, y):
    target_glyph = agent.current_level.get_monster_glyph(x, y)
    if target_glyph is None:
        yield Step.failed(f"There is no monster at ({x},{y}).")
        return

    tx, ty = x, y
    while True:
        target_neighbors = agent.current_level.get_neighbors(tx, ty)
        target_neighbors = [(x,y) for (x,y) in target_neighbors if agent.get_path_to(x,y, bump_into_unwalkables=False, avoid_monsters=True) is not None]
        if len(target_neighbors) == 0:
            yield Step.failed(f"Unable to reach the target at ({tx}, {ty}).")

        nx, ny = min(target_neighbors, key=lambda pos: agent.distance_to(pos[0], pos[1], bump_into_unwalkables=False, avoid_monsters=True))
        move_action = get_move_towards_action(agent, nx, ny, bump=False, avoid_monsters=True)
        # Should not happen because we checked the neighbors already, but safe is safe
        if move_action is None:
            yield Step.failed(f"Unable to reach the target at ({tx}, {ty}).")

        if move_action == actions.MiscDirection.WAIT:
            # We reached the target
            yield agent.step(actions.Command.FIGHT)
            yield agent.step(get_move_action(agent.blstats.x, agent.blstats.y, tx, ty))

            if agent.current_level.get_monster_glyph(tx, ty) != target_glyph:
                yield Step.completed(f"Killed the target.")
                return
        else:
            yield agent.step(move_action)

        # Check if target has moved
        if agent.current_level.get_monster_glyph(tx, ty) == target_glyph:
            continue

        found = False
        for nx, ny in agent.current_level.get_neighbors(tx, ty):
            if agent.current_level.get_monster_glyph(nx, ny) == target_glyph:
                tx, ty = nx, ny
                found = True
                break

        if not found:
            yield Step.failed(f"Lost track of the target")
            return

@skill(
    name="explore_level",
    description="Explores the level to find new rooms, as well as hidden doors and corridors.",
    parameters=[]
)
@fail_on_popup
def explore_level(agent: NetHackAgent, search_prio_limit=None, door_open_count=4):
    def compute_search_mask(prio_limit=0):
        level = agent.current_level

        prio = np.zeros(level.shape, np.float32)
        prio[:] = -1
        prio -= level.search_count ** 2 * 2

        is_on_door = np.isin(level.features, G.DOORS)
        stones = np.zeros(level.shape, np.int32)
        walls = np.zeros(level.shape, np.int32)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy != 0 or dx != 0:
                    stones += np.isin(translate(level.features, dy, dx), G.ROCKS)
                    walls += np.isin(translate(level.features, dy, dx), G.WALLS)

        prio += (is_on_door & (stones > 3)) * 250
        prio += (np.stack([translate(agent.get_walkable_mask(treat_boulder_unwalkable=False), y, x).astype(np.int32)
                            for y, x in [(1, 0), (-1, 0), (0, 1), (0, -1)]]).sum(0) <= 1) * 250
        prio[(stones == 0) & (walls == 0)] = -np.inf
        prio[~agent.get_walkable_mask(treat_boulder_unwalkable=False) | (agent.get_distance_map() == -1)] = -np.inf
        return prio, prio >= prio_limit
    
    while True:
        for step in open_neighbor_doors(agent):
            if step.is_done():
                break
            yield step

        dis = agent.get_distance_map()
        visit_mask = compute_visit_mask(agent, door_open_count=door_open_count)
        _, search_mask = compute_search_mask()
        explore_mask = (visit_mask | search_mask) & (dis != -1)

        dynamic_search_fallback = False
        if not explore_mask.any():
            dynamic_search_fallback = True
        else:
            # find all closest explore_mask tiles
            nonzero_y, nonzero_x = ((dis == dis[explore_mask].min()) & explore_mask).nonzero()
            if len(nonzero_y) == 0:
                dynamic_search_fallback = True

        if dynamic_search_fallback:
            search_prio, _ = compute_search_mask()
            if search_prio_limit is not None:
                search_prio[search_prio < search_prio_limit] = -np.inf
                search_prio -= dis * np.isfinite(search_prio) * 100
            else:
                search_prio -= dis * 4

            search_mask = np.isfinite(search_prio)
            explore_mask = (visit_mask | search_mask) & (dis != -1)
            if not explore_mask.any():
                yield Step.failed("Could not find anything to further explore.")
            nonzero_y, nonzero_x = ((search_prio == search_prio[explore_mask].max()) & explore_mask).nonzero()

        # select first closest to_explore tile
        i = 0
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        failed = False
        for step in move_to(agent, target_x, target_y):
            failed = failed or step.has_failed()
            if step.is_done():
                break
            yield step
        
        if not failed and search_mask[target_y, target_x] and not visit_mask[target_y, target_x]:
            yield agent.step(actions.Command.SEARCH)
            yield agent.step(actions.Command.SEARCH)
            yield agent.step(actions.Command.SEARCH)
            yield agent.step(actions.Command.SEARCH)
            yield agent.step(actions.Command.SEARCH)
        

### Skills for NetHack Commands
def create_command(command: IntEnum, command_name: str, command_description: str):
    @skill(
        name=command_name,
        description=command_description,
        parameters=[]
    )
    def command_skill(agent: NetHackAgent):
        yield agent.step(command)
        yield Step.completed()

    return command_skill

def create_inventory_command(command: IntEnum, command_name, command_description: str):
    @skill(
        name=command_name,
        description=command_description,
        parameters=[
            SkillParameter.string("item_letter", optional=True)
        ]
    )
    def command_skill(agent: NetHackAgent, item_letter: str=None):
        yield agent.step(command)
        if item_letter and agent.waiting_for_popup():
            yield from press_key(agent, key=item_letter)
        else:
            yield Step.completed()

    return command_skill

def create_position_command(command: IntEnum, command_name: str, command_description: str):
    @skill(
        name=command_name,
        description=command_description,
        parameters=[
            SkillParameter.integer("x", optional=True),
            SkillParameter.integer("y", optional=True)
        ]
    )
    def command_skill(agent: NetHackAgent, x: int=None, y: int=None):
        if x is not None and y is not None and (x != agent.blstats.x or y != agent.blstats.y):
            yield Step.think(f"Moving to ({x},{y})")

            failed = False
            for step in move_to(agent, x, y):
                failed = failed or step.has_failed()
                if step.is_done():
                    break
                yield step

            if failed or agent.blstats.x != x or agent.blstats.y != y:
                yield Step.failed(f"Failed to reach position ({x},{y}).")
                return
        
        yield agent.step(command)
        yield Step.completed()

    return command_skill

def create_direction_command(command: IntEnum, command_name: str, command_description: str):
    @skill(
        name=command_name,
        description=command_description,
        parameters=[
            SkillParameter.integer("x"),
            SkillParameter.integer("y")
        ]
    )
    def command_skill(agent: NetHackAgent, x: int=None, y: int=None):
        target_neighbors = agent.current_level.get_neighbors(x, y, include_diagonal=True)
        # Filter unreachable neighbors
        target_neighbors = [(x,y) for (x,y) in target_neighbors if agent.get_path_to(x,y) is not None]
        nx, ny = min(target_neighbors, key=lambda pos: agent.distance_to(pos[0], pos[1], bump_into_unwalkables=False))

        failed = False
        for step in move_to(agent, nx, ny):
            failed = failed or step.has_failed()
            if step.is_done():
                break
            yield step
        if failed or agent.blstats.x != nx or agent.blstats.y != ny:
            yield Step.failed(f"Failed to get close to ({x},{y})")
            return
            
        yield agent.step(command)
        yield agent.step(get_move_action(agent.blstats.x, agent.blstats.y, x, y))
        yield Step.completed()

    return command_skill

@skill(
    name="zap",
    description="Zap a wand in the given cardinal (n, e, s, w) or ordinal direction (ne, se, sw, nw) or target yourself using self.",
    parameters=[
        SkillParameter.string("item_letter"),
        SkillParameter.string("direction")
    ]
)
def zap(agent: NetHackAgent, item_letter, direction):
    yield agent.step(actions.Command.ZAP)
    for step in press_key(agent, key=item_letter):
        if step.is_done():
            break 
        yield step
    try:
        yield agent.step(direction_to_action(direction), f"Casting towards {direction}.")
        yield Step.completed()
    except:
        yield Step.failed(f"Given direction {direction} is invalid. Can only use (n, e, s, w, ne, se, sw, nw, self).")
    
@skill(
    name="rest",
    description="Rests n-moves while doing nothing or until something happens (default=5).",
    parameters=[
        SkillParameter.integer(name="count", optional=True)
    ]
)
@fail_on_popup
def rest(agent: NetHackAgent, count: int = 5):
    if count > 1:
        for step in type_text(agent, str(count)):
            if step.is_done():
                break 
            yield step
    yield agent.step(actions.MiscDirection.WAIT)

@skill(
    name="pray",
    description="Pray to the gods for help.",
    parameters=[]
)
def pray(agent: NetHackAgent, count: int = 5):
    yield agent.step(actions.Command.PRAY)
    if agent.current_game_message.startswith("Are you sure you want to pray?"):
        yield agent.step(RawKeyPress.KEYPRESS_Y)
    yield Step.completed()


pickup = create_position_command(actions.Command.PICKUP, "pickup", "Pickup things at your location or specify where you want to pickup an item.")
up = create_position_command(actions.MiscDirection.UP, "up", "Go up a staircase at your location or specify the position of the staircase you want to use.")
down = create_position_command(actions.MiscDirection.DOWN, "down", "Go down a staircase at your location or specify the position of the staircase you want to use.")
search = create_position_command(actions.Command.SEARCH, "search", "Search for unseen things around you or specify the position where you want to search.")
loot = create_position_command(actions.Command.LOOT, "loot", "Loot a box on the floor.")
offer = create_position_command(actions.Command.OFFER, "offer", "Offer a sacrifice to the gods.")

drop = create_inventory_command(actions.Command.DROP, "drop", "Drop an item.")
read = create_inventory_command(actions.Command.READ, "read", "Read a scroll, spellbook, or something else.")
puton = create_inventory_command(actions.Command.PUTON, "put_on", "Put on an accessory.")
remove = create_inventory_command(actions.Command.REMOVE, "remove", "Remove an accessory (ring, amulet, or blindfold).")
takeoff = create_inventory_command(actions.Command.TAKEOFF, "takeoff", "Take off one piece of armor.")
wield = create_inventory_command(actions.Command.WIELD, "wield", "Wield a weapon.")
wear = create_inventory_command(actions.Command.WEAR, "wear", "Wear a piece of armor.")
apply = create_inventory_command(actions.Command.APPLY, "apply", "Apply (use) a tool. If used on a wand, that wand will be broken, releasing its magic in the process.")
eat = create_inventory_command(actions.Command.EAT, "eat", "Eat something from your inventory or straight from the ground.")
drink = create_inventory_command(actions.Command.QUAFF, "drink", "Drink something from your inventory or straight from the ground.")
tip = create_inventory_command(actions.Command.TIP, "tip", "Tip out the content of a container.")
dip = create_inventory_command(actions.Command.DIP, "dip", "Dip an object into something. ")

kick = create_direction_command(actions.Command.KICK, "kick", "Kick something.")
open = create_direction_command(actions.Command.OPEN, "open", "Open a door.")
close = create_direction_command(actions.Command.CLOSE, "close", "Close a door.")

takeoffall = create_command(actions.Command.TAKEOFFALL, "take_off_all", "Remove all armor.")
cast = create_command(actions.Command.CAST, "cast", "Opens your spellbook to cast a spell.")
pay = create_command(actions.Command.PAY, "pay", "Pay your shopping bill.")
look = create_command(actions.Command.LOOK, "look", "Look at what is under you.")

ALL_COMMAND_SKILLS = [
    zap,
    pickup,
    up,
    down,
    #search,
    loot,
    offer,

    drop,
    read,
    puton,
    remove,
    takeoff,
    wield,
    wear,
    apply,
    eat,
    drink,
    tip,
    dip,

    kick,
    #open,
    #close,
    
    rest,

    #takeoffall, 
    cast,
    pay,
    pray,
    look
]