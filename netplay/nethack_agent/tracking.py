import netplay.nethack_utils.glyphs as G
from netplay.nethack_agent.describe import describe_glyph

from nle.nethack import actions, glyph_is_pet
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

import numpy as np
import re

from abc import abstractmethod
from collections import namedtuple, deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Generator, Optional, Iterator

LEVEL_WIDTH = 79
LEVEL_HEIGHT = 21
UNKNOWN_GLYPH_ID = -1

BLStats = namedtuple('BLStats', 'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask alignment')
Position = namedtuple("Position", "x y")

class NethackEvent:
    @abstractmethod
    def describe(self) -> str:
        pass

    def __str__(self):
        return self.describe()

class EventManager:
    def __init__(self):
        self.event_history: List[NethackEvent] = []

    def handle_event(self, event: NethackEvent):
        self.event_history.append(event)

    def clear_event_history(self) -> List[NethackEvent]:
        history = self.event_history
        self.event_history = []
        return history

@dataclass
class GameMessageEvent(NethackEvent):
    message: str

    def describe(self) -> str:
        return self.message

@dataclass
class DungeonLevelChangeEvent(NethackEvent):
    old_level: Tuple[int,int]
    new_level: Tuple[int,int]

    def describe(self) -> str:
        return f"Changed dungeon level from {self.old_level} to {self.new_level}"

@dataclass
class TeleportEvent(NethackEvent):
    start: Position
    to: Position

    def describe(self) -> str:
        start_x, start_y = self.start
        to_x, to_y = self.to
        return f"Teleported from ({start_x},{start_y}) to ({to_x},{to_y})"
    
@dataclass
class StatChangeEvent(NethackEvent):
    stat: str
    old_value: int
    new_value: int

    def difference(self):
        return self.new_value - self.old_value

    def describe(self) -> str:
        diff = self.difference()
        if diff > 0:
            return f"Gained {diff} {self.stat}"
        elif diff < 0:
            return f"Lost {abs(diff)} {self.stat}"

        raise ValueError(f"Expected old_value {self.old_value} to be different from new_value {self.new_value}.")
    
@dataclass
class LowHealthEvent(NethackEvent):
    def describe(self) -> str:
        return f"Health is low."

@dataclass
class NewGlyphEvent(NethackEvent):
    glyph: int
    position: Position

    def describe(self) -> str:
        glyph_descr = describe_glyph(self.glyph)
        x, y = self.position
        return f"A {glyph_descr} appeared at ({x},{y})."
    
@dataclass
class ItemEvent(NethackEvent):
    item_glyph: int
    position: Position

    def describe(self) -> str:
        item_descr = describe_glyph(self.item_glyph)
        x, y = self.position
        return f"Found a {item_descr} at ({x},{y})."


class RoomType(Enum):
    room = 0
    corridor = 1

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

@dataclass
class RoomData:
    OUTSIDE = 0
    INTERIOR = 1
    EXIT = 2
    ADJACENT = 3

    type: RoomType
    room_map: np.ndarray#[Tuple[int, int], int]

    def get_tiles(self) -> Generator[Tuple[int, int], None, None]:
        ys, xs = np.where(np.isin(self.room_map, [RoomData.INTERIOR, RoomData.EXIT]))
        return zip(xs, ys)
    
    def get_interior_tiles(self) -> Generator[Tuple[int, int], None, None]:
        ys, xs = np.where(self.room_map == RoomData.INTERIOR)
        return zip(xs, ys)
    
    def get_adjacent_tiles(self) -> Generator[Tuple[int, int], None, None]:
        ys, xs = np.where(self.room_map == RoomData.ADJACENT)
        return zip(xs, ys)
    
    def get_exit_tiles(self) -> Generator[Tuple[int, int], None, None]:
        ys, xs = np.where(self.room_map == RoomData.EXIT)
        return zip(xs, ys)

    def is_inside(self, x, y):
        return self.room_map[y, x] in [RoomData.INTERIOR, RoomData.EXIT]

    def get_room_mask(self):
        return np.isin(self.room_map, [RoomData.INTERIOR, RoomData.EXIT])

    def get_interior_mask(self):
        return self.room_map == RoomData.INTERIOR
    
    def get_adjacent_mask(self):
        return self.room_map == RoomData.ADJACENT
    
    def get_exit_mask(self):
        return self.room_map == RoomData.EXIT
    
    def get_wall_glyphs(self):
        if self.type == RoomType.room:
            return G.WALLS
        elif self.type == RoomType.corridor:
            return G.ROCKS
        
        raise ValueError(f"Unknown room type {self.type}.")
    
@dataclass
class NewRoomEvent(NethackEvent):
    room_id: int
    room_type: RoomType

    def describe(self) -> str:
        return f"Found a new {self.room_type} with ID {self.room_id}"
    
@dataclass
class MergedRoomsEvent(NethackEvent):
    new_room_id: int
    merged_ids: List[int]
    room_type: RoomType

    def describe(self) -> str:
        merged_ids_descr = f"[{','.join([str(x) for x in self.merged_ids])}]"
        return f"The {self.room_type}s {merged_ids_descr} are actually the same. They have been merged into a single {self.room_type} with ID {self.new_room_id}."

class RoomGraph:
    def __init__(self):
        self.rooms : Dict[int, RoomData] = {}
        self.old_id_mapping : Dict[int, int] = {}
        self.next_room_id = 0

    def add_room(self, data: RoomData) -> int:
        room_id = self.next_room_id
        self.next_room_id += 1
        self.rooms[room_id] = data
        return room_id

    def get_rooms(self) -> List[int]:
        return list(self.rooms.keys())

    def get_rooms_at(self, x: int, y: int) -> List[int]:
        return [r for r in self.rooms.values() if r.is_inside(x, y)]

    def get_first_room_at(self, x: int, y: int, prefer_id = None, ignore_id=None) -> int:
        ignore_id = self.resolve_id(ignore_id)
        prefer_id = self.resolve_id(prefer_id)

        room_ids = [id for id, room in self.rooms.items() if room.is_inside(x, y) and id != ignore_id]
        if prefer_id in room_ids:
            return prefer_id
        return room_ids[0] if len(room_ids) != 0 else None

    def get_room_data(self, room_id) -> RoomData:
        return self.rooms.get(self.resolve_id(room_id))
    
    def update(self, level: "Level", event_manager: EventManager) -> List[NethackEvent]:
        def overlap(old_room: RoomData, new_room: RoomData):
            return np.count_nonzero(old_room.get_room_mask() & new_room.get_room_mask())
        
        new_graph = RoomGraph.from_level(level)
        new_to_old = {new_room_id : [] for new_room_id in new_graph.get_rooms()}

        # Assign old rooms to new rooms based on their overlap
        for old_id, room_data in self.rooms.items():
            best_matching_id = max(new_graph.get_rooms(), key=lambda id: overlap(room_data, new_graph.get_room_data(id)))
            new_to_old[best_matching_id].append(old_id)

        self.rooms = {}
        for new_room_id, matching_old_room_ids in new_to_old.items():
            new_room = new_graph.get_room_data(new_room_id)
            if len(matching_old_room_ids) == 0:
                # This is an entirely new room
                event_manager.handle_event(NewRoomEvent(self.next_room_id, new_room.type))
                self.rooms[self.next_room_id] = new_room
                self.next_room_id += 1
            elif len(matching_old_room_ids) == 1:
                # This is an old room, possibly larger than the old one
                old_id = matching_old_room_ids[0]
                self.rooms[old_id] = new_room
            else:
                # This is a merge of multiple old rooms
                # Assign a new ID and remember which old id corresponds to which new id
                event_manager.handle_event(MergedRoomsEvent(self.next_room_id, matching_old_room_ids, new_room.type))

                new_id = self.next_room_id
                self.next_room_id += 1
                self.rooms[new_id] = new_room
                
                # Handle recurrent id mapping, which could happen if we merge a room multiple times
                past_ids = [old for old, new in self.old_id_mapping.items() if new in matching_old_room_ids]
                # Update mapping
                for id in [*matching_old_room_ids, *past_ids]:
                    self.old_id_mapping[id] = new_id

    def resolve_id(self, room_id):
        return self.old_id_mapping.get(room_id, room_id)
            
    @classmethod
    def from_level(cls, level: "Level"):
        graph = RoomGraph()
        occupation_map = np.full(level.shape, False, dtype=bool)
        for y in range(level.features.shape[0]):
            for x in range(level.features.shape[1]):
                if occupation_map[y,x]:
                    continue

                glyph = level.features[y,x]
                if glyph in G.FLOORS:
                    room_type = RoomType.room
                    boundary_glyphs = [*G.WALLS, *G.ROCKS, *G.CORRIDORS]
                    exit_glyphs = [*G.DOORS, *G.DOORWAYS]
                elif glyph in G.CORRIDORS:
                    room_type = RoomType.corridor
                    boundary_glyphs = [*G.WALLS, *G.ROCKS, *G.FLOORS]
                    exit_glyphs = [*G.DOORS, *G.DOORWAYS]
                else:
                    continue

                room_map = RoomGraph._floodfill_room(
                    level=level,
                    pos_x=x,
                    pos_y=y,
                    boundary_glyphs=boundary_glyphs,
                    exit_glyphs=exit_glyphs
                )
                room = RoomData(room_type, room_map)
                occupation_map[room.get_interior_mask()] = True
                graph.add_room(room)

        return graph

    @staticmethod
    def _floodfill_room(level: "Level", pos_x, pos_y, boundary_glyphs, exit_glyphs) -> np.array:
        room_map = np.full(level.shape, RoomData.OUTSIDE, dtype=int)
        open_list = deque([(pos_y, pos_x)])
        while len(open_list) != 0:
            y, x = open_list.popleft()
            if not level.is_in_bounds(x, y) or room_map[y, x] != RoomData.OUTSIDE:
                continue

            # Mark tile
            glyph = level.features[y,x] if level.features[y,x] not in [-1, G.SS.S_stone] else level.glyphs[y,x]
            if glyph in boundary_glyphs:
                room_map[y,x] = RoomData.ADJACENT
                continue # Do not floodfill neighbors for adjacent tiles
            elif glyph in exit_glyphs:
                room_map[y,x] = RoomData.EXIT
                # Mark neighbors as adjacent, but do not use them for further iterations
                for (nx, ny) in level.get_neighbors(x, y):
                    neighbor_glyph = level.get_feature_glyph(nx, ny)
                    if neighbor_glyph in boundary_glyphs:
                        room_map[ny, nx] = RoomData.ADJACENT
                continue
            else:
                room_map[y,x] = RoomData.INTERIOR

            # Add neighbors to queue
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                open_list.append((y + dy, x + dx))

        return room_map

# This class is inspired by the same class in autoascend
class Level:
    def __init__(self, dungeon_number, level_number, level_shape=(LEVEL_HEIGHT, LEVEL_WIDTH)):
        # General information about this level 
        self.dungeon_number = dungeon_number
        self.level_number = level_number
        self.shape = level_shape

        # Raw map information
        self.chars = np.full(self.shape, UNKNOWN_GLYPH_ID, np.int16)
        self.glyphs = np.full(self.shape, UNKNOWN_GLYPH_ID, np.int16)

        # Preprocessed map information
        # https://nethackwiki.com/wiki/Square
        # Everything that can be on a square at the same time
        # Although multiple objects can be on a square, we won't track that
        # Also engravings won't be tracked. As you see their message only when you stand on the square.
        self.features = np.full(self.shape, UNKNOWN_GLYPH_ID, dtype=np.int16)
        self.objects = np.full(self.shape, UNKNOWN_GLYPH_ID, dtype=np.int16)
        self.monsters = np.full(self.shape, UNKNOWN_GLYPH_ID, dtype=np.int16)

        # Agent information
        self.was_on = np.zeros(self.shape, bool)
        self.has_seen = np.zeros(self.shape, bool)
        self.door_open_attempts = np.zeros(self.shape, int)
        self.search_count = np.zeros(self.shape, int)
        self.graph = RoomGraph.from_level(self)

    def update(self, pos_x, pos_y, glyphs, chars, event_manager: EventManager):
        self.glyphs = glyphs
        self.chars = chars

        features_mask = G.is_dungeon_feature(glyphs)
        monster_mask = G.is_monster(glyphs)
        object_mask = G.is_object(glyphs)
        self.features[features_mask] = glyphs[features_mask]
        # We do not know what feature is below an monster/object if there was stone there previously. As stone both represents stone and unknown features
        stone_feature_mask = self.features == G.SS.S_stone
        self.features[stone_feature_mask & (monster_mask | object_mask)] = UNKNOWN_GLYPH_ID
        
        # Remove objects that we do not see anymore, but remember them if they are potentially blocked by monsters
        self.objects[object_mask] = glyphs[object_mask]
        self.objects[~object_mask & ~monster_mask] = UNKNOWN_GLYPH_ID
        # Only remember the monsters we can see
        monster_mask[pos_y, pos_x] = False # Do not track ourselves
        self.monsters[:] = UNKNOWN_GLYPH_ID
        self.monsters[monster_mask] = glyphs[monster_mask]

        self.was_on[pos_y, pos_x] = True
        # We've seen everything that contains an glyph other than stone
        self.has_seen[self.glyphs != G.SS.S_stone] = True
        # We've also seen tiles adjacent to us, including stone
        for (x,y) in self.get_neighbors(pos_x, pos_y):
            self.has_seen[y,x] = True

        self.graph.update(self, event_manager=event_manager)

    def get_monsters(self) -> Iterator[Tuple[int, Position]]:
        for y, x in zip(*np.where(self.monsters != UNKNOWN_GLYPH_ID)):
            yield (self.monsters[y,x], Position(x,y))

    def get_objects(self) -> Iterator[Tuple[int, Position]]:
        for y, x in zip(*np.where(self.objects != UNKNOWN_GLYPH_ID)):
            yield (self.objects[y,x], Position(x,y))

    def get_features(self, filter: List[int]) -> Iterator[Tuple[int, Position]]:
        for y, x in zip(*np.where(np.isin(self.features, filter))):
            yield (self.features[y,x], Position(x,y))

    def get_monster_glyph(self, x, y) -> Optional[int]:
        return self.monsters[y,x] if self.monsters[y,x] != UNKNOWN_GLYPH_ID else None
    
    def get_object_glyph(self, x, y) -> Optional[int]:
        return self.objects[y,x] if self.objects[y,x] != UNKNOWN_GLYPH_ID else None
    
    def get_feature_glyph(self, x, y) -> Optional[int]:
        return self.features[y,x] if self.features[y,x] != UNKNOWN_GLYPH_ID else None

    def get_walkable_mask(self, treat_boulder_unwalkable=True, treat_monster_unwalkable=False):
        walkable_mask = np.zeros(self.shape, dtype=bool)

        # Everything with a walkable feature is walkable
        walkable_features_mask = np.isin(self.features, G.WALKABLE_FEATURES)
        walkable_mask[walkable_features_mask] = True

        # If the feature is hidden, treat it as walkable
        hidden_feature_mask = self.features == UNKNOWN_GLYPH_ID
        walkable_mask[hidden_feature_mask] = True

        if treat_boulder_unwalkable:
            # Boulders will block you if there is no space to push them, so we treat them as always blocking
            boulders_mask = np.isin(self.objects, G.BOULDERS)
            walkable_mask[boulders_mask] = False

        if treat_monster_unwalkable:
            # Treat monsters unwalkable, but pets are fine
            monster_mask = self.monsters != UNKNOWN_GLYPH_ID
            pet_mask = glyph_is_pet(self.monsters)
            monster_mask[pet_mask] = False
            walkable_mask[monster_mask] = False

        return walkable_mask
    
    def get_diagonal_walkable_mask(self, treat_boulder_unwalkable=True, treat_monster_unwalkable=False):
        walkable_diagonally = self.get_walkable_mask(treat_boulder_unwalkable=treat_boulder_unwalkable, treat_monster_unwalkable=treat_monster_unwalkable)
        # Doors do not allow diagonal movement
        doors_mask = np.isin(self.features, G.DOORS)
        walkable_diagonally[doors_mask] = False
        return walkable_diagonally
    
    def is_in_bounds(self, x, y):
        return 0 <= x < self.shape[1] and 0 <= y < self.shape[0]
    
    def get_neighbors(self, x, y, include_diagonal=True):
        ret = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if not include_diagonal and abs(dy) + abs(dx) > 1:
                    continue

                nx = x + dx
                ny = y + dy
                if self.is_in_bounds(nx, ny):
                    ret.append((nx, ny))

        return ret
    
class ObservationEventDetector:
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager

        self.lang = NLELanguageObsv()
        self.last_blstats: BLStats = None

    def init(self, observation):
        self._copy_data(observation)

    def update(self, action, observation):
        game_message = self.lang.text_message(observation["tty_chars"]).decode("latin-1")
        if game_message:
            self.event_manager.handle_event(GameMessageEvent(game_message))
        self._generate_blstats_events(observation)
        self._copy_data(observation)

    def _generate_blstats_events(self, observation):
        old = self.last_blstats
        new = BLStats(*observation["blstats"])

        # Low Health - We only raise an event when we get below a percentage threshold
        # To avoid wrong messages when raising our max health, we use old.max_hitpoints in both cases to compute the percentage
        percentage = 0.4
        if (old.hitpoints / old.max_hitpoints) >= percentage and (new.hitpoints / old.max_hitpoints) < percentage:
            self.event_manager.handle_event(LowHealthEvent())

        # Did we change the level?
        old_level = (old.dungeon_number, old.level_number)
        new_level = (new.dungeon_number, new.level_number)
        if old_level != new_level:
            self.event_manager.handle_event(DungeonLevelChangeEvent(old_level, new_level))
        # Event for when we moved more than one tile
        elif abs(old.x - new.x) >= 2 or abs(old.y - new.y) >= 2:
            self.event_manager.handle_event(TeleportEvent((old.x, old.y), (new.x, new.y)))

        # Stats
        for stat in ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]:
            old_value = getattr(old, stat)
            new_value = getattr(new, stat)
            if old_value != new_value:
                self.event_manager.handle_event(StatChangeEvent(stat, old_value, new_value))

    def _copy_data(self, observation):
        self.last_blstats = BLStats(*observation["blstats"])

class Inventory:
    @dataclass
    class Item:
        letter: str
        count: int
        name: str

    def __init__(self):
        self.items: Dict[str, Inventory.Item] = None

    def init(self, observation):
        self.items = self._parse_observation(observation)

    def update(self, action, observation):
        self.items = self._parse_observation(observation)

    def _parse_observation(self, observation):
        inv_strs = observation["inv_strs"]
        inv_letters = observation["inv_letters"]
        num_items = len([x for x in inv_letters if x != 0])

        items = {}
        for i in range(num_items):
            letter = chr(inv_letters[i])
            item_description = "".join([chr(x) for x in inv_strs[i] if x != 0])
            count, name = re.findall("^(a|an|the|\d+)(?: )(.*)$", item_description)[0]
            count = {'a': 1, 'an': 1, 'the': 1}.get(count, count)

            items[letter] = Inventory.Item(letter, count, name)
            
        return items

class NewGlyphDetector:
    @dataclass
    class GlyphData:
        glyph: int
        position: Position
        last_seen: int

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.last_seen_glyphs: List[NewGlyphDetector.GlyphData] = []
        self.last_level_id = None
        self.last_timestep = 0

    def init(self, data: "NethackDataTracker"):
        self.level_id = (data.blstats.dungeon_number, data.blstats.level_number)
        self.last_timestep = data.blstats.time
        self.last_seen_glyphs = self._copy_glyph_stats(data)

    def update(self, data: "NethackDataTracker"):
        updated_glyphs = self._copy_glyph_stats(data)

        # Do not throw events when changing the level
        new_level_id = (data.blstats.dungeon_number, data.blstats.level_number)
        if new_level_id != self.last_level_id:
            self.last_level_id = new_level_id
            self.last_seen_glyphs = updated_glyphs
            return

        # Check for new glyphs
        for glyph in updated_glyphs:
            is_new = True
            for i, old_glyph in enumerate(self.last_seen_glyphs):
                if self._is_same_glyph(glyph, old_glyph):
                    is_new = False
                    # Do not use this monster again
                    self.last_seen_glyphs.pop(i)
                    break

            if is_new and not glyph_is_pet(glyph.glyph):
                self.event_manager.handle_event(NewGlyphEvent(glyph.glyph, glyph.position))
        
        # Remember glyphs for a while
        kept_glyphs = [m for m in self.last_seen_glyphs if (data.blstats.time - m.last_seen) < 10]
        self.last_seen_glyphs = [*kept_glyphs, *updated_glyphs]

    def _is_same_glyph(self, m1: GlyphData, m2: GlyphData) -> bool:
        if m1.glyph != m2.glyph:
            return False
        return True

    def _copy_glyph_stats(self, data: "NethackDataTracker") -> List["NewGlyphDetector.GlyphData"]:
        glyphs = [
            *data.current_level.get_monsters(),
            *data.current_level.get_objects(),
            *data.current_level.get_features(filter=[G.SS.S_dnstair, G.SS.S_dnladder, *G.ALTARS, *G.FOUNTAINS])
        ]
        glyphs = [
            NewGlyphDetector.GlyphData(
                glyph=glyph,
                position=position,
                last_seen=data.blstats.time
            )
            for (glyph, position) in glyphs
        ]

        return glyphs

class NethackDataTracker:
    def __init__(self, event_manager: EventManager=None):
        if event_manager is None:
            event_manager = EventManager()
        self.event_manager = event_manager
        self.event_detector = ObservationEventDetector(self.event_manager)
        self.glyph_detector = NewGlyphDetector(self.event_manager)
        
        self.blstats = None
        self.last_pray_time = None
        self.levels : Dict[Tuple[int, int], Level] = {}
        self.inventory = Inventory()

    def init(self, observation) -> List[NethackEvent]:
        self.blstats = BLStats(*observation["blstats"])
        self.event_detector.init(observation)
        self._update_level(observation)
        self.inventory.init(observation)
        self.glyph_detector.init(self)
        return self.event_manager.clear_event_history()

    def update(self, action, observation) -> List[NethackEvent]:
        if action == actions.Command.SEARCH:
            self.current_level.search_count[self.blstats.y, self.blstats.x] += 1

        self.blstats = BLStats(*observation["blstats"])
        self.event_detector.update(action, observation)
        self._update_level(observation)
        self.inventory.update(action, observation)
        self.glyph_detector.update(self)

        if action == actions.Command.PRAY:
            self.last_pray_time = self.blstats.time
        
        return self.event_manager.clear_event_history()

    def _update_level(self, observation):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        self.levels[key].update(
            self.blstats.x,
            self.blstats.y,
            observation["glyphs"],
            observation["chars"],
            event_manager=self.event_manager
        )

    @property
    def current_level(self) -> Level:
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        return self.levels[key]