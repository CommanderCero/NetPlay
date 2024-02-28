import netplay.nethack_agent.describe as describe
import netplay.nethack_utils.glyphs as G
from netplay.nethack_agent.agent import NetHackAgent
from netplay.core.descriptor import Descriptor
from netplay.nethack_agent.tracking import RoomData

from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

from inspect import cleandoc

DESCRIPTION_INCLUDE_FEATURE_GLYPHS = [
    *G.STAIRCASES,
    *G.LADDERS,
    *G.ALTARS,
    *G.SINKS,
    *G.FOUNTAINS,
    *G.TRAPS,
    *G.DOORS,
    *G.DRAWBRIDGES,
    *G.IRON_BARS,
    *G.THRONES,
    *G.GRAVES
]
    
class StatsDescriptor(Descriptor):
    def __init__(self):
        self.lang = NLELanguageObsv()

    def describe(self, agent: NetHackAgent) -> str:
        hunger_lookup = {
            0: "satiated",
            1: "not hungry",
            2: "hungry",
            3: "weak",
            4: "fainting",
            5: "fainted",
            6: "starved"
        }
         
        encumbrance_lookup = {
            0: "unencumbered",
            1: "burdened",
            2: "stressed",
            3: "strained",
            4: "overtaxed",
            5: "overloaded"
        }

        condition_map = {
            0x00000001: "stoned",
            0x00000002: "slimed",
            0x00000004: "strangled",
            0x00000008: "food poisoning",
            0x00000010: "terminally ill",
            0x00000020: "blind",
            0x00000040: "deaf",
            0x00000080: "stunned",
            0x00000100: "confused",
            0x00000200: "hallucinating",
            0x00000400: "levitating",
            0x00000800: "flying",
            0x00001000: "riding",
        }

        alignment_map = {
            0: "None",
            -1: "Chaotic",
            0: "Neutral",
            1: "Lawful"
        }

        conditions = [
            condition
            for mask, condition in condition_map.items()
            if agent.blstats.prop_mask & mask
        ]
        conditions = ", ".join(conditions) if len(conditions) > 0 else "None"

        return "\n".join([
            f'HP: {agent.blstats.hitpoints} / {agent.blstats.max_hitpoints}',
            f'armor class: {agent.blstats.armor_class}',
            f"strength: {agent.blstats.strength}",
            f"dexterity: {agent.blstats.dexterity}",
            f"constitution: {agent.blstats.constitution}",
            f"intelligence: {agent.blstats.intelligence}",
            f"wisdom: {agent.blstats.wisdom}",
            f"charisma: {agent.blstats.charisma}",
            f'energy: {agent.blstats.energy} / {agent.blstats.max_energy}',
            f"depth: {agent.blstats.level_number}",
            f"dungeon number: {agent.blstats.dungeon_number}",
            f"gold: {agent.blstats.gold}",
            f"level: {agent.blstats.experience_level}",
            f"exp: {agent.blstats.experience_points}",
            f"score: {agent.blstats.score}",
            f"encumbrance: {encumbrance_lookup[agent.blstats.carrying_capacity]}",
            f"hunger: {hunger_lookup[agent.blstats.hunger_state]}",
            f"alignment: {alignment_map[agent.blstats.alignment]}",
            f"conditions: {conditions}",
        ])

class GeneralContextDescriptor(Descriptor):
    def describe(self, agent: NetHackAgent) -> str:
        # texwrap.dedent doesn't work properly for this and I don't care enough to figure out why
        if agent.enable_finish_task_skill:
            return cleandoc("""
            You are an agent that is playing a partially observable rogue-like.
            You will be given tasks from the user which you have to fulfill.
            Do not act on your own, if you are done mark the given task as done.
            Always respond in first person.
            """.strip())
        else:
            return cleandoc("""
            You are an agent that is playing a partially observable rogue-like.
            Always respond in first person.
            """.strip())


class NetHackContextDescriptor(Descriptor):
    def describe(self, agent: NetHackAgent) -> str:
        # texwrap.dedent doesn't work properly for this and I don't care enough to figure out why
        if agent.enable_finish_task_skill:
            return cleandoc("""
            You are an agent that is playing the rogue-like NetHack.
            You will be given tasks from the user which you have to fulfill.
            Do not act on your own, if you are done mark the given task as done.
            Always respond in first person.
            """.strip())
        else:
            return cleandoc("""
            You are an agent that is playing the rogue-like NetHack.
            Always respond in first person.
            """.strip())
    
class AgentInformationDescriptor(Descriptor):
    def __init__(self):
        self.lang = NLELanguageObsv()

    def describe(self, agent: NetHackAgent) -> str:
        game_message = self.lang.text_message(agent.last_observation["tty_chars"]).decode("latin-1")
        avoid_monster_message = "Skills will try to avoid monsters." if agent.avoid_monsters else "Skills will fight monsters if they are in the way."
        pray_message = f"You prayed {agent.blstats.time - agent.data.last_pray_time} turns ago." if agent.data.last_pray_time else "You haven't prayed yet."
        return cleandoc(f"""
        You are at position ({agent.blstats.x}, {agent.blstats.y}).
        avoid_monster_flag is set to {agent.avoid_monsters}: {avoid_monster_message}
        Game message '{game_message if game_message else ""}'.
        {pray_message}
        """.strip())
    
class TaskDescriptor(Descriptor):
    def describe(self, agent: NetHackAgent) -> str:
        return agent.task if agent.task else "Win the game."
    
class ExplorationStatusDescriptor(Descriptor):
    def describe(self, agent: NetHackAgent) -> str:
        messages = []
        mentioned_positions = set()
        for room_id in agent.current_level.graph.get_rooms():
            room = agent.current_level.graph.get_room_data(room_id)

            unexplored_tiles = [(x,y)
                for (x,y) in room.get_adjacent_tiles()
                if agent.current_level.features[y,x] == G.SS.S_stone and not agent.current_level.has_seen[y,x]
            ]

            for x, y in unexplored_tiles:
                if self._is_reachable(agent, x, y):
                    messages.append(f"{room.type} {room_id} can be further explored.")
                    break

            for x, y in unexplored_tiles:
                res = self._get_blocking_glyph(agent, x, y)
                if res is None:
                    continue
                bx, by, blocking_glyph = res
                if (bx,by) in mentioned_positions:
                    continue
                
                mentioned_positions.add((bx,by))
                messages.append(f"{describe.describe_glyph(blocking_glyph)} at {(bx,by)} blocks progress in {room.type} {room_id}.")

        if len(messages) == 0:
            return "Nothing to explore, but the game has a lot of hidden passages."
        return "\n".join(messages)
    
    def _is_reachable(self, agent: NetHackAgent, x, y):
        for nx, ny in agent.current_level.get_neighbors(x, y):
            if agent.get_path_to(nx, ny, bump_into_unwalkables=False):
                return True
        return False
    
    def _get_blocking_glyph(self, agent: NetHackAgent, x, y):
        for nx, ny in agent.current_level.get_neighbors(x, y):
            if agent.current_level.get_feature_glyph(nx, ny) in G.CLOSED_DOORS:
                return nx, ny, agent.current_level.get_feature_glyph(nx, ny)
            elif agent.current_level.get_object_glyph(nx, ny) in G.BOULDERS:
                return nx, ny, agent.current_level.get_object_glyph(nx, ny)
        return None
        
class InventoryDescriptor(Descriptor):
    def __init__(self):
        self.lang = NLELanguageObsv()

    def describe(self, agent: NetHackAgent) -> str:
        inv_strs = agent.last_observation["inv_strs"]
        inv_letters = agent.last_observation["inv_letters"]
        tty_chars = agent.last_observation["tty_chars"]
        return self.lang.text_inventory(inv_strs, inv_letters).decode("latin-1")
    
class CurrentRoomDescriptor(Descriptor):
    def __init__(self, included_feature_glyphs=DESCRIPTION_INCLUDE_FEATURE_GLYPHS):
        self.included_feature_glyphs = included_feature_glyphs

    def describe(self, agent: NetHackAgent) -> str:
        room_id = agent.current_room_id
        room = agent.current_level.graph.get_room_data(room_id)
        # Describe all the tiles
        tile_descriptions = []
        for (x, y) in room.get_interior_tiles():
            tile_descriptions.append(self._describe_tile(agent, x, y))
        for (x, y) in room.get_exit_tiles():
            tile_descriptions.append(self._describe_exit_tile(agent, x, y, room_id))
        tile_descriptions = [f"-{desc}" for desc in tile_descriptions if desc is not None]

        return "\n".join([
            f"This is a {room.type} with id {room_id}.",
            f"{room.type} contains:",
            *tile_descriptions,
        ])
    
    def _describe_tile(self, agent, x, y):
        content_description = self._describe_tile_content(agent, x, y)
        if content_description is None:
            return None

        offset_description = describe.offset_to_compass((x - agent.blstats.x, y - agent.blstats.y))
        distance_description = self._describe_distance(agent, x, y)
        
        return f"{content_description} {offset_description} at ({x}, {y}) {distance_description}"
    
    def _describe_exit_tile(self, agent, x, y, current_room_id):
        content_description = self._describe_tile_content(agent, x, y)
        if content_description is None:
            return None
        
        distance_description = self._describe_distance(agent, x, y)

        exit_room_id = agent.current_level.graph.get_first_room_at(x, y, ignore_id=current_room_id)
        if exit_room_id is not None:
            exit_room = agent.current_level.graph.get_room_data(exit_room_id)
            return f"exit containing {content_description} {distance_description} at ({x}, {y}). It leads to {exit_room.type} {exit_room_id}."
        else:
            return f"exit containing {content_description} {distance_description} at ({x}, {y}). Unknown where this exit leads to."
    
    def _describe_tile_content(self, agent: NetHackAgent, x, y):
        # Collects the glyphs we want to describe for this tile
        feature_glyph = agent.current_level.get_feature_glyph(x, y)
        glyphs = [
            feature_glyph if feature_glyph in self.included_feature_glyphs else None,
            # Objects below the agent are shown in the game message.
            # We also have trouble detecting if an object below the agent disappeared.
            # So just do not include it in here.
            None if x == agent.blstats.x and y == agent.blstats.y else agent.current_level.get_object_glyph(x, y),
            agent.current_level.get_monster_glyph(x, y)
        ]
        glyphs = [g for g in glyphs if g is not None]
        if len(glyphs) == 0:
            return None
        
        content_description = ','.join([describe.describe_glyph(g) for g in glyphs])
        content_description = f"[{content_description}]"
        return content_description
    
    def _describe_distance(self, agent, x, y):
        distance = agent.distance_to(x, y)
        distance_description = f"reachable in {distance} steps" if distance != -1 else "currently unreachable"
        return distance_description
    
class OtherRoomsDescriptor(Descriptor):
    def describe(self, agent: NetHackAgent) -> str:
        other_rooms = [room_id for room_id in agent.current_level.graph.get_rooms() if room_id != agent.current_room_id]
        if len(other_rooms) == 0:
            return "No other known rooms"

        other_room_descriptions = []
        for room_id in other_rooms:
            room = agent.current_level.graph.get_room_data(room_id)

            # Collect all glyphs in the room
            glyphs = []
            for (x, y) in room.get_tiles():
                feature_glyph = agent.current_level.get_feature_glyph(x, y)
                if feature_glyph in DESCRIPTION_INCLUDE_FEATURE_GLYPHS:
                    glyphs.append(feature_glyph)
                glyphs.append(agent.current_level.get_monster_glyph(x, y))
                glyphs.append(agent.current_level.get_object_glyph(x, y))
            glyphs = [g for g in glyphs if g is not None]

            # Describe distance
            distances = [agent.distance_to(x,y) for (x,y) in room.get_tiles()]
            distances = [d for d in distances if d is not None]
            min_distance = min(distances, default=None)
            if min_distance is None:
                distance_description = "currently unreachable"
            else:
                distance_description = f"reachable in {min_distance} steps"

            # Describe
            glyph_descriptions = [describe.describe_glyph(g) for g in glyphs]
            glyph_descriptions = ", ".join(glyph_descriptions)
            other_room_descriptions.append(f"{room.type} with id {room_id} containing [{glyph_descriptions}] {distance_description}.")
        return "\n".join(other_room_descriptions)
    
class RoomsObjectFeatureDescriptor(Descriptor):
    """
    You are in (room/corridor) X which contains:
    (No objects or features)
    - X( and Z) at (x,y) (currently unreachable/reachable in X steps)
    - ...

    (Room/Corridor) X (currently unreachable/reachable in X steps) contains:
    (No objects or features)
    - X( and Z) at (x,y) (currently unreachable/reachable in X steps)
    - ...
    """
    def describe(self, agent: NetHackAgent) -> str:
        current_room_description = "Unable to determine in which room you are."
        room_descriptions = []
        for room_id in agent.current_level.graph.get_rooms():
            room = agent.current_level.graph.get_room_data(room_id)
            content_description = self._describe_room_content(agent, room)
            if room_id == agent.current_room_id:
                room_header = f"You are in {room.type} {room_id} which contains:"
                current_room_description = "\n".join([room_header, content_description])
            else:
                distance_description = self._describe_distance(self._get_min_room_distance(agent, room))
                room_header = f"{room.type} {room_id} {distance_description} contains:"
                room_descriptions.append("\n".join([room_header, self._describe_room_content(agent, room)]))

        return "\n\n".join([current_room_description, *room_descriptions])

    def _describe_room_content(self, agent: NetHackAgent, room: RoomData):
        descriptions = []
        for (x, y) in room.get_tiles():
            glyphs = []
            feature_glyph = agent.current_level.get_feature_glyph(x, y)
            if feature_glyph in DESCRIPTION_INCLUDE_FEATURE_GLYPHS:
                glyphs.append(feature_glyph)
            if agent.current_level.get_object_glyph(x, y):
                glyphs.append(agent.current_level.get_object_glyph(x, y))

            if len(glyphs) > 0:
                descriptions.append(self._describe_glyph_list(agent, glyphs, x, y))

        if len(descriptions) == 0:
            return "No objects or features"
        else:
            return "\n".join(descriptions)

    def _describe_glyph_list(self, agent: NetHackAgent, glyphs, x, y):
        glyphs_description = " and ".join([describe.describe_glyph(g) for g in glyphs])
        path = agent.get_path_to(x, y)
        distance_description = self._describe_distance(len(path) - 1 if path else None)
        return f"- {glyphs_description} at ({x},{y}) {distance_description}"

    def _get_min_room_distance(self, agent: NetHackAgent, room: RoomData):
        paths = [agent.get_path_to(x, y, bump_into_unwalkables=False) for x, y in room.get_tiles()]
        min_dist = min([len(p) for p in paths if p is not None], default=None)
        return min_dist
    
    def _describe_distance(self, dist):
        if dist is None:
            return "currently unreachable"
        else:
            return f"reachable in {dist} steps"
        
class CloseMonsterDescriptor(Descriptor):
    """
    (No monsters close to you)
    - tamed cat at (x,y) X steps northeast
    - ...
    """

    def __init__(self, distance_threshold=10):
        self.distance_threshold = distance_threshold

    def describe(self, agent: NetHackAgent) -> str:
        descriptions = []
        for glyph, pos in agent.current_level.get_monsters():
            path = agent.get_path_to(pos.x, pos.y)
            if path is None or len(path) - 1 > self.distance_threshold:
                continue

            compass_direction = describe.offset_to_compass(pos.x - agent.blstats.x, pos.y - agent.blstats.y)
            distance_description = f"{len(path) - 1} steps {compass_direction}"
            descriptions.append(f"- {describe.describe_glyph(glyph)} at ({pos.x},{pos.y}) {distance_description}")

        if len(descriptions) == 0:
            return "No monsters close to you"
        else:
            return "\n".join(descriptions)
    
class DistantMonsterDescriptor(Descriptor):
    """
    (No monsters in the distance)
    - tamed cat at (x,y) (currently unreachable/reachable in X steps)
    - ...
    """

    def __init__(self, distance_threshold=10):
        self.distance_threshold = distance_threshold

    def describe(self, agent: NetHackAgent) -> str:
        descriptions = []
        for glyph, pos in agent.current_level.get_monsters():
            path = agent.get_path_to(pos.x, pos.y)
            if path is None:
                descriptions.append(f"{describe.describe_glyph(glyph)} at ({pos.x},{pos.y}) currently unreachable")
            elif len(path) - 1 > self.distance_threshold:
                descriptions.append(f"- {describe.describe_glyph(glyph)} at ({pos.x},{pos.y}) reachable in {len(path) - 1} steps")
        
        if len(descriptions) == 0:
            return "No monsters in the distance"
        else:
            return "\n".join(descriptions)