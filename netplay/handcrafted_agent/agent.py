import netplay.nethack_agent.skills as skills
import netplay.nethack_agent.tracking as tracking
import netplay.nethack_utils.glyphs as G
from netplay.nethack_agent.describe import describe_glyph
from netplay.core.skill import Skill
from netplay.core.agent_base import NethackBaseAgent, AgentRenderer, Step, StepData, log_steps
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper, RawKeyPress

import nle.nethack as nh

import numpy as np

from enum import IntEnum
from collections import namedtuple
from typing import Iterator, Tuple, Any, Dict, List, Generator
import logging
logger = logging.getLogger(__name__)

Glyph = namedtuple("Glyph", "value pos")

class Hunger(IntEnum):
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4

    def is_hungry(self):
        return self.value >= Hunger.HUNGRY

class HandcraftedAgent(NethackBaseAgent):
    def __init__(self,
        env: NethackGymnasiumWrapper,
        log_folder: str,
        render=False
    ):
        super().__init__(env, log_folder, render=render)
        self.env = env
        self.running = False

    def init(self):
        super().init()
        self.pickup_mask = np.zeros(self.current_level.shape, dtype=bool)
        self.last_level = (self.blstats.dungeon_number, self.blstats.level_number)
        self.last_object_map = np.array(self.current_level.objects)
        self.kick_door_map = np.zeros(self.current_level.shape, dtype=np.int8)

        self.max_tries_per_gamestep = 10

    def step(self, action, thoughts=None):
        step = super().step(action, thoughts)

        current_level = (self.blstats.dungeon_number, self.blstats.level_number)
        if self.last_level != current_level:
            # Reset mask on new level
            self.pickup_mask = np.zeros(self.current_level.shape, dtype=bool)
            self.last_level = (self.blstats.dungeon_number, self.blstats.level_number)
            self.last_object_map = np.array(self.current_level.objects)
            self.kick_door_map = np.zeros(self.current_level.shape, dtype=np.int8)
        else:
            # Because we do not always succeed with picking up objects, we remember where we tried to pickup objects
            # We only try again when the object at that position changes
            changed_objects = np.where(self.last_object_map != self.current_level.objects)
            self.last_object_map = np.array(self.current_level.objects)
            self.pickup_mask[changed_objects] = False
        return step

    @log_steps
    def run(self) -> Iterator[Step]:
        self.running = True
        try_counter = 0
        current_gamestep = self.blstats.time
        while self.running:
            for step in self.run_next_skill():
                yield step

                if not self.running:
                    break
                if step.executed_action():
                    interrupt_events = [event for event in step.step_data.events if isinstance(event, (tracking.DungeonLevelChangeEvent, tracking.TeleportEvent, tracking.NewGlyphEvent))]
                    if len(interrupt_events) != 0:
                        yield Step.think(f"Interrupting skill to rethink because '{interrupt_events[0].describe()}'.")
                        break
                # Avoid endless loop
                if current_gamestep == self.blstats.time:
                    try_counter += 1
                    if try_counter >= self.max_tries_per_gamestep:
                        yield Step.failed(f"Unable to make progress for {self.max_tries_per_gamestep} steps.")
                        return
                else:
                    current_gamestep = self.blstats.time
                    try_counter = 0

    def run_next_skill(self) -> Tuple[Skill, Dict[str, Any]]:
        yield from self.handle_popups()

        if len(self.get_nearby_hostile_monsters()) > 0 and self.blstats.hitpoints > 3:
            yield from self.fight()
            return
        
        # Idk how many max hitpoints we have, so lets just keep them above 60%
        if (self.blstats.hitpoints / self.blstats.max_hitpoints) < 0.6:
            result = yield from self.try_heal()
            if not result.has_failed():
                return
            
        if Hunger(self.blstats.hunger_state).is_hungry():
            result = yield from self.try_eat()
            if not result.has_failed():
                return
            
        result = yield from self.try_pickup_useful_object()
        if not result.has_failed():
            return
        
        # Try going to the next level if we would have to start searching for hidden corridors
        dis = self.last_bfs_results.bfs_results.dist
        visit_mask = skills.compute_visit_mask(self) & (dis != -1)
        if not visit_mask.any():
            result = yield from self.move_to_next_level()
            if not result.has_failed():
                return
        
        yield Step.think("Exploring level")
        yield from self.explore()

    def handle_popups(self) -> Generator[Step, None, Step]:
        while True:
            if self.env.unwrapped.waiting_for_yn:
                yield self.step(RawKeyPress.KEYPRESS_y, "Pressing yes to resolve it.")
                if self.env.unwrapped.waiting_for_yn:
                    yield self.step(RawKeyPress.KEYPRESS_ESC, "Pressing ESC to abort.")
            if self.env.unwrapped.waiting_for_space:
                yield self.step(RawKeyPress.KEYPRESS_SPACE, "Pressing space to advance.")
                continue
            if self.env.unwrapped.waiting_for_line:
                yield self.step(RawKeyPress.KEYPRESS_ESC, "Skipping line input.")
                continue
            # Nothing left to do
            break
        return Step.completed()
    
    def try_heal(self) -> Generator[Step, None, Step]:
        # Not many !easy! ways to heal ourselves, except potions
        # Although unlikely this will even work, as potions are usually unidentified
        for letter, glyph, object_class in zip(self.last_observation["inv_letters"], self.last_observation["inv_glyphs"], self.last_observation["inv_oclasses"]):
            if object_class != nh.POTION_CLASS:
                continue
            obj_id = nh.glyph_to_obj(glyph)
            obj_description = nh.objdescr.from_idx(obj_id).oc_descr
            if "healing" in obj_description:
                yield Step.think(f"Drinking {chr(letter)} - {obj_description} to heal.")
                yield from skills.drink(self, chr(letter))
                return Step.completed()
            
        if self.data.last_pray_time is None or self.data.last_pray_time > 1000:
            yield Step.think("Praying")
            yield from skills.pray(self)
            return Step.completed()

        return Step.failed("Found nothing to heal.")

    def try_eat(self) -> Iterator[Step]:
        for letter, object_class in zip(self.last_observation["inv_letters"], self.last_observation["inv_oclasses"]):
            if object_class != nh.FOOD_CLASS:
                continue

            yield Step.think(f"Trying to eat item '{chr(letter)}'")
            yield from skills.eat(self, chr(letter))
            return Step.completed()

        return Step.failed("Nothing to eat in inventory.")
    
    def try_pickup_useful_object(self) -> Iterator[Step]:
        objects = self.get_useful_objects()
        # Filter all objects that we already tried picking up
        objects = [o for o in objects if not self.pickup_mask[o.pos.y,o.pos.x]]
        if len(objects) == 0:
            return Step.failed("No useful object to pickup.")
        
        closest_object = min(objects, key=lambda o: self.last_bfs_results.distance_to(o.pos.x, o.pos.y))
        yield Step.think(f"Trying to pickup {describe_glyph(closest_object.value)} at ({closest_object.pos.x},{closest_object.pos.y})")
        # Remember that we tried to pickup this object
        self.pickup_mask[closest_object.pos.y,closest_object.pos.x] = True
        yield from skills.pickup(self, x=closest_object.pos.x, y=closest_object.pos.y)
        return Step.completed()

    def fight(self) -> Iterator[Step]:
        yield Step.think(f"Fighting nearby monsters")
        while not self.waiting_for_popup():
            monsters = self.get_nearby_hostile_monsters()
            if len(monsters) == 0:
                return
            
            closest_monster = min(monsters, key=lambda m: self.last_bfs_results.distance_to(m.pos.x, m.pos.y))
            yield from skills.melee_attack(self, closest_monster.pos.x, closest_monster.pos.y)

    def move_to_next_level(self):
        staircases = list(self.current_level.get_features([G.SS.S_dnstair]))
        if len(staircases) == 0:
            return Step.failed("No staircase to go to the next level.")
        
        for _, position in staircases:
            yield Step.think(f"Moving to the next level using the staircase at ({position.x},{position.y}).")
            yield from skills.move_to(self, position.x, position.y)
            if self.blstats.x == position.x and self.blstats.y == position.y:
                yield from skills.down(self)
                return Step.completed()
        return Step.failed(f"Unable to reach any staircase")
    
    def explore(self):
        # Kick open doors
        potential_locked_doors = self.data.current_level.door_open_attempts >= 4
        currently_closed_doors = np.isin(self.current_level.features, G.CLOSED_DOORS)
        kicked_mask = self.kick_door_map < 5
        target_doors_mask = currently_closed_doors & potential_locked_doors & kicked_mask
        if np.any(target_doors_mask):
            target_positions = zip(*np.where(target_doors_mask))
            y, x = min(target_positions, key=lambda m: self.last_bfs_results.distance_to(m[1], m[0]))
            yield Step.think(f"Kicking open door at ({x}, {y})")
            kicked_mask[y,x] += 1
            yield from skills.kick(self, x, y)
            return

        yield from skills.explore_level(self)
    
    def get_useful_objects(self) -> List[Glyph]:
        objects = []
        for glyph, position in self.current_level.get_objects():
            object_id = nh.glyph_to_obj(glyph)
            object_class = ord(nh.objclass(object_id).oc_class)
            if object_class == nh.FOOD_CLASS and not nh.glyph_is_body(glyph):
                # Pickup food, but not corpses as they are heavy and will rot as we carry them
                objects.append(Glyph(glyph, position))
            elif object_class == nh.POTION_CLASS:
                objects.append(Glyph(glyph, position))
        return objects

    def get_nearby_hostile_monsters(self) -> List[Glyph]:
        monsters = []
        for glyph, position in self.current_level.get_monsters():
            is_close = self.last_bfs_results.distance_to(position.x, position.y) <= 10
            is_friendly = nh.glyph_is_pet(glyph)
            if is_close and not is_friendly:
                monsters.append(Glyph(glyph, position))
        return monsters    

    def _on_game_end(self, step_data: StepData):
        self.running = False