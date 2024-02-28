from netplay.nethack_agent.tracking import NethackDataTracker, NethackEvent, Level, BLStats
from netplay.nethack_agent.pathfinding import LevelPathfinder, BFSResults
from netplay.logging.step_logger import StepLogger
from netplay.logging.video_renderer import AgentVideoRenderer

import gymnasium as gym
import gradio

import os
from enum import Enum
from functools import wraps
from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Any, Optional, Iterator, Callable

STEP_JSON_LOG_FILE = "step.json"

@dataclass
class StepData:
    action: Any
    observation: Any
    reward: float
    done: bool
    truncated: bool
    info: Any
    events: List[NethackEvent]

class StepStatus(Enum):
    running = 0
    completed = 1
    failed = 2
    error = 3

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
class ThoughtType(Enum):
    System = 0
    AI = 1

@dataclass
class Step:
    status: StepStatus
    thoughts: str
    thought_type: ThoughtType
    step_data: Optional[StepData]

    def is_done(self):
        return self.status != StepStatus.running
    
    def has_completed(self):
        return self.status == StepStatus.completed

    def has_failed(self):
        return self.status in [StepStatus.failed, StepStatus.error]
    
    def has_thoughts(self):
        return self.thoughts != None

    def executed_action(self) -> bool:
        return self.step_data is not None

    @classmethod
    def completed(cls, thoughts: str=None, thought_type=ThoughtType.System):
        return Step(
            status=StepStatus.completed,
            thoughts=thoughts,
            thought_type=thought_type,
            step_data=None
        )
    
    @classmethod
    def failed(cls, thoughts: str=None, thought_type=ThoughtType.System):
        return Step(
            status=StepStatus.failed,
            thoughts=thoughts,
            thought_type=thought_type,
            step_data=None
        )
    
    @classmethod
    def error(cls, thoughts: str=None, thought_type=ThoughtType.System):
        return Step(
            status=StepStatus.error,
            thoughts=thoughts,
            thought_type=thought_type,
            step_data=None
        )

    @classmethod
    def act(cls, step_data: StepData, thoughts: str=None, thought_type=ThoughtType.System):
        return Step(
            status=StepStatus.running,
            thoughts=thoughts,
            thought_type=thought_type,
            step_data=step_data
        )
    
    @classmethod
    def think(cls, thoughts: str, thought_type=ThoughtType.System):
        return Step(
            status=StepStatus.running,
            thoughts=thoughts,
            thought_type=thought_type,
            step_data=None
        )
    
class AgentRenderer:
    @abstractmethod
    def init(self) -> List[gradio.components.Component]:
        pass

    @abstractmethod
    def update(self) -> List[any]:
        pass

def log_steps(fn: Callable[["NethackBaseAgent", str], Iterator[Step]]):
    @wraps(fn)
    def wrapper(self: NethackBaseAgent, *args, **kwargs) -> Iterator[Step]:
        for step in fn(self, *args, **kwargs):
            is_ai_thought=step.thought_type == ThoughtType.AI
            if step.executed_action():
                self.video_renderer.add_step(step.step_data.observation, step.step_data.action, step.thoughts, is_ai_thought=is_ai_thought)
            else:
                self.video_renderer.add_thoughts(step.thoughts, is_ai_thought=is_ai_thought)

            if step.thoughts:
                self.logger.log_json(
                    data={"thoughts": step.thoughts, "thought_type": step.thought_type.name},
                    file_name=STEP_JSON_LOG_FILE
                )

            yield step

    return wrapper
    
class NethackBaseAgent:
    def __init__(self,
        env: gym.Env,
        log_folder: str,
        render=False
    ):
        self.env = env

        self.data: NethackDataTracker = None
        self.last_bfs_results: LevelPathfinder = None
        self.last_observation = None
        self.current_room_id: int = None
        self.timestep = 0
        self.avoid_monsters = False

        self.logger = StepLogger(log_folder=log_folder)
        self.video_renderer = AgentVideoRenderer(video_path=os.path.join(log_folder, "video.mp4"), render=render)

    def init(self):
        observation, info = self.env.reset()
        self.last_observation = observation

        self.data = NethackDataTracker()
        self.data.init(observation)
        self.last_bfs_results = LevelPathfinder.from_level(self.blstats.x, self.blstats.y, self.current_level, self.can_squeeze)
        self.current_room_id = self.current_level.graph.get_first_room_at(self.blstats.x, self.blstats.y)

        self.video_renderer.init(observation)

    def step(self, action, thoughts=None) -> Step:
        self.timestep += 1
        self.logger.start_next_step()

        observation, reward, done, truncated, info = self.env.step(action)

        # Update
        events = self.data.update(action, observation)
        self.last_bfs_results = LevelPathfinder.from_level(self.blstats.x, self.blstats.y, self.current_level, self.can_squeeze)
        self.current_room_id = self.current_level.graph.get_first_room_at(self.blstats.x, self.blstats.y, prefer_id=self.current_room_id)

        step_data = StepData(
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
            events=events
        )

        if done:
            # Re-Initialize
            self._on_game_end(step_data)
            self.init()
        else:
            self.last_observation = observation
        return Step.act(step_data, thoughts)
    
    @abstractmethod
    def run(self) -> Iterator[Step]:
        pass

    def _on_game_end(self, step_data: StepData):
        pass

    def get_renderer(self) -> AgentRenderer:
        return None
    
    def set_current_room(self, room_id: int) -> bool:
        room_id = self.current_level.graph.resolve_id(room_id)
        room = self.current_level.graph.get_room_data(room_id)
        if room.is_inside(self.blstats.x, self.blstats.y):
            self.current_room_id = room_id
            return True
        
        return False
    
    def waiting_for_popup(self):
        if self.env.unwrapped.waiting_for_yn:
            return True
        if self.env.unwrapped.waiting_for_space:
            return True
        if self.env.unwrapped.waiting_for_line:
            return True
        return False
    
    def get_path_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters

        return self.last_bfs_results.get_path_to(x,y,bump_into_unwalkables, avoid_monsters=self.avoid_monsters)
    
    def distance_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters

        return self.last_bfs_results.distance_to(x,y,bump_into_unwalkables, avoid_monsters=self.avoid_monsters)
    
    def get_distance_map(self, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters
        return self.last_bfs_results.get_distance_map(avoid_monsters=avoid_monsters)
    
    def get_walkable_mask(self, treat_boulder_unwalkable=True):
        return self.current_level.get_walkable_mask(treat_boulder_unwalkable=treat_boulder_unwalkable)
    
    def get_diagonal_walkable_mask(self, treat_boulder_unwalkable=True, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters
        return self.current_level.get_diagonal_walkable_mask(treat_boulder_unwalkable=treat_boulder_unwalkable)
    
    def close(self):
        self.video_renderer.close()
        self.env.close()
    
    @property
    def showing_more_message(self) -> bool:
        # Seems like this is the only way to detect --More-- messages
        # As last_observation["message"] does not contain this
        for line in self.last_observation["tty_chars"]:
            line = "".join([chr(c) for c in line])
            if "--More--" in line:
                return True
        return False

    @property
    def current_game_message(self) -> str:
        return "".join([chr(c) for c in self.last_observation["message"] if c != 0])

    @property
    def can_squeeze(self) -> bool:
        return False # ToDo
    
    @property
    def blstats(self) -> BLStats:
        return self.data.blstats

    @property
    def current_level(self) -> Level:
        return self.data.current_level