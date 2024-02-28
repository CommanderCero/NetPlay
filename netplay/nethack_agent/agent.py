from netplay.core.agent_base import NethackBaseAgent, AgentRenderer, Step, StepData, log_steps, ThoughtType
from netplay.core.skill_repository import SkillRepository
from netplay.core.skill import skill, Skill
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper, RawKeyPress
from netplay.core.descriptor import Descriptor
from netplay.core.skill_selector import SkillSelector
import netplay.nethack_agent.tracking as tracking
import netplay.nethack_utils.glyphs as G

import numpy as np
from nle_language_wrapper import NLELanguageWrapper
from nle.nethack import actions
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from openai.error import RateLimitError

from typing import Tuple, Dict, Any, List, Optional, Iterator
from time import sleep
import warnings
import traceback
import logging
logger = logging.getLogger(__name__)

DESCRIPTION_INCLUDE_FEATURE_GLYPHS = [
    *G.STAIRCASES,
    *G.LADDERS,
    *G.ALTARS,
    *G.SINKS,
    *G.FOUNTAINS,
    *G.TRAPS,
    *G.DOORS,
    *G.DOORWAYS,
    *G.DRAWBRIDGES,
    *G.IRON_BARS,
    *G.THRONES,
    *G.GRAVES
]

@skill(
    "finish_task",
    "Use this skill when the task has been fulfilled. DO NOT CONTINUE playing without an task.",
    parameters=[]
)
def finish_task_skill(agent: "NetHackAgent"):
    assert False

class AgentMemory:
    def __init__(self, llm, token_limit=500):
        self.llm = llm
        self._token_limit = token_limit
        self._messages = []
        self._token_count = 0

    def get_messages(self) -> List[BaseMessage]:
        return [msg for (_, msg) in self._messages]

    def add_message(self, message: BaseMessage):
        size = self.llm.get_num_tokens(message.content)
        if size > self._token_limit:
            raise ValueError(f"The given memory contains {size} tokens, which exceeds the total token limit of {self._token_limit}")
        
        # Remove elements until there is space for the new memory
        while self._token_count + size > self._token_limit:
            self.pop_message()

        # Add new memory
        self._token_count += size
        self._messages.append((size, message))

    def add_user_message(self, content: str):
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.add_message(AIMessage(content=content))

    def add_system_message(self, content: str):
        self.add_message(SystemMessage(content=content))

    def pop_message(self):
        """Removes the oldest memory"""
        size, _ = self._messages.pop(0)
        self._token_count -= size

TASK_JSON_LOG_FILE = "task.json"
EXCEPTION_JSON_LOG_FILE = "exception.json"

class NetHackAgent(NethackBaseAgent):
    def __init__(self,
        env: NethackGymnasiumWrapper,
        state_descriptor: Descriptor,
        skill_selector: SkillSelector,
        llm,
        skills: SkillRepository,
        log_folder: str,
        max_memory_tokens = 500,
        max_skill_gamesteps = 100,
        max_tries_per_gamestep = 10,
        render=False,
        censor_nethack_messages=False,
        enable_finish_task_skill=True,
        update_hidden_objects=False
    ):
        super().__init__(env, log_folder, render)
        self.env = env
        self.state_descriptor = state_descriptor
        self.skill_selector = skill_selector
        self.llm = llm
        self.skills = skills
        self.message_history = AgentMemory(self.llm, token_limit=max_memory_tokens)
        self.task = None
        self.max_skill_gamesteps = max_skill_gamesteps
        self.max_tries_per_gamestep = max_tries_per_gamestep
        self.censor_nethack_messages = censor_nethack_messages
        self.enable_finish_task_skill = enable_finish_task_skill
        self.update_hidden_objects = update_hidden_objects

    def init(self):
        super().init()
        self.message_history.add_system_message(content="Started a new game.")
    
    def _on_game_end(self, step_data: StepData):
        if step_data.info["is_ascended"]:
            # Lol as if I ever need this case
            self.message_history.add_system_message(content="Congratulations, you ascended!")
        else:
            # ToDo Find a nicer way to access death information
            self.message_history.add_system_message(content=f"The game ended. Reason: {self.env.unwrapped.gym_env.nethack.how_done().name}.")

    def set_task(self, new_task: str):
        self.task = new_task
        self.message_history.add_user_message(new_task)
        self.video_renderer.add_thoughts(f"Received a new task: {new_task}")
        self.logger.log_json(
            data={"task": new_task},
            file_name=TASK_JSON_LOG_FILE
        )

    @log_steps
    def solve_manual_task(self, task: str):
        skill, kwargs = self.skills.parse_skill_call(task)
        strategy = self._execute_skill(skill, kwargs)
        strategy = self._skip_more_messages(strategy)
        strategy = self._update_objects(strategy)
        yield from self._execute_step_generator(strategy)

    @log_steps
    def run(self) -> Iterator[Step]:
        yield from self._execute_step_generator(self._solve_task())

    def _solve_task(self) -> Iterator[Step]:
        current_gamestep = self.blstats.time
        try_counter = 0
        while True:
            try:
                # Choose a skill
                skill_choice = self.skill_selector.choose_skill(self)

                # Are we done?
                if skill_choice.skill.name == finish_task_skill.skill.name:
                    self.task = None
                    yield Step.completed(f"Finished Task: {skill_choice.thoughts.speak}", thought_type=ThoughtType.AI)
                    return
                yield Step.think(skill_choice.thoughts.speak, thought_type=ThoughtType.AI)

                # Construct and run a generator that executes the skill and skips --more-- messages
                strategy = self._execute_skill(skill_choice.skill, skill_choice.skill_kwargs)
                strategy = self._skip_more_messages(strategy)
                strategy = self._update_objects(strategy)
                yield from strategy
            except RateLimitError as e:
                warnings.warn(f"Rate limit reached: {e}")
                print("Waiting for 1 minute...")
                sleep(60)
            except Exception as e:
                warnings.warn(f"A exception occured: {e}")
                self.logger.log_json(data={"exception": str(e), "with_traceback": traceback.format_exc()}, file_name=EXCEPTION_JSON_LOG_FILE)
                yield Step.think(f"A exception occured: {e}")

            # Avoid calling the llm forever without making progress
            if current_gamestep == self.blstats.time:
                try_counter += 1
                if try_counter >= self.max_tries_per_gamestep:
                    yield Step.failed(f"Unable to fulfill task, made no progress for {self.max_tries_per_gamestep} steps.")
                    return
            else:
                current_gamestep = self.blstats.time
                try_counter = 0

            
    def _skip_more_messages(self, generator: Iterator[Step]) -> Iterator[Step]:
        for step in generator:
            yield step
            if step.is_done():
                return
            
            while self.showing_more_message:
                yield self.step(RawKeyPress.KEYPRESS_SPACE)

    def _update_objects(self, generator: Iterator[Step]) -> Iterator[Step]:
        for step in generator:
            yield step

        # There was an issue with hidden objects still showing up in the environment description
        # This fixes it by asking the game to hide monsters
        # This updates our tracked data
        # Not sure if not yielding the steps will cause any issues, so far it hasn't
        if self.update_hidden_objects and not self.waiting_for_popup():
            self.step(actions.Command.EXTCMD)
            self.step(RawKeyPress.KEYPRESS_t)
            self.step(RawKeyPress.KEYPRESS_e)
            self.step(RawKeyPress.KEYPRESS_ENTER)
            self.step(RawKeyPress.KEYPRESS_c)
            self.step(RawKeyPress.KEYPRESS_ESC)
            self.step(RawKeyPress.KEYPRESS_ESC)

    def _execute_skill(self, skill: Skill, skill_kwargs: Dict[str, Any]) -> Iterator[Step]:
        kwargs_str = [str(x) for x in skill_kwargs.values()]
        skill_description = " ".join([skill.name, *kwargs_str])
        yield Step.think(f"Executing skill '{skill_description}'.")

        start_ingame_time = self.blstats.time
        for step in skill(self, **skill_kwargs):
            if step.is_done():
                thoughts = f"Skill '{skill_description}' {step.status}"
                thoughts += f": {step.thoughts}" if step.has_thoughts() else ""
                yield Step(step.status, thoughts, step.thought_type, step.step_data)
                return
            yield step
            
            if (self.blstats.time - start_ingame_time) >= self.max_skill_gamesteps:
                yield Step.think(f"Skill has been running for {(self.blstats.time - start_ingame_time)} timesteps without interruption. Rethinking.")
                return
            
            if step.executed_action():
                interrupt_event_types = (tracking.DungeonLevelChangeEvent, tracking.TeleportEvent, tracking.NewGlyphEvent, tracking.LowHealthEvent)
                interrupt_events = [event for event in step.step_data.events if isinstance(event, interrupt_event_types)]
                if len(interrupt_events) != 0:
                    yield Step.think(f"Interrupting skill to rethink because '{interrupt_events[0].describe()}'.")
                    return

    def _execute_step_generator(self, generator: Iterator[Step]) -> Iterator[Step]:
        # Wraps any step-generator to store updates in the LLMs memory
        for step in generator:
            game_ended = step.executed_action() and step.step_data.done
            if game_ended:
                break
            
            if step.has_thoughts():
                if step.thought_type == ThoughtType.System:
                    self.message_history.add_system_message(step.thoughts)
                elif step.thought_type == ThoughtType.AI:
                    self.message_history.add_ai_message(step.thoughts)

            if step.executed_action():
                # Store events
                if len(step.step_data.events) > 0:
                    messages = [event.describe() for event in step.step_data.events]
                    self.message_history.add_system_message(content="\n".join(messages))

            yield step

    def describe_current_state(self):
        return self.state_descriptor.describe(self)
    
    def describe_action(self, action):
        return NLELanguageWrapper.all_nle_action_map[action][0]
    
    def get_renderer(self):
        return NethackAgentRenderer(self)
    
import gradio as gr
from netplay.nethack_utils.nle_wrapper import render_ascii_map
from minihack.tiles.glyph_mapper import GlyphMapper

class NethackAgentRenderer(AgentRenderer):
    def __init__(self, agent: NetHackAgent):
        self.agent = agent
        self.mapper = GlyphMapper()

    def init(self):
        observation_size = gr.Textbox("", label="Observation Tokens: ",)
        with gr.Tab("Rooms"):
            initial_img = self._render_rooms()
            room_img = gr.Image(value=initial_img, height=initial_img.shape[0], width=initial_img.shape[1], show_download_button=False, show_label=False)
        with gr.Tab("Walkable"):
            initial_img = self._render_walkable_map()
            walkable_img = gr.Image(value=initial_img, height=initial_img.shape[0], width=initial_img.shape[1], show_download_button=False, show_label=False)
        with gr.Tab("Diagonal Walkable"):
            initial_img = self._render_diagonal_walkable_map()
            diagonal_walkable_img = gr.Image(value=initial_img, height=initial_img.shape[0], width=initial_img.shape[1], show_download_button=False, show_label=False)
        #memory_content = gr.TextArea(value=self.agent.memory.concatenate_memories(), interactive=False)
        skills = gr.TextArea(value="\n".join(reversed(self._get_memory_lines())), interactive=False)
        return [observation_size, room_img, walkable_img, diagonal_walkable_img, skills]

    def update(self):
        return [
            self._get_num_tokens(), 
            self._render_rooms(),
            self._render_walkable_map(),
            self._render_diagonal_walkable_map(),
            #self.agent.memory.concatenate_memories(), 
            "\n".join(reversed(self._get_memory_lines()))
        ]
    
    def _get_memory_lines(self):
        lines = []
        for message in self.agent.message_history.get_messages():
            lines.extend(message.content.split("\n"))
        return lines
    
    def _get_num_tokens(self):
        observation_token_count = self.agent.llm.get_num_tokens(self.agent.describe_current_state())
        return str(observation_token_count)

    def _render_rooms(self):
        level = self.agent.current_level
        room_ids = np.full(level.shape, " ", dtype=str)
        for room_id in level.graph.get_rooms():
            room = level.graph.get_room_data(room_id)
            room_ids[room.get_interior_mask()] = str(room_id)
            room_ids[room.get_exit_mask()] = "E"
        
        #room = level.graph.get_first_room_at(self.agent.blstats.x, self.agent.blstats.y)
        return render_ascii_map(room_ids, self.agent.env.font) #self.mapper._glyph_to_rgb(features)
    
    def _render_walkable_map(self):
        import netplay.nethack_agent.skills as sk
        return render_ascii_map(self.agent.get_walkable_mask().astype("|S1").astype(str), self.agent.env.font)
    
    def _render_diagonal_walkable_map(self):
        return render_ascii_map(self.agent.get_diagonal_walkable_mask().astype("|S1").astype(str), self.agent.env.font)