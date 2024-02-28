import netplay.nethack_agent.skills as sk
from netplay.nethack_agent.agent import NetHackAgent, finish_task_skill
from netplay.core.skill_repository import SkillRepository, Skill

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, BaseMessage

import json
import jsonschema
from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from typing import Tuple, Dict, Any, Optional, List

skill_call_schema = {
    "type" : "object",
    "properties" : {
        "thoughts": {
            "type": "object",
            "properties": {
                "observations": {type: "string"},
                "reasoning": {type: "string"},
                "speak": {type: "string"}
            },
            "required": ["observations", "reasoning", "speak"],
            "additionalProperties": False
        },
        "skill": {
            "type": "object",
            "properties": {
                "name": {type: "string"},
            },
            "required": ["name"]
        }
    },
    "additionalProperties": False,
    "required": ["thoughts", "skill"]
}

CHOOSE_SKILL_PROMPT = dedent("""
Choose an skill from the given list of skills.
Output your response in the following JSON format:
{
    "thoughts": {
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user>"
    }
    "skill": {
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }
}
""".strip())

POPUP_CHOOSE_SKILL_PROMPT = dedent("""
Resolve the popup by pressing keys.
If you want to close the popup abort it using ESC or confirm your choices using enter or space.
Output your response in the following JSON format:
{
    "thoughts": {
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user>"
    }
    "skill": {
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }
}
""".strip())

FIX_JSON_PROMPT = PromptTemplate(template=dedent("""
You were tasked to choose a skill from the given list of skills.
Your output:
{wrong_json}

Error message:
{error_message}

Fix the error and output your response in the following JSON format:
{{
    "thoughts": {{
        "observations": "<Relevant observations from your last action. Pay close attention to what you set out to do and compare that to the games current state.>",
        "reasoning": "<Plan ahead.>",
        "speak": "<Summary of thoughts, to say to user.>"
    }}
    "skill": {{
        "name": "<The name of the skill>",
        "<param1_name>": "<The value for this parameter>",
        "<param2_name>": "<The value for this parameter>",
    }}
}}
""".strip()), input_variables=["wrong_json", "error_message"])

CHOOSE_SKILL_LOG_FILE = "choose_skill_prompt.json"

@dataclass
class Thoughts:
    observations: str
    reasoning: str
    speak: str

@dataclass
class SkillSelection:
    thoughts: Thoughts
    skill: Skill
    skill_kwargs: Dict[str, Any]

def parse_json(json_str: str, skill_repo: SkillRepository) -> Tuple[Optional[Exception], Optional[SkillSelection]]:
    try:
        json_dict = json.loads(json_str)
        jsonschema.validate(instance=json_dict, schema=skill_call_schema)
    except json.JSONDecodeError as e:
        return e.msg, None
    except jsonschema.ValidationError as e:
        return e.message, None
    
    # Verify the skills parameters
    skill_name = json_dict["skill"]["name"]
    kwargs = {name : value for name, value in json_dict["skill"].items() if name != "name"}
    try:
        skill = skill_repo.get_skill(skill_name)
        skill.verify_kwargs(kwargs)
    except ValueError as e:
        return str(e), None
    
    thoughts = Thoughts(**json_dict["thoughts"])
    return None, SkillSelection(thoughts=thoughts, skill=skill, skill_kwargs=kwargs)

def construct_prompt(state_description: str, skills: SkillRepository, task: str) -> str:
    return "\n\n".join([
        state_description,
        f"Skills:\n{skills.get_skills_description()}",
        task
    ])

class SimpleSkillSelector:
    def __init__(self,
        llm,
        skills: SkillRepository,
        use_popup_prompt: bool=False
    ):
        self.llm = llm
        self.skills = skills
        self.use_popup_prompt = use_popup_prompt

    def choose_skill(self, agent: NetHackAgent) -> SkillSelection:
        if agent.waiting_for_popup() and self.use_popup_prompt:
            skills = [sk.press_key, sk.type_text]
            prompt = POPUP_CHOOSE_SKILL_PROMPT
        else:
            skills = agent.skills.skills.values()
            prompt = CHOOSE_SKILL_PROMPT

        if agent.enable_finish_task_skill:
            skills = [*skills, finish_task_skill]

        return self._internal_choose_skill(agent, SkillRepository(skills), prompt)

    def _internal_choose_skill(self, agent: NetHackAgent, skills: SkillRepository, prompt: str) -> SkillSelection:
        task_prompt = construct_prompt(agent.describe_current_state(), skills, prompt)
        messages = [
            *agent.message_history.get_messages(),
            SystemMessage(content=task_prompt)
        ]

        # Censoring
        if agent.censor_nethack_messages:
            messages = deepcopy(messages)
            for m in messages:
                m.content = m.content.replace("NetHack", "CENSORED")

        # Call and parse
        json_str = self.llm.predict_messages(messages).content
        agent.logger.log_json(
            data = {
                "prompt": messages[-1].content,
                "response": json_str,
                "context": [{m.type: m.content} for m in messages[:-1]]
            },
            file_name=CHOOSE_SKILL_LOG_FILE
        )

        error_message, skill_call = parse_json(json_str, skills)
        if error_message is None:
            return skill_call
        
        raise Exception(f"Unable to parse the JSON provided by the LLM. Error message: '{error_message}'.")