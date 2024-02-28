from netplay.core.skill import Skill
from netplay.core.agent_base import NethackBaseAgent

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any

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

class SkillSelector(ABC):
    @abstractmethod
    def choose_skill(self, agent: NethackBaseAgent) -> SkillSelection:
        pass