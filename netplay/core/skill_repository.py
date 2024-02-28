from netplay.core.skill import Skill, SKILL_IDENTIFIER

from typing import List, Union, Callable, Dict, Tuple, Any

class SkillRepository:
    def __init__(self, skills: List[Union[Skill, Callable]]):
        self.skills : Dict[str, Skill] = {}
        for skill in skills:
            self.add_skill(skill)
    
    def add_skill(self, skill: Union[Skill, Callable]):
        if isinstance(skill, Skill):
            self.skills[skill.name] = skill
        elif hasattr(skill, SKILL_IDENTIFIER):
            skill = skill.skill
            self.skills[skill.name] = skill
        else:
            raise ValueError(f"Expected a skill but received {skill}.")
        
    def get_skill(self, skill_name: str):
        if skill_name not in self.skills.keys():
            raise ValueError(f"A skill with the name '{skill_name}' does not exist.")
        return self.skills[skill_name]

    def parse_skill_call(self, skill_definition) -> Tuple[Skill, Dict[str, Any]]:
        parts = skill_definition.split(" ")
        name = parts[0]
        args = parts[1:]

        skill = self.skills[name]
        args = skill.parse_args(*args)
        return skill, args

    def parse_skill_json(self, skill_definition: Dict[str, Any]) -> Tuple[Skill, Dict[str, Any]]:
        skill_name = skill_definition["name"]
        kwargs = {name : value for name, value in skill_definition.items() if name != "name"}
        return self.skills[skill_name], kwargs
    
    def get_skills_description(self) -> str:
        skill_descriptions = [f"- {skill.generate_description()}" for skill in self.skills.values()]
        return "\n".join(skill_descriptions)