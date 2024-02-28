from netplay.core.agent_base import Step

from dataclasses import dataclass
from functools import wraps
from enum import Enum
from typing import Optional, Dict, Any, Iterator, List, Callable

SKILL_IDENTIFIER = "is_skill"

class SkillParameterType(Enum):
    string = 0
    integer = 1
    bool = 2

@dataclass
class SkillParameter:
    name: str
    type: SkillParameterType
    optional: bool = False

    def parse(self, value: str):
        if self.type == SkillParameterType.string:
            return value
        if self.type == SkillParameterType.integer:
            return int(value)
        if self.type == SkillParameterType.bool:
            return value.lower() == "true"

        raise ValueError(f"Invalid parameter type {self.type}")
    
    def get_type(self):
        if self.type == SkillParameterType.string:
            return str
        if self.type == SkillParameterType.integer:
            return int
        if self.type == SkillParameterType.bool:
            return bool
        
        raise ValueError(f"Invalid parameter type {self.type}")

    def __str__(self):
        if self.type == SkillParameterType.string:
            return self.name
        if self.type == SkillParameterType.integer:
            return self.name
        if self.type == SkillParameterType.bool:
            return self.name

        raise ValueError(f"Invalid parameter type {self.type}")

    @classmethod
    def integer(cls, name: str, optional=False):
        return cls(name, SkillParameterType.integer, optional=optional)
    
    @classmethod
    def string(cls, name: str, optional=False):
        return cls(name, SkillParameterType.string, optional=optional)
    
    @classmethod
    def bool(cls, name: str, optional=False):
        return cls(name, SkillParameterType.bool, optional=optional)

class Skill:
    def __init__(self,
        fn: Callable, 
        name: str,
        description: str,
        parameters: List[SkillParameter]
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, agent, **kwargs) -> Iterator[Step]:
        yield from self.fn(agent, **kwargs)

    def verify_kwargs(self, kwargs: Dict[str, Any]):
        params = {param.name : param for param in self.parameters}
        for arg_name, value in kwargs.items():
            if arg_name not in params.keys():
                raise ValueError(f"The skill '{self.name}' does not have a parameter '{arg_name}'.")
            if value is None and not params[arg_name].optional:
                raise ValueError(f"The parameter '{arg_name}' for the skill '{self.name}' cannot be null because it is not optional.")
            if value != None and not isinstance(value, params[arg_name].get_type()):
                raise ValueError(f"The type of '{arg_name}' for the skill '{self.name}' is {type(value)} instead of {params[arg_name].get_type()}.")

    def parse_args(self, *args) -> Dict[str, Any]:
        parsed_args = {}
        for param, value in zip(self.parameters, args):
            parsed_args[param.name] = param.parse(value)
        return parsed_args
    
    def generate_description(self) -> str:
        params = [
            f"{param.name}: {param.type.name if not param.optional else f'Optional[{param.type.name}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description.rstrip('.')}. Params: ({', '.join(params)})"

def skill(name, description, parameters):
    def decorator(skill_fn):
        @wraps(skill_fn)
        def wrapper(agent, *args, **kwargs):
            yield from skill_fn(agent, *args, **kwargs)

        skill = Skill(
            fn=skill_fn,
            name=name,
            description=description,
            parameters=parameters
        )

        setattr(wrapper, SKILL_IDENTIFIER, True)
        setattr(wrapper, "skill", skill)

        return wrapper
    return decorator