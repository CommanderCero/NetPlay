from netplay.core.agent_base import NethackBaseAgent

from abc import abstractmethod, ABC
from typing import Dict

class Descriptor(ABC):
    @abstractmethod
    def describe(self, agent: NethackBaseAgent) -> str:
        pass

class TitleValueDescriptor(Descriptor):
    def __init__(self, descriptors: Dict[str, Descriptor]):
        self.descriptors = descriptors

    def describe(self, agent: NethackBaseAgent) -> str:
        # Note dicts in Python 3.7 are ordered
        return "\n\n".join(
            f"{title}:\n{descriptor.describe(agent)}"
            for title, descriptor in self.descriptors.items()
        )

    def __getitem__(self, key):
        return self.descriptors[key]