from netplay.logging.step_logger import DEFAULT_STEP_FOLDER_FORMAT

import numpy as np

import os
import h5py
import re
from dataclasses import dataclass
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Iterator

DEFAULT_TRAJECTORY_FILE_NAME = "trajectories.h5py"

@dataclass
class AgentStepData:
    observation: Dict[str, Any]
    action: int
    reward: float
    next_observation: Dict[str, Any]
    done: bool
    

class AgentLogDataset:
    def __init__(self, trajectories: h5py.File, json_files: Dict[int, List[str]]):
        self.trajectories = trajectories
        self.json_files = json_files

        self.trajectory_groups = [self.trajectories["trajectories"][key] for key in self.trajectories["trajectories"].keys()]

    def __iter__(self) -> Iterator[AgentStepData]:
        for group in self.trajectory_groups:
            observations = group["observations"]
            actions = group["actions"]
            rewards = group["rewards"]

            for i in range(len(actions)):
                is_last_action = len(actions) - 1 == i
                yield AgentStepData(
                    observation=self._construct_obs(observations, i),
                    action=actions[i],
                    reward=rewards[i],
                    next_observation=self._construct_obs(observations, i),
                    done=is_last_action
                )

    def _construct_obs(self, obs_group, index):
        return {
            key : obs_group[key][index]
            for key in obs_group.keys()
        }

    def __len__(self):
        return sum([len(g["actions"]) for g in self.trajectory_groups])

    @classmethod
    def from_log_folder(
        cls,
        log_folder: str,
        trajectories_file_name = DEFAULT_TRAJECTORY_FILE_NAME,
        step_folder_format = DEFAULT_STEP_FOLDER_FORMAT
    ) -> "AgentLogDataset":
        trajectories = h5py.File(os.path.join(log_folder, trajectories_file_name), "r")
        
        step_folder_regex = re.compile(f"^{step_folder_format.format(step='(.*?)')}$")
        step_files_glob = os.path.join(log_folder, step_folder_format.format(step="*"), "*.json")
        json_files = defaultdict(lambda: [])
        for file in glob(step_files_glob):
            parent_folder = Path(file).parent.name
            step_number = step_folder_regex.match(parent_folder).group(1)
            json_files[int(step_number)].append(file)

        return cls(
            trajectories=trajectories,
            json_files=json_files
        )

    