
import numpy as np
from PIL import Image

import os
import json
from typing import Dict, Any

DEFAULT_STEP_FOLDER_FORMAT = "{step}"

class StepLogger:
    def __init__(self, log_folder: str, folder_format=DEFAULT_STEP_FOLDER_FORMAT):
        self.log_folder = log_folder
        self.folder_format = folder_format

        self.timestep = 0
        self.step_log_count = 0

        os.makedirs(log_folder, exist_ok=True)

    def start_next_step(self):
        self.timestep += 1
        self.step_log_count = 0

    def log_json(self,
        data: Dict[str, Any],
        file_name: str
    ):
        folder_path = self._create_folder(step=self.timestep)
        file_path = os.path.join(folder_path, f"{self.step_log_count}_{file_name}")

        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_data)

        self.step_log_count += 1

    def _create_folder(self, step: int):
        folder_name = self.folder_format.format(step=step)
        folder_path = os.path.join(self.log_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
