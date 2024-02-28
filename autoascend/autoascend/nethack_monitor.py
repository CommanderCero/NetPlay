import numpy as np
from gym import Wrapper

from nle.env import NLE

import h5py
from copy import deepcopy

class NethackH5PYMonitor(Wrapper):
    def __init__(self,
        env: NLE,
        file_path: str,
        monitored_obs_keys=["glyphs", "blstats", "tty_cursor", "inv_strs", "inv_letters", "tty_chars"]
    ):
        super().__init__(env)
        self._h5py_file = h5py.File(file_path, "w")
        self._trajectories_group = self._h5py_file.create_group(f"trajectories")
        self._monitored_obs_keys = monitored_obs_keys

        self._recorded_trajectories = 0
        self._observations = []
        self._actions = []
        self._rewards = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self._observations.append({key: deepcopy(value) for key, value in obs.items() if key in self._monitored_obs_keys})
        self._rewards.append(float(reward))
        self._actions.append(int(action))

        if done:
            self._flush_trajectory()

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._flush_trajectory()
        self._observations.append({key: deepcopy(value) for key, value in obs.items() if key in self._monitored_obs_keys})
        return obs
    
    def close(self):
        self.env.close()
        self._flush_trajectory()
        self._h5py_file.close()

    def seed(self, *args, **kwargs):
        self.env.seed(*args, **kwargs)

    def _flush_trajectory(self):
        if len(self._actions) == 0:
            # Still clear observations in case of multiple resets
            self._observations = []
            return
        
        current_trajectory = self._trajectories_group.create_group(str(self._recorded_trajectories))
        observations_group = current_trajectory.create_group("observations")
        for key in self._monitored_obs_keys:
            data = np.array([o[key] for o in self._observations])
            observations_group.create_dataset(key, data=data)
        current_trajectory.create_dataset("actions", data=np.array(self._actions))
        current_trajectory.create_dataset("rewards", data=np.array(self._rewards))
        
        self._observations = []
        self._actions = []
        self._rewards = []
        self._recorded_trajectories += 1

    def __getattr__(self, name):
        return getattr(self.env, name)
