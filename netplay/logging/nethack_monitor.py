from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper

import numpy as np
from gymnasium import Wrapper

import h5py
from copy import deepcopy

class NethackH5PYMonitor(Wrapper):
    def __init__(self,
        env: NethackGymnasiumWrapper,
        file_path: str,
        monitored_obs_keys=["glyphs", "blstats", "tty_cursor", "inv_strs", "inv_letters", "tty_chars"]
    ):
        super().__init__(env)
        self.h5py_file = h5py.File(file_path, "w")
        self.trajectories_group = self.h5py_file.create_group(f"trajectories")
        self.monitored_obs_keys = monitored_obs_keys

        self.recorded_trajectories = 0
        self.observations = []
        self.actions = []
        self.rewards = []

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self.observations.append({key : deepcopy(value) for key, value in obs.items() if key in self.monitored_obs_keys})
        self.rewards.append(float(reward))
        self.actions.append(int(action))

        if done:
            self._flush_trajectory()

        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._flush_trajectory()
        self.observations.append({key : deepcopy(value) for key, value in obs.items() if key in self.monitored_obs_keys})
        return obs, info
    
    def close(self):
        self._flush_trajectory()
        self.h5py_file.close()
        self.env.close()

    def _flush_trajectory(self):
        if len(self.actions) == 0:
            # Still clear observations in case of multiple resets
            self.observations = []
            return
        
        current_trajectory = self.trajectories_group.create_group(str(self.recorded_trajectories))
        observations_group = current_trajectory.create_group("observations")
        for key in self.monitored_obs_keys:
            data = np.array([o[key] for o in self.observations])
            observations_group.create_dataset(key, data=data)
        current_trajectory.create_dataset("actions", data=np.array(self.actions))
        current_trajectory.create_dataset("rewards", data=np.array(self.rewards))
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.recorded_trajectories += 1