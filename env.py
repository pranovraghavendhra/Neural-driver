import numpy as np
from metadrive import MetaDriveEnv


class DrivingEnv:
    def __init__(self, render=False, difficulty=0):
        configs = [
            dict(map=3, traffic_density=0.0),
            dict(map=4, traffic_density=0.05),
            dict(map=5, traffic_density=0.1),
            dict(map=6, traffic_density=0.2),
        ]
        curr = configs[min(difficulty, len(configs) - 1)]
        config = dict(
            use_render=render,
            manual_control=False,
            num_scenarios=100,
            start_seed=42,
            image_observation=False,
            crash_vehicle_done=True,
            crash_object_done=True,
            out_of_road_done=True,
            window_size=(600, 400),
            **curr
        )
        self.env       = MetaDriveEnv(config)
        self.n_actions = 2   # continuous [steer, throttle]

        obs          = self.env.reset()
        obs          = obs[0] if isinstance(obs, tuple) else obs
        self.obs_dim = obs.shape[0]
        print(f"Obs dim: {self.obs_dim}")

    def _shaped_reward(self, info):
        reward  = 0.0
        reward += info.get("step_reward", 0.0) * 3.0
        reward += float(info.get("route_completion", 0.0)) * 0.5
        v = info.get("velocity", 0.0)
        if v > 0.3:   reward += 0.1
        elif v < 0.05: reward -= 0.3
        if info.get("arrive_dest", False):  reward += 20.0
        if info.get("crash", False):        reward -= 5.0
        if info.get("out_of_road", False):  reward -= 5.0
        return float(np.clip(reward, -1.0, 1.0))

    def reset(self):
        result = self.env.reset()
        obs    = result[0] if isinstance(result, tuple) else result
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        steer    = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1],  0.0, 1.0))
        result   = self.env.step([steer, throttle])
        if len(result) == 5:
            obs, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, _, done, info = result
        return np.array(obs, dtype=np.float32), \
               self._shaped_reward(info), done, info

    def close(self):
        self.env.close()