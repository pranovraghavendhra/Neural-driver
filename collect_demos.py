"""
Collect expert demonstrations using MetaDrive's built-in IDM policy.
Run once: python collect_demos.py
"""
import numpy as np
import pickle
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

config = dict(
    use_render=False,
    manual_control=False,
    num_scenarios=100,
    start_seed=42,
    map=3,
    traffic_density=0.1,
    image_observation=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    out_of_road_done=True,
    agent_policy=IDMPolicy,   # built-in expert driver
)

env = MetaDriveEnv(config)
demos = []

print("Collecting expert demonstrations using IDM policy...")

for ep in range(50):
    result = env.reset()
    obs    = result[0] if isinstance(result, tuple) else result
    obs    = np.array(obs, dtype=np.float32)
    done   = False
    steps  = 0

    while not done and steps < 1000:
        result = env.step([0, 0])   # action ignored — IDM drives
        if len(result) == 5:
            next_obs, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_obs, _, done, info = result

        # get what action IDM actually took
        action = info.get("action", [0.0, 0.5])
        if hasattr(action, '__iter__'):
            action = list(action)
        else:
            action = [0.0, 0.5]

        demos.append((obs.copy(), np.array(action, dtype=np.float32)))
        obs   = np.array(next_obs, dtype=np.float32)
        steps += 1

    reason = "DEST" if info.get("arrive_dest") else \
             "CRASH" if info.get("crash") else "ROAD"
    print(f"  Ep {ep+1:>2} [{reason}] steps={steps:>4} "
          f"route={info.get('route_completion',0):.2f} "
          f"total={len(demos)}")

env.close()

with open("expert_demos.pkl", "wb") as f:
    pickle.dump(demos, f)

print(f"\nSaved {len(demos)} demonstrations to expert_demos.pkl")