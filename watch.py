import torch
import numpy as np
from env import DrivingEnv
from network import MLPActorCritic

MODEL_PATH = "checkpoint_810000.pt"
N_EPISODES = 10


def watch():
    print("Loading model:", MODEL_PATH)
    env = DrivingEnv(render=True, difficulty=1,)
    model = MLPActorCritic(obs_dim=env.obs_dim, n_actions=env.n_actions)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Model loaded.")
    except FileNotFoundError:
        print(f"'{MODEL_PATH}' not found. Run train.py first.")
        env.close()
        return

    model.eval()

    for ep in range(1, N_EPISODES + 1):
        obs       = env.reset()
        ep_reward = 0.0
        step      = 0
        done      = False
        print(f"\nEpisode {ep} starting...")

        while not done:
            with torch.no_grad():
                t_obs = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _, action_env = model.get_action(t_obs)

            obs, reward, done, info = env.step(action_env)
            ep_reward += reward
            step      += 1

            speed = 0.0
            try:
                speed = env.env.agent.speed
            except:
                speed = info.get("velocity", 0.0)

            print(f"\r  Step {step:>4} | Reward: {ep_reward:>6.3f} | "
                  f"Speed: {speed:>5.1f} | "
                  f"Route: {info.get('route_completion', 0.0):.3f}",
                  end="", flush=True)

        reason = "REACHED DEST" if info.get("arrive_dest", False) else \
                 "CRASHED"      if info.get("crash", False)        else \
                 "OUT OF ROAD"  if info.get("out_of_road", False)  else \
                 "ENDED"
        print(f"\n  [{reason}] Steps: {step} | Reward: {ep_reward:.3f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    watch()