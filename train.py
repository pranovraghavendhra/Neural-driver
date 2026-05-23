import torch
import torch.nn as nn
import numpy as np
import glob
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from env import DrivingEnv
from network import MLPActorCritic
from ppo import compute_gae, ppo_update, Memory

torch.set_num_threads(2)

HORIZON     = 2048
TOTAL_STEPS = 1_000_000
LR          = 3e-4
BATCH_SIZE  = 128
EPOCHS      = 5
SAVE_EVERY  = 10_000


def bc_from_demos(model, demo_path="expert_demos.pkl",
                  epochs=30, lr=1e-3, batch_size=256):
    """Train BC directly from recorded expert demonstrations."""
    print("=== Behavior Cloning from Expert Demos ===")

    with open(demo_path, "rb") as f:
        demos = pickle.load(f)

    print(f"  Loaded {len(demos)} demo transitions")

    obss    = np.array([d[0] for d in demos], dtype=np.float32)
    actions = np.array([d[1] for d in demos], dtype=np.float32)

    # Normalize actions to [0,1] for Beta
    # steer is [-1,1] → [0,1], throttle stays [0,1]
    actions[:, 0] = (actions[:, 0] + 1.0) / 2.0
    actions       = np.clip(actions, 0.01, 0.99)

    obs_t  = torch.FloatTensor(obss)
    act_t  = torch.FloatTensor(actions)
    ds     = TensorDataset(obs_t, act_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 999.0

    for ep in range(epochs):
        total_loss = 0.0
        for obs_b, act_b in loader:
            alpha, beta, _ = model(obs_b)
            dist = torch.distributions.Beta(alpha, beta)
            loss = -dist.log_prob(act_b).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "bc_best.pt")

        if ep % 5 == 0:
            print(f"  BC epoch {ep:>3} | Loss: {avg:.4f} "
                  f"| Best: {best_loss:.4f}")

    model.load_state_dict(torch.load("bc_best.pt", map_location="cpu"))
    print(f"=== BC Complete — best loss: {best_loss:.4f} ===\n")


def get_difficulty(step):
    for threshold, diff in reversed([
        (0, 0), (200_000, 1), (500_000, 2), (800_000, 3)
    ]):
        if step >= threshold:
            return diff
    return 0


def train():
    difficulty = 0
    env    = DrivingEnv(render=False, difficulty=difficulty)
    model  = MLPActorCritic(obs_dim=env.obs_dim, n_actions=env.n_actions)
    writer = SummaryWriter("runs/neural_driver_demo")

    checkpoints = sorted(
        glob.glob("checkpoint_*.pt"),
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if checkpoints:
        latest     = checkpoints[-1]
        model.load_state_dict(torch.load(latest, map_location="cpu"))
        start_step = int(latest.split("_")[1].split(".")[0])
        difficulty = get_difficulty(start_step)
        env.close()
        env = DrivingEnv(render=False, difficulty=difficulty)
        print(f"Resumed from {latest} at step {start_step}")
    else:
        start_step = 0
        print("Fresh start — running BC from expert demos...")
        bc_from_demos(model, demo_path="expert_demos.pkl")
        env.close()
        env = DrivingEnv(render=False, difficulty=difficulty)

    total_updates = TOTAL_STEPS // HORIZON
    optimizer     = torch.optim.Adam(model.parameters(),
                                     lr=LR, eps=1e-5)
    scheduler     = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1.0,
                        end_factor=0.01,
                        total_iters=total_updates)

    obs        = env.reset()
    step       = start_step
    episode    = 0
    ep_reward  = 0.0
    ep_rewards = []
    best_avg   = -999

    print(f"Obs dim: {env.obs_dim} | Actions: continuous")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("PPO training started.\n")

    while step < TOTAL_STEPS:

        new_diff = get_difficulty(step)
        if new_diff != difficulty:
            difficulty = new_diff
            env.close()
            env = DrivingEnv(render=False, difficulty=difficulty)
            obs = env.reset()
            print(f"\n  Curriculum → difficulty {difficulty} "
                  f"at step {step:,}\n")

        states_l, actions_l, lps_l = [], [], []
        rewards_l, values_l, dones_l = [], [], []

        for _ in range(HORIZON):
            t_obs = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, lp, value, action_env = model.get_action(t_obs)

            next_obs, reward, done, info = env.step(action_env)

            states_l.append(torch.FloatTensor(obs))
            actions_l.append(action)
            lps_l.append(lp)
            rewards_l.append(reward)
            values_l.append(value.item())
            dones_l.append(float(done))

            ep_reward += reward
            step      += 1

            if done:
                episode += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            t_obs      = torch.FloatTensor(obs).unsqueeze(0)
            _, _, last_v = model(t_obs)
            last_value = last_v.item()

        advantages = compute_gae(rewards_l, values_l,
                                 dones_l, last_value)
        memory = Memory(
            states     = torch.stack(states_l),
            actions    = torch.stack(actions_l),
            log_probs  = torch.stack(lps_l).detach(),
            rewards    = torch.tensor(rewards_l, dtype=torch.float32),
            advantages = torch.tensor(advantages, dtype=torch.float32),
            values     = torch.tensor(values_l,   dtype=torch.float32),
        )

        loss = ppo_update(model, optimizer, memory,
                          batch_size=BATCH_SIZE, epochs=EPOCHS)
        scheduler.step()

        if ep_rewards:
            avg = np.mean(ep_rewards[-20:])
            lr  = optimizer.param_groups[0]["lr"]
            writer.add_scalar("reward/avg_20ep", avg,  step)
            writer.add_scalar("train/loss",      loss, step)
            print(f"Step {step:>7,} | Ep {episode:>4} | "
                  f"Avg(20): {avg:>7.2f} | Loss: {loss:.4f} | "
                  f"LR: {lr:.2e} | Diff: {difficulty}")

            if avg > best_avg:
                best_avg = avg
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  ★ New best: {avg:.2f}")

        if step % SAVE_EVERY < HORIZON:
            ckpt = f"checkpoint_{(step//SAVE_EVERY)*SAVE_EVERY}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved {ckpt}")

    torch.save(model.state_dict(), "driver_model.pt")
    writer.close()
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train()