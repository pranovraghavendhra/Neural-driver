import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Memory(Dataset):
    def __init__(self, states, actions, log_probs, rewards, advantages, values):
        self.states     = states
        self.actions    = actions
        self.log_probs  = log_probs
        self.rewards    = rewards
        self.advantages = advantages
        self.values     = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],     self.actions[idx],
            self.log_probs[idx],  self.rewards[idx],
            self.advantages[idx], self.values[idx],
        )


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages     = [0] * len(rewards)
    last_advantage = 0
    for i in reversed(range(len(rewards))):
        delta          = rewards[i] + (1 - dones[i]) * gamma * last_value - values[i]
        advantages[i]  = delta + (1 - dones[i]) * gamma * lam * last_advantage
        last_value     = values[i]
        last_advantage = advantages[i]
    return advantages


def ppo_update(model, optimizer, memory, batch_size=128, epochs=5,
               clip=0.2, value_coef=0.5, entropy_coef=0.01):

    # normalize advantages — critical for stable loss
    adv               = memory.advantages
    memory.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    loader     = DataLoader(memory, batch_size=batch_size, shuffle=True)
    total_loss = 0.0

    for _ in range(epochs):
        for states, actions, old_log_probs, rewards, advantages, old_values in loader:

            value_target = (advantages + old_values).detach()

            log_probs, values, entropy = model.evaluate(states, actions)

            ratio       = (log_probs - old_log_probs).exp()
            surr1       = ratio * advantages
            surr2       = ratio.clamp(1 - clip, 1 + clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = nn.MSELoss()(values, value_target)
            loss        = policy_loss + value_coef * value_loss \
                        - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

    return total_loss / (epochs * len(loader))