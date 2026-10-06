# Autonomous Driving Agent
**Deep Reinforcement Learning with Behavior Cloning + PPO in MetaDrive**



---

## Overview

An end-to-end RL agent trained to drive in a 3D simulated environment using only onboard observations — no GPS, no hand-coded rules. The agent is pre-trained via Behavior Cloning on expert demonstrations, then fine-tuned with Proximal Policy Optimization (PPO) across a 4-stage difficulty curriculum.

**Final performance:** avg reward of 402 at Difficulty-3 (20% traffic density), episode lengths of 150–400 steps vs. 5–20 for an untrained agent.

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Framework | PyTorch 2.x |
| Simulator | MetaDrive 0.4.3 |
| RL Algorithm | PPO (custom implementation) |
| Expert Policy | IDM (Intelligent Driver Model) |
| Monitoring | TensorBoard |
| Numerics | NumPy |

---

## Approach

### Phase 1 — Behavior Cloning
1. Run MetaDrive's IDM expert policy to collect 200 episodes on Map-3
2. Filter episodes with `route_completion > 10%` → ~82,000 (obs, action) pairs
3. Train an MLP on the demonstrations using Beta NLL loss for 30 epochs
4. Save best BC checkpoint as the PPO initialisation

### Phase 2 — PPO Fine-Tuning
1. Load BC-initialised weights
2. Collect 2048-step rollouts per update cycle
3. Compute GAE advantages (γ=0.99, λ=0.95), normalise to zero mean / unit variance
4. Run 5 epochs of mini-batch PPO updates (clip=0.15, batch=128)
5. Apply linear LR decay: 3e-4 → 3e-6 over 1M steps
6. Curriculum: Difficulty 0 → 1 → 2 → 3

---

## Key Engineering Decisions

- **Beta distribution policy** — bounded continuous outputs for smooth steering; avoids hard action-bin artefacts
- **Reward clipping + gradient clipping** — prevents loss explosion (observed up to 285,000 without it)
- **Advantage normalisation** — stabilises training across curriculum difficulty jumps
- **5-step action smoothing at inference** — eliminates jitter from stochastic Beta sampling

---

## Results

| Metric | Untrained | BC + PPO Final |
|---|---|---|
| Avg Reward | −11 to −15 | 170 to 402 |
| Episode Length | 5–20 steps | 150–400 steps |
| Route Completion | < 2% | 15–40% |
| Turn Navigation | None | Consistent |
| Min PPO Loss | — | 0.42 |

Peak reward of **402** reached at step 812k on Difficulty-3.

---

## Evaluation

Run `watch.py` to load the best saved checkpoint and evaluate the agent in a live 3D MetaDrive window:

```bash
python watch.py
```

The evaluation script loads `best_model.pt`, applies 5-step action smoothing, and displays live reward and route completion statistics.

---

## Project Structure

```
.
├── train.py            # Main PPO training loop
├── watch.py            # Evaluation / visualisation
├── bc_train.py         # Behavior cloning pre-training
├── collect_demos.py    # IDM expert data collection
├── model.py            # Actor-Critic network definition
├── ppo.py              # PPO update logic
├── best_model.pt       # Best saved checkpoint
└── runs/               # TensorBoard logs
```

---

## Requirements

```bash
pip install torch metadrive-simulator numpy tensorboard
```

---
