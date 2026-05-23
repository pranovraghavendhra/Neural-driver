import torch
import glob
import numpy as np

best = torch.load("best_model.pt", map_location="cpu")

checkpoints = sorted(
    glob.glob("checkpoint_*.pt"),
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

print("Finding closest checkpoint to best_model.pt...\n")

results = []
for ckpt in checkpoints:
    ckpt_model = torch.load(ckpt, map_location="cpu")
    diff = sum(
        (best[k] - ckpt_model[k]).abs().mean().item()
        for k in best.keys()
    )
    results.append((diff, ckpt))

results.sort()
print("Top 5 closest checkpoints:")
for diff, ckpt in results[:5]:
    print(f"  {ckpt} — diff: {diff:.6f}")

print(f"\nBest model is closest to: {results[0][1]}")
print(f"So best_model.pt was saved between "
      f"{results[0][1]} and the next checkpoint")