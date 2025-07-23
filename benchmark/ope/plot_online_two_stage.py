import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from online_eval import online_eval_once
from opl_two_stage_train import train

methods = [
        "single_preference_is", "single_preference_kis", "multimodal_preference_is", "multimodal_preference_kis",
        "naive_cf", "online_policy", "random", "oracle"]

seeds = [0, 1, 2, 3, 4]
all_losses = {}
all_rewards = {}  # stores mean, std
raw_rewards = {}  # stores raw per-seed values

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

for method in methods:
    seed_losses = []
    scores = []

    for seed in seeds:
        print(f"[{method}] Training seed {seed}")
        set_seed(seed)

        if method not in {"random", "oracle"}:
            losses = train(method=method, seed=seed)
            seed_losses.append(losses)

        score = online_eval_once(method=method, seed=seed)
        scores.append(score)

    all_rewards[method] = (np.mean(scores), np.std(scores))
    raw_rewards[method] = scores

    if method not in {"random", "oracle"}:
        seed_losses = np.array(seed_losses)
        avg_loss = seed_losses.mean(axis=0)
        all_losses[method] = avg_loss

plt.figure(figsize=(10, 6))
for method in all_losses:
    plt.plot(all_losses[method], label=method.replace("_", " ").title())
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Loss Curves Averaged Over 5 Seeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_curves.png")

means = [all_rewards[m][0] for m in methods]
stds  = [all_rewards[m][1] for m in methods]

plt.figure(figsize=(12, 6))
plt.bar(methods, means, yerr=stds, capsize=5)
plt.title("Policy Evaluation: Average Top-5 Reward over 5 Seeds")
plt.ylabel("Average Top-5 Reward")
plt.xlabel("Training Method")
plt.ylim(0, max(means) + max(stds) + 0.2)

for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.02, f"{mean:.3f} ± {std:.3f}", ha='center')

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("two_stage_eval_seeds.png")

plt.figure(figsize=(12, 6))
for i, method in enumerate(methods):
    scores = raw_rewards[method]
    plt.scatter([i] * len(scores), scores, label=method if i == 0 else None, color='C{}'.format(i % 10))

plt.xticks(ticks=range(len(methods)), labels=methods, rotation=30, ha='right')
plt.ylabel("Top-5 Reward")
plt.title("Per-Seed Policy Reward across Methods")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("per_seed_rewards.png")
plt.show()
