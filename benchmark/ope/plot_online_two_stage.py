import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from online_eval import online_eval_once
from opl_two_stage_train import train

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_over_seeds(method: str, seeds=[0, 1, 2, 3, 4]):
    scores = []
    for seed in seeds:
        print(f"[Seed {seed}] Training {method}")
        train(method=method, seed=seed)
        score = online_eval_once(method, seed)
        scores.append(score)
    return np.mean(scores), np.std(scores)

def main():
    methods = [
        "single_preference_is", "single_preference_kis", "multimodal_preference_is", "multimodal_preference_kis",
        "naive_cf", "online_policy", "random", "oracle"]
    means, stds = [], []

    for method in methods:
        mean, std = evaluate_over_seeds(method)
        print(f"{method.upper()}: Mean = {mean:.4f}, Std = {std:.4f}")
        means.append(mean)
        stds.append(std)

    plt.figure(figsize=(6, 4))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Policy Evaluation: Average Reward over 5 Seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Training Method")
    plt.ylim(0, max(means) + max(stds) + 0.2)

    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.3f} ± {stds[i]:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "two_stage_eval_seeds.png"))
    plt.show()

if __name__ == "__main__":
    main()