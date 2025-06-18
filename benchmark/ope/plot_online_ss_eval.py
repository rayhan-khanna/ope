import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.action_policies import SoftmaxPolicy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def online_eval_once(policy_path: str, seed: int) -> float:
    set_seed(seed)
    device = "cpu"

    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        reward_std=0.5,
        action_policy=None,
        device=device,
        single_stage=True
    )

    policy = SoftmaxPolicy(dataset.dim_context)
    policy.load_state_dict(torch.load(policy_path))

    x_test = dataset.sample_context(10000)
    a_sampled = torch.tensor([
        policy.sample_action(x_i, dataset.action_context)
        for x_i in x_test
    ], device=device)

    r_true = dataset.reward_function(x_test, a_sampled)
    return r_true.mean().item()

def evaluate_over_seeds(method: str, seeds=[0,1,2,3,4]):
    path = f"{method}_policy.pt"
    scores = [online_eval_once(path, seed) for seed in seeds]
    return np.mean(scores), np.std(scores)

def main():
    methods = ["dm", "dr", "is"]
    means, stds = [], []

    for method in methods:
        mean, std = evaluate_over_seeds(method)
        print(f"{method.upper()}: Mean = {mean:.4f}, Std = {std:.4f}")
        means.append(mean)
        stds.append(std)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Online Evaluation over 5 Random Seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Method")
    plt.ylim(0, max(means) + max(stds) + 0.2)
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.3f} ± {stds[i]:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig("online_eval_seeds.png")
    plt.show()

if __name__ == "__main__":
    main()