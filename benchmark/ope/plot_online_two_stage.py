import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from obp.policy.action_policies import TwoStageRankingPolicy
from obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def online_eval_once(policy_path: str, seed: int) -> float:
    set_seed(seed)
    device = "cpu"

    # Setup dataset
    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        reward_std=0.5,
        action_policy=None,
        device=device,
        single_stage=False
    )

    # Recreate the architecture and load weights
    first_stage = TwoTowerFirstStagePolicy(input_dim=5, hidden_dim=64, n_actions=10)
    second_stage = SoftmaxSecondStagePolicy(context_dim=5, k=5)
    policy = TwoStageRankingPolicy(first_stage, second_stage, top_k=5, device=device)
    policy.load_state_dict(torch.load(policy_path))

    x_test = dataset.sample_context(10000)
    actions = torch.tensor([
        policy.sample_action(x_i, dataset.action_context)
        for x_i in x_test
    ], device=device)

    r = dataset.reward_function(x_test, actions)
    return r.mean().item()

def evaluate_over_seeds(method: str, seeds=[0, 1, 2, 3, 4]):
    path = f"{method}_policy.pt"
    scores = [online_eval_once(path, seed) for seed in seeds]
    return np.mean(scores), np.std(scores)

def main():
    methods = ["single_user_is", "iter_k_is", "iter_k_kis"]
    means, stds = [], []

    for method in methods:
        mean, std = evaluate_over_seeds(method)
        print(f"{method.upper()}: Mean = {mean:.4f}, Std = {std:.4f}")
        means.append(mean)
        stds.append(std)

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Two-Stage Policy: Online Evaluation over 5 Seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Training Method")
    plt.ylim(0, max(means) + max(stds) + 0.2)
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.3f} ± {stds[i]:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig("two_stage_eval_seeds.png")
    plt.show()

if __name__ == "__main__":
    main()