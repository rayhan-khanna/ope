import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def online_eval_once(policy_path: str, seed: int) -> float:
    set_seed(seed)
    device = "cpu"

    # setup dataset
    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        reward_std=0.5,
        action_policy=None,
        device=device,
        single_stage=False
    )

    # instantiate models
    first_stage = TwoTowerFirstStagePolicy(
        dim_context=5,
        num_items=10,
        emb_dim=64,
        top_k=5
    )
    second_stage = SoftmaxSecondStagePolicy(
        dim_context=5,
        emb_dim=64,
        first_stage_policy=first_stage
    )

    # load weights
    state_dicts = torch.load(policy_path)
    first_stage.load_state_dict(state_dicts["first_stage"])
    second_stage.load_state_dict(state_dicts["second_stage"])

    # evaluate
    x_test = dataset.sample_context(10000)
    candidates = first_stage.sample_topk(x_test)
    actions = second_stage.sample_output(x_test, candidates)

    rewards = dataset.reward_function(x_test, actions)
    return rewards.mean().item()

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

    # creating plot
    plt.figure(figsize=(6, 4))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Two-Stage Policy: Online Evaluation over 5 Seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Training Method")
    plt.ylim(0, max(means) + max(stds) + 0.2)
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.3f} ± {stds[i]:.3f}", ha='center')
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "two_stage_eval_seeds.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()