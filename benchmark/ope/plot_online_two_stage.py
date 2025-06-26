import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from custom_obp.models.cf_model import NaiveCF
from opl_two_stage_train import train

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def online_eval_once(method: str, seed: int) -> float:
    set_seed(seed)
    device = "cpu"

    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        reward_std=0.5,
        action_policy=None,
        device=device,
        single_stage=(method == "naive_cf")
    )

    dataset.user_embeddings = dataset.sample_context(1000)
    user_ids = torch.randint(0, 1000, size=(10000,), device=device)
    x_test = dataset.user_embeddings[user_ids] 

    if method == "naive_cf":
        model = NaiveCF(dim_context=dataset.dim_context, n_items=dataset.n_actions, emb_dim=32)
        model.load_state_dict(torch.load(f"{method}_policy_seed{seed}.pt"))
        model.eval()

        item_ids = torch.arange(dataset.n_actions)
        best_actions = []

        for i in range(len(x_test)):
            repeated_context = x_test[i].repeat(dataset.n_actions, 1)
            scores = model(repeated_context, item_ids)
            best_action = torch.argmax(scores).item()
            best_actions.append(best_action)
        
        best_actions = torch.tensor(best_actions, device=device)
        rewards = dataset.reward_function(user_ids, best_actions)
        return rewards.mean().item()

    if method == "random":
        actions = torch.randint(0, dataset.n_actions, (10000,), device=device)
        rewards = dataset.reward_function(user_ids, actions)
        return rewards.mean().item()
        
    else:
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

        state_dicts = torch.load(f"{method}_policy_seed{seed}.pt")
        dataset.action_context = state_dicts["action_context"]
        first_stage.load_state_dict(state_dicts["first_stage"])
        second_stage.load_state_dict(state_dicts["second_stage"])

        candidates = first_stage.sample_topk(x_test)
        local_index = second_stage.sample_output(x_test, candidates)
        actions = candidates[torch.arange(x_test.size(0), device=device), local_index]
        rewards = dataset.reward_function(user_ids, actions)

        return rewards.mean().item()

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
        "single_preference_is", "single_preference_kis", "iter_k_is", "iter_k_kis",
        "naive_cf", "online_policy", "random"]
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