import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from custom_obp.models.cf_model import NaiveCF

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
        single_stage=False
    )
    x_test = dataset.sample_context(10000)

    if method == "naive_cf":
        model = NaiveCF(n_users=1000, n_items=dataset.n_actions)
        model.load_state_dict(torch.load(f"{method}_policy.pt"))
        model.eval()

        user_ids = torch.randint(0, 1000, size=(10000,))
        item_ids = torch.arange(dataset.n_actions)

        best_actions = []
        for user_id in user_ids:
            user_tensor = user_id.repeat(dataset.n_actions)
            scores = model(user_tensor, item_ids)
            best_action = torch.argmax(scores).item()
            best_actions.append(best_action)

        best_actions = torch.tensor(best_actions)
        rewards = dataset.reward_function(x_test, best_actions)
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

        state_dicts = torch.load(f"{method}_policy.pt")
        dataset.action_context[:] = state_dicts["action_context"]
        first_stage.load_state_dict(state_dicts["first_stage"])
        second_stage.load_state_dict(state_dicts["second_stage"])

        candidates = first_stage.sample_topk(x_test)
        actions = second_stage.sample_output(x_test, candidates)
        rewards = dataset.reward_function(x_test, actions)
        return rewards.mean().item()

def evaluate_over_seeds(method: str, seeds=[0, 1, 2, 3, 4]):
    scores = [online_eval_once(method, seed) for seed in seeds]
    return np.mean(scores), np.std(scores)

def main():
    methods = ["single_preference_is", "iter_k_is", "iter_k_kis", "naive_cf"]
    means, stds = [], []

    for method in methods:
        mean, std = evaluate_over_seeds(method)
        print(f"{method.upper()}: Mean = {mean:.4f}, Std = {std:.4f}")
        means.append(mean)
        stds.append(std)

    # plotting
    plt.figure(figsize=(6, 4))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Policy Evaluation: Average Reward over 5 Seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Training Method")
    plt.ylim(0, max(means) + max(stds) + 0.2)

    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.3f} ± {stds[i]:.3f}", ha='center')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "two_stage_eval_seeds.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()