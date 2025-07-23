import torch
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from custom_obp.models.cf_model import NaiveCF
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def online_eval_once(method: str, seed: int) -> float:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomSyntheticBanditDataset(
        n_actions=1000,
        dim_context=5,
        top_k=10,
        reward_std=0.5,
        action_policy=None,
        device=device,
        single_stage=(method == "naive_cf")
    )

    dataset.user_embeddings = dataset.sample_context(1000).to(device)
    user_ids = torch.randint(0, 1000, size=(10000,), device=device)
    x_test = dataset.user_embeddings[user_ids] 

    if method == "naive_cf":
        model = NaiveCF(dim_context=dataset.dim_context, n_items=dataset.n_actions, emb_dim=32).to(device)
        model.load_state_dict(torch.load(f"{method}_policy_seed{seed}.pt", map_location=device))
        model.eval()

        item_ids = torch.arange(dataset.n_actions, device=device)
        rewards = []

        for i in range(len(x_test)):
            repeated_context = x_test[i].repeat(dataset.n_actions, 1)
            scores = model(repeated_context, item_ids)
            top5 = torch.topk(scores, k=5).indices
            reward_list = [dataset.reward_function(user_ids[i:i+1], top5[j:j+1]) for j in range(5)]
            avg_reward = torch.stack(reward_list).mean()
            rewards.append(avg_reward)

        return torch.tensor(rewards).mean().item()

    if method == "oracle":
        all_scores = torch.matmul(x_test, dataset.action_context.T)
        top5_actions = torch.topk(all_scores, k=5, dim=1).indices
        reward_list = [
            dataset.reward_function(user_ids, top5_actions[:, i])
            for i in range(5)
        ]
        rewards = torch.stack(reward_list, dim=1).mean(dim=1)
        return rewards.mean().item()

    if method == "random":
        actions = torch.randint(0, dataset.n_actions, (10000,), device=device)
        rewards = dataset.reward_function(user_ids, actions)
        top5_random = torch.randint(0, dataset.n_actions, (10000, 5), device=device)
        reward_list = [dataset.reward_function(user_ids, top5_random[:, i]) for i in range(5)]
        rewards = torch.stack(reward_list, dim=1).mean(dim=1)
        return rewards.mean().item()
        
    else:
        first_stage = TwoTowerFirstStagePolicy(
            dim_context=5,
            num_items=dataset.n_actions,
            emb_dim=64,
            top_k=10
        )
        second_stage = SoftmaxSecondStagePolicy(
            dim_context=5,
            emb_dim=64,
            first_stage_policy=first_stage
        )

        state_dicts = torch.load(f"{method}_policy_seed{seed}.pt", map_location=device)

        dataset.action_context = state_dicts["action_context"].to(device)

        first_stage = first_stage.to(device)
        second_stage = second_stage.to(device)

        first_stage.load_state_dict(state_dicts["first_stage"])
        second_stage.load_state_dict(state_dicts["second_stage"])

        candidates = first_stage.sample_topk(x_test)
        ranked_actions, _ = second_stage.rank_outputs(x_test, candidates)
        top5_actions = ranked_actions[:, :5]

        reward_list = [dataset.reward_function(user_ids, top5_actions[:, i]) for i in range(5)]
        rewards = torch.stack(reward_list, dim=1).mean(dim=1)

        return rewards.mean().item()
