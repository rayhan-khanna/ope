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
            if top5.dim() == 1:
                B, K = 1, top5.size(0)
            else:
                B, K = top5.shape

            user_ids_exp = user_ids[i].repeat(K)
            actions_exp = top5
            rewards_flat = dataset.reward_function(user_ids_exp, actions_exp)
            avg_reward = rewards_flat.mean()
            rewards.append(avg_reward)

        return torch.tensor(rewards).mean().item()

    if method == "oracle":
        gamma = 1.2
        k = 5
        item_vecs = dataset.action_context
        U, N = x_test.size(0), item_vecs.size(0)

        # precompute relevance 
        relevance = x_test @ item_vecs.T

        # precompute cosine similarity for vendi cscore
        items_norm = torch.nn.functional.normalize(item_vecs, p=2, dim=1)
        cos = items_norm @ items_norm.T

        # tack selected items per user
        selected = torch.zeros(U, 0, dtype=torch.long, device=device)
        mask = torch.zeros(U, N, dtype=torch.bool, device=device)

        for t in range(k):
            if t == 0:
                diversity = torch.zeros_like(relevance)
            else:
                K_sel = torch.stack([
                    cos[idx][:, idx] for idx in selected
                ])

                sims = cos[:, selected.reshape(-1)].T.reshape(U, N, t)                
                G = torch.zeros(U, N, t+1, t+1, device=device)
                G[:, :, :t, :t] = K_sel.unsqueeze(1).expand(-1, N, -1, -1)
                G[:, :, :t, t] = sims
                G[:, :, t, :t] = sims
                G[:, :, t, t] = 1.0

                G = G / (t + 1)
                eigvals = torch.linalg.eigvalsh(G.cpu()).to(device)
                entropy = -(eigvals.clamp_min(1e-12) * eigvals.clamp_min(1e-12).log()).sum(dim=-1)
                diversity = entropy.exp() 

            scores = relevance + gamma * diversity
            scores[mask] = -1e9  # mask picked items
            best = scores.argmax(dim=1) 

            selected = torch.cat([selected, best.unsqueeze(1)], dim=1)
            mask[torch.arange(U), best] = True

        user_ids_exp = user_ids.unsqueeze(1).repeat(1, k).reshape(-1)
        actions_exp = selected.reshape(-1)
        rewards = dataset.reward_function(user_ids_exp, actions_exp).view(U, k).mean(dim=1)
        return rewards.mean().item()

    if method == "random":
        actions = torch.randint(0, dataset.n_actions, (10000,), device=device)
        rewards = dataset.reward_function(user_ids, actions)
        top5_random = torch.randint(0, dataset.n_actions, (10000, 5), device=device)
        B, K = top5_random.shape
        user_ids_exp = user_ids.unsqueeze(1).repeat(1, K).reshape(-1)
        actions_exp = top5_random.reshape(-1)
        rewards_flat = dataset.reward_function(user_ids_exp, actions_exp)
        rewards = rewards_flat.view(B, K).mean(dim=1)
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

        B, K = top5_actions.shape
        user_ids_exp = user_ids.unsqueeze(1).repeat(1, K).reshape(-1)
        actions_exp = top5_actions.reshape(-1)

        rewards_flat = dataset.reward_function(user_ids_exp, actions_exp)
        rewards = rewards_flat.view(B, K).mean(dim=1)

        return rewards.mean().item()