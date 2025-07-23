import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from custom_obp.ope.gradients import TwoStageISGradient, KernelISGradient
from custom_obp.ope.estimators import TwoStageISEstimator, KernelISEstimator
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.action_policies import UniformRandomPolicy
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from custom_obp.models.cf_model import NaiveCF
from online_eval import online_eval_once

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 0.1):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value
    
def train(method, n_epochs=300, kernel_fn=None, seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = []

    # create dataset
    dataset = CustomSyntheticBanditDataset(
        n_actions=1000, dim_context=5, top_k=10, 
        reward_std=0.5, action_policy=UniformRandomPolicy(), device=device, 
        single_stage=(method=="naive_cf")
    )

    feedback = dataset.obtain_batch_bandit_feedback(n_samples=100000, n_users=1000)

    if method in {"random", "oracle"}:
        torch.save(feedback["user_id"], f"{method}_eval_user_ids.pt")
        torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
        return

    x = feedback["context"].to(device)
    a = feedback["action"].to(device)
    r = feedback["reward"].to(device)
    u = feedback["user_id"].to(device)
    action_context = dataset.action_context.to(device)

    if method == "naive_cf":
        model = NaiveCF(dim_context=dataset.dim_context, n_items=dataset.n_actions, emb_dim=32).to(device)
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            preds = model(x, a)
            loss = loss_fn(preds, r)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    mse = ((model(x, a) - r) ** 2).mean()
                    print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | MSE: {mse.item():.4f}")

        torch.save(model.state_dict(), f"{method}_policy_seed{seed}.pt")
        torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
        torch.save(dataset.action_context, f"{method}_action_context.pt")
        return losses

    elif method == "online_policy":
        first_stage = TwoTowerFirstStagePolicy(
            dim_context=5,
            num_items=dataset.n_actions,
            emb_dim=64,
            top_k=10
        ).to(device)
        second_stage = SoftmaxSecondStagePolicy(
            dim_context=5,
            emb_dim=64,
            first_stage_policy=first_stage
        ).to(device)

        optimizer = optim.Adam(list(first_stage.parameters()) + list(second_stage.parameters()), lr=1e-3)

        n_epochs = 1500
        n_steps_per_epoch = 10
        batch_size = 3000
        print_interval = 10
        entropy_coeff = 0.03
        max_grad_norm = 1.0

        losses = []
        rewards = []
        epoch_losses = []
        step_losses = []

        for epoch in range(n_epochs):
            for _ in range(n_steps_per_epoch):
                indices = torch.randint(0, len(x), (batch_size,))
                context = x[indices]
                user_ids = u[indices]
                candidates = first_stage.sample_topk_gumbel(context)
                ranked_actions, _ = second_stage.rank_outputs(context, candidates)
                top1_actions = ranked_actions[:, 0]
                reward = dataset.reward_function(user_ids, top1_actions)
                log_prob = second_stage.log_prob(context, top1_actions, dataset.action_context, A_k=candidates)
                loss = -(reward.detach() * log_prob).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                step_losses.append(loss.item())
                rewards.extend(reward.tolist())
                avg_epoch_loss = np.mean(step_losses)
                epoch_losses.append(avg_epoch_loss)

            if (epoch + 1) % print_interval == 0:
                avg_reward = np.mean(rewards[-print_interval * n_steps_per_epoch * batch_size:])
                torch.save({
                    "first_stage": first_stage.state_dict(),
                    "second_stage": second_stage.state_dict(),
                    "action_context": dataset.action_context
                }, f"{method}_policy_seed{seed}.pt")
                online_value = online_eval_once(method=method, seed=seed)
                print(f"[Epoch {epoch + 1}] Method: online_policy | Online: {online_value:.4f} | Avg Reward: {avg_reward:.4f} | Avg Loss: {avg_epoch_loss:.4f}")


        torch.save({
            "first_stage": first_stage.state_dict(),
            "second_stage": second_stage.state_dict(),
            "action_context": dataset.action_context
    	}, f"{method}_policy_seed{seed}.pt")

        torch.save(feedback["context"], f"{method}_eval_context.pt")
        torch.save(feedback["user_id"], f"{method}_eval_user_ids.pt")
        torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
        torch.save(dataset.action_context, f"{method}_action_context.pt")
        return np.array(epoch_losses)

    first_stage = TwoTowerFirstStagePolicy(
        dim_context=5,
        num_items=dataset.n_actions,
        emb_dim=64,
        top_k=10
    ).to(device)
    second_stage = SoftmaxSecondStagePolicy(
        dim_context=5,
        emb_dim=64,
        first_stage_policy=first_stage
    ).to(device)

    optimizer = optim.Adam(first_stage.parameters())
    density_model = ConstantMarginalDensityModel().to(device)
    n_steps_per_epoch = 10

    tau = 10.1

    def kernel_fn(y, y_i, _x, tau):
        vec_y = first_stage.item_embeddings(y)
        vec_yi = first_stage.item_embeddings(y_i)
        return torch.exp(-((vec_y - vec_yi).pow(2).sum(dim=-1)) / (2 * tau**2))

    for epoch in range(n_epochs):
        for _ in range(n_steps_per_epoch):
            optimizer.zero_grad()

            if method in {"single_preference_is", "single_preference_kis", "multimodal_preference_is", "multimodal_preference_kis"}:
                if method in ["single_preference_is", "single_preference_kis"]: 
                    n_pref_per_user = 1
                else: 
                    n_pref_per_user = 5
                    
                sampled_data = dataset.sample_k_prefs(feedback, n_pref_per_user)
                x_sampled = sampled_data["context"].to(device)
                a_sampled = sampled_data["action"].to(device)
                r_sampled = sampled_data["reward"].to(device)
                resampled_candidates = first_stage.sample_topk_gumbel(x_sampled)

                with torch.no_grad():
                    logged_emb = first_stage.item_embeddings(a_sampled)
                    cand_emb  = first_stage.item_embeddings(resampled_candidates)
                    dists = torch.cdist(logged_emb.unsqueeze(1), cand_emb, p=2)
                    min_dists = dists.min(dim=2).values.squeeze(1)
                    avg_min_dist = min_dists.mean().item()

                match = (resampled_candidates == a_sampled.unsqueeze(1))
                valid = match.any(dim=1)

                x_valid = x_sampled[valid]
                a_valid = a_sampled[valid]
                r_valid = r_sampled[valid]
                c_valid = resampled_candidates[valid]

                pi0_valid = torch.full_like(a_valid, 1.0 / dataset.n_actions, dtype=torch.float32).to(device)

                if method != "multimodal_preference_kis":
                    loss_fn = TwoStageISGradient(
                        first_stage=first_stage,
                        second_stage=second_stage,
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        behavior_pscore=pi0_valid,
                        candidates=c_valid
                    )
                else:
                    loss_fn = KernelISGradient(
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        first_stage=first_stage,
                        second_stage=second_stage,  
                        logging_policy=dataset.action_policy,
                        kernel=kernel_fn,
                        tau=tau,
                        marginal_density_model=density_model,
                        action_context=action_context,
                        candidates=c_valid
                    )
                    
            loss = loss_fn.estimate_policy_gradient()

            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                torch.save({
                    "first_stage": first_stage.state_dict(),
                    "second_stage": second_stage.state_dict(),
                    "action_context": dataset.action_context
                }, f"{method}_policy_seed{seed}.pt")

                if method in {"single_preference_is", "multimodal_preference_is"}:
                    eval_context = x_sampled
                    eval_action = a_sampled
                    eval_reward = r_sampled

                    eval_candidates = first_stage.sample_topk_gumbel(eval_context)
                    match = (eval_candidates == eval_action.unsqueeze(1))
                    valid = match.any(dim=1)

                    x_valid = eval_context[valid]
                    a_valid = eval_action[valid]
                    r_valid = eval_reward[valid]
                    c_valid = eval_candidates[valid]
                    pi0_valid = torch.full_like(a_valid, 1.0 / dataset.n_actions, dtype=torch.float32).to(device)

                    estimator = TwoStageISEstimator(
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        behavior_pscore=pi0_valid,
                        first_stage_policy=first_stage,
                        second_stage_policy=second_stage,
                        candidates=c_valid
                    )

                elif method in {"single_preference_kis", "multimodal_preference_kis"}:
                    eval_candidates = first_stage.sample_topk_gumbel(x_sampled)
                    match = (eval_candidates == a_sampled.unsqueeze(1))
                    valid = match.any(dim=1)

                    x_valid = x_sampled[valid]
                    a_valid = a_sampled[valid]
                    r_valid = r_sampled[valid]
                    c_valid = eval_candidates[valid]

                    estimator = KernelISEstimator(
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        first_stage=first_stage,
                        second_stage=second_stage,
                        logging_policy=dataset.action_policy,
                        kernel=kernel_fn,
                        tau=tau,
                        marginal_density_model=density_model,
                        action_context=action_context,
                        candidates=c_valid
                    )

                policy_value = estimator.estimate_policy_value()
                online_value = online_eval_once(method=method, seed=seed)
                print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f} | Online: {online_value:.4f} | AvgDist: {avg_min_dist:.4f}")
                # print(sf"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f} | Online: {online_value:.4f}")

    torch.save(dataset.action_context, f"{method}_action_context.pt")
    torch.save(x, f"{method}_eval_context.pt")
    torch.save(u, f"{method}_eval_user_ids.pt")
    torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
    return np.array(losses).reshape(n_epochs, -1).mean(axis=1) if method not in {"random", "oracle"} else []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_preference_is", "single_preference_kis", 
                                 "multimodal_preference_is", "multimodal_preference_kis", "naive_cf", 
                                 "online_policy", "random", "oracle"])
    args = parser.parse_args()

    train(method=args.method)
