import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from custom_obp.ope.gradients import TwoStageISGradient, KernelISGradient
from custom_obp.ope.estimators import TwoStageISEstimator, KernelISEstimator
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.action_policies import SoftmaxPolicy
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from custom_obp.models.cf_model import NaiveCF
from online_eval import online_eval_once

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 0.1):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value
    
class TrainableMarginalDensityModel(nn.Module):
    def __init__(self, dim_context, dim_action, top_k=None):
        super().__init__()
        self.top_k = top_k
        self.f = nn.Sequential(
            nn.Linear(dim_context + dim_action * (top_k if top_k else 1), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def predict(self, x, a_idx, action_context):
        if a_idx.dim() == 2:
            a_vec = action_context[a_idx]
            a_vec = a_vec.reshape(x.size(0), -1)
        else:
            a_vec = action_context[a_idx]

        x_cat = torch.cat([x, a_vec], dim=-1)
        return self.f(x_cat).squeeze(-1)  

def train(method, n_epochs=300, kernel_fn=None, seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax_logging_policy = SoftmaxPolicy(dim_context=5, temperature=1.0).to(device)
    losses = []

    # create dataset
    dataset = CustomSyntheticBanditDataset(
        n_actions=1000, dim_context=5, top_k=10, 
        reward_std=0.5, action_policy=softmax_logging_policy, device=device, 
        single_stage=(method=="naive_cf")
    )

    feedback = dataset.obtain_batch_bandit_feedback(n_samples=100000, n_users=1000)

    if method in {"random", "oracle"}:
        torch.save(feedback["user_id"], f"{method}_eval_user_ids.pt")
        torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
        return [], [] 

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
        return np.array(losses), []

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
        return np.array(epoch_losses), []

    mean_probs_per_epoch = []

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
    dim_context = dataset.dim_context
    dim_action = action_context.shape[1]

    # top_k = 5 here because we're going for 5 ranked action output
    density_model = TrainableMarginalDensityModel(dim_context, dim_action, top_k=5).to(device)
    n_steps_per_epoch =  10

    tau = 9.3 # 10.1 for the uniform logging policy case topk=10, etc.

    def kernel_fn(y, y_i, _x, tau):
        vec_y = first_stage.item_embeddings(y)
        vec_yi = first_stage.item_embeddings(y_i)
        if vec_y.dim() == 3:
            vec_y = vec_y.reshape(vec_y.size(0), -1)
            vec_yi = vec_yi.reshape(vec_yi.size(0), -1)
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
                a_sampled = sampled_data["action"].to(device)   # logged actions (DO NOT overwrite)
                r_sampled = sampled_data["reward"].to(device)

                if a_sampled.dim() == 3:
                    B, P, R = a_sampled.shape
                    x_sampled = x_sampled.repeat_interleave(P, 0)
                    a_sampled = a_sampled.view(B*P, R)
                    r_sampled = r_sampled.view(B*P, R) 

                resampled_candidates = first_stage.sample_topk_gumbel(x_sampled)
                logged_actions = a_sampled  
                if a_sampled.dim() == 2 and a_sampled.size(1) == 1:
                    # if it's really a single action, squeeze to (B,)
                    a_sampled = a_sampled.squeeze(1)
                    r_sampled = r_sampled.squeeze(1)
                    ranking = False
                else:
                    ranking = True

                # compute avg distance using logged actions vs candidates
                with torch.no_grad():
                    logged_emb = first_stage.item_embeddings(logged_actions)
                    cand_emb = first_stage.item_embeddings(resampled_candidates)
                    if logged_actions.dim() == 1:  
                        dists = torch.cdist(logged_emb.unsqueeze(1), cand_emb, p=2)
                        min_dists = dists.min(dim=2).values.squeeze(1)
                    else:  
                        B, R, D = logged_emb.shape
                        _, K, _ = cand_emb.shape
                        dists = torch.cdist(
                            logged_emb.reshape(B*R, 1, D),
                            cand_emb.repeat_interleave(R, dim=0),
                            p=2
                        ).view(B, R, K)
                        min_dists = dists.min(dim=2).values.mean(dim=1)
                    avg_min_dist = min_dists.mean().item()

                if a_sampled.dim() == 1: 
                    match = (resampled_candidates == a_sampled.unsqueeze(1))
                    valid = match.any(dim=1)
                else:                                      
                    match = torch.stack(
                        [(resampled_candidates == a_sampled[:, j].unsqueeze(1))
                        for j in range(a_sampled.size(1))], 
                        dim=1  
                    )
                    valid = match.any(dim=2).any(dim=1) 
                    # match = (resampled_candidates.unsqueeze(1) == a_sampled.unsqueeze(-1))
                    # valid = match.all(dim=2).all(dim=1)

                x_valid = x_sampled[valid]
                if x_valid.size(0) == 0:
                    continue 
                a_valid = a_sampled[valid]
                r_valid = r_sampled[valid]
                c_valid = resampled_candidates[valid]

                if ranking:
                    pi0_valid = None
                else:
                    probs_all = dataset.action_policy.probs(x_valid, action_context)
                    pi0_valid = probs_all[torch.arange(len(a_valid)), a_valid]

                if method in {"single_preference_is", "multimodal_preference_is"}:
                    loss_fn = TwoStageISGradient(
                        first_stage=first_stage,
                        second_stage=second_stage,
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        logging_policy=dataset.action_policy,
                        behavior_pscore=pi0_valid,
                        candidates=c_valid,
                        action_context=action_context,
                        ranking=ranking
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
                        candidates=c_valid,
                        ranking=ranking
                    )

            loss = loss_fn.estimate_policy_gradient()

            loss.backward()
            with torch.no_grad():
                probs = second_stage.calc_prob_given_output(x_valid, c_valid)
                match = (c_valid.unsqueeze(1) == a_valid.unsqueeze(-1))
                probs_exp = probs.unsqueeze(1)
                matched_probs = (match.float() * probs_exp).sum(dim=-1)
                avg_probs = matched_probs.mean(dim=1)
                mean_probs_per_epoch.append(avg_probs.mean().item())

            losses.append(loss.item())
            optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                torch.save({
                    "first_stage": first_stage.state_dict(),
                    "second_stage": second_stage.state_dict(),
                    "action_context": dataset.action_context
                }, f"{method}_policy_seed{seed}.pt")

            # Sample eval preferences
            if method in {"single_preference_is", "single_preference_kis"}:
                n_pref_eval = 1
            else:
                n_pref_eval = 5

            eval_data = dataset.sample_k_prefs(feedback, n_pref_eval)
            eval_context = eval_data["context"]
            eval_action = eval_data["action"]
            eval_reward = eval_data["reward"]
            eval_candidates = first_stage.sample_topk_gumbel(eval_context)
            if eval_action.dim() == 3:          # [B, P, R]  (multi‑preference)
                B, P, R = eval_action.shape
                eval_context  = eval_context.repeat_interleave(P, 0)   # keep aligned
                eval_action   = eval_action.view(B*P, R)               # [B·P, R]
                eval_reward   = eval_reward.view(B*P, R)
                eval_candidates = eval_candidates.repeat_interleave(P, 0)

            ranking_eval = not (eval_action.dim() == 2 and eval_action.size(1) == 1)
            if not ranking_eval:
                eval_action  = eval_action.squeeze(1)
                eval_reward  = eval_reward.squeeze(1)

            if eval_action.dim() == 1:  # single-action case
                match = (eval_candidates == eval_action.unsqueeze(1))
                valid = match.any(dim=1)
            else:  # ranking case
                match = torch.stack(
                    [(eval_candidates == eval_action[:, j].unsqueeze(1))
                    for j in range(eval_action.size(1))],
                    dim=1
                )
                valid = match.any(dim=2).any(dim=1)

            x_valid = eval_context[valid]
            if x_valid.size(0) == 0:
                print(f"[Epoch {epoch}] No valid samples for evaluation, skipping OPE.")
                continue
            a_valid = eval_action[valid]
            r_valid = eval_reward[valid]
            c_valid = eval_candidates[valid]

            if method in {"single_preference_is", "multimodal_preference_is"}:
                if not ranking_eval:  # single-action case
                    probs_all = dataset.action_policy.probs(x_valid, action_context)
                    pi0_valid = probs_all[torch.arange(len(a_valid)), a_valid]
                else:
                    pi0_valid = None

                estimator = TwoStageISEstimator(
                    context=x_valid,
                    actions=a_valid,
                    rewards=r_valid,
                    behavior_pscore=pi0_valid,
                    first_stage_policy=first_stage,
                    second_stage_policy=second_stage,
                    candidates=c_valid,
                    action_context=action_context,
                    logging_policy=dataset.action_policy,
                    ranking=ranking_eval
                )
            else:
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
                    candidates=c_valid,
                    ranking=ranking_eval
                )

            policy_value = estimator.estimate_policy_value()
            online_value = online_eval_once(method=method, seed=seed)
            # print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f} | Online: {online_value:.4f}")
            print(
                    f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} "
                    f"| OPE: {policy_value:.4f} | Online: {online_value:.4f} "
                    f"| AvgDist: {avg_min_dist:.4f}"
                )
            # print(sf"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f} | Online: {online_value:.4f}")

    torch.save(dataset.action_context, f"{method}_action_context.pt")
    torch.save(x, f"{method}_eval_context.pt")
    torch.save(u, f"{method}_eval_user_ids.pt")
    torch.save(dataset.user_embeddings, f"{method}_user_embeddings.pt")
    return (np.array(losses).reshape(n_epochs, -1).mean(axis=1), mean_probs_per_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_preference_is", "single_preference_kis", 
                                 "multimodal_preference_is", "multimodal_preference_kis", "naive_cf", 
                                 "online_policy", "random", "oracle"])
    args = parser.parse_args()

    train(method=args.method)