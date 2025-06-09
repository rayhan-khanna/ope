import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from gradients import TwoStageISGradient, KernelISGradient
from estimators import ImportanceSamplingEstimator, KernelISEstimator
from synthetic_bandit_dataset import CustomSyntheticBanditDataset
from action_policies import TwoStageRankingPolicy, UniformRandomPolicy
from two_stage_bandit import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 1.0):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value
    
def train(method: str, n_epochs=100, k_users=5, kernel_fn=None):
    device = "cpu"

    # Setup bandit dataset and logging policy
    dataset = CustomSyntheticBanditDataset(
        n_actions=10, dim_context=5, top_k=5, 
        reward_std=0.5, action_policy=UniformRandomPolicy(), device=device, 
        single_stage=False
    )
    feedback = dataset.sample_k_user_preferences(k_users)

    x = feedback["context"]
    a = feedback["action"]
    r = feedback["reward"]
    pi0 = feedback["pscore"]
    action_context = dataset.action_context
    candidates = dataset.candidate_selection(x)

    first_stage = TwoTowerFirstStagePolicy(
        dim_context=5,
        num_items=dataset.n_actions,
        emb_dim=64,
        top_k=5
    )
    second_stage = SoftmaxSecondStagePolicy(
        dim_context=5,
        emb_dim=64,
        first_stage_policy=first_stage
    )
    target_policy = TwoStageRankingPolicy(first_stage, second_stage, top_k=5, device=device)
    optimizer = optim.Adam(target_policy.parameters())

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if method == "single_user_is":
            loss_fn = TwoStageISGradient(
                policy=target_policy,
                context=x,
                actions=a,
                rewards=r,
                behavior_pscore=pi0,
                candidates=candidates
            )
            loss = loss_fn.estimate_policy_gradient()

        elif method in {"iter_k_is", "iter_k_kis"}:
            loss = 0.0
            for _ in range(k_users):
                # Sample preference dataset
                sampled_data = dataset.sample_k_user_preferences(k=k_users)
                x_sampled = sampled_data["context"]
                a_sampled = sampled_data["action"]
                r_sampled = sampled_data["reward"]
                pi0_sampled = sampled_data["pscore"]

                if method == "iter_k_is":
                    loss_fn = TwoStageISGradient(
                        policy=target_policy,
                        context=x_sampled,
                        actions=a_sampled,
                        rewards=r_sampled,
                        behavior_pscore=pi0_sampled,
                        candidates=sampled_data["candidates"]
                    )
                    loss += loss_fn.estimate_policy_gradient()

                elif method == "iter_k_kis":
                    data = list(zip(x_sampled, a_sampled, r_sampled))
                    marginal_model = ConstantMarginalDensityModel()
                    loss_fn = KernelISGradient(
                        data=data,
                        target_policy=target_policy,
                        logging_policy=dataset.action_policy,
                        kernel=kernel_fn,
                        tau=0.1,
                        marginal_density_model=marginal_model,
                        action_context=action_context
                    )
                    loss += loss_fn._estimate_policy_gradient()

            loss = loss / k_users

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
              if method in {"single_user_is", "iter_k_is"}:
                  logits = target_policy.first_stage(x)[0]
                  topk_idx = logits.topk(target_policy.top_k, dim=1).indices
                  probs_k  = target_policy.probs_given_topk(x, topk_idx)
                  mask = (topk_idx == a.unsqueeze(1))                 
                  selected_p = (probs_k * mask.float()).sum(dim=1)             
                  estimator = ImportanceSamplingEstimator(pi0, selected_p, r)
              elif method == "iter_k_kis":
                  data_for_eval = list(zip(x, a, r))
                  marginal_model = loss_fn.marginal_density_model
                  estimator = KernelISEstimator(
                      data=data_for_eval,
                      target_policy=target_policy,
                      logging_policy=dataset.action_policy,
                      kernel=kernel_fn,
                      tau=0.1,
                      marginal_density_model=marginal_model,
                      action_context=action_context
                  )
              
              policy_value = estimator.estimate_policy_value()
              print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f}")
    
    torch.save(target_policy.state_dict(), f"{method}_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_user_is", "iter_k_is", "iter_k_kis"])
    args = parser.parse_args()

    train(method=args.method)