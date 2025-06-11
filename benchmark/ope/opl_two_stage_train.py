import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from obp.ope.gradients import TwoStageISGradient, KernelISGradient
from obp.ope.estimators import ImportanceSamplingEstimator, KernelISEstimator
from obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from obp.policy.action_policies import TwoStageRankingPolicy, UniformRandomPolicy
from obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 0.1):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value
    
def train(method: str, n_epochs=300, k_users=5, kernel_fn=None):
    device = "cpu"

    # Setup bandit dataset and logging policy
    dataset = CustomSyntheticBanditDataset(
        n_actions=10, dim_context=5, top_k=5, 
        reward_std=0.5, action_policy=UniformRandomPolicy(), device=device, 
        single_stage=False
    )
    feedback = dataset.obtain_batch_bandit_feedback(n_samples=10000)

    x = feedback["context"]
    a = feedback["action"]
    r = feedback["reward"]
    pi0 = feedback["pscore"]
    action_context = dataset.action_context
    candidates = feedback["candidates"]

    if kernel_fn is None and method == "iter_k_kis":
        kernel_fn = lambda y, y_i, x, tau: torch.exp(
            -torch.norm(action_context[y] - action_context[y_i]) ** 2 / (2 * tau ** 2)
        )

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

    target_policy = TwoStageRankingPolicy(first_stage, second_stage, top_k=5, 
                                          action_context=action_context, device=device)
    optimizer = optim.Adam(target_policy.parameters())
    density_model = ConstantMarginalDensityModel()

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
            sampled_data = dataset.sample_k_user_batch_from_feedback(feedback, k_users, target_policy)
            x_sampled = sampled_data["context"]
            a_sampled = sampled_data["action"]
            r_sampled = sampled_data["reward"]
            pi0_sampled = sampled_data["pscore"]
            cand_sampled = sampled_data["candidates"]

            if method == "iter_k_is":
              loss_fn = TwoStageISGradient(
                  policy = target_policy,
                  context=x_sampled,
                  actions=a_sampled,
                  rewards=r_sampled,
                  behavior_pscore=pi0_sampled,
                  candidates=cand_sampled
              )

            else: 
              loss_fn = KernelISGradient(
                    context=x_sampled,
                    actions=a_sampled,
                    rewards=r_sampled,
                    target_policy=target_policy,
                    logging_policy=dataset.action_policy,
                    kernel=kernel_fn,
                    tau=0.3,
                    marginal_density_model=density_model,
                    action_context=action_context,
                    candidates=cand_sampled
                )
            loss = loss_fn.estimate_policy_gradient()

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
                estimator = KernelISEstimator(
                    context=x,
                    actions=a,
                    rewards=r,
                    target_policy=target_policy,
                    logging_policy=dataset.action_policy,
                    kernel=kernel_fn,
                    tau=0.3,
                    marginal_density_model=density_model,
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