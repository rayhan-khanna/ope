import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from custom_obp.ope.gradients import TwoStageISGradient, KernelISGradient
from custom_obp.ope.estimators import TwoStageISEstimator, KernelISEstimator
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.action_policies import UniformRandomPolicy
from custom_obp.policy.two_stage_policy import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 0.1):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value
    
def train(method: str, n_epochs=300, kernel_fn=None):
    device = "cpu"

    # create dataset
    dataset = CustomSyntheticBanditDataset(
        n_actions=10, dim_context=5, top_k=5, 
        reward_std=0.5, action_policy=UniformRandomPolicy(), device=device, 
        single_stage=False
    )
    feedback = dataset.obtain_batch_bandit_feedback(n_samples=10000, n_users=1000)

    x = feedback["context"]
    a = feedback["action"]
    r = feedback["reward"]
    # pi0 = feedback["pscore"]
    action_context = dataset.action_context

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

    optimizer = optim.Adam(first_stage.parameters())
    density_model = ConstantMarginalDensityModel()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if method == "single_preference_is":
            resampled_candidates = first_stage.sample_topk_gumbel(x)
            match = (resampled_candidates == a.unsqueeze(1))
            valid = match.any(dim=1)

            x_valid = x[valid]
            a_valid = a[valid]
            r_valid = r[valid]
            c_valid = resampled_candidates[valid]
            pi0_valid = torch.full_like(a_valid, 1.0 / first_stage.top_k, dtype=torch.float32)

            loss_fn = TwoStageISGradient(
                first_stage=first_stage,
                second_stage=second_stage,
                context=x_valid,
                actions=a_valid,
                rewards=r_valid,
                behavior_pscore=pi0_valid,
                candidates=c_valid,
            )

        elif method in {"iter_k_is", "iter_k_kis"}:
            n_pref_per_user = 1
            sampled_data = dataset.sample_k_prefs(feedback, n_pref_per_user)
            x_sampled = sampled_data["context"]
            a_sampled = sampled_data["action"]
            r_sampled = sampled_data["reward"]
            resampled_candidates = first_stage.sample_topk_gumbel(x_sampled)
            match = (resampled_candidates == a_sampled.unsqueeze(1))
            valid = match.any(dim=1)

            x_valid = x_sampled[valid]
            a_valid = a_sampled[valid]
            r_valid = r_sampled[valid]
            c_valid = resampled_candidates[valid]

            pi0_valid = torch.full_like(a_valid, 1.0 / first_stage.top_k, dtype=torch.float32)

            if method == "iter_k_is":
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
                    tau=0.1,
                    marginal_density_model=density_model,
                    action_context=action_context,
                    candidates=c_valid
                )
                
        loss = loss_fn.estimate_policy_gradient()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                if method == "single_preference_is":
                    eval_candidates = first_stage.sample_topk_gumbel(x)
                    match = (eval_candidates == a.unsqueeze(1))
                    valid = match.any(dim=1)

                    x_valid = x[valid]
                    a_valid = a[valid]
                    r_valid = r[valid]
                    c_valid = eval_candidates[valid]
                    pi0_valid = torch.full_like(a_valid, 1.0 / first_stage.top_k, dtype=torch.float32)

                    estimator = TwoStageISEstimator(
                        context=x_valid,
                        actions=a_valid,
                        rewards=r_valid,
                        behavior_pscore=pi0_valid,
                        first_stage_policy=first_stage,
                        second_stage_policy=second_stage,
                        candidates=c_valid
                    )
                    
                elif method == "iter_k_is":
                    eval_candidates = first_stage.sample_topk_gumbel(x_sampled)
                    match = (eval_candidates == a_sampled.unsqueeze(1))
                    valid = match.any(dim=1)

                    x_valid = x_sampled[valid]
                    a_valid = a_sampled[valid]
                    r_valid = r_sampled[valid]
                    c_valid = eval_candidates[valid]
                    pi0_valid = torch.full_like(a_valid, 1.0 / first_stage.top_k, dtype=torch.float32)

                    estimator = TwoStageISEstimator(
                        context=x_valid,    
                        actions=a_valid,
                        rewards=r_valid,
                        behavior_pscore=pi0_valid,
                        first_stage_policy=first_stage,
                        second_stage_policy=second_stage,
                        candidates=c_valid 
                    )

                elif method == "iter_k_kis":
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
                        tau=0.1,
                        marginal_density_model=density_model,
                        action_context=action_context,
                        candidates=c_valid
                    )

                policy_value = estimator.estimate_policy_value()
                print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f}")
    
    torch.save({
        "first_stage": first_stage.state_dict(),
        "second_stage": second_stage.state_dict()
    }, f"{method}_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_preference_is", "iter_k_is", "iter_k_kis"])
    args = parser.parse_args()

    train(method=args.method)