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
    
def train(method: str, n_epochs=300, k_users=5, kernel_fn=None):
    device = "cpu"

    # create dataset
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

    if method == "single_user_is":
        optimizer = optim.Adam(first_stage.parameters())
    else:
        optimizer = optim.Adam(first_stage.parameters(), lr=1e-5)

    density_model = ConstantMarginalDensityModel()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if method == "single_user_is":
            loss_fn = TwoStageISGradient(
                first_stage=first_stage,
                second_stage=second_stage,
                context=x,
                actions=a,
                rewards=r,
                behavior_pscore=pi0,
                candidates=candidates,
            )

            loss = loss_fn.estimate_policy_gradient()
            loss.backward()

        elif method in {"iter_k_is", "iter_k_kis"}:
            accumulated_loss = 0
            for _ in range(10):  # accumulate 10 batches per epoch
                sampled_data = dataset.sample_k_user_batch_from_feedback(feedback, k_users=20)
                x_sampled = sampled_data["context"]
                a_sampled = sampled_data["action"]
                r_sampled = sampled_data["reward"]
                pi0_sampled = sampled_data["pscore"]
                cand_sampled = sampled_data["candidates"]

                if method == "iter_k_is":
                    loss_fn = TwoStageISGradient(
                        first_stage=first_stage,
                        second_stage=second_stage,
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
                        first_stage=first_stage,
                        second_stage=second_stage,  
                        logging_policy=dataset.action_policy,
                        kernel=kernel_fn,
                        tau=0.3,
                        marginal_density_model=density_model,
                        action_context=action_context,
                        candidates=cand_sampled
                    )
                loss = loss_fn.estimate_policy_gradient()
                loss.backward()
                accumulated_loss += loss.item()

        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                if method == "single_user_is":
                    estimator = TwoStageISEstimator(
                        context=x,
                        actions=a,
                        rewards=r,
                        behavior_pscore=pi0,
                        first_stage_policy=first_stage,
                        second_stage_policy=second_stage,
                        candidates=candidates 
                    )

                elif method == "iter_k_is":
                    estimator = TwoStageISEstimator(
                        context=x_sampled,    
                        actions=a_sampled,
                        rewards=r_sampled,
                        behavior_pscore=pi0_sampled,
                        first_stage_policy=first_stage,
                        second_stage_policy=second_stage,
                        candidates=cand_sampled 
                    )

                elif method == "iter_k_kis":
                    estimator = KernelISEstimator(
                    context=x_sampled,
                    actions=a_sampled,
                    rewards=r_sampled,
                    first_stage=first_stage,
                    second_stage=second_stage,
                    logging_policy=dataset.action_policy,
                    kernel=kernel_fn,
                    tau=0.3,
                    marginal_density_model=density_model,
                    action_context=action_context,
                    candidates=cand_sampled
                )
                    
                policy_value = estimator.estimate_policy_value()
                if method != "single_user_is":
                    print(f"[Epoch {epoch}] Method: {method} | Avg Loss: {accumulated_loss / 10:.4f} | OPE: {policy_value:.4f}")
                else:
                    print(f"[Epoch {epoch}] Method: {method} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f}")
    
    torch.save({
        "first_stage": first_stage.state_dict(),
        "second_stage": second_stage.state_dict()
    }, f"{method}_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_user_is", "iter_k_is", "iter_k_kis"])
    args = parser.parse_args()

    train(method=args.method)