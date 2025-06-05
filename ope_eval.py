import torch
from action_policies import UniformRandomPolicy, TwoStageRankingPolicy
from two_stage_bandit import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy
from estimators import (
    ImportanceSamplingEstimator,
    DirectMethodEstimator,
    DoublyRobustEstimator,
    KernelISEstimator
)
import torch.nn as nn
import random
import numpy as np

class TrueRewardModel:
    def __init__(self, action_context: torch.Tensor):
        self.action_context = action_context

    def predict(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (contexts * self.action_context[actions]).sum(dim=1)

class ConstantMarginalDensityModel(nn.Module):
    def __init__(self, constant: float = 1.0):
        super().__init__()
        self.register_buffer("value", torch.tensor(constant))

    def predict(self, x, a_idx, action_context):
        return self.value

class LearnedMarginalDensityModel(nn.Module):
    """
    Trainable model estimating the logging marginal density using context 
    and action embeddings.
    """
    def __init__(self, context_dim, action_dim, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(context_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )

    def forward(self, x, action_embed):
        input = torch.cat([x, action_embed], dim=1)
        return self.model(input).squeeze()

    def predict(self, x, a_idx, action_context):
        a_embed = action_context[a_idx].unsqueeze(0)
        x = x.unsqueeze(0)
        return self.forward(x, a_embed)

def compute_target_pscores(policy, x, a, candidates):
    propensities = torch.zeros(len(x))
    for i in range(len(x)):
        x_i = x[i].unsqueeze(0)         
        a_i = a[i].item()
        A_k = candidates[i].unsqueeze(0) 

        with torch.no_grad():
            probs = policy.second_stage(x_i, A_k).squeeze()

        if a_i in A_k[0]:
            idx = (A_k[0] == a_i).nonzero(as_tuple=True)[0].item()
            propensities[i] = probs[idx]
        else:
            propensities[i] = 1e-8  

    return propensities

def estimate_true_policy_value(policy, reward_model, contexts):
    with torch.no_grad():
        ranked = policy.rank_topk(contexts) 
        rewards = reward_model.predict(contexts, ranked[:, 0])  # selecting top ranked item
        return rewards.mean().item()

def evaluate(method: str, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cpu"
    dataset = torch.load(f"{method}_data.pt")

    bandit = dataset["bandit"]
    x = bandit["context"]
    a_taken = bandit["action"]
    r = bandit["reward"]
    pi_b = bandit["pscore"]
    candidates = bandit["candidates"]
    action_context = dataset["action_context"]
    
    dim_context = x.shape[1]
    n_actions = action_context.shape[0]
    top_k = candidates.shape[1]

    model1 = TwoTowerFirstStagePolicy(dim_context, n_actions, 32, top_k)
    model1.load_state_dict(torch.load(f"{method}_first_stage.pt"))
    model1.eval()

    model2 = SoftmaxSecondStagePolicy(dim_context, 32, model1)
    model2.load_state_dict(torch.load(f"{method}_second_stage.pt"))
    model2.eval()

    policy = TwoStageRankingPolicy(model1, model2, top_k=top_k, device=device)
    rm = TrueRewardModel(action_context)

    true_val = estimate_true_policy_value(policy, rm, x) # ground truth

    if method == "mse":
        pi_t = compute_target_pscores(policy, x, a_taken, candidates)

        dm = DirectMethodEstimator(rm, policy, x, action_context)
        dr = DoublyRobustEstimator(rm, x, a_taken, pi_b, pi_t, r, policy, action_context)

        dm_val = dm.estimate_policy_value()
        dr_val = dr.estimate_policy_value()

        return {
            "DM": dm_val,
            "DR": dr_val,
            "True": true_val
        }

    elif method == "opl":
        pi_t = compute_target_pscores(policy, x, a_taken, candidates)
        is_estimator = ImportanceSamplingEstimator(pi_b, pi_t, r)
        is_val = is_estimator.estimate_policy_value()

        return {
            "IS": is_val,
            "True": true_val
        }

    elif method == "ma_et_al":
        kis_model = ConstantMarginalDensityModel()
        kis = KernelISEstimator(
            data=list(zip(x, a_taken, r)),
            target_policy=policy,
            logging_policy=UniformRandomPolicy(),
            # rbf kernel
            kernel = lambda y, y_i, x, tau: torch.exp(-torch.norm(action_context[y] - action_context[y_i]) ** 2 / (2 * tau ** 2)),
            tau=1.0,
            marginal_density_model=kis_model,
            action_context=action_context,
            num_epochs=10
        )
        kis_val = kis.estimate_policy_value()

        return {
            "KIS": kis_val,
            "True": true_val
        }