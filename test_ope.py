import torch
import torch.optim as optim
import numpy as np
from synthetic_bandit_dataset import CustomSyntheticBanditDataset
from action_policies import UniformRandomPolicy, SoftmaxPolicy, TwoStageRankingPolicy
from two_stage_bandit import TwoTowerFirstStagePolicy, SoftmaxSecondStagePolicy, ma_et_al_loss
from estimators import (
    ImportanceSamplingEstimator,
    DirectMethodEstimator,
    DoublyRobustEstimator, 
    KernelISEstimator
)
 
class TrueRewardModel:
    def __init__(self, action_context: torch.Tensor):
        self.action_context = action_context

    def predict(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (contexts * self.action_context[actions]).sum(dim=1)

# Dummy marginal‐density model for Kernel IS eval
class DummyDensityModel:
    """Always returns 1.0, and ignores updates (so KIS = IS)."""
    def predict(self, x, y):
        return torch.tensor(1.0, device=x.device)

    def update(self, x, y, loss):
        pass

# Compute π_t for logged actions
def compute_target_pscores(context, candidates, actions, action_context, policy: SoftmaxPolicy):
    """
    For each i: compute π_t(a_i | x_i) under `policy` over its candidate set.
    """
    propensities = torch.zeros(len(actions), device=context.device)
    temp = policy.temperature
    for i in range(len(actions)):
        x_i = context[i]
        c_i = candidates[i]
        emb = action_context[c_i]
        scores = (x_i.unsqueeze(0) * emb).sum(dim=1)
        exp_s = torch.exp(scores / temp)
        probs = exp_s / exp_s.sum()
        idx = (c_i == actions[i]).nonzero(as_tuple=True)[0]
        propensities[i] = probs[idx]
    return propensities

# Experiment
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    log_policy = UniformRandomPolicy()     
    target_policy = SoftmaxPolicy(temperature=1.0, random_state=0)

    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        reward_std=0.5,
        action_policy=log_policy,
        random_state=0,
        device="cpu"
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples=2000)

    x          = bandit_feedback["context"]     
    candidates = bandit_feedback["candidates"] 
    a_taken    = bandit_feedback["action"]      
    r          = bandit_feedback["reward"]      
    pi_b       = bandit_feedback["pscore"]     
    action_context    = dataset.action_context 

    # Compute π_t for each logged action
    pi_t = compute_target_pscores(x, candidates, a_taken, action_context, target_policy)

    # Initialize true reward model
    rm = TrueRewardModel(action_context)

    dim_context = dataset.dim_context
    num_items = dataset.n_actions
    emb_dim = 32
    top_k = dataset.top_k

    first_stage_model = TwoTowerFirstStagePolicy(dim_context, num_items, emb_dim, top_k)
    second_stage_model = SoftmaxSecondStagePolicy(dim_context, emb_dim, first_stage_model)
    optimizer = optim.Adam(first_stage_model.parameters())
    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = ma_et_al_loss(first_stage_model, second_stage_model, x, a_taken, pi_b, r, candidates)
        loss.backward()
        optimizer.step()
        #if epoch % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    two_stage_policy = TwoStageRankingPolicy(
        first_stage_model,      # your trained TwoTowerFirstStagePolicy
        second_stage_model,     # your trained SoftmaxSecondStagePolicy
        top_k=dataset.top_k,
        device="cpu"
    )

    # Initialize estimators
    ise = ImportanceSamplingEstimator(pi_b, pi_t, r)
    dm  = DirectMethodEstimator(rm, target_policy, x, action_context)
    dr  = DoublyRobustEstimator(rm, x, a_taken, pi_b, pi_t, r, target_policy, action_context)
    kis = KernelISEstimator(
        data=list(zip(x, a_taken, r)),
        target_policy=target_policy,
        logging_policy=log_policy,
        kernel=lambda y, y2, x, tau: torch.exp(-torch.norm(action_context[y] - action_context[y2])**2 / tau),
        tau=1.0,
        marginal_density_model=DummyDensityModel(),
        action_context=action_context, 
        num_epochs=3
    )

    dm_2stage = DirectMethodEstimator(rm, two_stage_policy, x, action_context)
    kis_2stage = KernelISEstimator(
        data=list(zip(x, a_taken, r)),
        target_policy=two_stage_policy,
        logging_policy=log_policy,
        kernel=lambda y, y2, xc, tau: torch.exp(-torch.norm(action_context[y] - action_context[y2])**2 / tau),
        tau=1.0,
        marginal_density_model=DummyDensityModel(),
        action_context=action_context,
        num_epochs=3
    )

    print(f"IS estimate: {ise.estimate_policy_value():.4f}")
    print(f"DM estimate: {dm.estimate_policy_value():.4f}")
    print(f"DR estimate: {dr.estimate_policy_value():.4f}")
    print(f"KIS estimate: {kis.estimate_policy_value():.4f}")
    print(f"DM (2-stage) estimate:  {dm_2stage.estimate_policy_value():.4f}")
    print(f"KIS (2-stage) estimate: {kis_2stage.estimate_policy_value():.4f}")