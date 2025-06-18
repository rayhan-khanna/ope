import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from custom_obp.ope.gradients import DMGradient, DRGradient, ISGradient
from custom_obp.ope.estimators import DirectMethodEstimator, DoublyRobustEstimator, ImportanceSamplingEstimator
from custom_obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from custom_obp.policy.action_policies import UniformRandomPolicy, SoftmaxPolicy

class RewardModel(nn.Module):
    def __init__(self, dim_context, emb_dim, n_actions):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, emb_dim)
        self.fc1 = nn.Linear(dim_context + emb_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, context, action_indices):
        a_emb = self.action_embed(torch.tensor(action_indices, 
                                               dtype=torch.long, 
                                               device=context.device))
        x = torch.cat([context, a_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

    def predict(self, context, actions):
        return self.forward(context, actions)
    
def train_model(method: str):
    device = "cpu"
    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=1,
        action_policy=UniformRandomPolicy(),
        reward_std=0.5,
        device=device
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples=10000, n_users=1000)
    x = bandit_feedback["context"]
    a_taken = bandit_feedback["action"]
    r = bandit_feedback["reward"]
    pi0 = bandit_feedback["pscore"]
    action_context = dataset.action_context

    dim_context = dataset.dim_context
    n_actions = dataset.n_actions
    emb_dim = 32

    reward_model = RewardModel(dim_context, emb_dim, n_actions).to(device)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)

    for epoch in range(300):
        reward_optimizer.zero_grad()
        pred_r = reward_model(x, a_taken)
        loss = nn.MSELoss()(pred_r, r)
        loss.backward()
        reward_optimizer.step()
        if epoch % 10 == 0:
            print(f"[RewardModel Epoch {epoch}] MSE Loss: {loss.item():.4f}")

    target_policy = SoftmaxPolicy(dim_context).to(device)
    optimizer = optim.Adam(target_policy.parameters())

    for epoch in range(300):
        optimizer.zero_grad()
        if method == "dm":
          loss_fn = DMGradient(
             reward_model=reward_model, 
             target_policy=target_policy, 
             context=x,
             action_context=action_context
          )
          loss = loss_fn.estimate_policy_gradient()

        elif method == "dr":
          loss_fn = DRGradient(
              reward_model=reward_model,
              context=x,
              actions=a_taken,
              behavior_pscore=pi0,
              target_policy=target_policy,
              rewards=r,
              action_context=action_context
          )
          loss = loss_fn.estimate_policy_gradient()

        elif method == "is":
          loss_fn = ISGradient(
              context=x,
              actions=a_taken,
              rewards=r,
              behavior_pscore=pi0,
              target_policy=target_policy,
              action_context=action_context
          )
          loss = loss_fn.estimate_policy_gradient()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pi_target = target_policy.probs(x, action_context)
            target_pscore = pi_target[torch.arange(len(x)), a_taken]

            if method == "dm":
                estimator = DirectMethodEstimator(reward_model, target_policy, x, action_context)
            elif method == "is":
                estimator = ImportanceSamplingEstimator(pi0, target_pscore, r)
            elif method == "dr":
                estimator = DoublyRobustEstimator(
                    reward_model=reward_model,
                    context=x,
                    actions=a_taken,
                    behavior_pscore=pi0,
                    target_pscore=target_pscore,
                    rewards=r,
                    target_policy=target_policy,
                    action_context=action_context
                )
            policy_value = estimator.estimate_policy_value()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] {method.upper()} | Loss: {loss.item():.4f} | OPE: {policy_value:.4f}")

    torch.save({"bandit": bandit_feedback, "action_context": action_context}, f"{method}_data.pt")
    torch.save(target_policy.state_dict(), f"{method}_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["dm", "dr", "is"])
    args = parser.parse_args()
    train_model(args.method)