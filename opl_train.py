import argparse
import torch
import torch.optim as optim
from obp.policy.two_stage_policy import (
    TwoTowerFirstStagePolicy,
    SoftmaxSecondStagePolicy,
    mse_loss,
    opl_loss,
    ma_et_al_loss
)
from obp.dataset.synthetic_bandit_dataset import CustomSyntheticBanditDataset
from obp.policy.action_policies import UniformRandomPolicy

def train_model(method: str):
    device = "cpu"
    dataset = CustomSyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        top_k=5,
        action_policy=UniformRandomPolicy(),
        reward_std=0.5,
        device=device
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples=2000)
    x = bandit_feedback["context"]
    a_taken = bandit_feedback["action"]
    r = bandit_feedback["reward"]
    pi0 = bandit_feedback["pscore"]
    candidates = bandit_feedback["candidates"]

    dim_context = dataset.dim_context
    num_items = dataset.n_actions
    emb_dim = 32
    top_k = dataset.top_k

    model1 = TwoTowerFirstStagePolicy(dim_context, num_items, emb_dim, top_k).to(device)
    model2 = SoftmaxSecondStagePolicy(dim_context, emb_dim, model1).to(device)
    optimizer = optim.Adam(model1.parameters())

    for epoch in range(100):
        optimizer.zero_grad()
        if method == "mse":
            loss = mse_loss(model1, x, a_taken, r)
        elif method == "opl":
            loss = opl_loss(model1, x, a_taken, pi0, r)
        elif method == "ma_et_al":
            loss = ma_et_al_loss(model1, model2, x, a_taken, pi0, r, candidates)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] {method.upper()} Loss: {loss.item():.4f}")

    torch.save({"bandit": bandit_feedback, "action_context": dataset.action_context}, f"{method}_data.pt")
    torch.save(model1.state_dict(), f"{method}_first_stage.pt")
    torch.save(model2.state_dict(), f"{method}_second_stage.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["mse", "opl", "ma_et_al"])
    args = parser.parse_args()
    train_model(args.method)