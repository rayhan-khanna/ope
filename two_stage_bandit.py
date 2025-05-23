import torch
import torch.nn as nn
import torch.nn.functional as F
from synthetic_bandit_dataset import CustomSyntheticBanditDataset
from action_policies import EpsilonGreedyPolicy
import torch.optim as optim

# candidate selection
class TwoTowerFirstStagePolicy(nn.Module):
    def __init__(self, dim_context, num_items, emb_dim, top_k):
        super().__init__()
        self.top_k = top_k  

        # context tower
        self.context_nn = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # item embeddings
        self.item_embeddings = nn.Embedding(num_items, emb_dim)

    def forward(self, x):
        context_repr = self.context_nn(x)  
        item_embs = self.item_embeddings.weight 

        # compute logits via inner product
        logits = torch.matmul(context_repr, item_embs.T) 
        probs = torch.softmax(logits, dim=1) 

        # select top-k 
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=1)

        return top_k_probs, top_k_indices

# action selection (softmax)
class SoftmaxSecondStagePolicy(nn.Module):
    def __init__(self, dim_context, emb_dim, first_stage_policy):
        super().__init__()
        self.first_stage_policy = first_stage_policy  

        self.fc = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU()
        )

    def forward(self, x, A_k, mask=None):
        context_repr = self.fc(x)  # transform context
        item_embs = self.first_stage_policy.item_embeddings(A_k)  # get top-k embeddings

        # compute inner product 
        scores = torch.matmul(item_embs, context_repr.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # compute softmax probabilities over top-k
        probs = torch.softmax(scores, dim=1)

        return probs
  
def mse_loss(model, x, a_taken, r):
    context_repr = model.context_nn(x)             
    item_emb = model.item_embeddings(a_taken)      
    pred_reward = (context_repr * item_emb).sum(dim=1) 
    return F.mse_loss(pred_reward, r)

def opl_loss(model, x, a_taken, pi0_probs, r):
    context_repr = model.context_nn(x)            
    item_embs = model.item_embeddings.weight       
    logits = torch.matmul(context_repr, item_embs.T) 
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs)
    log_prob_taken = log_probs[torch.arange(len(x)), a_taken]
    weight = r / pi0_probs 
    return -(weight.detach() * log_prob_taken).mean()

def sample_gumbel(shape, device):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U))

def gumbel_topk_sampling_over_candidates(candidate_logits, candidates, k):
    gumbel_noise = sample_gumbel(candidate_logits.shape, candidate_logits.device)
    noisy_logits = candidate_logits + gumbel_noise
    topk_pos = torch.topk(noisy_logits, k, dim=1).indices # positions w/in candidate set
    topk_indices = torch.gather(candidates, 1, topk_pos) # map to item indices
    return topk_indices

def ma_et_al_loss(model_stage1, model_stage2, x, a_taken, pi0_probs, r, candidates):
    # compute logits over all items
    context_repr = model_stage1.context_nn(x)
    item_embs = model_stage1.item_embeddings.weight
    logits = torch.matmul(context_repr, item_embs.T) 

    # restrict logits to candidate set
    candidate_logits = torch.gather(logits, 1, candidates) 

    # Gumbel sampling from candidate logits
    top_k = model_stage1.top_k
    topk_indices = gumbel_topk_sampling_over_candidates(candidate_logits, candidates, top_k) 

    # log probs for sampled top-k set
    log_probs = F.log_softmax(logits, dim=1)
    log_pi_theta1_Ak = log_probs.gather(1, topk_indices).sum(dim=1)

    # second-stage probs
    probs_topk = model_stage2(x, topk_indices)

    # mask for actions present in top-k
    match_mask = (topk_indices == a_taken.unsqueeze(1))
    action_in_topk = match_mask.any(dim=1)

    if action_in_topk.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=x.device)

    idx_valid = match_mask[action_in_topk].float().argmax(dim=1)
    probs_topk_valid = probs_topk[action_in_topk]
    pi_theta2_probs = probs_topk_valid[torch.arange(len(idx_valid)), idx_valid]

    log_pi_valid = log_pi_theta1_Ak[action_in_topk]
    pi0_valid = pi0_probs[action_in_topk]
    r_valid = r[action_in_topk]

    weight = pi_theta2_probs / pi0_valid
    loss = -weight.detach() * log_pi_valid * r_valid
    return loss.mean()

if __name__ == "__main__":
    # 
    # Experiment testing/results
    #

    # device = "cuda" if torch.cuda.is_available() else "cpu" # not needed since running on mac locally
    device = "cpu"

    # synthetic dataset
    policy = EpsilonGreedyPolicy(epsilon=0.1, random_state=42)
    dataset = CustomSyntheticBanditDataset(
        n_actions=500, 
        dim_context=10, 
        top_k=5, 
        action_policy=policy, 
        device=device
    )

    # generate bandit feedback 
    n_samples = 100
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples)

    # print bandit feedback (first two examples)
    # print("Bandit Feedback (first two examples):")
    # for key, value in bandit_feedback.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value[:2].cpu().numpy()}")
    #     else:
    #         print(f"{key}: {value}")

    dim_context = dataset.dim_context
    num_items = dataset.n_actions
    emb_dim = 32
    top_k = dataset.top_k

    # get data
    x = bandit_feedback["context"]
    a_taken = bandit_feedback["action"]
    r = bandit_feedback["reward"]
    pi0_probs = bandit_feedback["pscore"]
    candidates = bandit_feedback["candidates"]

    first_stage_model = TwoTowerFirstStagePolicy(dim_context, num_items, emb_dim, top_k).to(device)
    second_stage_model = SoftmaxSecondStagePolicy(dim_context, emb_dim, first_stage_model).to(device)
    optimizer = optim.Adam(first_stage_model.parameters())
    num_epochs = 100

    print("MSE Loss")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = mse_loss(first_stage_model, x, a_taken, r)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("OPL Loss")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = opl_loss(first_stage_model, x, a_taken, pi0_probs, r)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Ma Et Al. Loss")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = ma_et_al_loss(first_stage_model, second_stage_model, x, a_taken, pi0_probs, r, candidates)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # run loss tests -- used for initial debugging, now implemented full training loop 
    # print("Losses:")
    # mse = mse_loss(first_stage, x, a_taken, r)
    # print(f"MSE Loss: {mse.item():.4f}")

    # opl = opl_loss(first_stage, x, a_taken, pi0_probs, r)
    # print(f"OPL Loss: {opl.item():.4f}")

    # ma_loss = ma_et_al_loss(first_stage, second_stage, x, a_taken, pi0_probs, r, bandit_feedback["candidates"])
    # print(f"Ma et al. Loss: {ma_loss.item():.4f}")