import torch
import torch.nn as nn

# candidate selection
class TwoTowerFirstStagePolicy(nn.Module):
    def __init__(self, dim_context, num_items, emb_dim, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_items = num_items

        self.context_nn = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.item_embeddings = nn.Embedding(num_items, emb_dim)

    def forward(self, x):
        return self.sample_topk(x)

    def full_logits(self, x: torch.Tensor) -> torch.Tensor:
        context = self.context_nn(x)
        item_embs = self.item_embeddings.weight
        return torch.matmul(context, item_embs.T)
    
    def sample_topk_gumbel(self, x, return_prob=False):
        logits = self.full_logits(x)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        noisy_logits = logits + gumbel_noise
        top_k_probs, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=1)
        return (top_k_indices, top_k_probs) if return_prob else top_k_indices

    def calc_prob_given_topk(self, x, topk_indices):
        logits = self.full_logits(x)
        probs = torch.softmax(logits, dim=1)
        probs_topk = torch.gather(probs, 1, topk_indices)
        return probs_topk

    def log_prob_topk_set(self, x, topk_indices):
        B, k = topk_indices.shape
        logits = self.full_logits(x) 
        log_probs = torch.zeros(B, device=x.device)

        used_mask = torch.zeros_like(logits, dtype=torch.bool)

        for i in range(k):
            idx = topk_indices[:, i].unsqueeze(1)

            # mask prev selected items
            valid_logits = logits.masked_fill(used_mask, float('-inf'))
            denom = torch.logsumexp(valid_logits, dim=1)

            logit_i = torch.gather(logits, 1, idx).squeeze(1)
            log_probs += logit_i - denom

            used_mask = used_mask.scatter(1, idx, True)

        return log_probs

# action selection
class SoftmaxSecondStagePolicy(nn.Module):
    def __init__(self, dim_context, emb_dim, first_stage_policy):
        super().__init__()
        self.first_stage_policy = first_stage_policy
        self.fc = nn.Sequential(
            nn.Linear(dim_context, emb_dim),
            nn.ReLU()
        )

    def forward(self, x, A_k, mask=None):
        return self.calc_prob_given_output(x, A_k, mask)

    def sample_output(self, x, A_k, return_prob=False):
        probs = self.calc_prob_given_output(x, A_k)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        if return_prob:
            sampled_prob = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            return sampled, sampled_prob
        return sampled
    
    def calc_prob_given_output(self, x, A_k, mask=None):
        context_repr = self.fc(x)
        item_embs = self.first_stage_policy.item_embeddings(A_k)
        scores = torch.matmul(item_embs, context_repr.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        probs = torch.softmax(scores, dim=1)
        return probs