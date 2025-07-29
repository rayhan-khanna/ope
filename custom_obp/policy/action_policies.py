import torch
import torch.nn.functional as F
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseActionPolicy(ABC):
    """Abstract base class for Torch-native action policies."""

    @abstractmethod
    def select_action(self, candidates: torch.Tensor, context: torch.Tensor, action_context: torch.Tensor) -> int:
        pass

    @abstractmethod
    def sample_action(self, context: torch.Tensor, action_context: torch.Tensor) -> int:
        pass

    @abstractmethod
    def log_prob(self, context: torch.Tensor, action: int, action_context: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def probs(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        pass

    def sample_latent(self, context: torch.Tensor):
        raise NotImplementedError

class UniformRandomPolicy(BaseActionPolicy):
    def select_action(self, candidates, context, action_context):
        idx = torch.randint(0, len(candidates), (1,), device=action_context.device).item()
        return candidates[idx].item()

    def sample_action(self, context, action_context):
        return torch.randint(0, len(action_context), (1,), device=action_context.device).item()

    def log_prob(self, context, action, action_context):
        prob = 1.0 / action_context.size(0)
        return torch.log(torch.tensor(prob, dtype=torch.float32, device=action_context.device))

    def probs(self, context, action_context):
        n = action_context.size(0)
        return torch.ones(n, dtype=torch.float32, device=action_context.device) / n
    
    def log_prob_ranking(self, context, ranked_actions, candidates):
        B, K = ranked_actions.shape
        N = candidates.size(1)  # number of candidates
        device = candidates.device

        # log(1/N) + log(1/(N-1)) + ... + log(1/(N-K+1))
        log_prob_single = torch.sum(torch.log(1.0 / torch.arange(N, N-K, -1, device=device, dtype=torch.float32)))
        return log_prob_single.expand(B)  # same for each batch
    
    def sample_ranking(self, context, candidates, top_k=5):
        B, N = candidates.shape
        device = candidates.device
        noise = torch.rand(B, N, device=device) 

        # argsort noise to get random permutation indices
        perm_indices = torch.argsort(noise, dim=1)

        # reorder candidates by these permutations
        shuffled = candidates.gather(1, perm_indices) 

        # take top_k as ranking
        return shuffled[:, :top_k]
    
class SoftmaxPolicy(BaseActionPolicy, nn.Module):
    def __init__(self, dim_context, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.scorer = nn.Linear(dim_context, dim_context, bias=False)  

    def probs(self, context, action_context):
        logits = self.scorer(context) @ action_context.T
        return F.softmax(logits / self.temperature, dim=-1)
    
    def probs(self, context, action_context):
        # if action_context is [B, K, d], handle it per batch
        if action_context.dim() == 3:
            context_proj = self.scorer(context).unsqueeze(1)
            logits = torch.sum(context_proj * action_context, dim=-1)
        else:
            # standard case: action_context is [N_actions, d]
            logits = self.scorer(context) @ action_context.T
        return torch.softmax(logits / self.temperature, dim=-1)

    def sample_action(self, context, action_context):
        probs = self.probs(context.unsqueeze(0), action_context)[0]
        return torch.multinomial(probs, 1).item()

    def select_action(self, candidates, context, action_context):
        subset_context = action_context[candidates]
        scores = self.scorer(context.unsqueeze(0)) @ subset_context.T 
        probs = F.softmax(scores[0] / self.temperature, dim=0)
        return candidates[torch.multinomial(probs, 1)].item()

    def log_prob(self, context, actions, action_context):
        probs = self.probs(context, action_context)
        return torch.log(probs[torch.arange(len(context)), actions])
    
    def log_prob_ranking(self, context, ranked_actions, candidates, action_context):
        B, K = ranked_actions.shape
        mask = torch.ones_like(candidates, dtype=torch.bool)
        log_probs = torch.zeros(B, device=context.device)

        for pos in range(K):
            cand_embs = action_context[candidates]
            probs = self.probs(context, cand_embs)
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=1, keepdim=True)

            # gather log-prob of the selected action
            idx = ranked_actions[:, pos]
            candidate_indices = (candidates == idx.unsqueeze(1)).float().argmax(dim=1)
            log_probs += torch.log(probs[torch.arange(B), candidate_indices])

            # mask out the chosen item
            mask[torch.arange(B), candidate_indices] = False

        return log_probs
    
    def sample_ranking(self, context, candidates, action_context, top_k=5):
        B = context.size(0)
        mask = torch.ones_like(candidates, dtype=torch.bool)
        ranking = torch.zeros(B, top_k, dtype=torch.long, device=context.device)

        for pos in range(top_k):
            cand_embs = action_context[candidates]
            probs = self.probs(context, cand_embs)
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=1, keepdim=True)

            # sample actions from masked probs
            sampled_idx = torch.multinomial(probs, 1).squeeze(1)
            chosen_actions = candidates[torch.arange(B), sampled_idx]
            ranking[:, pos] = chosen_actions

            # mask out chosen action
            mask[torch.arange(B), sampled_idx] = False

        return ranking
    
# E-Greedy is outdated/unused at this point.
# class EpsilonGreedyPolicy(BaseActionPolicy):
#     def __init__(self, epsilon=0.1):
#         self.epsilon = epsilon

#     def select_action(self, candidates, context, action_context):
#         sampled = candidates[torch.randperm(len(candidates))[:min(10, len(candidates))]]
#         if torch.rand(1).item() < self.epsilon:
#             return sampled[torch.randint(0, len(sampled), (1,))].item()
#         scores = torch.matmul(action_context[sampled], context)
#         return sampled[torch.argmax(scores)].item()

#     def sample_action(self, context, action_context):
#         if torch.rand(1).item() < self.epsilon:
#             return torch.randint(0, len(action_context), (1,)).item()
#         scores = torch.matmul(action_context, context)
#         return torch.argmax(scores).item()

#     def log_prob(self, context, action, action_context):
#         scores = torch.matmul(action_context, context)
#         greedy_action = torch.argmax(scores).item()
#         n = action_context.size(0)
#         prob = (1 - self.epsilon) + self.epsilon / n if action == greedy_action else self.epsilon / n
#         return torch.log(torch.tensor(prob, dtype=torch.float32, device=action_context.device))

#     def probs(self, context, action_context):
#         scores = torch.matmul(action_context, context)
#         greedy_action = torch.argmax(scores).item()
#         n = action_context.size(0)
#         probs = torch.full((n,), self.epsilon / n, dtype=torch.float32)
#         probs[greedy_action] += (1.0 - self.epsilon)
#         return probs