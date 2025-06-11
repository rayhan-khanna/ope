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
        idx = torch.randint(0, len(candidates), (1,)).item()
        return candidates[idx].item()

    def sample_action(self, context, action_context):
        return torch.randint(0, len(action_context), (1,)).item()

    def log_prob(self, context, action, action_context):
        prob = 1.0 / action_context.size(0)
        return torch.log(torch.tensor(prob, dtype=torch.float32))

    def probs(self, context, action_context):
        n = action_context.size(0)
        return torch.ones(n, dtype=torch.float32) / n


class EpsilonGreedyPolicy(BaseActionPolicy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, candidates, context, action_context):
        sampled = candidates[torch.randperm(len(candidates))[:min(10, len(candidates))]]
        if torch.rand(1).item() < self.epsilon:
            return sampled[torch.randint(0, len(sampled), (1,))].item()
        scores = torch.matmul(action_context[sampled], context)
        return sampled[torch.argmax(scores)].item()

    def sample_action(self, context, action_context):
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, len(action_context), (1,)).item()
        scores = torch.matmul(action_context, context)
        return torch.argmax(scores).item()

    def log_prob(self, context, action, action_context):
        scores = torch.matmul(action_context, context)
        greedy_action = torch.argmax(scores).item()
        n = action_context.size(0)
        prob = (1 - self.epsilon) + self.epsilon / n if action == greedy_action else self.epsilon / n
        return torch.log(torch.tensor(prob, dtype=torch.float32))

    def probs(self, context, action_context):
        scores = torch.matmul(action_context, context)
        greedy_action = torch.argmax(scores).item()
        n = action_context.size(0)
        probs = torch.full((n,), self.epsilon / n, dtype=torch.float32)
        probs[greedy_action] += (1.0 - self.epsilon)
        return probs

class SoftmaxPolicy(BaseActionPolicy, nn.Module):
    def __init__(self, dim_context, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.scorer = nn.Linear(dim_context, dim_context, bias=False)  

    def probs(self, context, action_context):
        logits = self.scorer(context) @ action_context.T
        return F.softmax(logits / self.temperature, dim=-1)

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

class TwoStageRankingPolicy(BaseActionPolicy, nn.Module):
    def __init__(self, first_stage_model, second_stage_model, top_k, action_context, device="cpu"):
        super().__init__()
        self.first_stage = first_stage_model
        self.second_stage = second_stage_model
        self.top_k = top_k
        self.device = device
        self.action_context = action_context 

    def select_action(self, candidates, context, action_context):
        context = context.unsqueeze(0)
        _, topk = self.first_stage(context)  
        topk = topk[0]
        probs = self.second_stage(context, topk.unsqueeze(0))[0]
        return topk[torch.argmax(probs)].item()

    def sample_action(self, context: torch.Tensor, topk: torch.Tensor):
        probs = self.second_stage(context.unsqueeze(0), topk.unsqueeze(0))[0]
        action_idx = torch.multinomial(probs, 1).item()
        return topk[action_idx].item()
    
    def log_prob(self, context, actions, action_context):
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        batch_size = context.shape[0]
        candidates = torch.arange(action_context.shape[0], device=context.device).repeat(batch_size, 1)
        topk = self.sample_topk(context, candidates)
        probs = self.second_stage(context, topk)
        mask = (topk == actions.unsqueeze(1))
        selected_probs = probs[mask]
        return torch.log(selected_probs)

    def probs(self, context, action_context):
        logits = self.first_stage(context)[0]
        return torch.softmax(logits, dim=1)

    def sample_latent(self, context: torch.Tensor):
        topk = self.sample_topk(
            context,
            torch.arange(self.action_context.shape[0], device=context.device).repeat(context.size(0), 1)
        )
        return topk

    def sample_topk(self, context: torch.Tensor, candidates: torch.Tensor):
        logits = self._full_logits(context)
        candidate_logits = torch.gather(logits, 1, candidates)
        gumbel_noise = self.sample_gumbel(candidate_logits.shape, device=logits.device)
        noisy_logits = candidate_logits + gumbel_noise
        topk_pos = torch.topk(noisy_logits, self.top_k, dim=1).indices
        topk_indices = torch.gather(candidates, 1, topk_pos)
        return topk_indices

    def log_prob_topk_set(self, context: torch.Tensor, topk_indices: torch.Tensor):
        logits = self._full_logits(context)
        log_probs = F.log_softmax(logits, dim=1)
        log_pi_theta1_Ak = log_probs.gather(1, topk_indices).sum(dim=1)
        return log_pi_theta1_Ak

    def probs_given_topk(self, context_tensor: torch.Tensor, topk_indices: torch.Tensor):
        return self.second_stage(context_tensor, topk_indices)

    def rank_topk(self, context_tensor: torch.Tensor):
        with torch.no_grad():
            _, topk = self.first_stage(context_tensor)
            scores = self.second_stage(context_tensor, topk)
            gumbel = self.sample_gumbel(scores.shape, device=scores.device)
            noisy_scores = scores + gumbel
            ranked = topk[torch.arange(len(topk)).unsqueeze(1), torch.argsort(noisy_scores, dim=1, descending=True)]
        return ranked

    def _full_logits(self, x) -> torch.Tensor:
        return self.first_stage.full_logits(x)
    
    @staticmethod
    def sample_gumbel(shape, device):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U))