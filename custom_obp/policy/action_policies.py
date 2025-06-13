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