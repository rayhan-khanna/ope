from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseActionPolicy(ABC):
    """Abstract base class for action policies."""
    
    @abstractmethod
    def select_action(self, candidates, context, action_context):
        """
        Given top-K candidate actions and the context, select an action.

        Args:
            candidates (np.ndarray): Array of shape (top_k,) containing action indices.
            context (np.ndarray): Array of shape (dim_context,) representing user context.
            action_context (np.ndarray): Array of shape (n_actions, dim_context) for action embeddings.

        Returns:
            int: The chosen action.
        """
        pass

    @abstractmethod
    def sample_action(self, context, action_context):
        """
        Sample a single action from the full action space.
        """
        pass

class UniformRandomPolicy(BaseActionPolicy):
    """Uniform Random action selection policy."""

    def __init__(self, random_state=42):
        self.random_ = np.random.RandomState(random_state)

    def select_action(self, candidates, context, action_context):
        return self.random_.choice(candidates)

    def sample_action(self, context, action_context):
        return self.random_.choice(len(action_context))

class EpsilonGreedyPolicy(BaseActionPolicy):
    """Epsilon-Greedy action selection policy."""
    
    def __init__(self, epsilon=0.1, random_state=42):
        self.epsilon = epsilon
        self.random_ = np.random.RandomState(random_state)

    def select_action(self, candidates, context, action_context):
        # randomly pick 10 items from top-k
        sampled_candidates = self.random_.choice(candidates, size=min(10, len(candidates)), replace=False)

        if self.random_.rand() < self.epsilon:
            # exploration
            return self.random_.choice(sampled_candidates)
        else:
            # exploitation: action with the highest score (dot product)
            scores = np.dot(context, action_context[sampled_candidates].T)  
            return sampled_candidates[np.argmax(scores)]

    def sample_action(self, context, action_context):
        if self.random_.rand() < self.epsilon:
            return self.random_.choice(len(action_context))
        else:
            scores = np.dot(context, action_context.T)
            return np.argmax(scores)
        
class SoftmaxPolicy(BaseActionPolicy):
    """Softmax action selection policy."""
    
    def __init__(self, temperature=1.0, random_state=42):
        self.temperature = temperature
        self.random_ = np.random.RandomState(random_state)

    def select_action(self, candidates, context, action_context):
        sampled_candidates = self.random_.choice(candidates, size=min(10, len(candidates)), replace=False)
        scores = np.dot(context, action_context[sampled_candidates].T)

        exp_scores = np.exp(scores / self.temperature)  
        probabilities = exp_scores / np.sum(exp_scores) 

        return self.random_.choice(sampled_candidates, p=probabilities)
    
    def sample_action(self, context, action_context):
        scores = np.dot(context, action_context.T)
        exp_scores = np.exp(scores / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        return self.random_.choice(len(action_context), p=probabilities)
    
class TwoStageRankingPolicy(BaseActionPolicy):
    """
    Wraps a two-tower first-stage model and a second-stage softmax model
    to provide Plackett-Luce ranking over top-k candidates.
    """
    def __init__(self, first_stage_model, second_stage_model, top_k, device="cpu"):
        self.first_stage = first_stage_model
        self.second_stage = second_stage_model
        self.top_k = top_k
        self.device = device

    def select_action(self, candidates, context_np, action_context_np):
        context = torch.tensor(context_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        candidates = torch.tensor(candidates, dtype=torch.long).unsqueeze(0).to(self.device)
        probs = self.second_stage(context, candidates)
        action_idx = torch.multinomial(probs[0], 1).item()
        return candidates[0, action_idx].item()

    def sample_action(self, context_np, action_context_np):
        context = context_np.clone().detach().unsqueeze(0).to(self.device)
        _, topk = self.first_stage(context)
        probs = self.second_stage(context, topk)
        action_idx = torch.multinomial(probs[0], 1).item()
        return topk[0, action_idx].item()

    def rank_topk(self, context_tensor):
        """
        Plackett-Luce-style ranking using Gumbel top-k sampling.
        Returns a full ranking of top-k items for each context.
        """
        with torch.no_grad():
            _, topk = self.first_stage(context_tensor)  # [B, k]
            scores = self.second_stage(context_tensor, topk)  # [B, k]
            gumbel = self.sample_gumbel(scores.shape, device=scores.device)
            noisy_scores = scores + gumbel
            ranked = topk[torch.arange(len(topk)).unsqueeze(1), torch.argsort(noisy_scores, dim=1, descending=True)]
        return ranked

    @staticmethod
    def sample_gumbel(shape, device):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U))