from typing import Dict
import numpy as np
from obp.ope import BaseOffPolicyEstimator
import torch

class ImportanceSamplingEstimator(BaseOffPolicyEstimator):
    def __init__(self, behavior_policy_probs: np.ndarray, target_policy_probs: np.ndarray, rewards: np.ndarray):
        self.behavior_policy_probs = behavior_policy_probs
        self.target_policy_probs = target_policy_probs.squeeze()
        self.rewards = rewards

    def _estimate_round_rewards(self) -> np.ndarray:
        # IS weights (ratio of target policy prob to behavior policy prob)
        weights = self.target_policy_probs / self.behavior_policy_probs

        # Estimate round/weighted rewards
        weighted_rewards = self.rewards * weights
        return weighted_rewards

    def estimate_policy_value(self) -> float:
        weighted_rewards = self._estimate_round_rewards()
        policy_value = np.mean(weighted_rewards) # average of weighted rewards
        return policy_value

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        behavior_policy_probs = torch.tensor(self.behavior_policy_probs, dtype=torch.float32)
        target_policy_probs = torch.tensor(self.target_policy_probs, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        weights = target_policy_probs / behavior_policy_probs
        weighted_rewards = rewards * weights

        policy_value = torch.mean(weighted_rewards)

        return policy_value
    
    def estimate_interval(self) -> Dict[str, float]:
        bootstrap_samples = 1000
        n = len(self.rewards)
        bootstrap_estimates = np.zeros(bootstrap_samples)

        for i in range(bootstrap_samples):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_estimates[i] = np.mean(self.rewards[indices] * (self.target_policy_probs[indices] / self.behavior_policy_probs[indices]))

        lower_bound = np.percentile(bootstrap_estimates, 2.5)
        upper_bound = np.percentile(bootstrap_estimates, 97.5)
        return {"lower_bound": lower_bound, "upper_bound": upper_bound}