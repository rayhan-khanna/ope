from abc import ABC, abstractmethod
from typing import Dict
import torch

class BaseOffPolicyEstimator(ABC):
    """Base Class for OP Estimators."""

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the expected reward of the target policy."""
        pass

    @abstractmethod
    def estimate_policy_value_tensor(self) -> torch.Tensor:
        """Estimate the expected reward of the target policy."""
        pass

    @abstractmethod
    def estimate_interval(self, n_bootstrap: int = 1000) -> Dict[str, float]:
        """Estimate the 95% confidence interval for the policy value."""
        pass
    
class ImportanceSamplingEstimator(BaseOffPolicyEstimator):
    def __init__(self, behavior_pscore: torch.Tensor, target_pscore: torch.Tensor, rewards: torch.Tensor):
        """
        behavior_pscore: prob under behavior (logging) policy
        target_pscore: prob under target policy
        rewards: observed rewards
        """
        self.behavior_pscore = behavior_pscore
        self.target_pscore = target_pscore
        self.rewards = rewards

    def _estimate_round_rewards(self) -> torch.Tensor:
        weights = self.target_pscore / self.behavior_pscore
        weighted_rewards = self.rewards * weights
        return weighted_rewards

    def estimate_policy_value(self) -> float:
        weighted_rewards = self._estimate_round_rewards()
        return weighted_rewards.mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        weighted_rewards = self._estimate_round_rewards()
        return weighted_rewards.mean()

    def estimate_interval(self, n_bootstrap: int = 1000) -> Dict[str, float]:
        n = len(self.rewards)
        bootstrap_estimates = torch.zeros(n_bootstrap, device=self.rewards.device)

        for i in range(n_bootstrap):
            indices = torch.randint(0, n, (n,), device=self.rewards.device)
            sampled_rewards = self.rewards[indices]
            sampled_behavior = self.behavior_pscore[indices]
            sampled_target = self.target_pscore[indices]

            weights = sampled_target / sampled_behavior
            bootstrap_estimates[i] = (sampled_rewards * weights).mean()

        lower_bound = torch.quantile(bootstrap_estimates, 0.025).item()
        upper_bound = torch.quantile(bootstrap_estimates, 0.975).item()

        return {"lower_bound": lower_bound, "upper_bound": upper_bound}

class DirectMethodEstimator(BaseOffPolicyEstimator):
    def __init__(self, reward_model, target_policy, context):
        """
        reward_model: model that predicts expected reward given x, a 
        target_policy: model that samples actions from target policy
        context: contexts (x_i's)
        """
        self.reward_model = reward_model
        self.target_policy = target_policy
        self.context = context

    def _estimate_pred_rewards(self) -> torch.Tensor:
        pred_rewards = []
        for x_i in self.context:
            a_i = self.target_policy.sample_action(x_i)
            r_pred = self.reward_model.predict(x_i, a_i)
            pred_rewards.append(r_pred)
        return torch.stack(pred_rewards)

    def estimate_policy_value(self) -> float:
        predicted_rewards = self._estimate_pred_rewards()
        return predicted_rewards.mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        predicted_rewards = self._estimate_pred_rewards()
        return predicted_rewards.mean()

    def estimate_interval(self, n_bootstrap: int = 1000) -> Dict[str, float]:
        predicted_rewards = self._estimate_pred_rewards()
        n = len(predicted_rewards)
        bootstrap_estimates = torch.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            indices = torch.randint(0, n, (n,))
            sampled_preds = predicted_rewards[indices]
            bootstrap_estimates[i] = sampled_preds.mean()

        lower_bound = torch.quantile(bootstrap_estimates, 0.025).item()
        upper_bound = torch.quantile(bootstrap_estimates, 0.975).item()

        return {"lower_bound": lower_bound, "upper_bound": upper_bound}