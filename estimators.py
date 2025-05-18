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
        actions = self.target_policy.sample_action(self.context) 
        pred_rewards = self.reward_model.predict(self.context, actions) 
        return pred_rewards

    def estimate_policy_value(self) -> float:
        predicted_rewards = self._estimate_pred_rewards()
        return predicted_rewards.mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        predicted_rewards = self._estimate_pred_rewards()
        return predicted_rewards.mean()

class DoublyRobustEstimator(BaseOffPolicyEstimator):
    def __init__(self, reward_model, context, actions, 
                 behavior_pscore, target_pscore, rewards):
        """
        reward_model: model that predicts expected reward given x, a 
        context: contexts (x_i's)
        actions: logged actions
        behavior_pscore: prob under behavior (logging) policy
        target_pscore: prob under target policy
        rewards: observed rewards
        """
        self.reward_model = reward_model
        self.context = context
        self.behavior_pscore = behavior_pscore
        self.target_pscore = target_pscore
        self.rewards = rewards
        self.actions = actions

    def _estimate_dr_rewards(self) -> torch.Tensor:
        q_r_pred = self.reward_model.predict(self.context, self.actions)
        weights = self.target_pscore / self.behavior_pscore
        dr_reward = q_r_pred + weights * (self.rewards - q_r_pred)
        return dr_reward

    def estimate_policy_value(self) -> float:
        dr_reward = self._estimate_dr_rewards()
        return dr_reward.mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        dr_reward = self._estimate_dr_rewards()
        return dr_reward.mean()