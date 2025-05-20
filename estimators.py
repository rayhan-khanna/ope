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
        target_actions = self.target_policy.sample_action(self.context)
        q_target = self.reward_model.predict(self.context, target_actions)
        q_logged = self.reward_model.predict(self.context, self.actions)
        weights = self.target_pscore / self.behavior_pscore
        dr_reward = q_target + weights * (self.rewards - q_logged)
        return dr_reward

    def estimate_policy_value(self) -> float:
        dr_reward = self._estimate_dr_rewards()
        return dr_reward.mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        dr_reward = self._estimate_dr_rewards()
        return dr_reward.mean()

class KernelISEstimator(BaseOffPolicyEstimator):
    def __init__(self, data, target_policy, logging_policy, kernel, tau, marginal_density_model, num_epochs=10):
        """
        data: list of tuples (x_i, y_i, r_i)
        target_policy: evaluation policy π (supports sample_latent, sample_action, log_grad)
        logging_policy: logging policy π₀ (supports sample_action(x))
        kernel: function K(y, y', x, tau)
        tau: temperature for kernel
        marginal_density_model: model that approximates logging marginal density π₀(y | x)
        num_epochs: training iterations for h_model
        """
        self.data = data
        self.target_policy = target_policy
        self.logging_policy = logging_policy
        self.kernel = kernel
        self.tau = tau
        self.marginal_density_model = marginal_density_model
        self.num_epochs = num_epochs

    def estimateLMD(self):
        """Train h_model to estimate logging marginal density."""
        for _ in range(self.num_epochs):
            for x_i, y_i, _ in self.data:
                sampled_action = self.logging_policy.sample_action(x_i)
                k_val = self.kernel(y_i, sampled_action, x_i, self.tau)
                predicted_density = self.marginal_density_model.predict(x_i, y_i)
                loss = (predicted_density - k_val) ** 2
                self.marginal_density_model.update(x_i, y_i, loss)

    def _estimate_policy_gradient(self) -> torch.Tensor:
        grad = []
        for x_i, y_i, r_i in self.data:
            W_k, A_k = self.target_policy.sample_latent(x_i) 
            y = self.target_policy.sample_action(x_i, A_k, W_k)

            k_val = self.kernel(y, y_i, x_i, self.tau)
            density_estimate = self.marginal_density_model.predict(x_i, y_i)
            is_weight = k_val / density_estimate

            log_grad = self.target_policy.log_grad(W_k, A_k, x_i)
            grad.append(-is_weight.detach() * r_i.detach() * log_grad)

        return torch.stack(grad).mean()

    def estimate_policy_value(self) -> float:
        return self.estimate_policy_value_tensor().item()
    
    def estimate_policy_value_tensor(self) -> torch.Tensor:
        self.estimateLMD()
        return self._estimate_policy_gradient()