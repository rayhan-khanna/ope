from abc import ABC, abstractmethod
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
    def __init__(self, reward_model, target_policy, context, action_context):
        """
        reward_model: model that predicts expected reward given x, a 
        target_policy: model that samples actions from target policy
        context: contexts (x_i's)
        """
        self.reward_model = reward_model
        self.target_policy = target_policy
        self.context = context
        self.action_context = action_context

    def _estimate_pred_rewards(self) -> torch.Tensor:
        actions = [self.target_policy.sample_action(x_i, self.action_context) for x_i in self.context]
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
                 behavior_pscore, target_pscore, rewards, target_policy, action_context):
        """
        reward_model: model that predicts expected reward given x, a 
        context: contexts (x_i's)
        actions: logged actions
        behavior_pscore: prob under behavior (logging) policy
        target_pscore: prob under target policy
        rewards: observed rewards
        target_policy: model that samples actions from target policy
        action_context: action contexts
        """
        self.reward_model = reward_model
        self.context = context
        self.behavior_pscore = behavior_pscore
        self.target_pscore = target_pscore
        self.rewards = rewards
        self.target_policy = target_policy
        self.actions = actions
        self.action_context = action_context

    def _estimate_dr_rewards(self) -> torch.Tensor:
        target_actions = [self.target_policy.sample_action(x_i, self.action_context) for x_i in self.context]
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
    def __init__(self, data, target_policy, logging_policy, kernel, tau, marginal_density_model, action_context, num_epochs=10):
        """
        data: list of tuples (x_i, y_i, r_i)
        target_policy: evaluation policy π (supports sample_latent, sample_action, log_grad)
        logging_policy: logging policy π₀ (supports sample_action(x))
        kernel: function K(y, y', x, tau)
        tau: temperature for kernel
        marginal_density_model: model that approximates logging marginal density π₀(y | x)
        action_context: action contexts
        num_epochs: training iterations for h_model 
        """
        self.data = data
        self.target_policy = target_policy
        self.logging_policy = logging_policy
        self.kernel = kernel
        self.tau = tau
        self.marginal_density_model = marginal_density_model
        self.action_context = action_context
        self.num_epochs = num_epochs
        # to know if we actually need to train the density model (or if it's just constant)
        self.train_density_model = any(p.requires_grad for p in marginal_density_model.parameters()) 


    def estimateLMD(self):
        """Train h_model to estimate logging marginal density."""
        if not self.train_density_model:
            return
        optimizer = torch.optim.Adam(self.marginal_density_model.parameters(), lr=1e-3)
        for _ in range(self.num_epochs):
            for x_i, y_i, _ in self.data:
                optimizer.zero_grad()
                sampled_action = self.logging_policy.sample_action(x_i, self.action_context)
                k_val = self.kernel(y_i, sampled_action, x_i, self.tau)
                predicted_density = self.marginal_density_model.predict(x_i, y_i, self.action_context)
                loss = (predicted_density - k_val) ** 2
                loss.backward()
                optimizer.step()

    def kernel_is_value_estimate(self) -> torch.Tensor:
        values = []
        for x_i, y_i, r_i in self.data:
            k_val = self.kernel(y_i, y_i, x_i, self.tau)
            density_estimate = self.marginal_density_model.predict(x_i, y_i, self.action_context)
            is_weight = k_val / density_estimate
            values.append(is_weight * r_i)
        return torch.stack(values).mean()

    def estimate_policy_value(self) -> float:
        return self.estimate_policy_value_tensor().item()
    
    def estimate_policy_value_tensor(self) -> torch.Tensor:
        self.estimateLMD()
        return self.kernel_is_value_estimate()