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

### Two-stage estimators ###
    
class TwoStageISEstimator(BaseOffPolicyEstimator):
    def __init__(self, context, actions, rewards, behavior_pscore,
                 first_stage_policy, second_stage_policy, candidates):
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.behavior_pscore = behavior_pscore
        self.first_stage = first_stage_policy
        self.second_stage = second_stage_policy
        self.candidates = candidates

    def _estimate_round_rewards(self):
        values = []
        for i in range(len(self.context)):
            x_i = self.context[i].unsqueeze(0)
            a_i = self.actions[i]
            r_i = self.rewards[i]
            pi0_i = self.behavior_pscore[i]

            topk = self.candidates[i]
            probs = self.second_stage.calc_prob_given_output(x_i, topk.unsqueeze(0))[0]

            match_idx = (topk == a_i).nonzero(as_tuple=True)[0]

            if len(match_idx) == 0:
                continue  # skip if a_i not in A_k

            pi2 = probs[match_idx.item()]

            values.append((pi2 / pi0_i) * r_i)

        return torch.stack(values)

    def estimate_policy_value(self) -> float:
        return self._estimate_round_rewards().mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        return self._estimate_round_rewards().mean()
    
class KernelISEstimator(BaseOffPolicyEstimator):
    def __init__(self, context, actions, rewards, first_stage, second_stage,
                 logging_policy, kernel, tau, marginal_density_model, action_context, 
                 candidates, num_epochs=10):
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.first_stage = first_stage 
        self.second_stage = second_stage 
        self.logging_policy = logging_policy
        self.kernel = kernel
        self.tau = tau
        self.marginal_density_model = marginal_density_model
        self.action_context = action_context
        self.num_epochs = num_epochs
        self.candidates = candidates
        self.train_density_model = any(p.requires_grad for p in marginal_density_model.parameters())

    def estimateLMD(self):
        if not self.train_density_model:
            return
        optimizer = torch.optim.Adam(self.marginal_density_model.parameters(), lr=1e-3)
        for _ in range(self.num_epochs):
            for x_i, y_i in zip(self.context, self.actions):
                optimizer.zero_grad()
                sampled_action = self.logging_policy.sample_action(x_i, self.action_context)
                k_val = self.kernel(y_i, sampled_action, x_i, self.tau)
                predicted_density = self.marginal_density_model.predict(x_i, y_i, self.action_context)
                loss = (predicted_density - k_val) ** 2
                loss.backward()
                optimizer.step()
    
    def kernel_is_value_estimate(self):
        x = self.context
        y_log = self.actions
        r = self.rewards
        B = x.size(0)

        topk = self.candidates

        y_prime = torch.tensor([
            self.second_stage.sample_output(x[i].unsqueeze(0), topk[i].unsqueeze(0)).item()
            for i in range(B)
        ], device=x.device)

        values = []
        for i in range(B):
            k_val = self.kernel(y_prime[i], y_log[i], x[i], self.tau)
            density = self.marginal_density_model.predict(x[i], y_log[i], self.action_context)
            is_weight = k_val / density
            values.append(is_weight * r[i])

        return torch.stack(values).mean()

    def estimate_policy_value(self) -> float:
        return self.estimate_policy_value_tensor().item()
    
    def estimate_policy_value_tensor(self):
        self.estimateLMD()
        return self.kernel_is_value_estimate()