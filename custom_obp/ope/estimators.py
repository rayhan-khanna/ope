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
                 first_stage_policy, second_stage_policy, candidates, action_context,
                 logging_policy=None, ranking=False):
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.behavior_pscore = behavior_pscore
        self.first_stage = first_stage_policy
        self.second_stage = second_stage_policy
        self.logging_policy = logging_policy
        self.candidates = candidates
        self.action_context = action_context
        self.ranking = ranking

    def _estimate_round_rewards(self):
        if self.ranking:
            x = self.context
            ranked_actions = self.actions
            ranked_rewards = self.rewards
            r_agg = ranked_rewards.sum(dim=1)

            # joint log probs for IS ratio
            log_pi2 = self.second_stage.log_prob_ranking(x, ranked_actions, self.candidates)
            log_pi0 = self.logging_policy.log_prob_ranking(x, ranked_actions, self.candidates, self.action_context)

            weight = torch.exp(log_pi2 - log_pi0)
            log_pi1 = self.first_stage.log_prob_topk_set(x, self.candidates)

            return weight.detach() * r_agg * log_pi1

        else:
            probs = self.second_stage.calc_prob_given_output(self.context, self.candidates)
            match = (self.candidates == self.actions.unsqueeze(1))
            valid = match.any(dim=1)

            pi2 = probs[match]
            r_valid = self.rewards[valid]
            pi0_valid = self.behavior_pscore[valid]

            return ((pi2 / pi0_valid) * r_valid)

    def estimate_policy_value(self) -> float:
        return self._estimate_round_rewards().mean().item()

    def estimate_policy_value_tensor(self) -> torch.Tensor:
        return self._estimate_round_rewards().mean()
       
class KernelISEstimator(BaseOffPolicyEstimator):
    def __init__(self, context, actions, rewards, first_stage, second_stage,
                 logging_policy, kernel, tau, marginal_density_model, action_context, 
                 candidates, ranking=False, num_epochs=10):
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
        self.ranking = ranking
        self.train_density_model = any(p.requires_grad for p in marginal_density_model.parameters())

    def estimateLMD(self):
        if not self.train_density_model:
            return
        optimizer = torch.optim.Adam(self.marginal_density_model.parameters())

        for _ in range(self.num_epochs):
            x = self.context
            y_logged = self.actions             
            if self.ranking:
                y_logged = self.actions if self.actions.dim() == 2 else self.actions.unsqueeze(1)
                ranked_sampled, _ = self.second_stage.rank_outputs(self.context, self.candidates)
                y_sampled = ranked_sampled[:, :5] 
            else:
                y_logged = self.actions
                y_sampled = self.logging_policy.sample_action(self.context, self.action_context)
            k_val = self.kernel(y_logged, y_sampled, x, self.tau)
            pred_density = self.marginal_density_model.predict(x, y_logged, self.action_context)
            loss = ((pred_density - k_val) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def kernel_is_value_estimate(self):
        x = self.context
        log_pi1 = self.first_stage.log_prob_topk_set(x, self.candidates)

        if self.ranking:
            y_logged = self.actions
            r_agg = self.rewards.sum(dim=1)

            y_sampled = self.logging_policy.sample_ranking(x, self.candidates, self.action_context)
            k_val = self.kernel(y_sampled, y_logged, x, self.tau)

            density = self.marginal_density_model.predict(x, y_logged, self.action_context) 
            is_weight = (k_val / density)
            return (is_weight.detach() * r_agg * log_pi1).mean()

        else:
            y_logged = self.actions  
            r = self.rewards
            topk = self.candidates

            sampled_idx = self.second_stage.sample_output(x, topk)
            y_sampled = topk[torch.arange(x.size(0), device=x.device), sampled_idx]

            k_val = self.kernel(y_sampled, y_logged, x, self.tau)
            density = self.marginal_density_model.predict(x, y_logged, self.action_context)
            is_weight = k_val / density
            return (is_weight.detach() * r * log_pi1).mean()

    def estimate_policy_value_tensor(self):
        self.estimateLMD()
        return self.kernel_is_value_estimate()

    def estimate_policy_value(self) -> float:
        return self.estimate_policy_value_tensor().item()