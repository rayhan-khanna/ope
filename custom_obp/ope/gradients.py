import torch

class ISGradient:
    def __init__(self, context, actions, rewards, behavior_pscore, target_policy, action_context):
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.behavior_pscore = behavior_pscore
        self.target_policy = target_policy
        self.action_context = action_context

    def estimate_policy_gradient(self):
        pi_theta = self.target_policy.probs(self.context, self.action_context)
        target_pscore = pi_theta[torch.arange(len(self.context)), self.actions]
        weight = torch.clamp(target_pscore / self.behavior_pscore, max=10).detach()
        weighted_rewards = self.rewards * weight
        log_pi_theta = self.target_policy.log_prob(self.context, self.actions, self.action_context)
        loss = -(weighted_rewards * log_pi_theta).mean()
        return loss
    
class DMGradient:
    def __init__(self, reward_model, target_policy, context, action_context):
        self.reward_model = reward_model
        self.target_policy = target_policy
        self.context = context
        self.action_context = action_context

    def estimate_policy_gradient(self):
        sampled_actions = torch.tensor(
            [self.target_policy.sample_action(x_i, self.action_context) for x_i in self.context],
            device=self.context.device
        )
        rewards = self.reward_model.predict(self.context, sampled_actions).detach()
        log_probs = self.target_policy.log_prob(self.context, sampled_actions, self.action_context)
        loss = -(log_probs * rewards).mean()
        return loss
    
class DRGradient:
    def __init__(self, reward_model, context, actions, behavior_pscore, 
                 target_policy, rewards, action_context):
        self.reward_model = reward_model
        self.context = context
        self.actions = actions
        self.behavior_pscore = behavior_pscore
        self.rewards = rewards
        self.target_policy = target_policy
        self.action_context = action_context

    def estimate_policy_gradient(self):
        pi_e = self.target_policy.probs(self.context, self.action_context)

        sampled_actions = torch.tensor(
            [self.target_policy.sample_action(x_i, self.action_context) for x_i in self.context],
            device=self.context.device
        ).detach()

        q_target = self.reward_model.predict(self.context, sampled_actions).detach()
        log_pi_target = self.target_policy.log_prob(self.context, sampled_actions, self.action_context)
        reinforce = (q_target * log_pi_target).mean()

        q_logged = self.reward_model.predict(self.context, self.actions)
        target_pscore = pi_e[torch.arange(len(self.context)), self.actions].detach()
        weights = target_pscore / self.behavior_pscore

        log_pi_e = self.target_policy.log_prob(self.context, self.actions, self.action_context)
        dr = (weights * (self.rewards - q_logged.detach()) * log_pi_e).mean()
        loss = -(reinforce + dr)  
        return loss

class TwoStageISGradient:
    def __init__(self, first_stage, second_stage, context, actions, rewards,
                 logging_policy, behavior_pscore, action_context, candidates, ranking=False, top_k=5):
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.logging_policy = logging_policy
        self.behavior_pscore = behavior_pscore
        self.action_context = action_context
        self.candidates = candidates
        self.ranking = ranking
        self.top_k = top_k

    def estimate_policy_gradient(self):
        x = self.context
        device = x.device
        log_pi1 = self.first_stage.log_prob_topk_set(x, self.candidates)

        if self.ranking:
            ranked_actions = self.actions
            ranked_rewards = self.rewards

            # joint log prob of ranking under target policy
            log_pi2 = self.second_stage.log_prob_ranking(
                x, ranked_actions, self.candidates
            )  # sum over positions

            # joint log prob of ranking under behavior policy (approximate as product if independent)
            log_pi0 = self.logging_policy.log_prob_ranking(
                x, ranked_actions, self.candidates, self.action_context
            )

            # aggregated reward: sum (will need to adjust for diversity)
            r_agg = ranked_rewards.sum(dim=1)

            weight = torch.exp(log_pi2 - log_pi0)
            return -(weight.detach() * r_agg * log_pi1).mean()

        else:
            a = self.actions
            r = self.rewards
            pi0 = self.behavior_pscore
            topk = self.candidates

            probs_topk = self.second_stage.calc_prob_given_output(x, topk)
            index_in_topk = (topk == a.unsqueeze(1)).nonzero(as_tuple=True)

            valid_mask = torch.zeros(len(a), dtype=torch.bool, device=device)
            valid_mask[index_in_topk[0]] = True

            pi_theta2 = probs_topk[index_in_topk]

            log_pi1 = log_pi1[valid_mask]
            pi0 = pi0[valid_mask]
            r = r[valid_mask]

            weight = pi_theta2 / pi0
            return -(weight.detach() * r * log_pi1).mean()

class KernelISGradient:
    def __init__(self, first_stage, second_stage, context, actions, rewards, logging_policy,
                 kernel, tau, marginal_density_model, action_context, candidates,
                 ranking=False, num_epochs=10):
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.context = context
        self.actions = actions
        self.rewards = rewards
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

    def estimate_policy_gradient(self) -> torch.Tensor:
        self.estimateLMD()
        x = self.context
        B = x.shape[0]
        log_pi1 = self.first_stage.log_prob_topk_set(x, self.candidates)

        if self.ranking:
            y_logged = self.actions
            ranked_rewards = self.rewards
            r_agg = ranked_rewards.sum(dim=1)

            # sampled ranking from target policy
            y_sampled = self.second_stage.sample_ranking(x, self.candidates)

            # kernel similarity between logged and sampled rankings
            k_val = self.kernel(y_sampled, y_logged, x, self.tau)
            density = self.marginal_density_model.predict(x, y_logged, self.action_context) 
            is_weight = k_val / density
            return -(is_weight.detach() * r_agg * log_pi1).mean()

        else:
            y_logged = self.actions  
            r = self.rewards
            topk = self.candidates

            sampled_idx = self.second_stage.sample_output(x, topk)
            y_sampled = topk[torch.arange(B, device=x.device), sampled_idx]

            k_val = self.kernel(y_sampled, y_logged, x, self.tau)
            density = self.marginal_density_model.predict(x, y_logged, self.action_context)
            is_weight = k_val / density

            return -(is_weight.detach() * r * log_pi1).mean()
