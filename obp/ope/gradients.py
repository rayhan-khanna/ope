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
        weight = target_pscore / self.behavior_pscore
        weighted_rewards = self.rewards * weight.detach()
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
    def __init__(self, policy, context, actions, rewards, 
                 behavior_pscore, candidates):
        self.policy = policy
        self.context = context
        self.actions = actions
        self.rewards = rewards
        self.behavior_pscore = behavior_pscore
        self.candidates = candidates

    def estimate_policy_gradient(self):
        x = self.context
        a_taken = self.actions
        r = self.rewards
        pi0_probs = self.behavior_pscore
        candidates = self.candidates
        device = x.device

        # already sampled k candidates through iterative sampling
        topk_indices = self.candidates 

        # get log pi_1(A_k) using policy
        log_pi1 = self.policy.log_prob_topk_set(x, topk_indices)

        # get pi_2(a | x, A_k)
        probs_topk = self.policy.probs_given_topk(x, topk_indices)

        # filter to examples where a ∈ A_k
        match_mask = (topk_indices == a_taken.unsqueeze(1))
        action_in_topk = match_mask.any(dim=1)

        if action_in_topk.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        idx_valid = match_mask[action_in_topk].float().argmax(dim=1)
        probs_topk_valid = probs_topk[action_in_topk]
        pi_theta2_probs = probs_topk_valid[torch.arange(len(idx_valid)), idx_valid]

        # gather valid values
        log_pi1_valid = log_pi1[action_in_topk]
        pi0_valid = pi0_probs[action_in_topk]
        r_valid = r[action_in_topk]

        # compute weighted loss
        weight = pi_theta2_probs / pi0_valid
        loss = -weight.detach() * log_pi1_valid * r_valid
        return loss.mean()

class KernelISGradient:
    def __init__(self, context, actions, rewards, target_policy, logging_policy,
                 kernel, tau, marginal_density_model, action_context, num_epochs=10, candidates=None):
        self.context = context
        self.actions = actions 
        self.rewards = rewards
        self.target_policy = target_policy
        self.logging_policy = logging_policy
        self.kernel = kernel
        self.tau = tau
        self.marginal_density_model = marginal_density_model
        self.action_context = action_context
        self.num_epochs = num_epochs
        self.A_k = candidates
        self.train_density_model = any(p.requires_grad for p in marginal_density_model.parameters())

    def estimateLMD(self):
        """Train h_model to estimate logging marginal density."""
        if not self.train_density_model:
            return
        optimizer = torch.optim.Adam(self.marginal_density_model.parameters())
        for _ in range(self.num_epochs):
            for x_i, y_i in zip(self.context, self.actions):
                optimizer.zero_grad()
                sampled_action = self.logging_policy.sample_action(x_i, self.action_context)
                k_val = self.kernel(y_i, sampled_action, x_i, self.tau)
                predicted_density = self.marginal_density_model.predict(x_i, y_i, self.action_context)
                loss = (predicted_density - k_val) ** 2
                loss.backward()
                optimizer.step()

    def estimate_policy_gradient(self) -> torch.Tensor:
        self.estimateLMD()

        x = self.context 
        y_logged = self.actions  
        r = self.rewards
        B = x.shape[0] # batch size

        A_k = self.A_k

        y_sampled = torch.tensor([
            self.target_policy.sample_action(x[i], A_k[i]) for i in range(B)
        ], device=x.device)

        k_val = torch.tensor([
            self.kernel(y_sampled[i], y_logged[i], x[i], self.tau)
            for i in range(B)
        ], device=x.device)

        density = self.marginal_density_model.predict(x, y_logged, self.action_context)      
        is_weight = k_val / density    
        log_pi_theta1_Ak = self.target_policy.log_prob_topk_set(x, A_k)

        loss = -(is_weight.detach() * r * log_pi_theta1_Ak).mean()
        return loss