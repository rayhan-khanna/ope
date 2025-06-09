import torch
import torch.nn.functional as F

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
        return -(weighted_rewards * log_pi_theta).mean()
    
class DMGradient:
    def __init__(self, reward_model, target_policy, context, action_context):
        self.reward_model = reward_model
        self.target_policy = target_policy
        self.context = context
        self.action_context = action_context

    def estimate_policy_gradient(self):
        pi_theta = self.target_policy.probs(self.context, self.action_context) 

        batch_size = self.context.size(0)
        num_actions = self.action_context.size(0)

        all_rewards = torch.stack([
            self.reward_model.predict(
                self.context, 
                torch.full((batch_size,), a_idx, dtype=torch.long, device=self.context.device)
            ).detach() 
            for a_idx in range(num_actions)
        ], dim=1) 

        expected_reward = (pi_theta * all_rewards).sum(dim=1)

        return -expected_reward.mean()
    
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
        pi_e = self.target_policy.probs(self.context, self.action_context).detach()

        target_actions = torch.tensor(
            [self.target_policy.sample_action(x_i, self.action_context) for x_i in self.context],
            device=self.context.device
        )

        q_target = self.reward_model.predict(self.context, target_actions)
        q_logged = self.reward_model.predict(self.context, self.actions)

        target_pscore = pi_e[torch.arange(len(self.context)), self.actions]
        weights = (target_pscore / self.behavior_pscore)

        log_pi_e = self.target_policy.log_prob(self.context, self.actions, self.action_context)

        return -(q_target.mean() + ((weights * (self.rewards - q_logged)).detach() * log_pi_e).mean())

def sample_gumbel(shape, device):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U))

def gumbel_topk_sampling(candidate_logits, candidates, k):
    gumbel_noise = sample_gumbel(candidate_logits.shape, candidate_logits.device)
    noisy_logits = candidate_logits + gumbel_noise
    topk_pos = torch.topk(noisy_logits, k, dim=1).indices
    topk_indices = torch.gather(candidates, 1, topk_pos)
    return topk_indices

class TwoStageISGradient:
    def __init__(self, model_stage1, model_stage2, context, actions, rewards, 
                 behavior_pscore, candidates):
        self.model_stage1 = model_stage1
        self.model_stage2 = model_stage2
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

        # 1: compute logits over all items
        context_repr = self.model_stage1.context_nn(x)
        item_embs = self.model_stage1.item_embeddings.weight
        logits = torch.matmul(context_repr, item_embs.T) 

        # restrict to candidate set
        candidate_logits = torch.gather(logits, 1, candidates)

        # Gumbel top-k sampling
        top_k = self.model_stage1.top_k
        topk_indices = gumbel_topk_sampling(candidate_logits, candidates, top_k)

        # log π₁(A_k)
        log_probs = F.log_softmax(logits, dim=1)
        log_pi_theta1_Ak = log_probs.gather(1, topk_indices).sum(dim=1)

        # 2: compute pi_2(a | x, A_k)
        probs_topk = self.model_stage2(x, topk_indices)

        # only consider examples where a ∈ A_k
        match_mask = (topk_indices == a_taken.unsqueeze(1))
        action_in_topk = match_mask.any(dim=1)

        if action_in_topk.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        idx_valid = match_mask[action_in_topk].float().argmax(dim=1)
        probs_topk_valid = probs_topk[action_in_topk]
        pi_theta2_probs = probs_topk_valid[torch.arange(len(idx_valid)), idx_valid]

        log_pi1 = log_pi_theta1_Ak[action_in_topk]
        pi0_valid = pi0_probs[action_in_topk]
        r_valid = r[action_in_topk]

        weight = pi_theta2_probs / pi0_valid
        loss = -weight.detach() * log_pi1 * r_valid
        return loss.mean()

class KernelISGradient:
    def __init__(self, data, target_policy, logging_policy, kernel, tau, marginal_density_model, action_context, num_epochs=10):
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
        self.action_context = action_context
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

    def _estimate_policy_gradient(self) -> torch.Tensor:
        self.estimateLMD()
        grad = []
        for x_i, y_i, r_i in self.data:
            W_k, A_k = self.target_policy.sample_latent(x_i) 
            y = self.target_policy.sample_action(x_i, A_k, W_k)

            k_val = self.kernel(y, y_i, x_i, self.tau)
            density_estimate = self.marginal_density_model.predict(x_i, y_i, self.action_context)
            is_weight = k_val / density_estimate

            log_grad = self.target_policy.log_grad(W_k, A_k, x_i)
            grad.append(-is_weight.detach() * r_i.detach() * log_grad)

        return torch.stack(grad).mean()