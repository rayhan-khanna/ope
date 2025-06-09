import torch
from sklearn.utils import check_random_state
from action_policies import EpsilonGreedyPolicy
from action_policies import SoftmaxPolicy

class CustomSyntheticBanditDataset:
    """Generates synthetic bandit feedback dataset."""

    def __init__(self, 
                 n_actions=10000, 
                 dim_context=10, 
                 top_k=10, 
                 reward_std=1.0, 
                 random_state=42,
                 action_policy=None, 
                 device="cpu", 
                 single_stage=True): 
        
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.top_k = top_k
        self.reward_std = reward_std
        self.device = torch.device(device) 
        
        self.random_ = check_random_state(random_state)
        
        self.action_context = torch.tensor(
            self.random_.normal(size=(n_actions, dim_context)), dtype=torch.float32
        ).to(self.device)

        self.single_stage = single_stage

        if action_policy is None:
            self.action_policy = SoftmaxPolicy(dim_context)
        else:
            self.action_policy = action_policy

    def sample_context(self, n):
        """Sample n context vectors from a Gaussian distribution."""
        return torch.tensor(
            self.random_.normal(size=(n, self.dim_context)),
            dtype=torch.float32,
            device=self.device
        )

    def sample_policy_full(self, context):
        """Sample from full action space (no top-k restriction)."""
        return torch.tensor([
            self.action_policy.sample_action(context[i], self.action_context)
            for i in range(context.shape[0])
        ], dtype=torch.long).to(self.device)

    def generate_data(self, context):
        """Generate synthetic dataset."""
        if self.single_stage:
            actions = self.sample_policy_full(context)
            rewards = self.reward_function(context, actions)
            return {
                "context": context,
                "actions": actions,
                "rewards": rewards
            }
        else: 
            candidates = self.candidate_selection(context)
            actions = self.sample_policy(candidates, context)
            rewards = self.reward_function(context, actions)

            return {
                "context": context,
                "candidates": candidates,
                "actions": actions,
                "rewards": rewards
            }

    def candidate_selection(self, context):
        """Select top-K candidates based on inner product search."""
        scores = torch.matmul(context, self.action_context.T)  # inner product
        top_k_indices = torch.topk(scores, self.top_k, dim=1).indices
        return top_k_indices

    def sample_policy(self, candidates, context):
        """Use the provided policy to select an action."""
        return torch.tensor([
            self.action_policy.select_action(
                candidates[i].cpu().numpy(), 
                context[i].cpu().numpy(), 
                self.action_context.cpu().numpy()
            )
            for i in range(candidates.shape[0])
        ], dtype=torch.long).to(self.device)

    def reward_function(self, context, actions):
        """Compute expected reward."""
        mean_reward = torch.sum(context * self.action_context[actions], dim=1)
        noise = torch.randn_like(mean_reward) * self.reward_std
        return (mean_reward + noise).to(self.device)

    def obtain_batch_bandit_feedback(self, n_samples):
        """Obtain structured logged bandit feedback."""
        
        context = torch.tensor(
            self.random_.normal(size=(n_samples, self.dim_context)), dtype=torch.float32
        ).to(self.device)
        data = self.generate_data(context)

        
        if self.single_stage:
            pi_b_logits = torch.randn(n_samples, self.n_actions).to(self.device)
            pi_b = torch.softmax(pi_b_logits, dim=1)
            pscore = pi_b[torch.arange(n_samples), data["actions"]]

            return {
                "n_samples": n_samples,
                "context": data["context"],
                "action": data["actions"],
                "reward": data["rewards"],
                "pi_b": pi_b,
                "pscore": pscore
            }

        else:   
            expected_rewards = torch.zeros((n_samples, self.top_k), dtype=torch.float32).to(self.device)

            for i in range(n_samples):
                expected_rewards[i, :] = self.reward_function(
                    data["context"][i:i+1], data["candidates"][i]
                )

            # compute policy probabilities
            pi_b_logits = torch.tensor(
                self.random_.normal(size=data["candidates"].shape), dtype=torch.float32
            ).to(self.device)
            pi_b = torch.exp(pi_b_logits) / torch.sum(torch.exp(pi_b_logits), dim=1, keepdim=True)

            # get probability of chosen action
            chosen_prob = torch.tensor([
                pi_b[i, (data["candidates"][i] == data["actions"][i]).nonzero(as_tuple=True)[0]].item()
                for i in range(n_samples)
            ], dtype=torch.float32).to(self.device)

            return {
                "n_samples": n_samples,
                "context": data["context"],
                "candidates": data["candidates"],
                "action": data["actions"],
                "reward": data["rewards"],
                "expected_reward": expected_rewards,
                "pi_b": pi_b,
                "pscore": chosen_prob, 
            }