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
                 device="cpu"): 
        
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.top_k = top_k
        self.reward_std = reward_std
        self.device = torch.device(device) 
        
        self.random_ = check_random_state(random_state)
        
        self.action_context = torch.tensor(
            self.random_.normal(size=(n_actions, dim_context)), dtype=torch.float32
        ).to(self.device)
        
        if action_policy is None:
            print("No action policy provided: defaulting to SoftmaxPolicy.")
            self.action_policy = SoftmaxPolicy(temperature=1.0, random_state=random_state)
        else:
            self.action_policy = action_policy

    def generate_data(self, context):
        """Generate synthetic dataset."""
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

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = EpsilonGreedyPolicy(epsilon=0.1, random_state=42)
dataset = CustomSyntheticBanditDataset(n_actions=10000, dim_context=10, top_k=5, action_policy=policy, device=device)

# generate bandit feedback data
n_samples = 5
bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples)

# print first 2 values to see output
print("Bandit Feedback")
for key, value in bandit_feedback.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value[:2].cpu().numpy()}")  # numpy for readability
    else:
        print(f"{key}: {value}")