import numpy as np
from scipy.stats import truncnorm
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
                 action_policy=None): 
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.top_k = top_k
        self.reward_std = reward_std
        self.random_ = check_random_state(random_state)
        self.action_context = self.random_.normal(size=(n_actions, dim_context))
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
        scores = context @ self.action_context.T  # inner product 
        top_k_indices = np.argsort(scores, axis=1)[:, -self.top_k:]
        return top_k_indices

    def sample_policy(self, candidates, context):
        """Use the provided policy to select an action."""
        return np.array([self.action_policy.select_action(candidates[i], context[i], self.action_context)
                            for i in range(candidates.shape[0])])

    def reward_function(self, context, actions):
        """Compute expected reward."""
        mean_reward = np.sum(context * self.action_context[actions], axis=1)        
        return truncnorm.rvs(a=-1, b=1, loc=mean_reward, scale=self.reward_std, random_state=self.random_)
    
    def obtain_batch_bandit_feedback(self, n_samples):
        """Obtain structured logged bandit feedback, ensuring only this method generates context."""
        
        # generate user context and synthetic data
        context = self.random_.normal(size=(n_samples, self.dim_context))
        data = self.generate_data(context)

        expected_rewards = np.zeros((n_samples, self.top_k)) 

        for i in range(n_samples):
            expected_rewards[i, :] = self.reward_function(
                data["context"][i:i+1], data["candidates"][i]
            )

        # compute policy probabilities
        pi_b_logits = self.random_.normal(size=data["candidates"].shape) 
        pi_b = np.exp(pi_b_logits) / np.sum(np.exp(pi_b_logits), axis=1, keepdims=True)

        # get probability of chosen action
        chosen_prob = np.array([
            pi_b[i, list(data["candidates"][i]).index(data["actions"][i])]
            for i in range(n_samples)
        ])

        return {
            "n_rounds": n_samples,
            "context": data["context"],
            "candidates": data["candidates"],
            "action": data["actions"],
            "reward": data["rewards"],
            "expected_reward": expected_rewards,
            "pi_b": pi_b,
            "pscore": chosen_prob, 
        }

policy = EpsilonGreedyPolicy(epsilon=0.1, random_state=42)
dataset = CustomSyntheticBanditDataset(n_actions=10000, dim_context=10, top_k=5, action_policy=policy)

# generate bandit feedback data
n_samples = 5
bandit_feedback = dataset.obtain_batch_bandit_feedback(n_samples)

# print first 2 values to see output
print("Bandit Feedback")
for key, value in bandit_feedback.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value[:2]}")  
    else:
        print(f"{key}: {value}")