import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_random_state

class SyntheticPreferenceDataset:
    """Generates synthetic dataset for preference-adaptive OPL in non-stationary environments."""
    def __init__(self, 
                 n_actions=10000, 
                 dim_context=10, 
                 top_k=10, 
                 reward_std=1.0, 
                 random_state=42):
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.top_k = top_k
        self.reward_std = reward_std
        self.random_ = check_random_state(random_state)
        self.action_context = self.random_.normal(size=(n_actions, dim_context))

    def generate_data(self, context):
        """Generate synthetic dataset."""
        candidates = self.candidate_selection(context)
        actions = self.sample_policy(candidates)
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

    def sample_policy(self, candidates):
        """Sample an action from the top-K candidates using softmax."""
        logits = self.random_.normal(size=candidates.shape)
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        chosen_actions = np.array([self.random_.choice(candidates[i], p=probabilities[i]) for i in range(candidates.shape[0])])
        return chosen_actions

    def reward_function(self, context, actions):
        """Compute expected reward."""
        mean_reward = np.sum(context * self.action_context[actions], axis=1)        
        return truncnorm.rvs(a=-1, b=1, loc=mean_reward, scale=self.reward_std, random_state=self.random_)
    
    def obtain_batch_bandit_feedback(self, n_samples):
        """Obtain structured logged bandit feedback, ensuring only this method generates context."""
        
        # gen user context
        context = self.random_.normal(size=(n_samples, self.dim_context))

        # gen synthetic data 
        data = self.generate_data(context)

        expected_rewards = np.zeros((n_samples, self.top_k)) 

        for i in range(n_samples):
            expected_rewards[i, :] = dataset.reward_function(
                data["context"][i:i+1], data["candidates"][i]
            )

        # compute probs
        pi_b_logits = np.random.normal(size=data["candidates"].shape) 
        pi_b = np.exp(pi_b_logits) / np.sum(np.exp(pi_b_logits), axis=1, keepdims=True)

        # get prob of chosen action
        chosen_prob = np.array([pi_b[i, list(data["candidates"][i]).index(data["actions"][i])] for i in range(n_samples)])

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

# 
# Testing
#
    
dataset = SyntheticPreferenceDataset(n_actions=10000, dim_context=10, top_k=5)

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