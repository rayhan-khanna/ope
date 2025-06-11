import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from obp.dataset import SyntheticBanditDataset, logistic_reward_function

wandb.init(project="gumbel-max", name="training")

class NeuralNetworkPolicy(nn.Module):
    def __init__(self, context_dim: int, num_actions: int, hidden_dim: int = 32):
        super(NeuralNetworkPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, context):
        return self.model(context)

    def select_action_gumbel(self, context, epsilon=0.1):
        logits = self.forward(context.unsqueeze(0)) 
        logits = logits.detach().numpy()[0]  

        # exploration
        if np.random.rand() < epsilon:  
            return np.random.choice(len(logits))

        # adds Gumbel noise and takes argmax
        gumbel_noise = np.random.gumbel(size=len(logits))
        return np.argmax(logits + gumbel_noise) 

def train_online_policy(policy, optimizer, dataset, num_rounds=1000, epsilon=0.1):
    total_reward = 0
    for i in range(num_rounds):
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=1)
        context = torch.tensor(bandit_feedback["context"][0], dtype=torch.float32)
        reward = torch.tensor(bandit_feedback["reward"][0], dtype=torch.float32)

        # action selection -- gumbel-max
        action = policy.select_action_gumbel(context, epsilon=epsilon)
        action_tensor = torch.tensor(action, dtype=torch.long)

        action_logits = policy(context.unsqueeze(0))
        chosen_action_logit = action_logits[0, action_tensor]

        loss = -chosen_action_logit * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward.item()
        wandb.log({"Loss": loss.item(), "Total Reward": total_reward, "Round": i})

        if i % 100 == 0:
            print(f"Round {i}: Loss = {loss.item()}, Total Reward = {total_reward}")

# Dataset setup
context_dim = 5  
num_actions = 3  
dataset = SyntheticBanditDataset(
    n_actions=num_actions,
    dim_context=context_dim,
    reward_type="binary",
    reward_function=logistic_reward_function,
    random_state=42  
)

policy = NeuralNetworkPolicy(context_dim, num_actions)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

train_online_policy(policy, optimizer, dataset, num_rounds=1000, epsilon=0.1)

new_context = torch.randn(context_dim)
selected_action = policy.select_action_gumbel(new_context)
print(f"Selected action: {selected_action}")

wandb.finish()