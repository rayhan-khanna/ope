import matplotlib.pyplot as plt
from opl_two_stage_train import train

methods = [
    "single_preference_is", 
    "single_preference_kis", 
    "iter_k_is", 
    "iter_k_kis", 
    "naive_cf", 
    "online_policy"
]

all_losses = {}

for method in methods:
    print(f"Training: {method}")
    losses = train(method=method, seed=0)
    all_losses[method] = losses

plt.figure(figsize=(10, 6))

for method, losses in all_losses.items():
    plt.plot(losses, label=method.replace("_", " ").title())

plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Loss Curves Across Methods")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_curves.png")