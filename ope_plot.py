import matplotlib.pyplot as plt
import numpy as np
from ope_eval import evaluate

def clean_label(label):
    label = label.replace("_", " ")
    method, est = label.split(" - ")
    if method.lower() == "ma et al":
        method = "Ma et al."
    elif method.lower() == "opl":
        method = "OPL"
    elif method.lower() == "mse":
        method = "MSE"
    return f"{method} - {est}"

# Define seeds and methods to evaluate
seeds = [0, 1, 2, 3, 4]
methods = ["mse", "opl", "ma_et_al"]

# Store results: aggregate_results[method][metric] = [values across seeds]
aggregate_results = {}

# Collect results for each method and seed
for seed in seeds:
    for method in methods:
        result = evaluate(method, seed)
        for key, value in result.items():
            label = f"{method.upper()} - {key}" if key != "True" else "True"
            if label not in aggregate_results:
                aggregate_results[label] = []
            aggregate_results[label].append(value)

# Compute averages
flat_results = {}
true_values = []

for label, values in aggregate_results.items():
    mean_val = np.mean(values)
    if label == "True":
        true_values.append(mean_val)
    else:
        flat_results[label] = mean_val

# Plotting
labels, values = zip(*flat_results.items())
cleaned_labels = [clean_label(label) for label in labels]

plt.figure(figsize=(10, 6))
bars = plt.bar(cleaned_labels, values, color="skyblue", edgecolor="black")

# Annotate bar values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", 
             ha="center", va="bottom", fontsize=10)

# Plot true policy value
true_value = np.mean(true_values)
plt.axhline(y=true_value, color='red', linestyle='--', label='True Policy Value')
plt.text(len(cleaned_labels) - 0.5, true_value + 0.01, f"True: {true_value:.2f}", 
         color='red', fontsize=10, ha="right", va="bottom")

# Final touches
plt.xticks(rotation=45, ha="right")
plt.ylabel("Estimated Policy Value")
plt.title("OPE Estimators Comparison (Averaged Across Seeds)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()