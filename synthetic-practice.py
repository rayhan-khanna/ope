import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from obp.ope import DirectMethod, InverseProbabilityWeighting, DoublyRobust, SwitchDoublyRobust
from obp.dataset import SyntheticBanditDataset, logistic_reward_function, linear_reward_function
from custom_estimators import ImportanceSamplingEstimator  # Custom estimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
        
    # Load bandit config
    n_actions = cfg.bandit.n_actions
    dim_context = cfg.bandit.dim_context
    reward_type = cfg.bandit.reward_type
    reward_function_name = cfg.bandit.reward_function

    reward_function_map = {
        "logistic_reward_function": logistic_reward_function,
        "linear_reward_function": linear_reward_function
    }

    reward_function = reward_function_map[reward_function_name]

    # Define synthetic dataset
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_function=reward_function
    )

    # Define evaluation settings
    n_rounds_list = cfg.evaluation.n_rounds_list
    mse_results = {}

    # MODULARIZE THIS LOOP (AND CODE IN GENERAL)

    # Run OPE for different data sizes
    for n_rounds in n_rounds_list:
        # print(f"\nProcessing n_rounds={n_rounds}...")

        # Generate bandit feedback
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

        # Prepare training data from logged feedback
        X_train = np.hstack([bandit_feedback["context"], bandit_feedback["action"].reshape(-1, 1)])
        y_train = bandit_feedback["reward"]

        # Train logistic regression model
        model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LogisticRegression())
        model.fit(X_train, y_train)

        # Predict expected rewards for all actions
        n_rounds, n_actions = bandit_feedback["expected_reward"].shape
        X_test = np.repeat(bandit_feedback["context"], n_actions, axis=0)
        actions_test = np.tile(np.arange(n_actions), n_rounds).reshape(-1, 1)
        X_test = np.hstack([X_test, actions_test])

        # Get estimated probabilities
        predicted_probs = model.predict_proba(X_test)[:, 1] 
        estimated_rewards_by_reg_model = predicted_probs.reshape(n_rounds, n_actions, 1)

        # Non-uniform action distribution attempt (softmax)
        #action_scores = np.dot(bandit_feedback["context"], np.random.randn(dim_context, n_actions))
        #action_dist = np.exp(action_scores) / np.sum(np.exp(action_scores), axis=1, keepdims=True)
        #action_dist = action_dist.reshape(n_rounds, n_actions, 1)

        # Create a uniform action distribution
        action_dist = np.ones((n_rounds, n_actions, 1)) / n_actions
        
        estimator_map = {"DM": DirectMethod(), 
                         "IPS": InverseProbabilityWeighting(), 
                         "DR": DoublyRobust(), 
                         "MIPS": SwitchDoublyRobust(),
                         # "IS": ImportanceSamplingEstimator
                         }

        # Select estimators from config
        selected_estimators = {name: estimator_map[name] for name in cfg.ope.estimators}
        
        mse_scores = {}
        ope_results = {}

        # print("\nRunning OPE Estimators...")
        for name, estimator in selected_estimators.items():
            # if name == "IS":  # Handle your custom IS estimator
            #     behavior_policy_probs = bandit_feedback["pscore"] 
            #     target_policy_probs = action_dist  
            #     rewards = bandit_feedback["reward"]
            #     estimate = estimator(behavior_policy_probs, target_policy_probs, rewards).estimate_policy_value()
            if isinstance(estimator, DirectMethod):
                estimate = estimator.estimate_policy_value(
                    reward=bandit_feedback["reward"],
                    action=bandit_feedback["action"],
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
                )
            elif isinstance(estimator, InverseProbabilityWeighting):
                estimate = estimator.estimate_policy_value(
                    reward=bandit_feedback["reward"],
                    action=bandit_feedback["action"],
                    pscore=bandit_feedback["pscore"],
                    action_dist=action_dist
                )
            else:  # DR and MIPS
                estimate = estimator.estimate_policy_value(
                    reward=bandit_feedback["reward"],
                    action=bandit_feedback["action"],
                    pscore=bandit_feedback["pscore"],
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
                )

            # print(f"Estimate for {name}: {estimate}")
            ope_results[name] = estimate

            # Compute MSE
            mse = np.mean((estimate - bandit_feedback["expected_reward"].mean()) ** 2)
            mse_scores[name] = mse
            # print(f"MSE for {name}: {mse}")

        # Store MSE results
        mse_results[n_rounds] = mse_scores

    # Convert and plot results
    df_mse = pd.DataFrame.from_dict(mse_results, orient="index")

    plt.figure(figsize=(8, 5))
    for estimator in df_mse.columns:
        plt.plot(n_rounds_list, df_mse[estimator], marker="o", label=estimator)

    plt.xlabel("Number of Rounds")
    plt.ylabel("MSE")
    plt.title("MSE vs. Data Size for OPE Estimators")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()