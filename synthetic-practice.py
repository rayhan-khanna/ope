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

def run_ope(seed, dataset, n_rounds_list, n_actions, cfg):
    np.random.seed(seed)
    mse_results = {}
    for n_rounds in n_rounds_list:
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

        X_train = np.hstack([bandit_feedback["context"], bandit_feedback["action"].reshape(-1, 1)])
        y_train = bandit_feedback["reward"]

        model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LogisticRegression())
        model.fit(X_train, y_train)

        n_rounds, n_actions = bandit_feedback["expected_reward"].shape
        X_test = np.repeat(bandit_feedback["context"], n_actions, axis=0)
        actions_test = np.tile(np.arange(n_actions), n_rounds).reshape(-1, 1)
        X_test = np.hstack([X_test, actions_test])

        predicted_probs = model.predict_proba(X_test)[:, 1]
        estimated_rewards_by_reg_model = predicted_probs.reshape(n_rounds, n_actions, 1)

        # Compute action scores from the trained model
        action_scores = model.decision_function(X_test).reshape(n_rounds, n_actions)

        # Apply softmax 
        action_dist = np.exp(action_scores) / np.sum(np.exp(action_scores), axis=1, keepdims=True)

        action_dist = action_dist.reshape(n_rounds, n_actions, 1)

        estimator_map = {"DM": DirectMethod(), "IPS": InverseProbabilityWeighting(), "DR": DoublyRobust(), "MIPS": SwitchDoublyRobust()}
        selected_estimators = {name: estimator_map[name] for name in cfg.ope.estimators}

        mse_scores = {}
        for name, estimator in selected_estimators.items():
            estimate = estimator.estimate_policy_value(
                reward=bandit_feedback["reward"],
                action=bandit_feedback["action"],
                pscore=bandit_feedback["pscore"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
            )
            mse = np.mean((estimate - bandit_feedback["expected_reward"].mean()) ** 2)
            mse_scores[name] = mse

        mse_results[n_rounds] = mse_scores

    return mse_results   

def plot_results(df_mse_avg, df_mse_std, n_rounds_list):
    plt.figure(figsize=(8, 5))
    for estimator in df_mse_avg.columns:
        plt.errorbar(n_rounds_list, df_mse_avg[estimator], yerr=df_mse_std[estimator], marker="o", label=estimator)
    plt.xlabel("Number of Rounds")
    plt.ylabel("MSE")
    plt.xlim(left=min(n_rounds_list))
    plt.title("MSE vs. Data Size for OPE Estimators (with 20 Random Seeds)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    n_actions = cfg.bandit.n_actions
    dim_context = cfg.bandit.dim_context
    reward_type = cfg.bandit.reward_type
    reward_function_name = cfg.bandit.reward_function

    reward_function_map = {
        "logistic_reward_function": logistic_reward_function,
        "linear_reward_function": linear_reward_function
    }

    reward_function = reward_function_map[reward_function_name]
    n_rounds_list = cfg.evaluation.n_rounds_list
    seeds = list(range(20))  

    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_function=reward_function,
        random_state=42  
    )

    mse_results_all_seeds = [run_ope(seed, dataset, n_rounds_list, n_actions, cfg) for seed in seeds]
    df_mse_avg = pd.concat([pd.DataFrame.from_dict(res, orient="index") for res in mse_results_all_seeds]).groupby(level=0).mean()
    df_mse_std = pd.concat([pd.DataFrame.from_dict(res, orient="index") for res in mse_results_all_seeds]).groupby(level=0).std()

    plot_results(df_mse_avg, df_mse_std, n_rounds_list)

if __name__ == "__main__":
    main()