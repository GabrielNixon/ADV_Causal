import copy
import numpy as np
import matplotlib.pyplot as plt

from config import INVEST_OPTIONS
from experiment_configs import CONFIGS
from mrtt_env import MRTTWorld
from train_qlearner import QLearningInvestor, QLEARN_CONFIG

ACTION_LIST = list(INVEST_OPTIONS)


def build_env_from_config(exp_cfg, seed):
    return MRTTWorld(
        world_cfg=exp_cfg["world"],
        adv_cfg=exp_cfg["adversary"],
        seed=seed,
    )


def make_qlearn_eval_config(base_exp_cfg):
    exp_cfg = copy.deepcopy(base_exp_cfg)
    exp_cfg["investor"] = {"type": "qlearn"}
    return exp_cfg


def run_training_episode(agent, exp_cfg, seed):
    env = build_env_from_config(exp_cfg, seed=seed)
    total_reward = 0.0
    prev_repay_prop = None

    for t in range(env.horizon):
        state = agent.get_state(t, prev_repay_prop)
        action_idx = agent.choose_action(state)
        investment = float(ACTION_LIST[action_idx])

        out = env.step(investment, t)
        reward = out["investor_gain"]
        total_reward += reward

        next_prev_repay_prop = out["repay_prop"]
        done = (t == env.horizon - 1)
        next_state = None if done else agent.get_state(t + 1, next_prev_repay_prop)

        agent.update(state, action_idx, reward, next_state, done)
        prev_repay_prop = next_prev_repay_prop

    agent.decay_epsilon()
    return total_reward


def train_agent(exp_cfg, q_cfg, seed_offset=0):
    agent = QLearningInvestor(
        alpha=q_cfg["alpha"],
        gamma=q_cfg["gamma"],
        epsilon=q_cfg["epsilon"],
        epsilon_min=q_cfg["epsilon_min"],
        epsilon_decay=q_cfg["epsilon_decay"],
        seed=seed_offset,
    )

    returns = []
    for ep in range(q_cfg["n_train"]):
        ep_return = run_training_episode(agent, exp_cfg, seed=seed_offset + ep)
        returns.append(ep_return)

    return agent, np.array(returns)


def evaluate_agent(agent, exp_cfg, n_eval_episodes=200, seed=100000):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    returns = []
    mean_investments = []
    mean_repay_props = []

    for ep in range(n_eval_episodes):
        env = build_env_from_config(exp_cfg, seed=seed + ep)
        total_reward = 0.0
        prev_repay_prop = None
        investments = []
        repay_props = []

        for t in range(env.horizon):
            state = agent.get_state(t, prev_repay_prop)
            action_idx = agent.choose_action(state)
            investment = float(ACTION_LIST[action_idx])

            out = env.step(investment, t)
            total_reward += out["investor_gain"]
            investments.append(investment)
            repay_props.append(out["repay_prop"])
            prev_repay_prop = out["repay_prop"]

        returns.append(total_reward)
        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))

    agent.epsilon = old_epsilon

    return {
        "returns": np.array(returns),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
    }


def moving_average(x, window=100):
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(train_returns_ind, train_returns_coord, window=100, filename="qlearning_learning_curves.png"):
    ma_ind = moving_average(train_returns_ind, window=window)
    ma_coord = moving_average(train_returns_coord, window=window)

    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(ma_ind) + 1), ma_ind, label="Train in Independent")
    plt.plot(np.arange(1, len(ma_coord) + 1), ma_coord, label="Train in Coordinated")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return (moving average)")
    plt.title("Q-Learning Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_eval_bar(results_dict, filename="qlearning_cross_world_bars.png"):
    labels = list(results_dict.keys())
    means = [results_dict[k]["returns"].mean() for k in labels]
    stds = [results_dict[k]["returns"].std() for k in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean Return")
    plt.title("Q-Learner Cross-World Evaluation")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_eval_histograms(results_dict, filename="qlearning_return_histograms.png"):
    plt.figure(figsize=(9, 5))

    for label, result in results_dict.items():
        plt.hist(result["returns"], bins=30, alpha=0.45, label=label)

    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    plt.title("Evaluation Return Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_behavior_bars(results_dict, metric_key, ylabel, title, filename):
    labels = list(results_dict.keys())
    means = [results_dict[k][metric_key].mean() for k in labels]
    stds = [results_dict[k][metric_key].std() for k in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def summarize(label, result):
    print(f"\n{label}")
    print(f"Mean return     : {result['returns'].mean():.3f}")
    print(f"Std return      : {result['returns'].std():.3f}")
    print(f"Mean investment : {result['mean_investments'].mean():.3f}")
    print(f"Mean repay prop : {result['mean_repay_props'].mean():.3f}")


if __name__ == "__main__":
    q_cfg = QLEARN_CONFIG

    ind_cfg = make_qlearn_eval_config(CONFIGS["baseline_independent"])
    coord_cfg = make_qlearn_eval_config(CONFIGS["baseline_coordinated"])

    print("Training agent in Independent world...")
    agent_ind, train_returns_ind = train_agent(
        exp_cfg=ind_cfg,
        q_cfg=q_cfg,
        seed_offset=0,
    )

    print("Training agent in Coordinated world...")
    agent_coord, train_returns_coord = train_agent(
        exp_cfg=coord_cfg,
        q_cfg=q_cfg,
        seed_offset=50000,
    )

    eval_results = {
        "Ind→Ind": evaluate_agent(agent_ind, ind_cfg, n_eval_episodes=q_cfg["n_eval"], seed=200000),
        "Ind→Coord": evaluate_agent(agent_ind, coord_cfg, n_eval_episodes=q_cfg["n_eval"], seed=300000),
        "Coord→Coord": evaluate_agent(agent_coord, coord_cfg, n_eval_episodes=q_cfg["n_eval"], seed=400000),
        "Coord→Ind": evaluate_agent(agent_coord, ind_cfg, n_eval_episodes=q_cfg["n_eval"], seed=500000),
    }

    for label, result in eval_results.items():
        summarize(label, result)

    plot_learning_curves(
        train_returns_ind,
        train_returns_coord,
        window=100,
        filename="qlearning_learning_curves.png",
    )

    plot_eval_bar(
        eval_results,
        filename="qlearning_cross_world_bars.png",
    )

    plot_eval_histograms(
        eval_results,
        filename="qlearning_return_histograms.png",
    )

    plot_behavior_bars(
        eval_results,
        metric_key="mean_investments",
        ylabel="Mean Investment",
        title="Mean Investment Across Evaluation Settings",
        filename="qlearning_mean_investment_bars.png",
    )

    plot_behavior_bars(
        eval_results,
        metric_key="mean_repay_props",
        ylabel="Mean Repayment Proportion",
        title="Mean Repayment Proportion Across Evaluation Settings",
        filename="qlearning_mean_repayprop_bars.png",
    )

    print("\nSaved plots:")
    print(" - qlearning_learning_curves.png")
    print(" - qlearning_cross_world_bars.png")
    print(" - qlearning_return_histograms.png")
    print(" - qlearning_mean_investment_bars.png")
    print(" - qlearning_mean_repayprop_bars.png")