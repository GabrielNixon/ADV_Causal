import numpy as np
import matplotlib.pyplot as plt

from plot_learning_results import (
    train_agent as train_q_agent,
    evaluate_agent as evaluate_q_agent,
    make_qlearn_eval_config,
    QLEARN_CONFIG,
)
from train_rnn_investor import (
    train_agent as train_rnn_agent,
    evaluate_agent as evaluate_rnn_agent,
    make_rnn_eval_config,
    RNN_CONFIG,
)
from experiment_configs import CONFIGS


def moving_average(x, window=100):
    x = np.asarray(x)
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_comparison(q_train_ind, q_train_coord, rnn_train_ind, rnn_train_coord, window=100):
    q_ind = moving_average(q_train_ind, window)
    q_coord = moving_average(q_train_coord, window)
    rnn_ind = moving_average(rnn_train_ind, window)
    rnn_coord = moving_average(rnn_train_coord, window)

    plt.figure(figsize=(10, 6))
    plt.plot(q_ind, label="Q: Train in Independent")
    plt.plot(q_coord, label="Q: Train in Coordinated")
    plt.plot(rnn_ind, label="RNN: Train in Independent")
    plt.plot(rnn_coord, label="RNN: Train in Coordinated")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return (moving average)")
    plt.title("Q-Learner vs RNN Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q_vs_rnn_training_curves.png", dpi=220)
    plt.close()


def plot_metric_bars(q_results, rnn_results, metric_key, ylabel, title, filename):
    labels = list(q_results.keys())
    x = np.arange(len(labels))
    width = 0.38

    q_means = [q_results[k][metric_key].mean() for k in labels]
    rnn_means = [rnn_results[k][metric_key].mean() for k in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, q_means, width=width, label="Q-Learner")
    plt.bar(x + width / 2, rnn_means, width=width, label="RNN")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_round_profiles(q_results, rnn_results, filename_prefix="q_vs_rnn"):
    settings = list(q_results.keys())

    plt.figure(figsize=(10, 6))
    for label in settings:
        q_df = q_results[label]["round_logs"]
        r_df = rnn_results[label]["round_logs"]

        q_round = q_df.groupby("round")[["investment"]].mean().reset_index()
        r_round = r_df.groupby("round")[["investment"]].mean().reset_index()

        plt.plot(q_round["round"], q_round["investment"], marker="o", linestyle="--", label=f"Q {label}")
        plt.plot(r_round["round"], r_round["investment"], marker="o", label=f"RNN {label}")

    plt.xlabel("Round")
    plt.ylabel("Mean investment")
    plt.title("Q-Learner vs RNN: Mean Investment by Round")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_round_investment.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    for label in settings:
        q_df = q_results[label]["round_logs"]
        r_df = rnn_results[label]["round_logs"]

        q_round = q_df.groupby("round")[["repay_prop"]].mean().reset_index()
        r_round = r_df.groupby("round")[["repay_prop"]].mean().reset_index()

        plt.plot(q_round["round"], q_round["repay_prop"], marker="o", linestyle="--", label=f"Q {label}")
        plt.plot(r_round["round"], r_round["repay_prop"], marker="o", label=f"RNN {label}")

    plt.xlabel("Round")
    plt.ylabel("Mean repayment proportion")
    plt.title("Q-Learner vs RNN: Mean Repayment by Round")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_round_repayment.png", dpi=220)
    plt.close()


def summarize_group(name, results):
    print(f"\n{name}")
    for label, result in results.items():
        print(
            f"{label:>12} | "
            f"Return={result['returns'].mean():.3f} | "
            f"Investment={result['mean_investments'].mean():.3f} | "
            f"Repay={result['mean_repay_props'].mean():.3f}"
        )


if __name__ == "__main__":
    q_cfg = QLEARN_CONFIG
    rnn_cfg = RNN_CONFIG

    ind_q_cfg = make_qlearn_eval_config(CONFIGS["baseline_independent"])
    coord_q_cfg = make_qlearn_eval_config(CONFIGS["baseline_coordinated"])

    ind_rnn_cfg = make_rnn_eval_config(CONFIGS["baseline_independent"])
    coord_rnn_cfg = make_rnn_eval_config(CONFIGS["baseline_coordinated"])

    print("Training Q-Learner in Independent world...")
    q_agent_ind, q_train_ind = train_q_agent(
        exp_cfg=ind_q_cfg,
        q_cfg=q_cfg,
        seed_offset=0,
    )

    print("Training Q-Learner in Coordinated world...")
    q_agent_coord, q_train_coord = train_q_agent(
        exp_cfg=coord_q_cfg,
        q_cfg=q_cfg,
        seed_offset=50000,
    )

    print("Training RNN in Independent world...")
    rnn_agent_ind, rnn_train_ind, _ = train_rnn_agent(
        exp_cfg=ind_rnn_cfg,
        rnn_cfg=rnn_cfg,
        seed_offset=100000,
    )

    print("Training RNN in Coordinated world...")
    rnn_agent_coord, rnn_train_coord, _ = train_rnn_agent(
        exp_cfg=coord_rnn_cfg,
        rnn_cfg=rnn_cfg,
        seed_offset=150000,
    )

    q_results = {
        "Ind→Ind": evaluate_q_agent(q_agent_ind, ind_q_cfg, n_eval_episodes=q_cfg["n_eval"], seed=200000),
        "Ind→Coord": evaluate_q_agent(q_agent_ind, coord_q_cfg, n_eval_episodes=q_cfg["n_eval"], seed=300000),
        "Coord→Coord": evaluate_q_agent(q_agent_coord, coord_q_cfg, n_eval_episodes=q_cfg["n_eval"], seed=400000),
        "Coord→Ind": evaluate_q_agent(q_agent_coord, ind_q_cfg, n_eval_episodes=q_cfg["n_eval"], seed=500000),
    }

    rnn_results = {
        "Ind→Ind": evaluate_rnn_agent(rnn_agent_ind, ind_rnn_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=600000),
        "Ind→Coord": evaluate_rnn_agent(rnn_agent_ind, coord_rnn_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=700000),
        "Coord→Coord": evaluate_rnn_agent(rnn_agent_coord, coord_rnn_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=800000),
        "Coord→Ind": evaluate_rnn_agent(rnn_agent_coord, ind_rnn_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=900000),
    }

    summarize_group("Q-Learner", q_results)
    summarize_group("RNN", rnn_results)

    plot_training_comparison(
        q_train_ind,
        q_train_coord,
        rnn_train_ind,
        rnn_train_coord,
        window=100,
    )

    plot_metric_bars(
        q_results,
        rnn_results,
        metric_key="returns",
        ylabel="Mean return",
        title="Q-Learner vs RNN: Cross-World Mean Returns",
        filename="q_vs_rnn_return_bars.png",
    )

    plot_metric_bars(
        q_results,
        rnn_results,
        metric_key="mean_investments",
        ylabel="Mean investment",
        title="Q-Learner vs RNN: Mean Investment",
        filename="q_vs_rnn_investment_bars.png",
    )

    plot_metric_bars(
        q_results,
        rnn_results,
        metric_key="mean_repay_props",
        ylabel="Mean repayment proportion",
        title="Q-Learner vs RNN: Mean Repayment Proportion",
        filename="q_vs_rnn_repayment_bars.png",
    )

    plot_round_profiles(
        q_results,
        rnn_results,
        filename_prefix="q_vs_rnn",
    )

    print("\nSaved:")
    print(" - q_vs_rnn_training_curves.png")
    print(" - q_vs_rnn_return_bars.png")
    print(" - q_vs_rnn_investment_bars.png")
    print(" - q_vs_rnn_repayment_bars.png")
    print(" - q_vs_rnn_round_investment.png")
    print(" - q_vs_rnn_round_repayment.png")