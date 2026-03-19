import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mrtt_env import MRTTWorld
from experiment_configs import CONFIGS
from behavioral_investor import BehavioralRNNInvestor


def make_eval_config(base_exp_cfg):
    exp_cfg = copy.deepcopy(base_exp_cfg)
    exp_cfg["investor"] = {"type": "behavioral_rnn"}
    return exp_cfg


def build_env_from_config(exp_cfg, seed):
    return MRTTWorld(
        world_cfg=exp_cfg["world"],
        adv_cfg=exp_cfg["adversary"],
        seed=seed,
    )


def run_behavioral_eval(exp_cfg, n_episodes=200, seed=0, greedy=True):
    env = build_env_from_config(exp_cfg, seed=seed)
    investor = BehavioralRNNInvestor(
        model_path="behavioral_rnn.pt",
        horizon=exp_cfg["world"]["horizon"],
        endowment=exp_cfg["world"]["endowment"],
        greedy=False,
        seed=seed,
    )

    returns = []
    mean_investments = []
    mean_repay_props = []
    round_logs = []

    for ep in range(n_episodes):
        env = build_env_from_config(exp_cfg, seed=seed + ep)
        investor.reset()

        total_reward = 0.0
        investments = []
        repay_props = []

        for t in range(env.horizon):
            investment = investor.act(t)

            out = env.step(investment, t)
            reward = float(out["investor_gain"])
            repay_prop = float(out["repay_prop"])

            investor.observe_outcome(
                action=investment,
                reward=reward,
                repay_prop=repay_prop,
            )

            total_reward += reward
            investments.append(investment)
            repay_props.append(repay_prop)

            round_logs.append({
                "episode": ep,
                "round": t + 1,
                "world": exp_cfg["world"]["type"],
                "investment": float(investment),
                "repay_prop": repay_prop,
                "investor_gain": reward,
                "trustee_gain": float(out["trustee_gain"]),
                "investor_total_running": float(total_reward),
            })

        returns.append(total_reward)
        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))

    return {
        "returns": np.array(returns),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
        "round_logs": pd.DataFrame(round_logs),
    }


def summarize(label, result):
    print(f"\n{label}")
    print(f"Mean return     : {result['returns'].mean():.3f}")
    print(f"Std return      : {result['returns'].std():.3f}")
    print(f"Mean investment : {result['mean_investments'].mean():.3f}")
    print(f"Mean repay prop : {result['mean_repay_props'].mean():.3f}")


def plot_round_profiles(results_dict, prefix="behavioral_rnn"):
    plt.figure(figsize=(10, 5))
    for label, result in results_dict.items():
        df = result["round_logs"]
        round_summary = df.groupby("round")[["investment"]].mean().reset_index()
        plt.plot(round_summary["round"], round_summary["investment"], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel("Mean investment")
    plt.title("Behavioral RNN Investor: Mean Investment by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_investment.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    for label, result in results_dict.items():
        df = result["round_logs"]
        round_summary = df.groupby("round")[["repay_prop"]].mean().reset_index()
        plt.plot(round_summary["round"], round_summary["repay_prop"], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel("Mean repayment proportion")
    plt.title("Behavioral RNN Investor: Mean Repayment by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_repayment.png", dpi=220)
    plt.close()


def plot_summary_bars(results_dict, prefix="behavioral_rnn"):
    labels = list(results_dict.keys())
    x = np.arange(len(labels))

    returns = [results_dict[k]["returns"].mean() for k in labels]
    investments = [results_dict[k]["mean_investments"].mean() for k in labels]
    repay_props = [results_dict[k]["mean_repay_props"].mean() for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(x, returns)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean return")
    plt.title("Behavioral RNN Investor: Mean Returns")
    plt.tight_layout()
    plt.savefig(f"{prefix}_return_bars.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, investments)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean investment")
    plt.title("Behavioral RNN Investor: Mean Investment")
    plt.tight_layout()
    plt.savefig(f"{prefix}_investment_bars.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, repay_props)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean repayment proportion")
    plt.title("Behavioral RNN Investor: Mean Repayment Proportion")
    plt.tight_layout()
    plt.savefig(f"{prefix}_repayment_bars.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    ind_cfg = make_eval_config(CONFIGS["baseline_independent"])
    coord_cfg = make_eval_config(CONFIGS["baseline_coordinated"])

    eval_results = {
        "BehRNN→Ind": run_behavioral_eval(ind_cfg, n_episodes=200, seed=200000, greedy=True),
        "BehRNN→Coord": run_behavioral_eval(coord_cfg, n_episodes=200, seed=300000, greedy=True),
    }

    for label, result in eval_results.items():
        summarize(label, result)

    plot_round_profiles(eval_results, prefix="behavioral_rnn")
    plot_summary_bars(eval_results, prefix="behavioral_rnn")

    print("\nSaved:")
    print(" - behavioral_rnn_round_investment.png")
    print(" - behavioral_rnn_round_repayment.png")
    print(" - behavioral_rnn_return_bars.png")
    print(" - behavioral_rnn_investment_bars.png")
    print(" - behavioral_rnn_repayment_bars.png")