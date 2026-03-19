import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mrtt_env import MRTTWorld
from behavioral_investor import BehavioralRNNInvestor
from behavioral_bandit_adversary import BehavioralBanditAdversary


def make_behavioral_bandit_config(world_type="independent", objective="max"):
    return {
        "world": {
            "type": world_type,
            "horizon": 10,
            "endowment": 20.0,
        },
        "adversary": {
            "objective": objective,
            "epsilon": 0.10,
            "alpha": 0.10,
        },
        "investor": {
            "type": "behavioral_rnn",
        },
    }


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def build_trustee_adversary(exp_cfg, seed=0):
    return BehavioralBanditAdversary(
        objective=exp_cfg["adversary"]["objective"],
        epsilon=exp_cfg["adversary"].get("epsilon", 0.10),
        alpha=exp_cfg["adversary"].get("alpha", 0.10),
        seed=seed,
    )


def run_training(exp_cfg, n_episodes=3000, seed=0, investor_greedy=False):
    trustee = build_trustee_adversary(exp_cfg, seed=seed)

    env = MRTTWorld(
        world_cfg=exp_cfg["world"],
        seed=seed,
        adversary=trustee,
    )

    investor = BehavioralRNNInvestor(
        model_path="behavioral_rnn.pt",
        horizon=exp_cfg["world"]["horizon"],
        endowment=exp_cfg["world"]["endowment"],
        greedy=investor_greedy,
        seed=seed,
    )

    investor_totals = []
    trustee_totals = []
    earning_gaps = []
    mean_repay_props = []
    mean_investments = []
    q_history = []
    round_logs = []

    for ep in range(n_episodes):
        env.reset()
        investor.reset()

        total_investor = 0.0
        total_trustee = 0.0
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

            total_investor += reward
            total_trustee += float(out["trustee_gain"])
            investments.append(float(investment))
            repay_props.append(repay_prop)

            round_logs.append({
                "episode": ep,
                "round": t + 1,
                "investment": float(investment),
                "repay_prop": repay_prop,
                "repayment": float(out["repayment"]),
                "tripled_amount": float(out["tripled_amount"]),
                "investor_gain": float(out["investor_gain"]),
                "trustee_gain": float(out["trustee_gain"]),
            })

        investor_totals.append(total_investor)
        trustee_totals.append(total_trustee)
        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))
        earning_gaps.append(abs(total_trustee - total_investor))
        q_history.append(trustee.q_values.copy())

        if (ep + 1) % 300 == 0:
            best_action, qvals = trustee.greedy_action_for_state()
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Investor mean(last100)={np.mean(investor_totals[-100:]):.3f} | "
                f"Trustee mean(last100)={np.mean(trustee_totals[-100:]):.3f} | "
                f"Gap mean(last100)={np.mean(earning_gaps[-100:]):.3f} | "
                f"Repay mean(last100)={np.mean(mean_repay_props[-100:]):.3f} | "
                f"Investment mean(last100)={np.mean(mean_investments[-100:]):.3f} | "
                f"Greedy repay={best_action:.2f} | "
                f"Q={np.round(qvals, 2)}"
            )

    return {
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "earning_gaps": np.array(earning_gaps),
        "mean_repay_props": np.array(mean_repay_props),
        "mean_investments": np.array(mean_investments),
        "round_logs": pd.DataFrame(round_logs),
        "q_history": np.array(q_history),
        "trustee": trustee,
    }


def summarize(label, results):
    print(f"\n{label}")
    print(f"Investor total mean : {results['investor_totals'].mean():.3f}")
    print(f"Investor total std  : {results['investor_totals'].std():.3f}")
    print(f"Trustee total mean  : {results['trustee_totals'].mean():.3f}")
    print(f"Trustee total std   : {results['trustee_totals'].std():.3f}")
    print(f"Gap mean            : {results['earning_gaps'].mean():.3f}")
    print(f"Mean repay prop     : {results['mean_repay_props'].mean():.3f}")
    print(f"Mean investment     : {results['mean_investments'].mean():.3f}")
    best_action, qvals = results["trustee"].greedy_action_for_state()
    print(f"Final greedy repay  : {best_action:.3f}")
    print(f"Final q-values      : {np.round(qvals, 3)}")


def plot_training_curves(max_results, fair_results, window=100, prefix="behavioral_bandit_adv"):
    max_inv = moving_average(max_results["investor_totals"], window)
    max_tru = moving_average(max_results["trustee_totals"], window)
    fair_inv = moving_average(fair_results["investor_totals"], window)
    fair_tru = moving_average(fair_results["trustee_totals"], window)

    max_gap = moving_average(max_results["earning_gaps"], window)
    fair_gap = moving_average(fair_results["earning_gaps"], window)

    max_rep = moving_average(max_results["mean_repay_props"], window)
    fair_rep = moving_average(fair_results["mean_repay_props"], window)

    max_invest = moving_average(max_results["mean_investments"], window)
    fair_invest = moving_average(fair_results["mean_investments"], window)

    plt.figure(figsize=(10, 5))
    plt.plot(max_tru, label="MAX trustee")
    plt.plot(max_inv, label="MAX investor")
    plt.plot(fair_tru, label="FAIR trustee")
    plt.plot(fair_inv, label="FAIR investor")
    plt.xlabel("Episode")
    plt.ylabel("Episode total (moving average)")
    plt.title("Behavioral Bandit Trustee: Totals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_totals.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(max_gap, label="MAX gap")
    plt.plot(fair_gap, label="FAIR gap")
    plt.xlabel("Episode")
    plt.ylabel("Absolute earning gap")
    plt.title("Behavioral Bandit Trustee: Absolute Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_gap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(max_rep, label="MAX repay")
    plt.plot(max_invest, label="MAX invest")
    plt.plot(fair_rep, label="FAIR repay")
    plt.plot(fair_invest, label="FAIR invest")
    plt.xlabel("Episode")
    plt.ylabel("Moving average")
    plt.title("Behavioral Bandit Trustee: Repayment and Investment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_behavior.png", dpi=220)
    plt.close()


def plot_round_profiles(max_results, fair_results, prefix="behavioral_bandit_adv"):
    max_df = max_results["round_logs"]
    fair_df = fair_results["round_logs"]

    max_round = max_df.groupby("round")[["investment", "repay_prop", "investor_gain", "trustee_gain"]].mean().reset_index()
    fair_round = fair_df.groupby("round")[["investment", "repay_prop", "investor_gain", "trustee_gain"]].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(max_round["round"], max_round["repay_prop"], marker="o", label="MAX repay")
    plt.plot(fair_round["round"], fair_round["repay_prop"], marker="o", label="FAIR repay")
    plt.xlabel("Round")
    plt.ylabel("Mean repayment proportion")
    plt.title("Repayment by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_repayment.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(max_round["round"], max_round["investment"], marker="o", label="MAX invest")
    plt.plot(fair_round["round"], fair_round["investment"], marker="o", label="FAIR invest")
    plt.xlabel("Round")
    plt.ylabel("Mean investment")
    plt.title("Investment by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_investment.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(max_round["round"], max_round["trustee_gain"], marker="o", label="MAX trustee gain")
    plt.plot(max_round["round"], max_round["investor_gain"], marker="o", label="MAX investor gain")
    plt.plot(fair_round["round"], fair_round["trustee_gain"], marker="o", linestyle="--", label="FAIR trustee gain")
    plt.plot(fair_round["round"], fair_round["investor_gain"], marker="o", linestyle="--", label="FAIR investor gain")
    plt.xlabel("Round")
    plt.ylabel("Mean gain")
    plt.title("Per-Round Gains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_gains.png", dpi=220)
    plt.close()


def plot_repayment_heatmaps(max_results, fair_results, prefix="behavioral_bandit_adv"):
    for label, results in [("max", max_results), ("fair", fair_results)]:
        df = results["round_logs"]

        heat = (
            df.groupby(["repay_prop", "round"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        heat_norm = heat.div(heat.sum(axis=0), axis=1).fillna(0.0)

        plt.figure(figsize=(8, 5))
        plt.imshow(heat_norm.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        plt.xticks(range(len(heat_norm.columns)), heat_norm.columns)
        plt.yticks(range(len(heat_norm.index)), [str(v) for v in heat_norm.index])
        plt.xlabel("Round")
        plt.ylabel("Repayment proportion")
        plt.title(f"{label.upper()} normalized repayment heatmap")
        plt.colorbar(label="Normalized frequency within round")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{label}_heatmap.png", dpi=220)
        plt.close()


def plot_q_values(max_results, fair_results, prefix="behavioral_bandit_adv"):
    max_q = max_results["q_history"]
    fair_q = fair_results["q_history"]
    actions = [0.0, 0.25, 0.5, 0.75, 1.0]

    plt.figure(figsize=(10, 5))
    for i, a in enumerate(actions):
        plt.plot(max_q[:, i], label=f"MAX repay {a:.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("MAX Trustee Q-values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_max_qvalues.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, a in enumerate(actions):
        plt.plot(fair_q[:, i], label=f"FAIR repay {a:.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("FAIR Trustee Q-values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_fair_qvalues.png", dpi=220)
    plt.close()


def save_round_logs(max_results, fair_results, prefix="behavioral_bandit_adv"):
    max_results["round_logs"].to_csv(f"{prefix}_max_round_logs.csv", index=False)
    fair_results["round_logs"].to_csv(f"{prefix}_fair_round_logs.csv", index=False)


if __name__ == "__main__":
    max_cfg = make_behavioral_bandit_config(world_type="independent", objective="max")
    fair_cfg = make_behavioral_bandit_config(world_type="independent", objective="fair")

    print("Training MAX bandit trustee against frozen behavioral investor...")
    max_results = run_training(
        max_cfg,
        n_episodes=3000,
        seed=0,
        investor_greedy=False,
    )
    summarize("MAX bandit trustee vs behavioral investor", max_results)

    print("\nTraining FAIR bandit trustee against frozen behavioral investor...")
    fair_results = run_training(
        fair_cfg,
        n_episodes=3000,
        seed=50000,
        investor_greedy=False,
    )
    summarize("FAIR bandit trustee vs behavioral investor", fair_results)

    plot_training_curves(max_results, fair_results, window=100, prefix="behavioral_bandit_adv")
    plot_round_profiles(max_results, fair_results, prefix="behavioral_bandit_adv")
    plot_repayment_heatmaps(max_results, fair_results, prefix="behavioral_bandit_adv")
    plot_q_values(max_results, fair_results, prefix="behavioral_bandit_adv")
    save_round_logs(max_results, fair_results, prefix="behavioral_bandit_adv")

    print("\nSaved:")
    print(" - behavioral_bandit_adv_totals.png")
    print(" - behavioral_bandit_adv_gap.png")
    print(" - behavioral_bandit_adv_behavior.png")
    print(" - behavioral_bandit_adv_round_repayment.png")
    print(" - behavioral_bandit_adv_round_investment.png")
    print(" - behavioral_bandit_adv_round_gains.png")
    print(" - behavioral_bandit_adv_max_heatmap.png")
    print(" - behavioral_bandit_adv_fair_heatmap.png")
    print(" - behavioral_bandit_adv_max_qvalues.png")
    print(" - behavioral_bandit_adv_fair_qvalues.png")
    print(" - behavioral_bandit_adv_max_round_logs.csv")
    print(" - behavioral_bandit_adv_fair_round_logs.csv")