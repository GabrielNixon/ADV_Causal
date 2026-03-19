import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mrtt_env import MRTTWorld
from behavioral_investor import BehavioralRNNInvestor
from behavioral_contextual_bandit_adversary import BehavioralContextualBanditAdversary


def make_behavioral_contextual_config(world_type="independent", objective="max"):
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
            "coord_weight": 0.70,
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
    return BehavioralContextualBanditAdversary(
        objective=exp_cfg["adversary"]["objective"],
        world_type=exp_cfg["world"]["type"],
        epsilon=exp_cfg["adversary"].get("epsilon", 0.10),
        alpha=exp_cfg["adversary"].get("alpha", 0.10),
        coord_weight=exp_cfg["adversary"].get("coord_weight", 0.70),
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
                "world": exp_cfg["world"]["type"],
                "objective": exp_cfg["adversary"]["objective"],
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

        if (ep + 1) % 300 == 0:
            probe_low = trustee.greedy_action_for_context(round_idx=0, investment=5.0)[0]
            probe_mid = trustee.greedy_action_for_context(round_idx=4, investment=10.0)[0]
            probe_high = trustee.greedy_action_for_context(round_idx=8, investment=20.0)[0]

            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"{exp_cfg['world']['type']} | {exp_cfg['adversary']['objective']} | "
                f"Investor mean(last100)={np.mean(investor_totals[-100:]):.3f} | "
                f"Trustee mean(last100)={np.mean(trustee_totals[-100:]):.3f} | "
                f"Gap mean(last100)={np.mean(earning_gaps[-100:]):.3f} | "
                f"Repay mean(last100)={np.mean(mean_repay_props[-100:]):.3f} | "
                f"Investment mean(last100)={np.mean(mean_investments[-100:]):.3f} | "
                f"Probe repay (r1,i5)={probe_low:.2f} | "
                f"(r5,i10)={probe_mid:.2f} | "
                f"(r9,i20)={probe_high:.2f}"
            )

    return {
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "earning_gaps": np.array(earning_gaps),
        "mean_repay_props": np.array(mean_repay_props),
        "mean_investments": np.array(mean_investments),
        "round_logs": pd.DataFrame(round_logs),
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

    trustee = results["trustee"]
    p1 = trustee.greedy_action_for_context(round_idx=0, investment=5.0)[0]
    p2 = trustee.greedy_action_for_context(round_idx=4, investment=10.0)[0]
    p3 = trustee.greedy_action_for_context(round_idx=8, investment=20.0)[0]
    print(f"Greedy repay probes : (r1,i5)={p1:.2f}, (r5,i10)={p2:.2f}, (r9,i20)={p3:.2f}")


def plot_round_profiles(results_dict, prefix="behavioral_contextual"):
    plt.figure(figsize=(11, 5))
    for label, results in results_dict.items():
        df = results["round_logs"]
        round_summary = df.groupby("round")[["repay_prop"]].mean().reset_index()
        plt.plot(round_summary["round"], round_summary["repay_prop"], marker="o", label=label)
    plt.xlabel("Round")
    plt.ylabel("Mean repayment proportion")
    plt.title("Repayment by Round Across Worlds and Objectives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_repayment.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5))
    for label, results in results_dict.items():
        df = results["round_logs"]
        round_summary = df.groupby("round")[["investment"]].mean().reset_index()
        plt.plot(round_summary["round"], round_summary["investment"], marker="o", label=label)
    plt.xlabel("Round")
    plt.ylabel("Mean investment")
    plt.title("Investment by Round Across Worlds and Objectives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_investment.png", dpi=220)
    plt.close()


def plot_summary_bars(results_dict, prefix="behavioral_contextual"):
    labels = list(results_dict.keys())
    x = np.arange(len(labels))

    trustee_means = [results_dict[k]["trustee_totals"].mean() for k in labels]
    investor_means = [results_dict[k]["investor_totals"].mean() for k in labels]
    gap_means = [results_dict[k]["earning_gaps"].mean() for k in labels]
    repay_means = [results_dict[k]["mean_repay_props"].mean() for k in labels]

    plt.figure(figsize=(11, 5))
    plt.bar(x, trustee_means)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Trustee total mean")
    plt.title("Trustee Totals Across Worlds and Objectives")
    plt.tight_layout()
    plt.savefig(f"{prefix}_trustee_totals.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, investor_means)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Investor total mean")
    plt.title("Investor Totals Across Worlds and Objectives")
    plt.tight_layout()
    plt.savefig(f"{prefix}_investor_totals.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, gap_means)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Absolute earning gap")
    plt.title("Gap Across Worlds and Objectives")
    plt.tight_layout()
    plt.savefig(f"{prefix}_gap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, repay_means)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Mean repayment proportion")
    plt.title("Repayment Across Worlds and Objectives")
    plt.tight_layout()
    plt.savefig(f"{prefix}_repay_mean.png", dpi=220)
    plt.close()


def plot_heatmaps(results_dict, prefix="behavioral_contextual"):
    for label, results in results_dict.items():
        safe_label = label.replace(" ", "_").replace("→", "_").replace("/", "_")
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
        plt.title(f"{label} normalized repayment heatmap")
        plt.colorbar(label="Normalized frequency within round")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{safe_label}_heatmap.png", dpi=220)
        plt.close()


def save_logs(results_dict, prefix="behavioral_contextual"):
    for label, results in results_dict.items():
        safe_label = label.replace(" ", "_").replace("→", "_").replace("/", "_")
        results["round_logs"].to_csv(f"{prefix}_{safe_label}_round_logs.csv", index=False)


if __name__ == "__main__":
    configs = {
        "Ind-MAX": make_behavioral_contextual_config(world_type="independent", objective="max"),
        "Ind-FAIR": make_behavioral_contextual_config(world_type="independent", objective="fair"),
        "Coord-MAX": make_behavioral_contextual_config(world_type="coordinated", objective="max"),
        "Coord-FAIR": make_behavioral_contextual_config(world_type="coordinated", objective="fair"),
    }

    seeds = {
        "Ind-MAX": 0,
        "Ind-FAIR": 50000,
        "Coord-MAX": 100000,
        "Coord-FAIR": 150000,
    }

    results = {}

    for label, cfg in configs.items():
        print(f"\nTraining {label}...")
        results[label] = run_training(
            cfg,
            n_episodes=3000,
            seed=seeds[label],
            investor_greedy=False,
        )
        summarize(label, results[label])

    plot_round_profiles(results, prefix="behavioral_contextual")
    plot_summary_bars(results, prefix="behavioral_contextual")
    plot_heatmaps(results, prefix="behavioral_contextual")
    save_logs(results, prefix="behavioral_contextual")

    print("\nSaved:")
    print(" - behavioral_contextual_round_repayment.png")
    print(" - behavioral_contextual_round_investment.png")
    print(" - behavioral_contextual_trustee_totals.png")
    print(" - behavioral_contextual_investor_totals.png")
    print(" - behavioral_contextual_gap.png")
    print(" - behavioral_contextual_repay_mean.png")
    print(" - behavioral_contextual_*_heatmap.png")
    print(" - behavioral_contextual_*_round_logs.csv")