import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from adversaries import build_adversary
from mrtt_env import MRTTWorld


class BaseInvestor:
    def reset(self):
        pass

    def act(self, round_idx, prev_repay_prop):
        raise NotImplementedError


class FixedInvestor(BaseInvestor):
    def __init__(self, value):
        self.value = float(value)

    def act(self, round_idx, prev_repay_prop):
        return self.value


class RandomInvestor(BaseInvestor):
    def __init__(self, rng):
        self.rng = rng
        self.actions = np.array([0, 5, 10, 15, 20], dtype=float)

    def act(self, round_idx, prev_repay_prop):
        return float(self.rng.choice(self.actions))


class ReactiveInvestor(BaseInvestor):
    def __init__(self, thresholds=None, actions=None):
        self.thresholds = thresholds or [0.25, 0.50, 0.75]
        self.actions = actions or {
            "start": 10.0,
            "low": 5.0,
            "midlow": 10.0,
            "midhigh": 15.0,
            "high": 20.0,
        }

    def act(self, round_idx, prev_repay_prop):
        if prev_repay_prop is None:
            return self.actions["start"]
        if prev_repay_prop >= self.thresholds[2]:
            return self.actions["high"]
        if prev_repay_prop >= self.thresholds[1]:
            return self.actions["midhigh"]
        if prev_repay_prop >= self.thresholds[0]:
            return self.actions["midlow"]
        return self.actions["low"]


def make_rl_adversary_config(world_type="coordinated", investor_type="reactive"):
    return {
        "world": {
            "type": world_type,
            "horizon": 10,
            "endowment": 20.0,
        },
        "adversary": {
            "fair": {
                "type": "rl",
                "objective": "fair",
                "alpha": 0.10,
                "gamma": 0.95,
                "epsilon": 0.10,
            },
            "mid": {
                "type": "rl",
                "objective": "mixed",
                "alpha": 0.10,
                "gamma": 0.95,
                "epsilon": 0.10,
            },
            "max": {
                "type": "rl",
                "objective": "max",
                "alpha": 0.10,
                "gamma": 0.95,
                "epsilon": 0.10,
            },
            "worlds": {
                "independent": {
                    "aggregation": "sample",
                },
                "coordinated": {
                    "aggregation": "min",
                    "shared_regime_prob": 0.90,
                    "seed": 123,
                },
            },
        },
        "investor": {
            "type": investor_type,
        },
    }


def build_investor_from_config(investor_cfg, rng):
    investor_type = investor_cfg["type"]

    if investor_type == "reactive":
        return ReactiveInvestor(
            thresholds=investor_cfg.get("thresholds"),
            actions=investor_cfg.get("actions"),
        )
    if investor_type == "random":
        return RandomInvestor(rng)
    if investor_type == "fixed":
        return FixedInvestor(investor_cfg["value"])

    raise ValueError(f"unknown investor type: {investor_type}")


def run_training(exp_cfg, n_episodes=3000, seed=0):
    persistent_adversary = build_adversary(exp_cfg["world"]["type"], exp_cfg["adversary"])
    env = MRTTWorld(
        world_cfg=exp_cfg["world"],
        seed=seed,
        adversary=persistent_adversary,
    )
    investor = build_investor_from_config(exp_cfg["investor"], rng=env.rng)

    investor_totals = []
    trustee_totals = []
    mean_repay_props = []
    mean_investments = []

    fair_scores = []
    mid_scores = []
    max_scores = []

    visited_states = {
        "fair": defaultdict(int),
        "mid": defaultdict(int),
        "max": defaultdict(int),
    }

    round_logs = []

    for ep in range(n_episodes):
        env.reset()
        prev_repay_prop = None
        total_investor = 0.0
        total_trustee = 0.0
        investments = []
        repay_props = []
        fair_ep = []
        mid_ep = []
        max_ep = []

        for t in range(env.horizon):
            investment = investor.act(t, prev_repay_prop)
            out = env.step(investment, t)

            total_investor += out["investor_gain"]
            total_trustee += out["trustee_gain"]
            investments.append(investment)
            repay_props.append(out["repay_prop"])

            details = out.get("adv_details", {})

            for adv_name in ["fair", "mid", "max"]:
                if adv_name in details and "state" in details[adv_name]:
                    visited_states[adv_name][details[adv_name]["state"]] += 1

            fair_val = np.nan
            mid_val = np.nan
            max_val = np.nan

            if "fair" in details and "repay_prop" in details["fair"]:
                fair_val = details["fair"]["repay_prop"]
                fair_ep.append(fair_val)

            if "mid" in details and "repay_prop" in details["mid"]:
                mid_val = details["mid"]["repay_prop"]
                mid_ep.append(mid_val)

            if "max" in details and "repay_prop" in details["max"]:
                max_val = details["max"]["repay_prop"]
                max_ep.append(max_val)

            round_logs.append({
                "episode": ep,
                "round": t + 1,
                "world": exp_cfg["world"]["type"],
                "investment": float(investment),
                "repay_prop": float(out["repay_prop"]),
                "investor_gain": float(out["investor_gain"]),
                "trustee_gain": float(out["trustee_gain"]),
                "fair_repay_prop": fair_val,
                "mid_repay_prop": mid_val,
                "max_repay_prop": max_val,
            })

            prev_repay_prop = out["repay_prop"]

        investor_totals.append(total_investor)
        trustee_totals.append(total_trustee)
        mean_repay_props.append(np.mean(repay_props))
        mean_investments.append(np.mean(investments))

        fair_scores.append(np.mean(fair_ep) if fair_ep else np.nan)
        mid_scores.append(np.mean(mid_ep) if mid_ep else np.nan)
        max_scores.append(np.mean(max_ep) if max_ep else np.nan)

        if (ep + 1) % 300 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Investor mean(last100)={np.mean(investor_totals[-100:]):.3f} | "
                f"Trustee mean(last100)={np.mean(trustee_totals[-100:]):.3f} | "
                f"Repay mean(last100)={np.mean(mean_repay_props[-100:]):.3f} | "
                f"Fair={np.nanmean(fair_scores[-100:]):.3f} | "
                f"Mid={np.nanmean(mid_scores[-100:]):.3f} | "
                f"Max={np.nanmean(max_scores[-100:]):.3f}"
            )

    return {
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "mean_repay_props": np.array(mean_repay_props),
        "mean_investments": np.array(mean_investments),
        "fair_scores": np.array(fair_scores),
        "mid_scores": np.array(mid_scores),
        "max_scores": np.array(max_scores),
        "persistent_adversary": persistent_adversary,
        "visited_states": visited_states,
        "round_logs": pd.DataFrame(round_logs),
    }


def moving_average(x, window=100):
    x = np.asarray(x)
    if len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_training_curves(results, prefix="rl_adv", window=100):
    inv = moving_average(results["investor_totals"], window)
    tru = moving_average(results["trustee_totals"], window)
    rep = moving_average(results["mean_repay_props"], window)
    invst = moving_average(results["mean_investments"], window)
    fair = moving_average(results["fair_scores"], window)
    mid = moving_average(results["mid_scores"], window)
    maxv = moving_average(results["max_scores"], window)

    plt.figure(figsize=(9, 5))
    plt.plot(inv, label="Investor total")
    plt.plot(tru, label="Trustee total")
    plt.xlabel("Episode")
    plt.ylabel("Episode total (moving average)")
    plt.title("RL Adversary Training: Investor vs Trustee Totals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_totals.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rep, label="Mean repayment proportion")
    plt.plot(invst, label="Mean investment")
    plt.xlabel("Episode")
    plt.ylabel("Moving average")
    plt.title("RL Adversary Training: Repayment and Investment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_behavior.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(fair, label="Fair adversary")
    plt.plot(mid, label="Mid adversary")
    plt.plot(maxv, label="Max adversary")
    plt.xlabel("Episode")
    plt.ylabel("Chosen repayment")
    plt.title("RL Adversary Policies Over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_adversary_policies.png", dpi=200)
    plt.close()


def plot_round_profiles(results, prefix="rl_adv"):
    df = results["round_logs"]

    round_summary = (
        df.groupby("round")[["investment", "repay_prop", "investor_gain", "trustee_gain"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(round_summary["round"], round_summary["investment"], marker="o", label="Investment")
    plt.plot(round_summary["round"], round_summary["repay_prop"] * 20.0, marker="o", label="Repayment proportion × 20")
    plt.xlabel("Round")
    plt.ylabel("Mean value")
    plt.title("Mean Investment and Repayment by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_profiles.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(round_summary["round"], round_summary["investor_gain"], marker="o", label="Investor gain")
    plt.plot(round_summary["round"], round_summary["trustee_gain"], marker="o", label="Trustee gain")
    plt.xlabel("Round")
    plt.ylabel("Mean gain")
    plt.title("Mean Gains by Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_round_gains.png", dpi=200)
    plt.close()


def plot_repayment_heatmap(results, prefix="rl_adv"):
    df = results["round_logs"]

    heat = (
        df.groupby(["repay_prop", "round"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    plt.imshow(heat.values, aspect="auto", origin="lower")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), [str(x) for x in heat.index])
    plt.xlabel("Round")
    plt.ylabel("Repayment proportion")
    plt.title("Repayment Frequency Heatmap")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(f"{prefix}_repayment_heatmap.png", dpi=200)
    plt.close()


def plot_adversary_heatmaps(results, prefix="rl_adv"):
    df = results["round_logs"]

    for col, name in [
        ("fair_repay_prop", "fair"),
        ("mid_repay_prop", "mid"),
        ("max_repay_prop", "max"),
    ]:
        sub = df.dropna(subset=[col])

        heat = (
            sub.groupby([col, "round"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        plt.figure(figsize=(8, 5))
        plt.imshow(heat.values, aspect="auto", origin="lower")
        plt.xticks(range(len(heat.columns)), heat.columns)
        plt.yticks(range(len(heat.index)), [str(x) for x in heat.index])
        plt.xlabel("Round")
        plt.ylabel("Repayment proportion")
        plt.title(f"{name.capitalize()} Adversary Repayment Heatmap")
        plt.colorbar(label="Count")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{name}_heatmap.png", dpi=200)
        plt.close()


def plot_summary_bars(results, prefix="rl_adv"):
    labels = [
        "Investor total",
        "Trustee total",
        "Mean investment",
        "Mean repay %",
    ]
    means = [
        results["investor_totals"].mean(),
        results["trustee_totals"].mean(),
        results["mean_investments"].mean(),
        results["mean_repay_props"].mean() * 100.0,
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(labels)), means)
    plt.xticks(np.arange(len(labels)), labels, rotation=20)
    plt.ylabel("Mean value")
    plt.title("Summary Statistics")
    plt.tight_layout()
    plt.savefig(f"{prefix}_summary_bars.png", dpi=200)
    plt.close()


def plot_paper_style_panel(results, prefix="rl_adv"):
    df = results["round_logs"]

    round_summary = (
        df.groupby("round")[["investment", "repay_prop", "investor_gain", "trustee_gain"]]
        .mean()
        .reset_index()
    )

    fair_heat = (
        df.dropna(subset=["fair_repay_prop"])
        .groupby(["fair_repay_prop", "round"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.plot(round_summary["round"], round_summary["investment"], marker="o", label="Investment")
    plt.plot(round_summary["round"], round_summary["repay_prop"] * 20.0, marker="o", label="Repayment proportion × 20")
    plt.xlabel("Round")
    plt.ylabel("Mean value")
    plt.title("A. Mean Investment and Repayment")
    plt.legend()

    plt.subplot(2, 2, 2)
    vals = [
        results["investor_totals"].mean(),
        results["trustee_totals"].mean(),
        results["mean_investments"].mean(),
        results["mean_repay_props"].mean() * 100.0,
    ]
    labs = ["Investor total", "Trustee total", "Mean invest", "Mean repay %"]
    plt.bar(np.arange(len(labs)), vals)
    plt.xticks(np.arange(len(labs)), labs, rotation=20)
    plt.ylabel("Mean value")
    plt.title("B. Summary Statistics")

    plt.subplot(2, 2, 3)
    plt.plot(round_summary["round"], round_summary["investor_gain"], marker="o", label="Investor gain")
    plt.plot(round_summary["round"], round_summary["trustee_gain"], marker="o", label="Trustee gain")
    plt.xlabel("Round")
    plt.ylabel("Mean gain")
    plt.title("C. Mean Gains by Round")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.imshow(fair_heat.values, aspect="auto", origin="lower")
    plt.xticks(range(len(fair_heat.columns)), fair_heat.columns)
    plt.yticks(range(len(fair_heat.index)), [str(x) for x in fair_heat.index])
    plt.xlabel("Round")
    plt.ylabel("Repayment proportion")
    plt.title("D. Fair Adversary Heatmap")
    plt.colorbar(label="Count")

    plt.tight_layout()
    plt.savefig(f"{prefix}_paper_panel.png", dpi=200)
    plt.close()


def save_round_logs(results, filename):
    results["round_logs"].to_csv(filename, index=False)


def summarize_results(label, results):
    print(f"\n{label}")
    print(f"Investor total mean : {results['investor_totals'].mean():.3f}")
    print(f"Investor total std  : {results['investor_totals'].std():.3f}")
    print(f"Trustee total mean  : {results['trustee_totals'].mean():.3f}")
    print(f"Trustee total std   : {results['trustee_totals'].std():.3f}")
    print(f"Mean repay prop     : {results['mean_repay_props'].mean():.3f}")
    print(f"Mean investment     : {results['mean_investments'].mean():.3f}")
    print(f"Fair action mean    : {np.nanmean(results['fair_scores']):.3f}")
    print(f"Mid action mean     : {np.nanmean(results['mid_scores']):.3f}")
    print(f"Max action mean     : {np.nanmean(results['max_scores']):.3f}")


def print_top_visited_policies(persistent_adversary, visited_states, top_k=15):
    print("\nTop visited adversary states and greedy actions")

    adv_map = {adv.name: adv for adv in persistent_adversary.adversaries}

    for adv_name, state_counts in visited_states.items():
        print(f"\nAdversary: {adv_name}")
        adv = adv_map[adv_name]
        ranked = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

        for state, count in ranked:
            best_action, qvals = adv.greedy_action_for_state(state)
            print(
                f"state={state} | visits={count} | greedy_action={best_action} | "
                f"qvals={np.round(qvals, 3)}"
            )


def summarize_policy_on_visited_states(persistent_adversary, visited_states):
    print("\nMean greedy action over visited states (weighted by visits)")

    adv_map = {adv.name: adv for adv in persistent_adversary.adversaries}

    for adv_name, state_counts in visited_states.items():
        adv = adv_map[adv_name]

        weighted_sum = 0.0
        total = 0

        for state, count in state_counts.items():
            best_action, _ = adv.greedy_action_for_state(state)
            weighted_sum += best_action * count
            total += count

        mean_action = weighted_sum / total if total > 0 else float("nan")
        print(f"{adv_name}: {mean_action:.3f}")


if __name__ == "__main__":
    coordinated_cfg = make_rl_adversary_config("coordinated", "reactive")
    independent_cfg = make_rl_adversary_config("independent", "reactive")

    print("Training RL adversaries in coordinated world...")
    coord_results = run_training(coordinated_cfg, n_episodes=3000, seed=0)
    summarize_results("Coordinated RL-adversary world", coord_results)
    plot_training_curves(coord_results, prefix="rl_adv_coord", window=100)
    plot_round_profiles(coord_results, prefix="rl_adv_coord")
    plot_repayment_heatmap(coord_results, prefix="rl_adv_coord")
    plot_adversary_heatmaps(coord_results, prefix="rl_adv_coord")
    plot_summary_bars(coord_results, prefix="rl_adv_coord")
    plot_paper_style_panel(coord_results, prefix="rl_adv_coord")
    save_round_logs(coord_results, "rl_adv_coord_round_logs.csv")
    print_top_visited_policies(
        coord_results["persistent_adversary"],
        coord_results["visited_states"],
        top_k=15,
    )
    summarize_policy_on_visited_states(
        coord_results["persistent_adversary"],
        coord_results["visited_states"],
    )

    print("\nTraining RL adversaries in independent world...")
    ind_results = run_training(independent_cfg, n_episodes=3000, seed=50000)
    summarize_results("Independent RL-adversary world", ind_results)
    plot_training_curves(ind_results, prefix="rl_adv_ind", window=100)
    plot_round_profiles(ind_results, prefix="rl_adv_ind")
    plot_repayment_heatmap(ind_results, prefix="rl_adv_ind")
    plot_adversary_heatmaps(ind_results, prefix="rl_adv_ind")
    plot_summary_bars(ind_results, prefix="rl_adv_ind")
    plot_paper_style_panel(ind_results, prefix="rl_adv_ind")
    save_round_logs(ind_results, "rl_adv_ind_round_logs.csv")
    print_top_visited_policies(
        ind_results["persistent_adversary"],
        ind_results["visited_states"],
        top_k=15,
    )
    summarize_policy_on_visited_states(
        ind_results["persistent_adversary"],
        ind_results["visited_states"],
    )

    print("\nSaved plots:")
    print(" - rl_adv_coord_totals.png")
    print(" - rl_adv_coord_behavior.png")
    print(" - rl_adv_coord_adversary_policies.png")
    print(" - rl_adv_coord_round_profiles.png")
    print(" - rl_adv_coord_round_gains.png")
    print(" - rl_adv_coord_repayment_heatmap.png")
    print(" - rl_adv_coord_fair_heatmap.png")
    print(" - rl_adv_coord_mid_heatmap.png")
    print(" - rl_adv_coord_max_heatmap.png")
    print(" - rl_adv_coord_summary_bars.png")
    print(" - rl_adv_coord_paper_panel.png")
    print(" - rl_adv_coord_round_logs.csv")
    print(" - rl_adv_ind_totals.png")
    print(" - rl_adv_ind_behavior.png")
    print(" - rl_adv_ind_adversary_policies.png")
    print(" - rl_adv_ind_round_profiles.png")
    print(" - rl_adv_ind_round_gains.png")
    print(" - rl_adv_ind_repayment_heatmap.png")
    print(" - rl_adv_ind_fair_heatmap.png")
    print(" - rl_adv_ind_mid_heatmap.png")
    print(" - rl_adv_ind_max_heatmap.png")
    print(" - rl_adv_ind_summary_bars.png")
    print(" - rl_adv_ind_paper_panel.png")
    print(" - rl_adv_ind_round_logs.csv")