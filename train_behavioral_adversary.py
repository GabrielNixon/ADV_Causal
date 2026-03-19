import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adversaries import build_single_adversary
from mrtt_env import MRTTWorld
from behavioral_investor import BehavioralRNNInvestor


class TrusteeAdapter:
    def __init__(self, base_adversary):
        self.base = base_adversary
        self._last_repay_prop = None
        self._last_adv_details = None

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()
        self._last_repay_prop = None
        self._last_adv_details = None

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        repay_prop, adv_details = self.base.get_repay_prop(
            round_idx,
            investment,
            rng,
            context=context,
        )
        self._last_repay_prop = repay_prop
        self._last_adv_details = adv_details
        return repay_prop, adv_details

    def observe_outcome(
        self,
        state_bundle=None,
        final_repay_prop=None,
        investor_gain=None,
        trustee_gain=None,
        next_context=None,
        done=None,
    ):
        if not hasattr(self.base, "observe_outcome"):
            return

        method = self.base.observe_outcome
        sig = inspect.signature(method)
        params = sig.parameters

        candidate_kwargs = {
            "state_bundle": state_bundle,
            "adv_details": state_bundle,
            "final_repay_prop": final_repay_prop,
            "repay_prop": final_repay_prop,
            "investor_gain": investor_gain,
            "trustee_gain": trustee_gain,
            "reward": trustee_gain,
            "next_context": next_context,
            "done": done,
        }

        filtered = {}
        for name in params:
            if name == "self":
                continue
            if name in candidate_kwargs:
                filtered[name] = candidate_kwargs[name]

        try:
            method(**filtered)
        except TypeError:
            fallback_kwargs = {}
            if "reward" in params:
                fallback_kwargs["reward"] = trustee_gain
            if "repay_prop" in params:
                fallback_kwargs["repay_prop"] = final_repay_prop
            if "done" in params:
                fallback_kwargs["done"] = done
            method(**fallback_kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_behavioral_adversary_config(world_type="independent", objective="max"):
    return {
        "world": {
            "type": world_type,
            "horizon": 10,
            "endowment": 20.0,
        },
        "adversary": {
            "type": "rl",
            "objective": objective,
            "alpha": 0.10,
            "gamma": 0.95,
            "epsilon": 0.10,
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
    objective = exp_cfg["adversary"]["objective"]

    if objective == "fair":
        adv_name = "fair"
    elif objective == "max":
        adv_name = "max"
    elif objective == "mixed":
        adv_name = "mid"
    else:
        raise ValueError(f"unknown objective: {objective}")

    trustee = build_single_adversary(
        adv_name,
        exp_cfg["adversary"],
        seed=seed,
    )
    return TrusteeAdapter(trustee)


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
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Investor mean(last100)={np.mean(investor_totals[-100:]):.3f} | "
                f"Trustee mean(last100)={np.mean(trustee_totals[-100:]):.3f} | "
                f"Gap mean(last100)={np.mean(earning_gaps[-100:]):.3f} | "
                f"Repay mean(last100)={np.mean(mean_repay_props[-100:]):.3f} | "
                f"Investment mean(last100)={np.mean(mean_investments[-100:]):.3f}"
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


def plot_training_curves(max_results, fair_results, window=100, prefix="behavioral_adv"):
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
    plt.title("Behavioral Investor: Trustee vs Investor Totals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_totals.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(max_gap, label="MAX gap")
    plt.plot(fair_gap, label="FAIR gap")
    plt.xlabel("Episode")
    plt.ylabel("Absolute earning gap")
    plt.title("Behavioral Investor: Absolute Earning Gap")
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
    plt.title("Behavioral Investor: Repayment and Investment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_behavior.png", dpi=220)
    plt.close()


def plot_round_profiles(max_results, fair_results, prefix="behavioral_adv"):
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


def plot_repayment_heatmaps(max_results, fair_results, prefix="behavioral_adv"):
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


def save_round_logs(max_results, fair_results, prefix="behavioral_adv"):
    max_results["round_logs"].to_csv(f"{prefix}_max_round_logs.csv", index=False)
    fair_results["round_logs"].to_csv(f"{prefix}_fair_round_logs.csv", index=False)


if __name__ == "__main__":
    max_cfg = make_behavioral_adversary_config(world_type="independent", objective="max")
    fair_cfg = make_behavioral_adversary_config(world_type="independent", objective="fair")

    print("Training MAX adversary against frozen behavioral investor...")
    max_results = run_training(
        max_cfg,
        n_episodes=3000,
        seed=0,
        investor_greedy=False,
    )
    summarize("MAX adversary vs behavioral investor", max_results)

    print("\nTraining FAIR adversary against frozen behavioral investor...")
    fair_results = run_training(
        fair_cfg,
        n_episodes=3000,
        seed=50000,
        investor_greedy=False,
    )
    summarize("FAIR adversary vs behavioral investor", fair_results)

    plot_training_curves(max_results, fair_results, window=100, prefix="behavioral_adv")
    plot_round_profiles(max_results, fair_results, prefix="behavioral_adv")
    plot_repayment_heatmaps(max_results, fair_results, prefix="behavioral_adv")
    save_round_logs(max_results, fair_results, prefix="behavioral_adv")

    print("\nSaved:")
    print(" - behavioral_adv_totals.png")
    print(" - behavioral_adv_gap.png")
    print(" - behavioral_adv_behavior.png")
    print(" - behavioral_adv_round_repayment.png")
    print(" - behavioral_adv_round_investment.png")
    print(" - behavioral_adv_round_gains.png")
    print(" - behavioral_adv_max_heatmap.png")
    print(" - behavioral_adv_fair_heatmap.png")
    print(" - behavioral_adv_max_round_logs.csv")
    print(" - behavioral_adv_fair_round_logs.csv")