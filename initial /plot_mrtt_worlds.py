import numpy as np
import matplotlib.pyplot as plt
from mrtt_world import MRTTWorld

def collect_histories(world_type, investor_type="reactive", n_episodes=500, seed=0):
    investor_totals = []
    trustee_totals = []
    investments_by_round = []
    repay_props_by_round = []
    investor_gain_by_round = []
    trustee_gain_by_round = []

    for ep in range(n_episodes):
        env = MRTTWorld(world_type=world_type, seed=seed + ep)
        out = env.simulate_episode(investor_type=investor_type)

        investor_totals.append(out["total_investor"])
        trustee_totals.append(out["total_trustee"])

        investments_by_round.append([row["investment"] for row in out["history"]])
        repay_props_by_round.append([row["repay_prop"] for row in out["history"]])
        investor_gain_by_round.append([row["investor_gain"] for row in out["history"]])
        trustee_gain_by_round.append([row["trustee_gain"] for row in out["history"]])

    return {
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "investments_by_round": np.array(investments_by_round),
        "repay_props_by_round": np.array(repay_props_by_round),
        "investor_gain_by_round": np.array(investor_gain_by_round),
        "trustee_gain_by_round": np.array(trustee_gain_by_round),
    }

def plot_line_with_band(x, arr, title, ylabel, filename, label):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_compare_lines(x, arr1, arr2, title, ylabel, filename, label1, label2):
    mean1 = arr1.mean(axis=0)
    std1 = arr1.std(axis=0)

    mean2 = arr2.mean(axis=0)
    std2 = arr2.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean1, label=label1)
    plt.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.2)

    plt.plot(x, mean2, label=label2)
    plt.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.2)

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_hist(arr1, arr2, title, xlabel, filename, label1, label2):
    plt.figure(figsize=(8, 5))
    plt.hist(arr1, bins=30, alpha=0.6, label=label1)
    plt.hist(arr2, bins=30, alpha=0.6, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

if __name__ == "__main__":
    n_episodes = 500
    rounds = np.arange(1, 11)

    ind = collect_histories(
        world_type="independent",
        investor_type="reactive",
        n_episodes=n_episodes,
        seed=0
    )

    coord = collect_histories(
        world_type="coordinated",
        investor_type="reactive",
        n_episodes=n_episodes,
        seed=10000
    )

    plot_hist(
        ind["investor_totals"],
        coord["investor_totals"],
        title="Investor Total Earnings",
        xlabel="Total Investor Earnings",
        filename="investor_totals_hist.png",
        label1="Independent",
        label2="Coordinated"
    )

    plot_hist(
        ind["trustee_totals"],
        coord["trustee_totals"],
        title="Trustee Total Earnings",
        xlabel="Total Trustee Earnings",
        filename="trustee_totals_hist.png",
        label1="Independent",
        label2="Coordinated"
    )

    plot_compare_lines(
        rounds,
        ind["investments_by_round"],
        coord["investments_by_round"],
        title="Average Investment by Round",
        ylabel="Investment",
        filename="investment_by_round.png",
        label1="Independent",
        label2="Coordinated"
    )

    plot_compare_lines(
        rounds,
        ind["repay_props_by_round"],
        coord["repay_props_by_round"],
        title="Average Repayment Proportion by Round",
        ylabel="Repayment Proportion",
        filename="repay_prop_by_round.png",
        label1="Independent",
        label2="Coordinated"
    )

    plot_compare_lines(
        rounds,
        ind["investor_gain_by_round"],
        coord["investor_gain_by_round"],
        title="Investor Gain by Round",
        ylabel="Investor Gain",
        filename="investor_gain_by_round.png",
        label1="Independent",
        label2="Coordinated"
    )

    plot_compare_lines(
        rounds,
        ind["trustee_gain_by_round"],
        coord["trustee_gain_by_round"],
        title="Trustee Gain by Round",
        ylabel="Trustee Gain",
        filename="trustee_gain_by_round.png",
        label1="Independent",
        label2="Coordinated"
    )

    print("Saved plots:")
    print(" - investor_totals_hist.png")
    print(" - trustee_totals_hist.png")
    print(" - investment_by_round.png")
    print(" - repay_prop_by_round.png")
    print(" - investor_gain_by_round.png")
    print(" - trustee_gain_by_round.png")