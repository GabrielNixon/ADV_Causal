import numpy as np
from mrtt_world import MRTTWorld

def run_many(world_type, investor_type="reactive", n_episodes=500, seed=0):
    investor_totals = []
    trustee_totals = []
    mean_investments = []
    mean_repay_props = []

    for ep in range(n_episodes):
        env = MRTTWorld(world_type=world_type, seed=seed + ep)
        out = env.simulate_episode(investor_type=investor_type)

        investor_totals.append(out["total_investor"])
        trustee_totals.append(out["total_trustee"])

        investments = [row["investment"] for row in out["history"]]
        repay_props = [row["repay_prop"] for row in out["history"]]

        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))

    return {
        "world_type": world_type,
        "investor_type": investor_type,
        "n_episodes": n_episodes,
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
    }

def summarize(results):
    print(f"\nWORLD: {results['world_type']}")
    print(f"Investor type: {results['investor_type']}")
    print(f"Episodes: {results['n_episodes']}")
    print(f"Investor total mean: {results['investor_totals'].mean():.3f}")
    print(f"Investor total std : {results['investor_totals'].std():.3f}")
    print(f"Trustee total mean : {results['trustee_totals'].mean():.3f}")
    print(f"Trustee total std  : {results['trustee_totals'].std():.3f}")
    print(f"Mean investment    : {results['mean_investments'].mean():.3f}")
    print(f"Mean repay prop    : {results['mean_repay_props'].mean():.3f}")

if __name__ == "__main__":
    n_episodes = 500
    investor_type = "reactive"

    res_ind = run_many(
        world_type="independent",
        investor_type=investor_type,
        n_episodes=n_episodes,
        seed=0
    )

    res_coord = run_many(
        world_type="coordinated",
        investor_type=investor_type,
        n_episodes=n_episodes,
        seed=10000
    )

    summarize(res_ind)
    summarize(res_coord)

    diff_investor = res_ind["investor_totals"].mean() - res_coord["investor_totals"].mean()
    diff_trustee = res_coord["trustee_totals"].mean() - res_ind["trustee_totals"].mean()

    print("\nCOMPARISON")
    print(f"Investor advantage (independent - coordinated): {diff_investor:.3f}")
    print(f"Trustee advantage (coordinated - independent): {diff_trustee:.3f}")