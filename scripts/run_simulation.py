import numpy as np
from mrtt_env import MRTTWorld
from learners import build_investor

def simulate_episode(world_type="independent", investor_type="reactive", horizon=10, seed=None):
    env = MRTTWorld(world_type=world_type, horizon=horizon, seed=seed)
    investor = build_investor(investor_type, rng=env.rng)

    history = []
    prev_repay_prop = None
    total_investor = 0.0
    total_trustee = 0.0

    for t in range(env.horizon):
        investment = investor.act(t, prev_repay_prop)
        out = env.step(investment, t)

        history.append(out)
        total_investor += out["investor_gain"]
        total_trustee += out["trustee_gain"]
        prev_repay_prop = out["repay_prop"]

    return {
        "world_type": world_type,
        "investor_type": investor_type,
        "history": history,
        "total_investor": total_investor,
        "total_trustee": total_trustee,
    }


def simulate_many(world_type="independent", investor_type="reactive", n_episodes=500, seed=0):
    investor_totals = []
    trustee_totals = []
    mean_investments = []
    mean_repay_props = []

    for ep in range(n_episodes):
        out = simulate_episode(
            world_type=world_type,
            investor_type=investor_type,
            seed=seed + ep
        )

        investor_totals.append(out["total_investor"])
        trustee_totals.append(out["total_trustee"])
        mean_investments.append(np.mean([row["investment"] for row in out["history"]]))
        mean_repay_props.append(np.mean([row["repay_prop"] for row in out["history"]]))

    return {
        "world_type": world_type,
        "investor_type": investor_type,
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
    }


if __name__ == "__main__":
    for world_type in ["independent", "coordinated"]:
        out = simulate_episode(world_type=world_type, investor_type="reactive", seed=42)
        print(f"\nWORLD: {world_type}")
        print("Investor total:", out["total_investor"])
        print("Trustee total:", out["total_trustee"])
        for row in out["history"]:
            print(
                row["round"],
                row["investment"],
                row["repay_prop"],
                row["investor_gain"],
                row["trustee_gain"]
            )