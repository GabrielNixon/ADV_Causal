import numpy as np
from mrtt_env import MRTTWorld
from learners import build_investor

def simulate_episode(exp_cfg, seed=None):
    env = MRTTWorld(
        world_cfg=exp_cfg["world"],
        adv_cfg=exp_cfg["adversary"],
        seed=seed,
    )
    investor = build_investor(exp_cfg["investor"], rng=env.rng)

    history = []
    total_investor = 0.0
    total_trustee = 0.0
    prev_repay_prop = None

    for t in range(env.horizon):
        investment = investor.act(t, prev_repay_prop)
        out = env.step(investment, t)

        history.append(out)
        total_investor += out["investor_gain"]
        total_trustee += out["trustee_gain"]
        prev_repay_prop = out["repay_prop"]

    return {
        "history": history,
        "total_investor": total_investor,
        "total_trustee": total_trustee,
        "world_type": exp_cfg["world"]["type"],
        "investor_type": exp_cfg["investor"]["type"],
    }


def simulate_many(exp_cfg, n_episodes=500, seed=0):
    investor_totals = []
    trustee_totals = []
    mean_investments = []
    mean_repay_props = []

    for ep in range(n_episodes):
        out = simulate_episode(exp_cfg, seed=seed + ep)

        investor_totals.append(out["total_investor"])
        trustee_totals.append(out["total_trustee"])
        mean_investments.append(np.mean([row["investment"] for row in out["history"]]))
        mean_repay_props.append(np.mean([row["repay_prop"] for row in out["history"]]))

    return {
        "investor_totals": np.array(investor_totals),
        "trustee_totals": np.array(trustee_totals),
        "mean_repay_props": np.array(mean_repay_props),
        "mean_investments": np.array(mean_investments),
        "fair_scores": np.array(fair_scores),
        "mid_scores": np.array(mid_scores),
        "max_scores": np.array(max_scores),
        "persistent_adversary": persistent_adversary,
    }