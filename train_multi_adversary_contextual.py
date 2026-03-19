import numpy as np
import pandas as pd
import random

from mrtt_env import MRTTWorld
from behavioral_investor import BehavioralRNNInvestor
from behavioral_contextual_bandit_adversary import BehavioralContextualBanditAdversary


class NullAdversary:
    def reset(self):
        pass


def make_cfg(world_type):
    return {
        "world": {
            "type": world_type,
            "horizon": 10,
            "endowment": 20.0,
        },
        "adversary": {
            "epsilon": 0.10,
            "alpha": 0.10,
            "coord_weight": 0.70,
        },
    }


def build_adv(kind, world_type, cfg, seed):
    return BehavioralContextualBanditAdversary(
        objective=kind,
        world_type=world_type,
        epsilon=cfg["adversary"]["epsilon"],
        alpha=cfg["adversary"]["alpha"],
        coord_weight=cfg["adversary"]["coord_weight"],
        seed=seed,
    )


def run_experiment(world_type, pairing, n_episodes=3000, seed=0):
    cfg = make_cfg(world_type)

    adv1 = build_adv(pairing[0], world_type, cfg, seed)
    adv2 = build_adv(pairing[1], world_type, cfg, seed + 1)
    adversaries = [adv1, adv2]

    env = MRTTWorld(
        world_cfg=cfg["world"],
        seed=seed,
        adversary=NullAdversary(),
    )

    investor = BehavioralRNNInvestor(
        model_path="behavioral_rnn.pt",
        horizon=cfg["world"]["horizon"],
        endowment=cfg["world"]["endowment"],
        greedy=False,
        seed=seed,
    )

    logs = []
    investor_totals = []
    trustee_totals = []

    for ep in range(n_episodes):
        env.reset()
        investor.reset()

        ep_records = []
        total_investor = 0.0
        total_trustee = 0.0

        for t in range(env.horizon):
            adv_id = random.choice([0, 1])
            adv = adversaries[adv_id]

            investment = investor.act(t)

            repay_prop, adv_details = adv.get_repay_prop(t, investment, None)
            out = env.step_manual(investment, repay_prop, t)

            investor.observe_outcome(
                action=investment,
                reward=out["investor_gain"],
                repay_prop=repay_prop,
            )

            total_investor += out["investor_gain"]
            total_trustee += out["trustee_gain"]

            ep_records.append({
                "adv_id": adv_id,
                "adv": adv,
                "state": adv_details,
                "investor_gain": out["investor_gain"],
                "trustee_gain": out["trustee_gain"],
            })

            logs.append({
                "episode": ep,
                "round": t + 1,
                "world": world_type,
                "pair": f"{pairing[0]}-{pairing[1]}",
                "active_adv": adv_id,
                "adv_type": pairing[adv_id],
                "investment": float(investment),
                "repay_prop": float(repay_prop),
                "investor_gain": float(out["investor_gain"]),
                "trustee_gain": float(out["trustee_gain"]),
            })

        for record in ep_records:
            adv = record["adv"]
            state = record["state"]

            if adv.objective == "max":
                reward_signal_trustee = total_trustee
                reward_signal_investor = total_investor
            else:
                reward_signal_trustee = total_trustee
                reward_signal_investor = total_investor

            adv.observe_outcome(
                state_bundle=state,
                investor_gain=reward_signal_investor,
                trustee_gain=reward_signal_trustee,
            )

        investor_totals.append(total_investor)
        trustee_totals.append(total_trustee)

        if (ep + 1) % 500 == 0:
            print(
                f"{world_type} | {pairing} | Ep {ep+1} | "
                f"Inv={np.mean(investor_totals[-100:]):.2f} | "
                f"Tru={np.mean(trustee_totals[-100:]):.2f}"
            )

    return {
        "logs": pd.DataFrame(logs),
        "investor": np.array(investor_totals),
        "trustee": np.array(trustee_totals),
    }


if __name__ == "__main__":
    setups = [
        ("independent", ("max", "max")),
        ("independent", ("fair", "fair")),
        ("independent", ("max", "fair")),
        ("coordinated", ("max", "max")),
        ("coordinated", ("fair", "fair")),
        ("coordinated", ("max", "fair")),
    ]

    all_results = {}

    for i, (world, pair) in enumerate(setups):
        print(f"\nRunning {world} | {pair}")
        res = run_experiment(world, pair, seed=1000 * i)
        key = f"{world}_{pair[0]}_{pair[1]}"
        all_results[key] = res

        print(f"{key}")
        print("Investor mean:", res["investor"].mean())
        print("Trustee mean :", res["trustee"].mean())

        res["logs"].to_csv(f"{key}_logs.csv", index=False)

    print("\nDONE. All 6 experiments saved.")