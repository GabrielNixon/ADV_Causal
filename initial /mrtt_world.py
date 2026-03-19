import numpy as np

INVEST_OPTIONS = np.array([0, 5, 10, 15, 20], dtype=float)
REPAY_OPTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

class MRTTWorld:
    def __init__(self, world_type="independent", horizon=10, endowment=20.0, seed=None):
        assert world_type in ["independent", "coordinated"]
        self.world_type = world_type
        self.horizon = horizon
        self.endowment = endowment
        self.rng = np.random.default_rng(seed)

        self.adv_names = ["fair", "mid", "max"]
        self.base_strength = {
            "fair": 0.75,
            "mid": 0.50,
            "max": 0.20,
        }
        self.shared_regime = 0.0

    def reset(self):
        self.t = 0
        self.shared_regime = 0.0

    def investor_policy(self, investor_type, prev_repay_prop=None):
        if investor_type == "fixed_high":
            return 20.0
        if investor_type == "fixed_mid":
            return 10.0
        if investor_type == "random":
            return float(self.rng.choice(INVEST_OPTIONS))
        if investor_type == "reactive":
            if prev_repay_prop is None:
                return 10.0
            if prev_repay_prop >= 0.75:
                return 20.0
            if prev_repay_prop >= 0.50:
                return 15.0
            if prev_repay_prop >= 0.25:
                return 10.0
            return 5.0
        raise ValueError("unknown investor_type")

    def _sample_shared_regime(self):
        return self.rng.normal(0.0, 0.15)

    def _adversary_score(self, adv_name, round_idx, investment, shared_signal=0.0):
        base = self.base_strength[adv_name]

        if adv_name == "max":
            time_effect = -0.03 * round_idx
        elif adv_name == "mid":
            time_effect = -0.01 * round_idx
        else:
            time_effect = 0.0

        invest_effect = 0.01 * (investment / 5.0)

        if self.world_type == "independent":
            noise = self.rng.normal(0.0, 0.12)
            score = base + time_effect + invest_effect + noise
        else:
            local_noise = self.rng.normal(0.0, 0.04)
            score = base + time_effect + invest_effect + shared_signal + local_noise

        return np.clip(score, 0.0, 1.0)

    def _final_repayment_prop(self, round_idx, investment):
        if self.world_type == "coordinated":
            shared_signal = self._sample_shared_regime()
        else:
            shared_signal = 0.0

        scores = []
        for adv in self.adv_names:
            s = self._adversary_score(
                adv_name=adv,
                round_idx=round_idx,
                investment=investment,
                shared_signal=shared_signal
            )
            scores.append(s)

        avg_score = float(np.mean(scores))
        repay_prop = float(REPAY_OPTIONS[np.argmin(np.abs(REPAY_OPTIONS - avg_score))])

        return repay_prop, dict(zip(self.adv_names, scores))

    def step(self, investment, round_idx):
        tripled = 3.0 * investment
        repay_prop, adv_scores = self._final_repayment_prop(round_idx, investment)
        repayment = tripled * repay_prop

        investor_gain = self.endowment - investment + repayment
        trustee_gain = tripled - repayment

        return {
            "round": round_idx,
            "investment": investment,
            "tripled_amount": tripled,
            "repay_prop": repay_prop,
            "repayment": repayment,
            "investor_gain": investor_gain,
            "trustee_gain": trustee_gain,
            "adv_scores": adv_scores,
        }

    def simulate_episode(self, investor_type="reactive"):
        self.reset()

        history = []
        prev_repay_prop = None
        total_investor = 0.0
        total_trustee = 0.0

        for t in range(self.horizon):
            investment = self.investor_policy(investor_type, prev_repay_prop)
            out = self.step(investment, t)
            history.append(out)

            total_investor += out["investor_gain"]
            total_trustee += out["trustee_gain"]
            prev_repay_prop = out["repay_prop"]

        return {
            "world_type": self.world_type,
            "investor_type": investor_type,
            "history": history,
            "total_investor": total_investor,
            "total_trustee": total_trustee,
        }


if __name__ == "__main__":
    for world_type in ["independent", "coordinated"]:
        env = MRTTWorld(world_type=world_type, seed=42)
        episode = env.simulate_episode(investor_type="reactive")
        print("\nWORLD:", world_type)
        print("Investor total:", episode["total_investor"])
        print("Trustee total:", episode["total_trustee"])
        for row in episode["history"]:
            print(
                row["round"],
                row["investment"],
                row["repay_prop"],
                row["investor_gain"],
                row["trustee_gain"]
            )