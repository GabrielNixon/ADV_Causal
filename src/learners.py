from config import INVEST_OPTIONS

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

    def act(self, round_idx, prev_repay_prop):
        return float(self.rng.choice(INVEST_OPTIONS))


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


def build_investor(investor_cfg, rng=None):
    investor_type = investor_cfg["type"]

    if investor_type == "fixed":
        return FixedInvestor(investor_cfg["value"])

    if investor_type == "random":
        if rng is None:
            raise ValueError("rng required for random investor")
        return RandomInvestor(rng)

    if investor_type == "reactive":
        return ReactiveInvestor(
            thresholds=investor_cfg.get("thresholds"),
            actions=investor_cfg.get("actions"),
        )

    raise ValueError(f"unknown investor_type: {investor_type}")