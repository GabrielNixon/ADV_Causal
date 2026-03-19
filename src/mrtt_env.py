import numpy as np
from adversaries import build_adversary


class MRTTWorld:
    def __init__(self, world_cfg, adv_cfg=None, seed=None, adversary=None):
        self.world_type = world_cfg["type"]
        self.horizon = world_cfg["horizon"]
        self.endowment = world_cfg["endowment"]
        self.rng = np.random.default_rng(seed)

        if adversary is not None:
            self.adversary = adversary
        else:
            self.adversary = build_adversary(self.world_type, adv_cfg)

        self.reset()

    def reset(self):
        if self.adversary is not None:
            self.adversary.reset()
        self.prev_investment = None
        self.prev_repay_prop = None

    def step(self, investment, round_idx):
        context = {
            "prev_investment": self.prev_investment,
            "prev_repay_prop": self.prev_repay_prop,
        }

        tripled = 3.0 * investment
        repay_prop, adv_details = self.adversary.get_repay_prop(
            round_idx,
            investment,
            self.rng,
            context=context,
        )

        repayment = tripled * repay_prop
        investor_gain = self.endowment - investment + repayment
        trustee_gain = tripled - repayment

        done = (round_idx == self.horizon - 1)

        next_context = {
            "round_idx": round_idx + 1,
            "investment": investment,
            "prev_investment": investment,
            "prev_repay_prop": repay_prop,
        }

        self.adversary.observe_outcome(
            state_bundle=adv_details,
            final_repay_prop=repay_prop,
            investor_gain=investor_gain,
            trustee_gain=trustee_gain,
            next_context=next_context,
            done=done,
        )

        self.prev_investment = investment
        self.prev_repay_prop = repay_prop

        return {
            "round": round_idx,
            "investment": investment,
            "tripled_amount": tripled,
            "repay_prop": repay_prop,
            "repayment": repayment,
            "investor_gain": investor_gain,
            "trustee_gain": trustee_gain,
            "adv_details": adv_details,
            "done": done,
        }

    def step_manual(self, investment, repay_prop, round_idx):
        tripled = 3.0 * investment
        repayment = tripled * repay_prop

        investor_gain = self.endowment - investment + repayment
        trustee_gain = tripled - repayment

        done = (round_idx == self.horizon - 1)

        self.prev_investment = investment
        self.prev_repay_prop = repay_prop

        return {
            "round": round_idx,
            "investment": investment,
            "tripled_amount": tripled,
            "repay_prop": repay_prop,
            "repayment": repayment,
            "investor_gain": investor_gain,
            "trustee_gain": trustee_gain,
            "done": done,
        }