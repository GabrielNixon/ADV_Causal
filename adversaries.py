import numpy as np
from collections import defaultdict
from config import REPAY_OPTIONS


class BaseAdversary:
    def reset(self):
        pass

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        raise NotImplementedError

    def observe_outcome(self, *args, **kwargs):
        pass


class FixedPolicyAdversary(BaseAdversary):
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def get_score(self, round_idx, investment, rng):
        base = self.params["base"]
        time_slope = self.params["time_slope"]
        invest_slope = self.params["invest_slope"]
        noise_std = self.params["noise_std"]

        score = (
            base
            + time_slope * round_idx
            + invest_slope * (investment / 5.0)
            + rng.normal(0.0, noise_std)
        )
        return float(np.clip(score, 0.0, 1.0))

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        score = self.get_score(round_idx, investment, rng)
        repay_prop = float(REPAY_OPTIONS[np.argmin(np.abs(REPAY_OPTIONS - score))])
        return repay_prop, {"mode": "fixed", "score": score}


class QLearningTrusteeAdversary(BaseAdversary):
    def __init__(
        self,
        name,
        objective="max",
        alpha=0.1,
        gamma=0.95,
        epsilon=0.10,
        seed=0,
    ):
        self.name = name
        self.objective = objective
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(len(REPAY_OPTIONS), dtype=float))

    def reset(self):
        pass

    def get_state(self, round_idx, investment, context=None):
        prev_investment = None if context is None else context.get("prev_investment")
        prev_repay = None if context is None else context.get("prev_repay_prop")
        return (round_idx, investment, prev_investment, prev_repay)

    def choose_action_idx(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(REPAY_OPTIONS)))
        qvals = self.q_table[state]
        best = np.flatnonzero(qvals == qvals.max())
        return int(self.rng.choice(best))

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        state = self.get_state(round_idx, investment, context)
        action_idx = self.choose_action_idx(state)
        repay_prop = float(REPAY_OPTIONS[action_idx])

        info = {
            "mode": "rl",
            "state": state,
            "action_idx": action_idx,
            "repay_prop": repay_prop,
            "objective": self.objective,
        }
        return repay_prop, info

    def compute_adversary_reward(self, investor_gain, trustee_gain, repay_prop):
        gap = abs(trustee_gain - investor_gain)
        exploit_bonus = 1.0 - repay_prop

        if self.objective == "fair":
            return -gap

        if self.objective == "mixed":
            return 0.5 * trustee_gain - 0.5 * gap

        if self.objective == "max":
            return trustee_gain + 5.0 * exploit_bonus

        raise ValueError(f"unknown objective: {self.objective}")

    def observe_outcome(self, state, action_idx, reward, next_state, done):
        q_sa = self.q_table[state][action_idx]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.alpha * (target - q_sa)

    def greedy_action_for_state(self, state):
        qvals = self.q_table[state]
        best_idx = int(np.argmax(qvals))
        return float(REPAY_OPTIONS[best_idx]), qvals.copy()


class IndependentCompositeAdversary(BaseAdversary):
    def __init__(self, adversaries, aggregation="mean"):
        self.adversaries = adversaries
        self.aggregation = aggregation

    def reset(self):
        for adv in self.adversaries:
            adv.reset()

    def aggregate_scores(self, repay_props, rng):
        if self.aggregation == "mean":
            val = float(np.mean(repay_props))
        elif self.aggregation == "min":
            val = float(np.min(repay_props))
        elif self.aggregation == "max":
            val = float(np.max(repay_props))
        elif self.aggregation == "sample":
            val = float(rng.choice(repay_props))
        else:
            raise ValueError(f"unknown aggregation: {self.aggregation}")

        return float(REPAY_OPTIONS[np.argmin(np.abs(REPAY_OPTIONS - val))])

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        repay_props = []
        details = {}

        for adv in self.adversaries:
            repay_prop, info = adv.get_repay_prop(round_idx, investment, rng, context=context)
            repay_props.append(repay_prop)
            details[adv.name] = info

        final_repay_prop = self.aggregate_scores(repay_props, rng)
        return final_repay_prop, details

    def observe_outcome(self, state_bundle, final_repay_prop, investor_gain, trustee_gain, next_context, done):
        for adv in self.adversaries:
            if adv.name not in state_bundle:
                continue

            info = state_bundle[adv.name]
            if info.get("mode") != "rl":
                continue

            state = info["state"]
            action_idx = info["action_idx"]
            next_state = None if done else adv.get_state(
                next_context["round_idx"],
                next_context["investment"],
                next_context,
            )
            reward = adv.compute_adversary_reward(
                investor_gain,
                trustee_gain,
                info["repay_prop"],
            )
            adv.observe_outcome(state, action_idx, reward, next_state, done)


class CoordinatedCompositeAdversary(BaseAdversary):
    def __init__(self, adversaries, aggregation="mean", shared_regime_prob=0.85, seed=0):
        self.adversaries = adversaries
        self.aggregation = aggregation
        self.shared_regime_prob = shared_regime_prob
        self.rng = np.random.default_rng(seed)
        self.current_regime = 0

    def reset(self):
        self.current_regime = int(self.rng.integers(0, len(REPAY_OPTIONS)))
        for adv in self.adversaries:
            adv.reset()

    def maybe_shift_regime(self):
        if self.rng.random() > self.shared_regime_prob:
            self.current_regime = int(self.rng.integers(0, len(REPAY_OPTIONS)))

    def aggregate_scores(self, repay_props, rng):
        if self.aggregation == "mean":
            val = float(np.mean(repay_props))
        elif self.aggregation == "min":
            val = float(np.min(repay_props))
        elif self.aggregation == "max":
            val = float(np.max(repay_props))
        elif self.aggregation == "sample":
            val = float(rng.choice(repay_props))
        else:
            raise ValueError(f"unknown aggregation: {self.aggregation}")

        return float(REPAY_OPTIONS[np.argmin(np.abs(REPAY_OPTIONS - val))])

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        self.maybe_shift_regime()

        repay_props = []
        details = {}

        for adv in self.adversaries:
            repay_prop, info = adv.get_repay_prop(round_idx, investment, rng, context=context)

            regime_target = float(REPAY_OPTIONS[self.current_regime])
            adjusted = 0.7 * repay_prop + 0.3 * regime_target
            adjusted = float(REPAY_OPTIONS[np.argmin(np.abs(REPAY_OPTIONS - adjusted))])

            repay_props.append(adjusted)
            info["coordinated_adjusted"] = adjusted
            info["shared_regime_idx"] = self.current_regime
            details[adv.name] = info

        final_repay_prop = self.aggregate_scores(repay_props, rng)
        return final_repay_prop, details

    def observe_outcome(self, state_bundle, final_repay_prop, investor_gain, trustee_gain, next_context, done):
        for adv in self.adversaries:
            if adv.name not in state_bundle:
                continue

            info = state_bundle[adv.name]
            if info.get("mode") != "rl":
                continue

            state = info["state"]
            action_idx = info["action_idx"]
            next_state = None if done else adv.get_state(
                next_context["round_idx"],
                next_context["investment"],
                next_context,
            )
            reward = adv.compute_adversary_reward(
                investor_gain,
                trustee_gain,
                info["repay_prop"],
            )
            adv.observe_outcome(state, action_idx, reward, next_state, done)


def build_single_adversary(name, cfg, seed=0):
    adv_type = cfg.get("type", "fixed")

    if adv_type == "fixed":
        return FixedPolicyAdversary(name, cfg)

    if adv_type == "rl":
        return QLearningTrusteeAdversary(
            name=name,
            objective=cfg.get("objective", "max"),
            alpha=cfg.get("alpha", 0.1),
            gamma=cfg.get("gamma", 0.95),
            epsilon=cfg.get("epsilon", 0.10),
            seed=seed,
        )

    raise ValueError(f"unknown adversary type: {adv_type}")


def build_adversary(world_type, adv_cfg):
    base_adversaries = [
        build_single_adversary("fair", adv_cfg["fair"], seed=1),
        build_single_adversary("mid", adv_cfg["mid"], seed=2),
        build_single_adversary("max", adv_cfg["max"], seed=3),
    ]

    worlds_cfg = adv_cfg.get("worlds", {})

    if world_type == "independent":
        ind_cfg = worlds_cfg.get("independent", {})
        return IndependentCompositeAdversary(
            base_adversaries,
            aggregation=ind_cfg.get("aggregation", "mean"),
        )

    if world_type == "coordinated":
        coord_cfg = worlds_cfg.get("coordinated", {})
        return CoordinatedCompositeAdversary(
            base_adversaries,
            aggregation=coord_cfg.get("aggregation", "mean"),
            shared_regime_prob=coord_cfg.get("shared_regime_prob", 0.85),
            seed=coord_cfg.get("seed", 123),
        )

    raise ValueError(f"unknown world_type: {world_type}")