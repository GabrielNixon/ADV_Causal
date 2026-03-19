import numpy as np


class BehavioralContextualBanditAdversary:
    def __init__(
        self,
        objective="max",
        world_type="independent",
        epsilon=0.10,
        alpha=0.10,
        coord_weight=0.70,
        seed=0,
    ):
        self.objective = objective
        self.world_type = world_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.coord_weight = coord_weight
        self.rng = np.random.default_rng(seed)

        self.actions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

        self.global_q = np.zeros(len(self.actions), dtype=float)
        self.global_counts = np.zeros(len(self.actions), dtype=float)

        self.context_q = {}
        self.context_counts = {}

    def reset(self):
        pass

    def _investment_bucket(self, investment):
        investment = float(investment)
        if investment <= 2.5:
            return 0
        if investment <= 7.5:
            return 5
        if investment <= 12.5:
            return 10
        if investment <= 17.5:
            return 15
        return 20

    def _context_key(self, round_idx, investment):
        inv_bucket = self._investment_bucket(investment)
        return (int(round_idx), int(inv_bucket))

    def _ensure_context(self, key):
        if key not in self.context_q:
            self.context_q[key] = np.zeros(len(self.actions), dtype=float)
            self.context_counts[key] = np.zeros(len(self.actions), dtype=float)

    def _scores_for_context(self, key):
        self._ensure_context(key)

        if self.world_type == "independent":
            return self.context_q[key].copy()

        if self.world_type == "coordinated":
            return (
                self.coord_weight * self.global_q
                + (1.0 - self.coord_weight) * self.context_q[key]
            )

        raise ValueError(f"unknown world_type: {self.world_type}")

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        key = self._context_key(round_idx, investment)
        scores = self._scores_for_context(key)

        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(len(self.actions)))
        else:
            action_idx = int(np.argmax(scores))

        repay_prop = float(self.actions[action_idx])

        adv_details = {
            "action_idx": action_idx,
            "round_idx": int(round_idx),
            "investment": float(investment),
            "investment_bucket": self._investment_bucket(investment),
            "context_key": key,
            "repay_prop": repay_prop,
            "scores": scores.copy(),
        }
        return repay_prop, adv_details

    def observe_outcome(
        self,
        state_bundle=None,
        final_repay_prop=None,
        investor_gain=None,
        trustee_gain=None,
        next_context=None,
        done=None,
        **kwargs,
    ):
        action_idx = int(state_bundle["action_idx"])
        key = state_bundle["context_key"]
        self._ensure_context(key)

        investor_gain = float(investor_gain)
        trustee_gain = float(trustee_gain)

        if self.objective == "max":
            reward = trustee_gain
        elif self.objective == "fair":
            reward = -abs(trustee_gain - investor_gain)
        else:
            reward = trustee_gain

        self.context_counts[key][action_idx] += 1.0
        self.context_q[key][action_idx] += self.alpha * (
            reward - self.context_q[key][action_idx]
        )

        if self.world_type == "coordinated":
            self.global_counts[action_idx] += 1.0
            self.global_q[action_idx] += self.alpha * (
                reward - self.global_q[action_idx]
            )

    def greedy_action_for_context(self, round_idx, investment):
        key = self._context_key(round_idx, investment)
        scores = self._scores_for_context(key)
        best_idx = int(np.argmax(scores))
        return float(self.actions[best_idx]), scores.copy()

    def greedy_action_for_state(self, state=None):
        scores = self.global_q.copy() if self.world_type == "coordinated" else self.global_q.copy()
        if np.allclose(scores, 0.0):
            return float(self.actions[0]), scores
        best_idx = int(np.argmax(scores))
        return float(self.actions[best_idx]), scores.copy()