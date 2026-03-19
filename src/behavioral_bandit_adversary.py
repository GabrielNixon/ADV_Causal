import numpy as np


class BehavioralBanditAdversary:
    def __init__(self, objective="max", epsilon=0.10, alpha=0.10, seed=0):
        self.objective = objective
        self.epsilon = epsilon
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.actions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        self.q_values = np.zeros(len(self.actions), dtype=float)
        self.counts = np.zeros(len(self.actions), dtype=float)

    def reset(self):
        pass

    def get_repay_prop(self, round_idx, investment, rng, context=None):
        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(len(self.actions)))
        else:
            action_idx = int(np.argmax(self.q_values))

        repay_prop = float(self.actions[action_idx])

        adv_details = {
            "action_idx": action_idx,
            "round_idx": round_idx,
            "investment": float(investment),
            "repay_prop": repay_prop,
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

        if self.objective == "max":
            reward = float(trustee_gain)
        elif self.objective == "fair":
            reward = -abs(float(trustee_gain) - float(investor_gain))
        else:
            reward = float(trustee_gain)

        self.counts[action_idx] += 1.0
        self.q_values[action_idx] += self.alpha * (reward - self.q_values[action_idx])

    def greedy_action_for_state(self, state=None):
        best_idx = int(np.argmax(self.q_values))
        return float(self.actions[best_idx]), self.q_values.copy()