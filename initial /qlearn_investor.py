import numpy as np
from collections import defaultdict
from config import INVEST_OPTIONS
from mrtt_env import MRTTWorld

ACTION_LIST = list(INVEST_OPTIONS)

class QLearningInvestor:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(len(ACTION_LIST), dtype=float))

    def get_state(self, round_idx, prev_repay_prop):
        return (round_idx, prev_repay_prop)

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTION_LIST)))
        qvals = self.q_table[state]
        best_actions = np.flatnonzero(qvals == qvals.max())
        return int(self.rng.choice(best_actions))

    def update(self, state, action_idx, reward, next_state, done):
        q_sa = self.q_table[state][action_idx]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_training_episode(agent, world_type, seed):
    env = MRTTWorld(world_type=world_type, seed=seed)
    prev_repay_prop = None
    total_reward = 0.0

    for t in range(env.horizon):
        state = agent.get_state(t, prev_repay_prop)
        action_idx = agent.choose_action(state)
        investment = float(ACTION_LIST[action_idx])

        out = env.step(investment, t)
        reward = out["investor_gain"]
        total_reward += reward

        next_prev_repay_prop = out["repay_prop"]
        done = (t == env.horizon - 1)
        next_state = None if done else agent.get_state(t + 1, next_prev_repay_prop)

        agent.update(state, action_idx, reward, next_state, done)
        prev_repay_prop = next_prev_repay_prop

    agent.decay_epsilon()
    return total_reward


def evaluate_agent(agent, world_type, n_eval_episodes=200, seed=100000):
    returns = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for ep in range(n_eval_episodes):
        env = MRTTWorld(world_type=world_type, seed=seed + ep)
        prev_repay_prop = None
        total_reward = 0.0

        for t in range(env.horizon):
            state = agent.get_state(t, prev_repay_prop)
            action_idx = agent.choose_action(state)
            investment = float(ACTION_LIST[action_idx])

            out = env.step(investment, t)
            total_reward += out["investor_gain"]
            prev_repay_prop = out["repay_prop"]

        returns.append(total_reward)

    agent.epsilon = old_epsilon
    return np.array(returns)


if __name__ == "__main__":
    n_train = 5000

    agent_ind = QLearningInvestor(seed=0)
    for ep in range(n_train):
        run_training_episode(agent_ind, "independent", ep)

    agent_coord = QLearningInvestor(seed=1)
    for ep in range(n_train):
        run_training_episode(agent_coord, "coordinated", 50000 + ep)

    eval_ind = evaluate_agent(agent_ind, "independent", n_eval_episodes=500, seed=200000)
    eval_coord = evaluate_agent(agent_coord, "coordinated", n_eval_episodes=500, seed=300000)

    print("\nEVALUATION RESULTS")
    print(f"Independent-trained mean: {eval_ind.mean():.3f}")
    print(f"Independent-trained std : {eval_ind.std():.3f}")
    print(f"Coordinated-trained mean: {eval_coord.mean():.3f}")
    print(f"Coordinated-trained std : {eval_coord.std():.3f}")