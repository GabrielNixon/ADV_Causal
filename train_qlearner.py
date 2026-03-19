import copy
import numpy as np
from collections import defaultdict

from config import INVEST_OPTIONS
from experiment_configs import CONFIGS
from mrtt_env import MRTTWorld

ACTION_LIST = list(INVEST_OPTIONS)
REPAY_STATES = [None, 0.0, 0.25, 0.5, 0.75, 1.0]

QLEARN_CONFIG = {
    "alpha": 0.10,
    "gamma": 0.95,
    "epsilon": 1.00,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "n_train": 5000,
    "n_eval": 500,
}


class QLearningInvestor:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
    ):
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
        best_idxs = np.flatnonzero(qvals == qvals.max())
        return int(self.rng.choice(best_idxs))

    def update(self, state, action_idx, reward, next_state, done):
        current_q = self.q_table[state][action_idx]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action_idx] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def build_env_from_config(exp_cfg, seed):
    return MRTTWorld(
        world_cfg=exp_cfg["world"],
        adv_cfg=exp_cfg["adversary"],
        seed=seed,
    )


def run_training_episode(agent, exp_cfg, seed):
    env = build_env_from_config(exp_cfg, seed=seed)
    total_reward = 0.0
    prev_repay_prop = None

    for t in range(env.horizon):
        state = agent.get_state(t, prev_repay_prop)
        action_idx = agent.choose_action(state)
        investment = float(ACTION_LIST[action_idx])

        out = env.step(investment, t)
        reward = out["investor_gain"]
        total_reward += reward

        next_prev_repay_prop = out["repay_prop"]
        done = (t == env.horizon - 1)

        if done:
            next_state = None
        else:
            next_state = agent.get_state(t + 1, next_prev_repay_prop)

        agent.update(state, action_idx, reward, next_state, done)
        prev_repay_prop = next_prev_repay_prop

    agent.decay_epsilon()
    return total_reward


def evaluate_agent(agent, exp_cfg, n_eval_episodes=200, seed=100000):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    returns = []
    mean_investments = []
    mean_repay_props = []

    for ep in range(n_eval_episodes):
        env = build_env_from_config(exp_cfg, seed=seed + ep)
        total_reward = 0.0
        prev_repay_prop = None
        investments = []
        repay_props = []

        for t in range(env.horizon):
            state = agent.get_state(t, prev_repay_prop)
            action_idx = agent.choose_action(state)
            investment = float(ACTION_LIST[action_idx])

            out = env.step(investment, t)
            total_reward += out["investor_gain"]

            investments.append(investment)
            repay_props.append(out["repay_prop"])
            prev_repay_prop = out["repay_prop"]

        returns.append(total_reward)
        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))

    agent.epsilon = old_epsilon

    return {
        "returns": np.array(returns),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
    }


def print_policy_snapshot(agent, title, horizon=10):
    print(f"\n{title}")
    for round_idx in range(horizon):
        parts = []
        for repay in REPAY_STATES:
            state = (round_idx, repay)
            qvals = agent.q_table[state]
            best_idx = int(np.argmax(qvals))
            best_action = ACTION_LIST[best_idx]
            parts.append(f"{repay}:{int(best_action)}")
        print(f"Round {round_idx}: " + " | ".join(parts))


def summarize_eval(label, result):
    print(f"\n{label}")
    print(f"Mean return       : {result['returns'].mean():.3f}")
    print(f"Std return        : {result['returns'].std():.3f}")
    print(f"Mean investment   : {result['mean_investments'].mean():.3f}")
    print(f"Mean repay prop   : {result['mean_repay_props'].mean():.3f}")


def train_agent(exp_cfg, q_cfg, seed_offset=0):
    agent = QLearningInvestor(
        alpha=q_cfg["alpha"],
        gamma=q_cfg["gamma"],
        epsilon=q_cfg["epsilon"],
        epsilon_min=q_cfg["epsilon_min"],
        epsilon_decay=q_cfg["epsilon_decay"],
        seed=seed_offset,
    )

    train_returns = []

    for ep in range(q_cfg["n_train"]):
        ep_return = run_training_episode(agent, exp_cfg, seed=seed_offset + ep)
        train_returns.append(ep_return)

        if (ep + 1) % 1000 == 0:
            print(
                f"Episode {ep + 1}/{q_cfg['n_train']} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"recent mean return={np.mean(train_returns[-100:]):.3f}"
            )

    return agent, np.array(train_returns)


def make_qlearn_eval_config(base_exp_cfg):
    exp_cfg = copy.deepcopy(base_exp_cfg)
    exp_cfg["investor"] = {"type": "qlearn"}
    return exp_cfg


if __name__ == "__main__":
    q_cfg = QLEARN_CONFIG

    train_ind_cfg = make_qlearn_eval_config(CONFIGS["baseline_independent"])
    train_coord_cfg = make_qlearn_eval_config(CONFIGS["baseline_coordinated"])

    print("Training Q-learner in baseline_independent...")
    agent_ind, train_returns_ind = train_agent(
        exp_cfg=train_ind_cfg,
        q_cfg=q_cfg,
        seed_offset=0,
    )

    print("\nTraining Q-learner in baseline_coordinated...")
    agent_coord, train_returns_coord = train_agent(
        exp_cfg=train_coord_cfg,
        q_cfg=q_cfg,
        seed_offset=50000,
    )

    print("\n================ SAME-WORLD EVALUATION ================")
    eval_ind_on_ind = evaluate_agent(
        agent_ind,
        train_ind_cfg,
        n_eval_episodes=q_cfg["n_eval"],
        seed=200000,
    )
    eval_coord_on_coord = evaluate_agent(
        agent_coord,
        train_coord_cfg,
        n_eval_episodes=q_cfg["n_eval"],
        seed=300000,
    )

    summarize_eval("Independent-trained on Independent", eval_ind_on_ind)
    summarize_eval("Coordinated-trained on Coordinated", eval_coord_on_coord)

    print("\n================ CROSS-WORLD EVALUATION ================")
    eval_ind_on_coord = evaluate_agent(
        agent_ind,
        train_coord_cfg,
        n_eval_episodes=q_cfg["n_eval"],
        seed=400000,
    )
    eval_coord_on_ind = evaluate_agent(
        agent_coord,
        train_ind_cfg,
        n_eval_episodes=q_cfg["n_eval"],
        seed=500000,
    )

    summarize_eval("Independent-trained on Coordinated", eval_ind_on_coord)
    summarize_eval("Coordinated-trained on Independent", eval_coord_on_ind)

    print_policy_snapshot(
        agent_ind,
        title="Greedy policy snapshot: agent trained in Independent world",
        horizon=train_ind_cfg["world"]["horizon"],
    )

    print_policy_snapshot(
        agent_coord,
        title="Greedy policy snapshot: agent trained in Coordinated world",
        horizon=train_coord_cfg["world"]["horizon"],
    )