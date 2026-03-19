import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import INVEST_OPTIONS, DEFAULT_HORIZON, DEFAULT_ENDOWMENT
from experiment_configs import CONFIGS
from mrtt_env import MRTTWorld

ACTION_LIST = list(INVEST_OPTIONS)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env_from_config(exp_cfg, seed):
    return MRTTWorld(
        world_cfg=exp_cfg["world"],
        adv_cfg=exp_cfg["adversary"],
        seed=seed,
    )


def make_rnn_eval_config(base_exp_cfg):
    exp_cfg = copy.deepcopy(base_exp_cfg)
    exp_cfg["investor"] = {"type": "rnn"}
    return exp_cfg


class RNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        logits = self.policy_head(out[:, -1, :])
        return logits, hidden


class RNNInvestor:
    def __init__(
        self,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        device=None,
        seed=0,
        horizon=DEFAULT_HORIZON,
        endowment=DEFAULT_ENDOWMENT,
    ):
        set_all_seeds(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.horizon = horizon
        self.endowment = endowment

        self.model = RNNPolicy(
            input_dim=4,
            hidden_dim=hidden_dim,
            n_actions=len(ACTION_LIST),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def encode_step(self, round_idx, prev_repay_prop, prev_action, prev_reward):
        round_scaled = float(round_idx) / max(self.horizon - 1, 1)

        repay_scaled = -1.0 if prev_repay_prop is None else float(prev_repay_prop)

        if prev_action is None:
            action_scaled = -1.0
        else:
            action_scaled = float(prev_action) / float(self.endowment)

        if prev_reward is None:
            reward_scaled = 0.0
        else:
            reward_scaled = float(prev_reward) / float(self.endowment)

        return [round_scaled, repay_scaled, action_scaled, reward_scaled]

    def build_prefix_tensor(self, history):
        x = torch.tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        return x

    def choose_action(self, history, greedy=False):
        x = self.build_prefix_tensor(history)
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            action_idx = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action_idx] + 1e-8)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            return action_idx, log_prob, entropy

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action.item(), log_prob, entropy

    def discounted_returns(self, rewards):
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_episode(self, log_probs, rewards, entropies):
        returns = self.discounted_returns(rewards)

        policy_loss = 0.0
        entropy_term = 0.0

        for log_prob, ret, ent in zip(log_probs, returns, entropies):
            policy_loss = policy_loss - log_prob * ret
            entropy_term = entropy_term + ent

        loss = policy_loss - self.entropy_coef * entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())


RNN_CONFIG = {
    "hidden_dim": 64,
    "lr": 1e-3,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "n_train": 3000,
    "n_eval": 200,
}


def run_training_episode(agent, exp_cfg, seed):
    env = build_env_from_config(exp_cfg, seed=seed)

    total_reward = 0.0
    history = []
    rewards = []
    log_probs = []
    entropies = []

    prev_repay_prop = None
    prev_action = None
    prev_reward = None

    for t in range(env.horizon):
        step_input = agent.encode_step(
            round_idx=t,
            prev_repay_prop=prev_repay_prop,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        history.append(step_input)

        action_idx, log_prob, entropy = agent.choose_action(history, greedy=False)
        investment = float(ACTION_LIST[action_idx])

        out = env.step(investment, t)
        reward = float(out["investor_gain"])

        total_reward += reward
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)

        prev_repay_prop = float(out["repay_prop"])
        prev_action = investment
        prev_reward = reward

    loss = agent.update_episode(log_probs, rewards, entropies)
    return total_reward, loss


def train_agent(exp_cfg, rnn_cfg, seed_offset=0):
    horizon = exp_cfg["world"]["horizon"]
    endowment = exp_cfg["world"]["endowment"]

    agent = RNNInvestor(
        hidden_dim=rnn_cfg["hidden_dim"],
        lr=rnn_cfg["lr"],
        gamma=rnn_cfg["gamma"],
        entropy_coef=rnn_cfg["entropy_coef"],
        seed=seed_offset,
        horizon=horizon,
        endowment=endowment,
    )

    returns = []
    losses = []

    for ep in range(rnn_cfg["n_train"]):
        ep_return, ep_loss = run_training_episode(agent, exp_cfg, seed=seed_offset + ep)
        returns.append(ep_return)
        losses.append(ep_loss)

        if (ep + 1) % 300 == 0:
            print(
                f"Episode {ep+1}/{rnn_cfg['n_train']} | "
                f"Return mean(last100)={np.mean(returns[-100:]):.3f} | "
                f"Loss mean(last100)={np.mean(losses[-100:]):.3f}"
            )

    return agent, np.array(returns), np.array(losses)


def evaluate_agent(agent, exp_cfg, n_eval_episodes=200, seed=100000):
    returns = []
    mean_investments = []
    mean_repay_props = []
    round_logs = []

    for ep in range(n_eval_episodes):
        env = build_env_from_config(exp_cfg, seed=seed + ep)

        total_reward = 0.0
        history = []
        investments = []
        repay_props = []

        prev_repay_prop = None
        prev_action = None
        prev_reward = None

        for t in range(env.horizon):
            step_input = agent.encode_step(
                round_idx=t,
                prev_repay_prop=prev_repay_prop,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )
            history.append(step_input)

            action_idx, _, _ = agent.choose_action(history, greedy=True)
            investment = float(ACTION_LIST[action_idx])

            out = env.step(investment, t)
            reward = float(out["investor_gain"])

            total_reward += reward
            investments.append(investment)
            repay_props.append(float(out["repay_prop"]))

            round_logs.append({
                "episode": ep,
                "round": t + 1,
                "world": exp_cfg["world"]["type"],
                "investment": investment,
                "repay_prop": float(out["repay_prop"]),
                "investor_gain": reward,
                "trustee_gain": float(out["trustee_gain"]),
                "investor_total_running": float(total_reward),
                "prev_action_used": -1.0 if prev_action is None else float(prev_action),
                "prev_reward_used": 0.0 if prev_reward is None else float(prev_reward),
                "prev_repay_used": -1.0 if prev_repay_prop is None else float(prev_repay_prop),
            })

            prev_repay_prop = float(out["repay_prop"])
            prev_action = investment
            prev_reward = reward

        returns.append(total_reward)
        mean_investments.append(np.mean(investments))
        mean_repay_props.append(np.mean(repay_props))

    return {
        "returns": np.array(returns),
        "mean_investments": np.array(mean_investments),
        "mean_repay_props": np.array(mean_repay_props),
        "round_logs": pd.DataFrame(round_logs),
    }


def moving_average(x, window=100):
    x = np.asarray(x)
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_curves(train_returns_ind, train_returns_coord, train_losses_ind, train_losses_coord, window=100):
    ma_ret_ind = moving_average(train_returns_ind, window=window)
    ma_ret_coord = moving_average(train_returns_coord, window=window)
    ma_loss_ind = moving_average(train_losses_ind, window=window)
    ma_loss_coord = moving_average(train_losses_coord, window=window)

    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(ma_ret_ind) + 1), ma_ret_ind, label="Train in Independent")
    plt.plot(np.arange(1, len(ma_ret_coord) + 1), ma_ret_coord, label="Train in Coordinated")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return (moving average)")
    plt.title("RNN Investor Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rnn_learning_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(ma_loss_ind) + 1), ma_loss_ind, label="Train in Independent")
    plt.plot(np.arange(1, len(ma_loss_coord) + 1), ma_loss_coord, label="Train in Coordinated")
    plt.xlabel("Episode")
    plt.ylabel("Loss (moving average)")
    plt.title("RNN Investor Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rnn_training_loss.png", dpi=220)
    plt.close()


def summarize(label, result):
    print(f"\n{label}")
    print(f"Mean return     : {result['returns'].mean():.3f}")
    print(f"Std return      : {result['returns'].std():.3f}")
    print(f"Mean investment : {result['mean_investments'].mean():.3f}")
    print(f"Mean repay prop : {result['mean_repay_props'].mean():.3f}")


if __name__ == "__main__":
    rnn_cfg = RNN_CONFIG

    ind_cfg = make_rnn_eval_config(CONFIGS["baseline_independent"])
    coord_cfg = make_rnn_eval_config(CONFIGS["baseline_coordinated"])

    print("Training RNN investor in Independent world...")
    agent_ind, train_returns_ind, train_losses_ind = train_agent(
        exp_cfg=ind_cfg,
        rnn_cfg=rnn_cfg,
        seed_offset=0,
    )

    print("Training RNN investor in Coordinated world...")
    agent_coord, train_returns_coord, train_losses_coord = train_agent(
        exp_cfg=coord_cfg,
        rnn_cfg=rnn_cfg,
        seed_offset=50000,
    )

    eval_results = {
        "Ind→Ind": evaluate_agent(agent_ind, ind_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=200000),
        "Ind→Coord": evaluate_agent(agent_ind, coord_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=300000),
        "Coord→Coord": evaluate_agent(agent_coord, coord_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=400000),
        "Coord→Ind": evaluate_agent(agent_coord, ind_cfg, n_eval_episodes=rnn_cfg["n_eval"], seed=500000),
    }

    for label, result in eval_results.items():
        summarize(label, result)

    plot_training_curves(
        train_returns_ind,
        train_returns_coord,
        train_losses_ind,
        train_losses_coord,
        window=100,
    )

    print("\nSaved plots:")
    print(" - rnn_learning_curves.png")
    print(" - rnn_training_loss.png")