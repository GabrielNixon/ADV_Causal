import numpy as np
import torch
import torch.nn as nn

from config import INVEST_OPTIONS, DEFAULT_HORIZON, DEFAULT_ENDOWMENT

ACTION_LIST = list(INVEST_OPTIONS)


class BehavioralRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=3, n_actions=5):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        logits = self.head(h_n[-1])
        return logits


class BehavioralRNNInvestor:
    def __init__(
        self,
        model_path="behavioral_rnn.pt",
        horizon=DEFAULT_HORIZON,
        endowment=DEFAULT_ENDOWMENT,
        greedy=True,
        seed=0,
    ):
        self.horizon = horizon
        self.endowment = endowment
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

        checkpoint = torch.load(model_path, map_location="cpu")
        hidden_size = checkpoint["hidden_size"]
        input_size = checkpoint["input_size"]
        actions = checkpoint["actions"]

        self.actions = actions
        self.model = BehavioralRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            n_actions=len(actions),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.reset()

    def reset(self):
        self.history = []
        self.prev_action = None
        self.prev_reward = None
        self.prev_repay_prop = None

    def encode_step(self, round_idx, prev_repay_prop, prev_action, prev_reward):
        return [
            round_idx / max(self.horizon - 1, 1),
            prev_repay_prop if prev_repay_prop is not None else 0.0,
            (prev_action / self.endowment) if prev_action is not None else 0.0,
            (prev_reward / self.endowment) if prev_reward is not None else 0.0,
        ]

    def _build_input(self, round_idx):
        step_vec = self.encode_step(
            round_idx=round_idx,
            prev_repay_prop=self.prev_repay_prop,
            prev_action=self.prev_action,
            prev_reward=self.prev_reward,
        )
        seq = self.history + [step_vec]
        x = torch.tensor([seq], dtype=torch.float32)
        lengths = torch.tensor([len(seq)], dtype=torch.long)
        return x, lengths, step_vec

    def act(self, round_idx, prev_repay_prop=None):
        x, lengths, step_vec = self._build_input(round_idx)

        with torch.no_grad():
            logits = self.model(x, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        action_idx = int(self.rng.choice(len(self.actions), p=probs))
        action = float(self.actions[action_idx])

        self.history.append(step_vec)
        return action

    def observe_outcome(self, action, reward, repay_prop):
        self.prev_action = float(action)
        self.prev_reward = float(reward)
        self.prev_repay_prop = float(repay_prop)