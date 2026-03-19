import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ACTIONS = [0, 5, 10, 15, 20]
DATA_PATH = "behavioral_rnd_dataset.csv"
MODEL_PATH = "behavioral_rnn.pt"

SEED = 42
HIDDEN_SIZE = 16
DROPOUT = 0.20
LR = 0.003
EPOCHS = 120
BATCH_SIZE = 64
TRAIN_FRAC = 0.8


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_to_idx(a):
    a = float(a)
    closest = min(ACTIONS, key=lambda x: abs(x - a))
    return ACTIONS.index(closest)


def idx_to_action(idx):
    return ACTIONS[int(idx)]


def encode(row, prev):
    return [
        row["round"] / 9.0,
        prev["repay_prop"] if prev is not None else 0.0,
        prev["investment"] / 20.0 if prev is not None else 0.0,
        prev["reward"] / 20.0 if prev is not None else 0.0,
    ]


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = max(lengths).item()

    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append([[0.0, 0.0, 0.0, 0.0]] * pad_len + seq)

    x = torch.tensor(padded, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)
    return x, lengths, y


class BehavioralRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, n_actions=5, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        h = self.dropout(h_n[-1])
        logits = self.head(h)
        return logits


def split_by_episode(df, train_frac=TRAIN_FRAC, seed=SEED):
    episode_ids = sorted(df["episode_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    n_train = int(len(episode_ids) * train_frac)
    train_ids = set(episode_ids[:n_train])
    test_ids = set(episode_ids[n_train:])

    train_df = df[df["episode_id"].isin(train_ids)].copy()
    test_df = df[df["episode_id"].isin(test_ids)].copy()
    return train_df, test_df


def build_examples(df):
    episodes = df.groupby("episode_id")
    X = []
    y = []
    meta = []

    for episode_id, ep in episodes:
        ep = ep.sort_values("round").reset_index(drop=True)
        history = []

        for t in range(len(ep)):
            prev = ep.iloc[t - 1] if t > 0 else None
            history.append(encode(ep.iloc[t], prev))

            if t == 0:
                continue

            X.append(history[:-1].copy())
            y.append(action_to_idx(ep.iloc[t]["investment"]))
            meta.append(
                {
                    "episode_id": episode_id,
                    "round": int(ep.iloc[t]["round"]),
                    "investment_raw": float(ep.iloc[t]["investment"]),
                    "investment_bucket": idx_to_action(action_to_idx(ep.iloc[t]["investment"])),
                }
            )

    return X, y, pd.DataFrame(meta)


def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, y)
            acc = compute_accuracy(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            total_n += bs

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return total_loss / total_n, total_acc / total_n, np.array(all_preds), np.array(all_targets)


def main():
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(DATA_PATH)
    train_df, test_df = split_by_episode(df)

    X_train, y_train, meta_train = build_examples(train_df)
    X_test, y_test, meta_test = build_examples(test_df)

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    counts = np.bincount(y_train, minlength=len(ACTIONS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = BehavioralRNN(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        n_actions=len(ACTIONS),
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    best_test_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_n = 0

        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, y)
            acc = compute_accuracy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = y.size(0)
            epoch_loss += loss.item() * bs
            epoch_acc += acc * bs
            total_n += bs

        train_loss = epoch_loss / total_n
        train_acc = epoch_acc / total_n
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "actions": ACTIONS,
                "input_size": 4,
                "hidden_size": HIDDEN_SIZE,
            }

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f} | "
                f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.3f}"
            )

    torch.save(best_state, MODEL_PATH)

    model.load_state_dict(best_state["model_state_dict"])
    final_test_loss, final_test_acc, preds, targets = evaluate(model, test_loader, criterion, device)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Behavioral RNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("behavioral_rnn_training.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train accuracy")
    plt.plot(test_accs, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Behavioral RNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("behavioral_rnn_accuracy.png", dpi=220)
    plt.close()

    results_df = meta_test.copy().reset_index(drop=True)
    results_df["target_idx"] = targets
    results_df["pred_idx"] = preds
    results_df["target_action"] = results_df["target_idx"].apply(idx_to_action)
    results_df["pred_action"] = results_df["pred_idx"].apply(idx_to_action)
    results_df.to_csv("behavioral_rnn_test_predictions.csv", index=False)

    confusion = pd.crosstab(
        results_df["target_action"],
        results_df["pred_action"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    confusion.to_csv("behavioral_rnn_confusion_matrix.csv")

    print("\nClass counts (train):")
    for i, a in enumerate(ACTIONS):
        print(f"  {a:>2}: {int(counts[i])}")

    print("\nSaved:")
    print(" - behavioral_rnn.pt")
    print(" - behavioral_rnn_training.png")
    print(" - behavioral_rnn_accuracy.png")
    print(" - behavioral_rnn_test_predictions.csv")
    print(" - behavioral_rnn_confusion_matrix.csv")
    print(f"\nFinal test accuracy: {final_test_acc:.3f}")


if __name__ == "__main__":
    main()