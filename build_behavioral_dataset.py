import os
import pandas as pd
import numpy as np

BASE_DIR = "mrtt/RND"   # <-- IMPORTANT: use RND first

rows = []

for folder in os.listdir(BASE_DIR):
    path = os.path.join(BASE_DIR, folder, "data", "output.csv")
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    # clean columns
    df["repay"] = df["repay"].apply(lambda x: float(str(x).strip("[]")))
    df["investment"] = df["investment"].astype(float)

    # compute repay proportion
    df["repay_prop"] = df.apply(
        lambda r: r["repay"] / (3.0 * r["investment"]) if r["investment"] > 0 else 0.0,
        axis=1,
    )

    # compute reward
    df["reward"] = 20.0 - df["investment"] + df["repay"]

    df["round"] = np.arange(len(df))
    df["episode_id"] = folder

    rows.append(df[["episode_id", "round", "investment", "repay_prop", "reward"]])

dataset = pd.concat(rows, ignore_index=True)
dataset.to_csv("behavioral_rnd_dataset.csv", index=False)

print("Saved:", dataset.shape)