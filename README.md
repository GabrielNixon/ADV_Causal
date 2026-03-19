# Multi-Agent Trust Game with Behavioral Investor

## Overview

This project builds a multi-agent trust game environment where:

- The **investor** is a learned behavioral agent (trained using an RNN)
- The **trustees (adversaries)** are adaptive agents with different objectives:
  - **MAX** → maximize own payoff
  - **FAIR** → reduce inequality between investor and trustee

We simulate interactions across:

- **Two world structures**
  - `independent`
  - `coordinated`

- **Three adversary compositions**
  - MAX–MAX
  - FAIR–FAIR
  - MAX–FAIR

Total: **6 experimental settings**

---

## Project Structure

### Environment

**`mrtt_env.py`**
- Defines trust game mechanics:
  - investment → tripling → repayment
- Two modes:
  - `step()` → single adversary
  - `step_manual()` → multi-adversary (used in final experiments)

---

### Behavioral Investor

**`behavioral_investor.py`**
- GRU-based model that predicts investment behavior

**`train_behavioral_rnn.py`**
- Trains the behavioral model
- Outputs:
  - `behavioral_rnn.pt`
  - training curves
  - evaluation metrics

**`build_behavioral_dataset.py`**
- Builds training dataset from raw logs

---

### Adversaries

#### Baseline
**`behavioral_bandit_adversary.py`**
- Global policy (no context)

#### Main Model
**`behavioral_contextual_bandit_adversary.py`**
- Uses context:
  - round
  - investment bucket
- Supports:
  - `independent` → local policies
  - `coordinated` → shared + local policies

---

### Training Scripts

#### Single-Adversary
- `train_behavioral_bandit_adversary.py`
- `train_behavioral_contextual_worlds.py`

#### Multi-Agent (Final)
- `train_multi_adversary_contextual.py`

Features:
- 2 trustees
- 1 investor
- delayed (episode-level) learning
- runs all 6 settings

---

## Key Design Ideas

### Behavioral Investor
- Not reward-maximizing
- Learned from data
- Captures bounded rationality

### Trustee Objectives

- **MAX**
  - maximize trustee payoff

- **FAIR**
  - minimize:
    ```
    |trustee_gain - investor_gain|
    ```

---

### Contextual Learning

Trustees condition on:
- round
- investment level

---

### Delayed Reward

- Learning happens **after full episode**
- Enables:
  - trust-building
  - long-term strategies

---

### Multi-Agent Setup

- Two trustees
- One chosen randomly per round
- Both learn from full episode outcome

---

## Results

### Independent World

#### MAX–MAX
- Investor mean: **120.32**
- Trustee mean: **317.23**

→ Low trust, high exploitation

---

#### FAIR–FAIR
- Investor mean: **251.88**
- Trustee mean: **222.94**

→ Cooperative, stable system

---

#### MAX–FAIR
- Investor mean: **217.09**
- Trustee mean: **241.47**

→ Intermediate regime

---

### Coordinated World

#### MAX–MAX
- Investor mean: **126.86**
- Trustee mean: **308.30**

→ Still exploitative, slightly softened

---

#### FAIR–FAIR
- Investor mean: **260.55**
- Trustee mean: **220.85**

→ Most cooperative outcome

---

#### MAX–FAIR
- Investor mean: **215.86**
- Trustee mean: **248.10**

→ Mixed behavior, trustee advantage remains

---

## Key Findings

### 1. Adversary Composition Drives Behavior

- MAX–MAX → exploitative regime
- FAIR–FAIR → cooperative regime
- MAX–FAIR → intermediate regime

---

### 2. World Structure Matters Less Than Composition

- Independent → more variation
- Coordinated → more stable behavior

But:
> Composition effects are stronger than world effects

---

### 3. Behavioral Investor Changes Dynamics

- Not fully rational
- Responds to patterns
- Enables realistic trust dynamics

---

## How to Run

### Train Investor

```bash
python train_behavioral_rnn.py
