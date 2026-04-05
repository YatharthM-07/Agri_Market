---
title: AgriMarket Optimizer v1
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# AgriMarket Optimizer v1

**Team 404** | Reinforcement Learning Environment | Hackathon Submission

---

## Overview

AgriMarket Optimizer is a Gymnasium-compatible Reinforcement Learning environment that simulates crop warehouse management in Indian agriculture. An AI agent acts as a business manager for a warehouse stocked with three perishable crops — **wheat**, **corn**, and **tomatoes** — and must decide each day whether to hold or sell each crop to maximize total revenue.

The environment captures a genuine real-world tension: crops deteriorate over time (tomatoes last only 3 days!), market prices fluctuate daily, and sudden crash warnings require immediate action. The agent must balance freshness urgency, price timing, and market risk signals simultaneously.

---

## Installation

```bash
git clone https://huggingface.co/NSRexe/agrimarket-optimizer-v1
cd agrimarket-optimizer-v1
pip install -r requirements.txt
```

**Optional — DQN agent (requires PyTorch):**
```bash
pip install torch
```

---

## Quick Start

```python
from env import AgriMarketEnv

env = AgriMarketEnv(task="task1")
state, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # random agent
    state, reward, done, truncated, info = env.step(action)
    env.render()
```

---

## State & Action Spaces

### Observation Space — `Box(10,)` float32

| Index | Feature | Range |
|-------|---------|-------|
| 0 | `wheat_quantity` (units) | 0–100 |
| 1 | `corn_quantity` (units) | 0–80 |
| 2 | `tomato_quantity` (units) | 0–50 |
| 3 | `wheat_freshness` (days remaining) | 0–10 |
| 4 | `corn_freshness` (days remaining) | 0–7 |
| 5 | `tomato_freshness` (days remaining) | 0–3 |
| 6 | `wheat_price` ($/unit) | 3.0–9.0 |
| 7 | `corn_price` ($/unit) | 2.0–7.0 |
| 8 | `tomato_price` ($/unit) | 5.0–15.0 |
| 9 | `news_feed` | 0=Normal, 1=Rain, 2=Crash |

### Action Space — `Discrete(4)`

| Action | Description |
|--------|-------------|
| 0 | **Hold** — do nothing, wait for better prices |
| 1 | **Sell Wheat** — liquidate all wheat at today's price |
| 2 | **Sell Corn** — liquidate all corn at today's price |
| 3 | **Sell Tomatoes** — liquidate all tomatoes at today's price |

---

## Reward Function

| Event | Reward |
|-------|--------|
| Sell crop (standard) | `+price × quantity` |
| Sell crop at peak price (>80th percentile) | `+price × quantity × 1.5` |
| Sell on crash day (ignored yesterday's warning) | `+price × quantity × 0.3` |
| Crop rots (freshness = 0, quantity > 0) | `-50` per crop |
| Hold through crash event (task3) | `-100` |
| Hold action (small discouragement) | `-0.1` |

---

## Tasks

### Task 1 — Basic Sales (Easy)
- **Prices**: Fixed (wheat=$5, corn=$3, tomatoes=$8)
- **News**: Disabled
- **Goal**: Sell all inventory before crops rot
- **Success**: 0 rot events in 90%+ of 100 evaluation episodes
- **Key learning**: Tomatoes (3 days) → Corn (7 days) → Wheat (10 days)

### Task 2 — Profit Maximization (Medium)
- **Prices**: Dynamic (sine wave + noise)
- **News**: Rain signals only
- **Goal**: Achieve $1,000+ total profit per episode
- **Success**: Reach $1,000 in 70%+ of 100 evaluation episodes
- **Key learning**: Hold on low-price days, sell on high-price peaks

### Task 3 — Risk Management (Hard)
- **Prices**: Dynamic with crash events (80% drop)
- **News**: Full (normal / rain / crash warning)
- **Goal**: Detect crash warnings and liquidate immediately
- **Success**: Sell within 1 timestep of crash warning in 80%+ of crash episodes
- **Key learning**: `news_feed == 2` → emergency liquidation overrides profit timing

---

## Training Example

```bash
# Train on a single task
python train.py --task task1

# Train all three tasks sequentially
python train.py --all

# Use DQN agent (requires PyTorch)
python train.py --task task3 --agent dqn
```

### Expected Training Output
```
============================================================
  Training on TASK1 | Agent: QLEARNING
  Episodes: 1000 | Success: 0 rots in 90%+ of episodes
============================================================
  Ep    0 | Avg Reward:   -95.30 | Avg Profit:  $  0.00 | Avg Rots: 3.00 | Epsilon: 1.000
  Ep  100 | Avg Reward:   234.56 | Avg Profit: $312.00 | Avg Rots: 0.12 | Epsilon: 0.607
  Ep  200 | Avg Reward:   512.89 | Avg Profit: $540.00 | Avg Rots: 0.00 | Epsilon: 0.368
  Ep  500 | Avg Reward:   823.45 | Avg Profit: $840.00 | Avg Rots: 0.00 | Epsilon: 0.082
```

---

## Evaluation

Judges can reproduce evaluation with:

```python
from env import AgriMarketEnv
from agent import QLearningAgent
import pickle

# Load trained agent
agent = QLearningAgent()
agent.load("models/task1_qlearning_agent.pkl")
agent.epsilon = 0.0   # greedy

env = AgriMarketEnv(task="task1", seed=42)
profits = []

for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, _, done, _, _ = env.step(action)
    profits.append(env.total_profit)

print(f"Zero-rot rate: {sum(env.rot_events == 0 for _ in range(100))}%")
print(f"Mean profit: ${sum(profits)/len(profits):.2f}")
```

---

## Citation

```
Team 404 — AgriMarket Optimizer v1
Yatharth (Environment Logic Lead)
Purvanshi (Agent Training Lead)
Hackathon Submission, April 2026
```

---

## License

MIT License
