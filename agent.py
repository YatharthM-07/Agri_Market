"""
AgriMarket Optimizer - Agent Implementations
Team 404

QLearningAgent: tabular Q-learning, good for Task 1 and 2.
DQNAgent: deep Q-network with PyTorch, recommended for Task 3.
"""

import numpy as np
from collections import deque
import random


class QLearningAgent:
    """Tabular Q-learning with a sparse dictionary Q-table.
    Continuous prices are discretized into 5 bins to keep state space manageable.
    """

    # Price bin edges per crop (lo, hi from env)
    PRICE_BINS = {
        "wheat":    [3.0, 4.2, 5.4, 6.6, 7.8, 9.0],
        "corn":     [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "tomatoes": [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
    }
    CROPS = ["wheat", "corn", "tomatoes"]

    def __init__(
        self,
        action_size=4,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Sparse Q-table: key = discretized state tuple, value = np.array(action_size)
        self.q_table = {}

    def _discretize(self, state):
        """Convert float state vector (10,) to a hashable tuple of integers."""
        qtys = [int(state[i]) // 10 for i in range(3)]          # buckets of 10
        fresh = [int(state[i + 3]) for i in range(3)]
        prices = []
        for i, crop in enumerate(self.CROPS):
            bins = self.PRICE_BINS[crop]
            bucket = int(np.digitize(state[6 + i], bins)) - 1
            bucket = max(0, min(bucket, len(bins) - 2))
            prices.append(bucket)
        news = int(state[9])
        return tuple(qtys + fresh + prices + [news])

    def _get_q(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key]

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        key = self._discretize(state)
        return int(np.argmax(self._get_q(key)))

    def update(self, state, action, reward, next_state, done):
        key = self._discretize(state)
        next_key = self._discretize(next_state)

        q_vals = self._get_q(key)
        next_q_vals = self._get_q(next_key)

        target = reward
        if not done:
            target += self.gamma * np.max(next_q_vals)

        q_vals[action] += self.alpha * (target - q_vals[action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                },
                f,
            )

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _QNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
            )

        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        """DQN agent with experience replay and a target network."""

        def __init__(
            self,
            state_size=10,
            action_size=4,
            gamma=0.95,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            batch_size=64,
            memory_size=10000,
            target_update_freq=10,
        ):
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self._train_steps = 0

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.policy_net = _QNetwork(state_size, action_size).to(self.device)
            self.target_net = _QNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = deque(maxlen=memory_size)

        def get_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return int(self.policy_net(t).argmax(dim=1).item())

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def update(self, state, action, reward, next_state, done):
            self.remember(state, action, reward, next_state, done)
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_t = torch.BoolTensor(dones).to(self.device)

            current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states_t).max(1)[0]
                next_q[dones_t] = 0.0
                target_q = rewards_t + self.gamma * next_q

            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._train_steps += 1
            if self._train_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def save(self, path):
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "epsilon": self.epsilon,
                },
                path,
            )

        def load(self, path):
            data = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(data["policy_net"])
            self.target_net.load_state_dict(data["target_net"])
            self.epsilon = data["epsilon"]

except ImportError:
    # PyTorch not installed — DQNAgent is unavailable
    class DQNAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for DQNAgent. "
                "Install with: pip install torch"
            )
