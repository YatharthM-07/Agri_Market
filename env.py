"""
AgriMarket Optimizer v1 - Reinforcement Learning Environment
Team 404 | Hackathon Submission

Gym environment for crop warehouse management — agent decides when to sell
perishable crops to maximize revenue while managing freshness and market crashes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AgriMarketEnv(gym.Env):
    """
    Warehouse env for 3 perishable crops (wheat, corn, tomatoes).
    Tasks: task1=fixed prices, task2=dynamic prices, task3=crash events.
    """

    metadata = {"render_modes": ["human"]}

    # Crop order is fixed throughout: [wheat, corn, tomatoes]
    CROPS = ["wheat", "corn", "tomatoes"]

    VALID_TASKS = ("task1", "task2", "task3")

    def __init__(self, task="task1", seed=None):
        super().__init__()

        if task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of {self.VALID_TASKS}")

        self.task = task
        self._np_rng = np.random.default_rng(seed)

        self.initial_inventory = {"wheat": 100, "corn": 80, "tomatoes": 50}
        self.max_freshness = {"wheat": 10, "corn": 7, "tomatoes": 3}

        self.price_range = {
            "wheat": (3.0, 9.0),
            "corn": (2.0, 7.0),
            "tomatoes": (5.0, 15.0),
        }

        # Base prices used for sine-wave generation in task2/3
        self.base_prices = {
            "wheat": 6.0,
            "corn": 4.5,
            "tomatoes": 10.0,
        }

        self.fixed_prices = {"wheat": 5.0, "corn": 3.0, "tomatoes": 8.0}
        self.max_steps = 15

        # Action: 0=Hold, 1=SellWheat, 2=SellCorn, 3=SellTomatoes
        self.action_space = spaces.Discrete(4)

        # Observation: [wheat_qty, corn_qty, tomato_qty,
        #               wheat_fresh, corn_fresh, tomato_fresh,
        #               wheat_price, corn_price, tomato_price,
        #               news_feed]
        self.observation_space = spaces.Box(
            low=0.0, high=200.0, shape=(10,), dtype=np.float32
        )

        self.inventory = {}
        self.freshness = {}
        self.market_prices = {}
        self.news_feed = 0
        self.prev_news_feed = 0
        self.total_profit = 0.0
        self.day = 0
        self.rot_events = 0
        self.crash_warnings_heeded = 0
        self.crash_warnings_received = 0

        # Price history for percentile calculation (task2/3 peak bonus)
        self._price_history = {c: [] for c in self.CROPS}

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self.inventory = dict(self.initial_inventory)
        self.freshness = dict(self.max_freshness)
        self.market_prices = self._generate_prices()
        self.news_feed = 0
        self.prev_news_feed = 0
        self.total_profit = 0.0
        self.day = 0
        self.rot_events = 0
        self.crash_warnings_heeded = 0
        self.crash_warnings_received = 0
        self._price_history = {c: [self.market_prices[c]] for c in self.CROPS}

        return self._get_obs(), {}

    def step(self, action, quantity: float = 1.0):
        """
        Parameters
        ----------
        action   : int   — 0=Hold, 1=SellWheat, 2=SellCorn, 3=SellTomatoes
        quantity : float — fraction of inventory to sell (0.0–1.0, default 1.0).
                          Values outside [0,1] are clamped.
                          Gymnasium evaluators call step(action) and get full sells
                          (quantity defaults to 1.0 — fully backward compatible).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        quantity = float(np.clip(quantity, 0.0, 1.0))

        self.day += 1
        reward = 0.0

        # Track crash warning BEFORE action — agent can see news_feed in state
        crash_active = (self.news_feed == 2)
        if crash_active:
            self.crash_warnings_received += 1

        info = {
            "day":      self.day,
            "action":   action,
            "quantity": quantity,
            "sold":     None,
            "rot":      [],
            "crash_warned": crash_active,
        }

        # 1. Decay freshness for crops still in inventory
        for crop in self.CROPS:
            if self.inventory[crop] > 0:
                self.freshness[crop] -= 1

        # 2. Execute sell action (partial or full)
        if action > 0:
            crop  = self.CROPS[action - 1]
            total = self.inventory[crop]
            # sell at least 1 unit if there is stock; round down for partial
            qty   = max(1, int(total * quantity)) if total > 0 else 0
            price = self.market_prices[crop]

            if qty > 0:
                sale_reward = price * qty
                multiplier  = self._get_price_multiplier(crop, price)
                sale_reward *= multiplier
                reward += sale_reward
                self.total_profit      += sale_reward
                self.inventory[crop]   -= qty
                info["sold"] = (crop, qty, price, multiplier)

                # Heeded = agent sells on the same step it sees crash warning
                if crash_active:
                    self.crash_warnings_heeded += 1

        # 3. Rot check — penalty for any crop that hit freshness <= 0
        for crop in self.CROPS:
            if self.freshness[crop] <= 0 and self.inventory[crop] > 0:
                reward -= 50.0
                self.rot_events += 1
                self.inventory[crop] = 0
                self.freshness[crop] = 0
                info["rot"].append(crop)

        # 4. Risk penalty: if crash warning was active yesterday and agent held,
        #    and crash actually materialises today (task3 only)
        if self.task == "task3" and self.prev_news_feed == 2 and action == 0:
            reward -= 100.0

        # 5. Small holding penalty to discourage infinite waiting
        if action == 0:
            reward -= 0.1

        # 6. Update market prices and news for next step
        self.prev_news_feed = self.news_feed
        self.market_prices = self._generate_prices()
        self.news_feed = self._generate_news()
        for crop in self.CROPS:
            self._price_history[crop].append(self.market_prices[crop])

        done = self._is_done()
        truncated = False

        return self._get_obs(), reward, done, truncated, info

    def render(self, mode="human"):
        inv_str = " | ".join(
            f"{c[:2].upper()}: qty={self.inventory[c]}, fresh={self.freshness[c]}, "
            f"${self.market_prices[c]:.2f}"
            for c in self.CROPS
        )
        news_map = {0: "Normal", 1: "Rain(+)", 2: "CRASH(!)"}
        print(
            f"Day {self.day:2d} | {inv_str} | News: {news_map[self.news_feed]} "
            f"| Profit: ${self.total_profit:.2f}"
        )

    def _get_obs(self):
        return np.array(
            [self.inventory[c] for c in self.CROPS]
            + [self.freshness[c] for c in self.CROPS]
            + [self.market_prices[c] for c in self.CROPS]
            + [float(self.news_feed)],
            dtype=np.float32,
        )

    def _is_done(self):
        all_empty = all(self.inventory[c] == 0 for c in self.CROPS)
        all_rotten = all(self.freshness[c] <= 0 for c in self.CROPS)
        return all_empty or all_rotten or self.day >= self.max_steps

    def _generate_prices(self):
        if self.task == "task1":
            return dict(self.fixed_prices)

        # task2 / task3: sine wave + uniform noise
        prices = {}
        for crop in self.CROPS:
            lo, hi = self.price_range[crop]
            base = self.base_prices[crop]
            trend = np.sin(self.day * 0.5) * base * 0.3
            noise = self._np_rng.uniform(-0.3, 0.3) * base
            raw = base + trend + noise
            prices[crop] = float(np.clip(raw, lo, hi))

        # task3: if crash warning was given, crash prices 80%
        if self.task == "task3" and self.news_feed == 2:
            for crop in self.CROPS:
                prices[crop] = max(
                    self.price_range[crop][0], prices[crop] * 0.2
                )

        return prices

    def _generate_news(self):
        if self.task == "task1":
            return 0  # No news in task1

        r = self._np_rng.random()
        if self.task == "task2":
            # task2: only rain signal, no crash
            if r < 0.85:
                return 0
            return 1
        else:
            # task3: full news
            if r < 0.70:
                return 0   # Normal
            elif r < 0.85:
                return 1   # Rain — prices may rise
            else:
                return 2   # Crash warning! (received counted in step())

    def _get_price_multiplier(self, crop, price):
        """Return sale multiplier based on price quality."""
        history = self._price_history[crop]
        if len(history) < 5:
            return 1.0  # Not enough history yet

        p80 = float(np.percentile(history, 80))

        if self.task in ("task2", "task3") and price >= p80:
            return 1.5  # Peak pricing bonus

        if self.task == "task3" and self.prev_news_feed == 2:
            return 0.3  # Sold on crash day (should have sold sooner)

        return 1.0
