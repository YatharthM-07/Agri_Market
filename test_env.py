"""
AgriMarket Optimizer — Test Suite
Team 404

Run with:
    python test_env.py
"""

import numpy as np
import sys

from env import AgriMarketEnv
from agent import QLearningAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    tag = f"[{status} ] {name}"
    if detail:
        tag += f"  ({detail})"
    print(tag)
    results.append(condition)


# ---------------------------------------------------------------------------
# 1. Unit Tests — Environment
# ---------------------------------------------------------------------------

def test_reset():
    """reset() must return a (10,) float32 array and initialise inventory."""
    env = AgriMarketEnv(task="task1", seed=42)
    state, info = env.reset()

    check("reset returns shape (10,)",   state.shape == (10,),      f"got {state.shape}")
    check("reset dtype float32",         state.dtype == np.float32, f"got {state.dtype}")
    check("reset wheat qty = 100",       env.inventory["wheat"]   == 100)
    check("reset corn qty = 80",         env.inventory["corn"]    == 80)
    check("reset tomato qty = 50",       env.inventory["tomatoes"] == 50)
    check("reset wheat freshness = 10",  env.freshness["wheat"]   == 10)
    check("reset corn freshness = 7",    env.freshness["corn"]    == 7)
    check("reset tomato freshness = 3",  env.freshness["tomatoes"] == 3)
    check("reset day = 0",               env.day == 0)
    check("reset news_feed = 0",         env.news_feed == 0)
    check("reset total_profit = 0",      env.total_profit == 0.0)


def test_step_shapes():
    """step() must return (state, reward, done, truncated, info) with correct types."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    next_state, reward, done, truncated, info = env.step(0)  # Hold

    check("step next_state shape (10,)", next_state.shape == (10,))
    check("step reward is float",        isinstance(reward, float))
    check("step done is bool",           isinstance(done, bool))
    check("step truncated is bool",      isinstance(truncated, bool))
    check("step info is dict",           isinstance(info, dict))


def test_freshness_decay():
    """Freshness must decrement by 1 each step for non-empty crops."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    env.step(0)   # Hold — all crops still in inventory

    check("wheat freshness decremented",   env.freshness["wheat"]    == 9)
    check("corn freshness decremented",    env.freshness["corn"]     == 6)
    check("tomato freshness decremented",  env.freshness["tomatoes"] == 2)


def test_sell_wheat():
    """Action 1 must zero wheat inventory and return positive reward."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    _, reward, _, _, info = env.step(1)   # Sell wheat

    check("sell wheat: inventory zeroed",  env.inventory["wheat"] == 0)
    check("sell wheat: reward > 0",        reward > 0, f"reward={reward:.2f}")
    check("sell wheat: corn untouched",    env.inventory["corn"] == 80)
    check("sell wheat: tomatoes untouched",env.inventory["tomatoes"] == 50)


def test_sell_corn():
    """Action 2 must zero corn inventory and return positive reward."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    _, reward, _, _, _ = env.step(2)

    check("sell corn: inventory zeroed",   env.inventory["corn"] == 0)
    check("sell corn: reward > 0",         reward > 0, f"reward={reward:.2f}")


def test_sell_tomatoes():
    """Action 3 must zero tomato inventory and return positive reward."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    _, reward, _, _, _ = env.step(3)

    check("sell tomatoes: inventory zeroed", env.inventory["tomatoes"] == 0)
    check("sell tomatoes: reward > 0",       reward > 0, f"reward={reward:.2f}")


def test_rot_penalty():
    """Tomatoes (freshness=3) must rot and trigger -50 penalty after 3 holds."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    total_penalty = 0.0
    rot_info_captured = []

    for _ in range(4):                          # freshness: 3→2→1→0(rot)→already gone
        _, reward, _, _, info = env.step(0)
        if info["rot"]:
            total_penalty += reward
            rot_info_captured = info["rot"]

    check("tomatoes rotted",                env.inventory["tomatoes"] == 0)
    check("rot fires at correct step",      "tomatoes" in rot_info_captured)
    check("rot penalty is -50",             total_penalty <= -50.0, f"penalty={total_penalty:.2f}")


def test_termination_all_sold():
    """Episode must terminate when all inventory reaches zero."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    env.step(1)   # sell wheat
    env.step(2)   # sell corn
    _, _, done, _, _ = env.step(3)   # sell tomatoes — all empty

    check("done when all sold", done)


def test_termination_max_steps():
    """Episode must terminate after 15 steps regardless of inventory."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        _, _, done, _, _ = env.step(0)
        steps += 1

    check("done within 15 steps", steps <= 15, f"steps={steps}")


def test_task1_fixed_prices():
    """Task1 prices must be constant (wheat=5, corn=3, tomatoes=8)."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    for _ in range(5):
        env.step(0)
        check("task1 wheat price fixed at 5.0",    env.market_prices["wheat"]    == 5.0)
        check("task1 corn price fixed at 3.0",     env.market_prices["corn"]     == 3.0)
        check("task1 tomato price fixed at 8.0",   env.market_prices["tomatoes"] == 8.0)


def test_task2_dynamic_prices():
    """Task2 prices must vary between steps (not all identical)."""
    env = AgriMarketEnv(task="task2", seed=42)
    env.reset()
    prices = []
    for _ in range(10):
        env.step(0)
        prices.append(env.market_prices["wheat"])

    check("task2 prices vary", len(set(round(p, 4) for p in prices)) > 1,
          f"unique prices={len(set(round(p,4) for p in prices))}")


def test_task3_crash_price_drop():
    """When news_feed==2, task3 prices must drop to ~20% of normal."""
    env = AgriMarketEnv(task="task3", seed=42)
    env.reset()

    # Force a crash warning and grab prices before/after
    env.news_feed = 2
    normal_prices = dict(env.market_prices)
    env.step(0)   # trigger price generation with crash active

    crashed = any(
        env.market_prices[c] < normal_prices[c] * 0.5
        for c in ["wheat", "corn", "tomatoes"]
        if normal_prices[c] > 0
    )
    check("task3 prices drop on crash", crashed)


def test_action_space():
    """action_space must be Discrete(4) and contain actions 0-3."""
    env = AgriMarketEnv()
    check("action_space size = 4", env.action_space.n == 4)
    for a in range(4):
        check(f"action {a} valid", env.action_space.contains(a))


def test_observation_space():
    """Observation space bounds must contain the reset state."""
    env = AgriMarketEnv(task="task1", seed=42)
    state, _ = env.reset()
    check("state in obs space", env.observation_space.contains(state),
          f"state={state}")


# ---------------------------------------------------------------------------
# 2. Integration Tests
# ---------------------------------------------------------------------------

def test_full_episode_no_crash():
    """A full task1 episode must complete without Python errors."""
    env = AgriMarketEnv(task="task1", seed=42)
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated) and steps < 20:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    check("full episode completes",    done or truncated)
    check("episode within 15 steps",   steps <= 15, f"steps={steps}")


def test_optimal_task1_strategy():
    """Selling in order Tomatoes→Corn→Wheat should yield 0 rots."""
    env = AgriMarketEnv(task="task1", seed=42)
    env.reset()
    done = False
    truncated = False

    env.step(3)  # sell tomatoes immediately
    env.step(2)  # sell corn
    env.step(1)  # sell wheat

    check("optimal task1: 0 rots", env.rot_events == 0, f"rots={env.rot_events}")
    check("optimal task1: all sold", all(env.inventory[c] == 0 for c in env.CROPS))


def test_multi_task_episodes():
    """All three tasks must run full episodes without raising exceptions."""
    for task in ["task1", "task2", "task3"]:
        env = AgriMarketEnv(task=task, seed=42)
        state, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        try:
            while not (done or truncated) and steps < 20:
                state, _, done, truncated, _ = env.step(env.action_space.sample())
                steps += 1
            check(f"{task} runs without error", True)
        except Exception as e:
            check(f"{task} runs without error", False, str(e))


# ---------------------------------------------------------------------------
# 3. Agent Tests
# ---------------------------------------------------------------------------

def test_agent_action_range():
    """Agent must return actions within [0, 3]."""
    env = AgriMarketEnv(task="task1", seed=42)
    agent = QLearningAgent()
    state, _ = env.reset()

    for _ in range(20):
        action = agent.get_action(state)
        check("agent action in [0,3]", 0 <= action <= 3, f"action={action}")
        state, _, done, _, _ = env.step(action)
        if done:
            state, _ = env.reset()


def test_agent_learns():
    """After 200 episodes on task1, late rewards must exceed early rewards."""
    env = AgriMarketEnv(task="task1", seed=42)
    agent = QLearningAgent(epsilon_decay=0.99)
    early, late = [], []

    for ep in range(200):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done or truncated)
            state = next_state
            ep_reward += reward
        agent.decay_epsilon()
        if ep < 20:
            early.append(ep_reward)
        elif ep >= 180:
            late.append(ep_reward)

    check("agent improves over training",
          np.mean(late) > np.mean(early),
          f"early={np.mean(early):.1f}  late={np.mean(late):.1f}")


def test_agent_epsilon_decay():
    """Epsilon must decrease after each call to decay_epsilon()."""
    agent = QLearningAgent(epsilon_start=1.0, epsilon_decay=0.99)
    prev = agent.epsilon
    for _ in range(10):
        agent.decay_epsilon()
        check("epsilon decreases", agent.epsilon < prev, f"eps={agent.epsilon:.4f}")
        prev = agent.epsilon

    agent_floor = QLearningAgent(epsilon_start=0.01, epsilon_min=0.01)
    for _ in range(5):
        agent_floor.decay_epsilon()
    check("epsilon respects min floor", agent_floor.epsilon == 0.01)


def test_agent_save_load(tmp_path="/tmp/test_agent.pkl"):
    """Saved and reloaded Q-table must produce identical actions."""
    env = AgriMarketEnv(task="task1", seed=42)
    agent = QLearningAgent()
    state, _ = env.reset()

    # Populate Q-table a little
    for _ in range(30):
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

    agent.save(tmp_path)

    agent2 = QLearningAgent()
    agent2.load(tmp_path)
    agent2.epsilon = 0.0
    agent.epsilon = 0.0

    state, _ = env.reset()
    for _ in range(5):
        a1 = agent.get_action(state)
        a2 = agent2.get_action(state)
        check("save/load: actions match", a1 == a2, f"a1={a1} a2={a2}")
        state, _, done, _, _ = env.step(a1)
        if done:
            state, _ = env.reset()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    suites = [
        ("Unit — reset()",               test_reset),
        ("Unit — step() shapes",         test_step_shapes),
        ("Unit — freshness decay",        test_freshness_decay),
        ("Unit — sell wheat",             test_sell_wheat),
        ("Unit — sell corn",              test_sell_corn),
        ("Unit — sell tomatoes",          test_sell_tomatoes),
        ("Unit — rot penalty",            test_rot_penalty),
        ("Unit — termination all sold",   test_termination_all_sold),
        ("Unit — termination max steps",  test_termination_max_steps),
        ("Unit — task1 fixed prices",     test_task1_fixed_prices),
        ("Unit — task2 dynamic prices",   test_task2_dynamic_prices),
        ("Unit — task3 crash drop",       test_task3_crash_price_drop),
        ("Unit — action space",           test_action_space),
        ("Unit — observation space",      test_observation_space),
        ("Integration — full episode",    test_full_episode_no_crash),
        ("Integration — optimal task1",   test_optimal_task1_strategy),
        ("Integration — all tasks run",   test_multi_task_episodes),
        ("Agent — action range",          test_agent_action_range),
        ("Agent — learns over time",      test_agent_learns),
        ("Agent — epsilon decay",         test_agent_epsilon_decay),
        ("Agent — save / load",           test_agent_save_load),
    ]

    for label, fn in suites:
        print(f"\n--- {label} ---")
        fn()

    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED  ←")
    else:
        print("  — all clear!")
    print(f"{'='*50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_all()
