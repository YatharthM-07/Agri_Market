"""
AgriMarket Optimizer — Ask the AI
Team 404

Run: python3 ask.py
The trained agent tells you what to do with your crops.
"""

import numpy as np
from agent import QLearningAgent

ACTION_NAMES = {
    0: "⏳ HOLD     — wait for better prices",
    1: "🌾 SELL WHEAT",
    2: "🌽 SELL CORN",
    3: "🍅 SELL TOMATOES",
}

NEWS_MAP = {
    "0": 0, "normal": 0, "n": 0,
    "1": 1, "rain": 1, "r": 1,
    "2": 2, "crash": 2, "c": 2,
}


def load_agent(task: str) -> QLearningAgent:
    path = f"models/{task}_qlearning_agent.pkl"
    agent = QLearningAgent()
    agent.load(path)
    agent.epsilon = 0.0   # fully greedy — no random exploration
    return agent


def get_float(prompt: str, default: float) -> float:
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return float(val) if val else default
    except ValueError:
        return default


def get_int(prompt: str, default: int) -> int:
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return int(val) if val else default
    except ValueError:
        return default


def build_state(task: str) -> np.ndarray:
    print("\n  📦 Enter your warehouse situation (press Enter to use defaults):\n")

    wheat_qty   = get_int  ("Wheat quantity  (0-100)", 100)
    corn_qty    = get_int  ("Corn  quantity  (0-80) ",  80)
    tomato_qty  = get_int  ("Tomato quantity (0-50) ",  50)

    print()
    wheat_fresh  = get_int ("Wheat freshness  (days left, max 10)", 10)
    corn_fresh   = get_int ("Corn  freshness  (days left, max 7) ",  7)
    tomato_fresh = get_int ("Tomato freshness (days left, max 3) ",  3)

    if task in ("task2", "task3"):
        print()
        wheat_price  = get_float("Wheat price  ($/unit, range 3-9) ",  5.0)
        corn_price   = get_float("Corn  price  ($/unit, range 2-7) ",  3.5)
        tomato_price = get_float("Tomato price ($/unit, range 5-15)", 10.0)
    else:
        wheat_price, corn_price, tomato_price = 5.0, 3.0, 8.0   # task1 fixed prices

    news = 0
    if task == "task3":
        print()
        raw = input("  Market news — 0=Normal / 1=Rain / 2=CRASH WARNING [0]: ").strip().lower()
        news = NEWS_MAP.get(raw, 0)

    return np.array([
        wheat_qty, corn_qty, tomato_qty,
        wheat_fresh, corn_fresh, tomato_fresh,
        wheat_price, corn_price, tomato_price,
        news,
    ], dtype=np.float32)


def explain(state: np.ndarray, action: int) -> str:
    qtys   = state[0:3]
    fresh  = state[3:6]
    prices = state[6:9]
    news   = int(state[9])
    names  = ["Wheat", "Corn", "Tomatoes"]

    reasons = []

    if news == 2 and action != 0:
        reasons.append("🚨 CRASH WARNING detected — sell before prices drop 80%!")
    if fresh[2] <= 1 and action == 3:
        reasons.append("⚡ Tomatoes expire in 1 day — must sell NOW or lose them")
    if fresh[1] <= 1 and action == 2:
        reasons.append("⚡ Corn expires in 1 day — must sell NOW or lose them")
    if fresh[0] <= 2 and action == 1:
        reasons.append("⚡ Wheat expires in 2 days — must sell NOW or lose them")
    if action != 0:
        crop_idx = action - 1
        value = qtys[crop_idx] * prices[crop_idx]
        reasons.append(f"💰 You'll earn ${value:.2f} ({names[crop_idx]} qty={int(qtys[crop_idx])} × ${prices[crop_idx]:.2f}/unit)")
    if action == 0:
        reasons.append("📈 Prices not optimal yet — waiting for a better moment")

    return "\n".join(f"     {r}" for r in reasons)


def main():
    print("\n" + "═" * 52)
    print("  🌾 AgriMarket Optimizer — Ask the AI")
    print("  Team 404  |  Trained Q-Learning Agent")
    print("═" * 52)

    print("\n  Which task are you playing?")
    print("  1 → Task 1 (Fixed prices, learn to avoid rot)")
    print("  2 → Task 2 (Dynamic prices, maximize profit)")
    print("  3 → Task 3 (Crash events, risk management)")

    choice = input("\n  Your choice [3]: ").strip()
    task_map = {"1": "task1", "2": "task2", "3": "task3", "": "task3"}
    task = task_map.get(choice, "task3")

    print(f"\n  ✅ Loading trained agent for {task.upper()}...")
    agent = load_agent(task)
    print(f"  ✅ Agent loaded — epsilon={agent.epsilon} (fully greedy)")

    while True:
        print("\n" + "─" * 52)
        state = build_state(task)

        action = agent.get_action(state)

        print("\n" + "═" * 52)
        print(f"  🤖 AI DECISION:  {ACTION_NAMES[action]}")
        print("═" * 52)
        print("\n  Why?")
        print(explain(state, action))

        # Show current portfolio value
        print(f"\n  📊 Portfolio snapshot:")
        crops = [("Wheat", 0), ("Corn", 1), ("Tomatoes", 2)]
        total_value = 0
        for name, i in crops:
            val = state[i] * state[6+i]
            total_value += val
            days = int(state[3+i])
            print(f"     {name:10s}: qty={int(state[i]):3d}  value=${val:7.2f}  freshness={days}d left")
        print(f"     {'TOTAL':10s}: ${total_value:.2f}")

        print()
        again = input("  Ask again? (y/n) [y]: ").strip().lower()
        if again == "n":
            print("\n  Good luck farming! 🌾\n")
            break


if __name__ == "__main__":
    main()
