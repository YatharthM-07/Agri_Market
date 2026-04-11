"""
AgriMarket Optimizer - Training & Evaluation Script
Team 404

Usage: python train.py --task task1 [--agent dqn] [--all]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from env import AgriMarketEnv
from agent import QLearningAgent, DQNAgent

SEED = 42
np.random.seed(SEED)


TASK_CONFIG = {
    "task1": {
        "episodes": 1000,
        "eval_episodes": 100,
        "epsilon_decay": 0.995,
        "success_label": "0 rots in 90%+ of episodes",
    },
    "task2": {
        "episodes": 2000,
        "eval_episodes": 100,
        "epsilon_decay": 0.997,
        "success_label": "Profit >= $1000 in 70%+ of episodes",
    },
    "task3": {
        "episodes": 5000,
        "eval_episodes": 100,
        "epsilon_decay": 0.998,
        "success_label": "Respond to crash warnings in 80%+ of crash episodes",
    },
}


def run_episode(env, agent, training=True):
    """Run a single episode. Returns a dict of metrics."""
    state, _ = env.reset(seed=None)
    total_reward = 0.0
    done = False
    truncated = False
    steps = 0
    action_log = []

    while not (done or truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        if training:
            agent.update(state, action, reward, next_state, done or truncated)

        action_log.append({
            "day": info["day"],
            "action": action,
            "reward": reward,
            "crash_warned": info["crash_warned"],
            "sold": info["sold"],
            "rot": info["rot"],
        })

        state = next_state
        total_reward += reward
        steps += 1

    if training:
        agent.decay_epsilon()

    return {
        "total_reward": total_reward,
        "total_profit": env.total_profit,
        "rot_events": env.rot_events,
        "steps": steps,
        "epsilon": agent.epsilon,
        "crash_warnings_received": env.crash_warnings_received,
        "crash_warnings_heeded": env.crash_warnings_heeded,
        "action_log": action_log,
    }


def train(task="task1", agent_type="qlearning", verbose=True):
    """Train an agent on the specified task."""
    config = TASK_CONFIG[task]
    n_episodes = config["episodes"]

    env = AgriMarketEnv(task=task, seed=SEED)

    if agent_type == "dqn":
        agent = DQNAgent(
            state_size=10,
            action_size=4,
            gamma=0.95,
            epsilon_decay=config["epsilon_decay"],
        )
    else:
        agent = QLearningAgent(
            action_size=4,
            epsilon_decay=config["epsilon_decay"],
        )

    rewards_log = []
    profit_log = []
    rot_log = []
    epsilon_log = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training on {task.upper()} | Agent: {agent_type.upper()}")
        print(f"  Episodes: {n_episodes} | Success: {config['success_label']}")
        print(f"{'='*60}")

    for ep in range(n_episodes):
        metrics = run_episode(env, agent, training=True)
        rewards_log.append(metrics["total_reward"])
        profit_log.append(metrics["total_profit"])
        rot_log.append(metrics["rot_events"])
        epsilon_log.append(metrics["epsilon"])

        if verbose and ep % 100 == 0:
            window = min(100, ep + 1)
            avg_r = np.mean(rewards_log[-window:])
            avg_p = np.mean(profit_log[-window:])
            avg_rot = np.mean(rot_log[-window:])
            print(
                f"  Ep {ep:4d} | Avg Reward: {avg_r:8.2f} | "
                f"Avg Profit: ${avg_p:7.2f} | "
                f"Avg Rots: {avg_rot:.2f} | "
                f"Epsilon: {metrics['epsilon']:.3f}"
            )

    training_data = {
        "rewards": rewards_log,
        "profits": profit_log,
        "rots": rot_log,
        "epsilons": epsilon_log,
    }

    return agent, env, training_data


def evaluate(task, agent, n_episodes=100, verbose=True):
    """Run greedy evaluation and report success metrics."""
    env = AgriMarketEnv(task=task, seed=SEED + 999)
    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy during eval

    rewards, profits, rots = [], [], []
    crash_heeded, crash_received = 0, 0

    for _ in range(n_episodes):
        metrics = run_episode(env, agent, training=False)
        rewards.append(metrics["total_reward"])
        profits.append(metrics["total_profit"])
        rots.append(metrics["rot_events"])
        crash_heeded += metrics["crash_warnings_heeded"]
        crash_received += metrics["crash_warnings_received"]

    agent.epsilon = saved_eps

    results = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_profit": np.mean(profits),
        "zero_rot_rate": np.mean([r == 0 for r in rots]),
        "profit_1000_rate": np.mean([p >= 1000 for p in profits]),
        "crash_heed_rate": (
            crash_heeded / crash_received if crash_received > 0 else None
        ),
        "n_crash_episodes": crash_received,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluation — {task.upper()} ({n_episodes} episodes)")
        print(f"{'='*60}")
        print(f"  Mean Reward  : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Profit  : ${results['mean_profit']:.2f}")
        print(f"  Zero-Rot Rate: {results['zero_rot_rate']*100:.1f}%  (target: 90% for task1)")
        print(f"  $1000+ Rate  : {results['profit_1000_rate']*100:.1f}%  (target: 70% for task2)")
        if results["crash_heed_rate"] is not None:
            print(
                f"  Crash Heed   : {results['crash_heed_rate']*100:.1f}%  "
                f"(target: 80% for task3, n={results['n_crash_episodes']} warnings)"
            )
        print(f"{'='*60}\n")

    return results


def plot_training(task, training_data, save_path=None):
    """Plot reward curve and rot events over training."""
    rewards = training_data["rewards"]
    profits = training_data["profits"]
    rots = training_data["rots"]
    epsilons = training_data["epsilons"]

    window = 50
    smoothed_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_p = np.convolve(profits, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"AgriMarket Optimizer — {task.upper()} Training Curve", fontsize=14)

    axes[0, 0].plot(smoothed_r, color="steelblue")
    axes[0, 0].set_title("Smoothed Reward (window=50)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(smoothed_p, color="forestgreen")
    axes[0, 1].axhline(1000, color="red", linestyle="--", alpha=0.7, label="$1000 target")
    axes[0, 1].set_title("Smoothed Profit (window=50)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total Profit ($)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rots, alpha=0.4, color="tomato")
    axes[1, 0].plot(
        np.convolve(rots, np.ones(window) / window, mode="valid"),
        color="darkred", linewidth=2
    )
    axes[1, 0].set_title("Rot Events per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("# Crops Rotted")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epsilons, color="darkorange")
    axes[1, 1].set_title("Epsilon (Exploration Rate)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def demo_episode(task, agent, render=True):
    """Run and display one greedy episode step by step."""
    env = AgriMarketEnv(task=task, seed=7)
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    state, _ = env.reset()
    done = False
    truncated = False
    action_names = ["HOLD", "SELL_WHEAT", "SELL_CORN", "SELL_TOMATOES"]

    print(f"\n--- Demo Episode ({task.upper()}) ---")
    if render:
        env.render()

    while not (done or truncated):
        action = agent.get_action(state)
        state, reward, done, truncated, info = env.step(action)
        if render:
            print(f"  Action: {action_names[action]}  |  Reward: {reward:+.2f}")
            env.render()

    print(f"\n  Final Profit: ${env.total_profit:.2f} | Rot Events: {env.rot_events}")
    agent.epsilon = saved_eps


def main():
    parser = argparse.ArgumentParser(description="Train AgriMarket RL agent")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    parser.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    parser.add_argument("--all", action="store_true", help="Train all 3 tasks sequentially")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tasks = ["task1", "task2", "task3"] if args.all else [args.task]
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    all_results = {}

    for task in tasks:
        agent, env, training_data = train(task=task, agent_type=args.agent)

        if not args.no_plot:
            plot_training(task, training_data, save_path=f"../plots/{task}_curve.png")

        eval_results = evaluate(task, agent)
        all_results[task] = eval_results

        model_path = f"../models/{task}_{args.agent}_agent.pkl"
        if hasattr(agent, "save"):
            agent.save(model_path)
            print(f"  Model saved: {model_path}")

        demo_episode(task, agent)

    if len(tasks) > 1:
        print("\n" + "="*60)
        print("  SUMMARY TABLE")
        print("="*60)
        print(f"  {'Task':<8} | {'Mean Profit':>12} | {'Zero-Rot%':>10} | {'$1000+%':>8} | {'Crash%':>8}")
        print("  " + "-"*58)
        for t, r in all_results.items():
            cr = f"{r['crash_heed_rate']*100:.1f}%" if r["crash_heed_rate"] is not None else "  N/A"
            print(
                f"  {t:<8} | ${r['mean_profit']:>11.2f} | "
                f"{r['zero_rot_rate']*100:>9.1f}% | "
                f"{r['profit_1000_rate']*100:>7.1f}% | "
                f"{cr:>8}"
            )
        print("="*60)


if __name__ == "__main__":
    main()
