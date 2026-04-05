"""
AgriMarket Optimizer v1 — Main Entry Point
Team 404

The single command to run, train, evaluate, or demo the environment.

Usage:
    python main.py                          # Interactive menu
    python main.py train --task task1
    python main.py train --task task2
    python main.py train --task task3
    python main.py train --all              # All 3 tasks sequentially
    python main.py eval  --task task1       # Evaluate saved model
    python main.py demo  --task task3       # Watch one greedy episode
    python main.py random                   # Random agent sanity check
"""

import argparse
import os
import sys

import numpy as np

from env import AgriMarketEnv
from agent import QLearningAgent
from train import train, evaluate, demo_episode, plot_training


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_train(args):
    tasks = ["task1", "task2", "task3"] if args.all else [args.task]
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    for task in tasks:
        agent, env, training_data = train(
            task=task, agent_type=args.agent, verbose=True
        )

        plot_training(task, training_data, save_path=f"../plots/{task}_curve.png")
        evaluate(task, agent, n_episodes=100)

        model_path = f"../models/{task}_{args.agent}_agent.pkl"
        if hasattr(agent, "save"):
            agent.save(model_path)
            print(f"  Model saved → {model_path}")

        demo_episode(task, agent, render=True)


def cmd_eval(args):
    model_path = f"../models/{args.task}_{args.agent}_agent.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print(f"        Run: python main.py train --task {args.task} first.")
        sys.exit(1)

    agent = QLearningAgent()
    agent.load(model_path)
    agent.epsilon = 0.0  # greedy

    evaluate(args.task, agent, n_episodes=100)


def cmd_demo(args):
    model_path = f"../models/{args.task}_{args.agent}_agent.pkl"
    if os.path.exists(model_path):
        agent = QLearningAgent()
        agent.load(model_path)
        agent.epsilon = 0.0
        print(f"Loaded model: {model_path}")
    else:
        print(f"[WARN] No saved model found at {model_path}. Using untrained agent.")
        agent = QLearningAgent()

    demo_episode(args.task, agent, render=True)


def cmd_random(_args):
    """Run a random agent on all tasks — sanity check that env works."""
    print("\nRandom agent sanity check\n" + "=" * 40)
    action_names = ["HOLD", "SELL_WHEAT", "SELL_CORN", "SELL_TOMATOES"]

    for task in ["task1", "task2", "task3"]:
        env = AgriMarketEnv(task=task, seed=42)
        state, _ = env.reset()
        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            step += 1

        print(
            f"  {task} | steps={step} | profit=${env.total_profit:.2f} "
            f"| rots={env.rot_events} | state_shape={state.shape}"
        )

    print("\nAll tasks ran without errors. Environment is healthy.")


def cmd_interactive():
    """Simple interactive menu when no sub-command is given."""
    print("\n" + "=" * 50)
    print("  AgriMarket Optimizer v1 — Team 404")
    print("=" * 50)
    print("  1. Train Task 1 (Basic Sales)")
    print("  2. Train Task 2 (Profit Maximization)")
    print("  3. Train Task 3 (Risk Management)")
    print("  4. Train All Tasks")
    print("  5. Evaluate saved model")
    print("  6. Demo episode (greedy agent)")
    print("  7. Random agent sanity check")
    print("  q. Quit")
    print("=" * 50)

    choice = input("\nEnter choice: ").strip().lower()

    task_map = {"1": "task1", "2": "task2", "3": "task3"}

    if choice in task_map:
        task = task_map[choice]
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        agent, env, data = train(task=task, agent_type="qlearning", verbose=True)
        plot_training(task, data, save_path=f"../plots/{task}_curve.png")
        evaluate(task, agent, n_episodes=100)
        agent.save(f"models/{task}_qlearning_agent.pkl")
        demo_episode(task, agent, render=True)

    elif choice == "4":
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        for task in ["task1", "task2", "task3"]:
            agent, env, data = train(task=task, agent_type="qlearning", verbose=True)
            plot_training(task, data, save_path=f"../plots/{task}_curve.png")
            evaluate(task, agent, n_episodes=100)
            agent.save(f"models/{task}_qlearning_agent.pkl")

    elif choice == "5":
        task = input("  Task (task1/task2/task3): ").strip()
        model_path = f"../models/{task}_qlearning_agent.pkl"
        if not os.path.exists(model_path):
            print(f"[ERROR] No model at {model_path}. Train first.")
            return
        agent = QLearningAgent()
        agent.load(model_path)
        agent.epsilon = 0.0
        evaluate(task, agent, n_episodes=100)

    elif choice == "6":
        task = input("  Task (task1/task2/task3): ").strip()
        model_path = f"../models/{task}_qlearning_agent.pkl"
        agent = QLearningAgent()
        if os.path.exists(model_path):
            agent.load(model_path)
            agent.epsilon = 0.0
        demo_episode(task, agent, render=True)

    elif choice == "7":
        cmd_random(None)

    elif choice == "q":
        print("Bye!")
    else:
        print("Invalid choice.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AgriMarket Optimizer v1 — Team 404",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train", help="Train an agent")
    p_train.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    p_train.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    p_train.add_argument("--all", action="store_true", help="Train all 3 tasks")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a saved model")
    p_eval.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    p_eval.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")

    # demo
    p_demo = subparsers.add_parser("demo", help="Watch one greedy episode")
    p_demo.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    p_demo.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")

    # random
    subparsers.add_parser("random", help="Random agent sanity check")

    args = parser.parse_args()

    np.random.seed(42)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "random":
        cmd_random(args)
    else:
        # No sub-command — launch interactive menu
        cmd_interactive()


if __name__ == "__main__":
    main()
