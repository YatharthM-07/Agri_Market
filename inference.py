"""
AgriMarket Optimizer v1 — Inference Script
Team 404

Connects to the running OpenEnv HTTP server via REST API.
The evaluator spins up our Docker container (server at port 7860),
then runs this script to interact with the environment.

Required env vars:
  ENV_BASE_URL  — OpenEnv server URL  (default: http://localhost:7860)
  API_BASE_URL  — LLM API base URL    (default: https://api.openai.com/v1)
  MODEL_NAME    — LLM model name      (default: gpt-4o-mini)
  HF_TOKEN      — LLM API key

Usage:
  python inference.py                    # LLM agent, all tasks
  python inference.py --no-llm           # Rule-based agent (no API needed)
  python inference.py --task task3       # Single task
  python inference.py --episodes 5       # Override episode count
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import textwrap
from typing import List, Optional

import numpy as np
import requests
from openai import OpenAI

# ── Config from env vars ───────────────────────────────────────────────────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"

TEMPERATURE   = 0.0
MAX_TOKENS    = 10
EVAL_EPISODES = 20
ACTION_PATTERN = re.compile(r"\b([0-3])\b")

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert agricultural trading agent managing a crop warehouse.
    Each day you choose ONE action to maximise profit and avoid spoilage.

    STATE (10 values you receive each day):
      [0] wheat_quantity   — units in stock  (0-100)
      [1] corn_quantity    — units in stock  (0-80)
      [2] tomato_quantity  — units in stock  (0-50)
      [3] wheat_freshness  — days left before rot  (max 10)
      [4] corn_freshness   — days left before rot  (max 7)
      [5] tomato_freshness — days left before rot  (max 3)  ← spoils fastest!
      [6] wheat_price      — $/unit today  (3–9)
      [7] corn_price       — $/unit today  (2–7)
      [8] tomato_price     — $/unit today  (5–15)
      [9] news_feed        — 0=Normal, 1=Rain/prices_rising, 2=CRASH_WARNING

    ACTIONS:
      0 = HOLD          (do nothing, risk spoilage)
      1 = SELL_WHEAT    (sell all wheat at today's price)
      2 = SELL_CORN     (sell all corn at today's price)
      3 = SELL_TOMATOES (sell all tomatoes at today's price)

    CRITICAL RULES:
      - news_feed == 2 → CRASH WARNING: sell highest-value crop immediately
      - tomato_freshness <= 1 → sell tomatoes NOW or they rot
      - corn_freshness <= 1   → sell corn NOW
      - Sell higher-priced crops first when freshness allows

    Reply with ONLY a single digit: 0, 1, 2, or 3.
""").strip()


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post(path: str, body: dict = {}) -> dict:
    r = requests.post(f"{ENV_BASE_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def _get(path: str) -> dict:
    r = requests.get(f"{ENV_BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

def wait_for_server(retries: int = 10) -> bool:
    for i in range(retries):
        try:
            _get("/health")
            return True
        except Exception:
            time.sleep(1)
    return False


# ── Decision logic ─────────────────────────────────────────────────────────────

def rule_based_action(state: list) -> int:
    """Deterministic rule-based agent — no LLM needed."""
    wq, cq, tq = state[0], state[1], state[2]
    wf, cf, tf = state[3], state[4], state[5]
    wp, cp, tp = state[6], state[7], state[8]
    news = int(state[9])

    if news == 2:
        values = [wq * wp, cq * cp, tq * tp]
        best = int(np.argmax(values))
        if values[best] > 0:
            return best + 1

    if tf <= 1 and tq > 0: return 3
    if cf <= 1 and cq > 0: return 2
    if wf <= 2 and wq > 0: return 1

    if tp >= 13.0 and tq > 0: return 3
    if wp >= 8.0  and wq > 0: return 1
    if cp >= 6.0  and cq > 0: return 2

    if tq > 0: return 3
    if cq > 0: return 2
    if wq > 0: return 1
    return 0


def llm_action(client: OpenAI, state: list, day: int, history: List[str]) -> int:
    """Ask the LLM for an action. Falls back to rule-based on any error."""
    crops = ["wheat", "corn", "tomatoes"]
    news_str = {0: "Normal", 1: "Rain — prices rising",
                2: "*** CRASH WARNING — SELL NOW ***"}.get(int(state[9]), "Normal")
    lines = [f"Day {day} | Market: {news_str}"]
    for i, crop in enumerate(crops):
        if state[i] > 0:
            lines.append(
                f"  {crop:9s}: qty={int(state[i]):3d}  "
                f"freshness={int(state[3+i])}d  price=${state[6+i]:.2f}"
            )
        else:
            lines.append(f"  {crop:9s}: SOLD")
    if history:
        lines.append("Recent: " + " | ".join(history[-3:]))
    lines.append("Your action (0/1/2/3):")

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = resp.choices[0].message.content or ""
        m = ACTION_PATTERN.search(raw.strip())
        if m:
            return int(m.group(1))
    except Exception as exc:
        print(json.dumps({"type": "warn", "msg": f"LLM error: {exc}"}), flush=True)

    return rule_based_action(state)


# ── Episode runner ─────────────────────────────────────────────────────────────

ACTION_NAMES = ["HOLD", "SELL_WHEAT", "SELL_CORN", "SELL_TOMATOES"]

def run_episode(episode_num: int, task: str, seed: int,
                use_llm: bool, client: Optional[OpenAI]) -> dict:

    # Reset environment via HTTP
    obs = _post("/reset", {"seed": seed, "task": task})

    print(json.dumps({
        "type":    "START",
        "episode": episode_num,
        "task":    task,
        "seed":    seed,
    }), flush=True)

    history: List[str] = []
    step = 0

    while not obs["done"]:
        state = obs["state"]
        day   = obs["day"]

        action = llm_action(client, state, day, history) if use_llm \
                 else rule_based_action(state)

        obs = _post("/step", {"action": action})

        print(json.dumps({
            "type":         "STEP",
            "step":         step,
            "day":          obs["day"],
            "action":       action,
            "action_name":  ACTION_NAMES[action],
            "reward":       round(obs["reward"], 4),
            "total_profit": round(obs["total_profit"], 2),
            "rot_events":   obs["rot_events"],
            "done":         obs["done"],
            "crash_warned": obs["info"].get("crash_warned", False),
            "msp_applied":  obs["info"].get("msp_applied", False),
        }), flush=True)

        history.append(f"Day {obs['day']}: {ACTION_NAMES[action]}")
        step += 1

    print(json.dumps({
        "type":                     "END",
        "episode":                  episode_num,
        "task":                     task,
        "total_profit":             round(obs["total_profit"], 2),
        "rot_events":               obs["rot_events"],
        "crash_warnings_heeded":    obs.get("crash_warnings_heeded", 0),
        "crash_warnings_received":  obs.get("crash_warnings_received", 0),
        "steps":                    step,
    }), flush=True)

    return {
        "total_profit":            obs["total_profit"],
        "rot_events":              obs["rot_events"],
        "crash_warnings_heeded":   obs.get("crash_warnings_heeded", 0),
        "crash_warnings_received": obs.get("crash_warnings_received", 0),
        "steps":                   step,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AgriMarket Optimizer v1 — Inference")
    parser.add_argument("--task",     choices=["task1", "task2", "task3", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--no-llm",   action="store_true", help="Use rule-based agent")
    args = parser.parse_args()

    # Wait for env server to be ready
    if not wait_for_server():
        print(json.dumps({"type": "error", "msg": f"Server not reachable at {ENV_BASE_URL}"}))
        sys.exit(1)

    use_llm = not args.no_llm
    client  = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if use_llm else None
    tasks   = ["task1", "task2", "task3"] if args.task == "all" else [args.task]

    print(json.dumps({
        "type":    "config",
        "env_url": ENV_BASE_URL,
        "agent":   f"llm:{MODEL_NAME}" if use_llm else "rule-based",
        "tasks":   tasks,
        "episodes_per_task": args.episodes,
    }), flush=True)

    for task in tasks:
        for ep in range(args.episodes):
            try:
                run_episode(
                    episode_num = ep,
                    task        = task,
                    seed        = 42 + ep,
                    use_llm     = use_llm,
                    client      = client,
                )
            except Exception as exc:
                print(json.dumps({
                    "type": "error", "episode": ep, "task": task, "msg": str(exc)
                }), flush=True)


if __name__ == "__main__":
    main()
