"""
AgriMarket Optimizer v1 — OpenEnv-compatible HTTP Server
Team 404

Pure FastAPI server — no openenv_core dependency.
Works on Python 3.9+ locally and Python 3.10 in Docker.

Start: uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import uuid
import uvicorn
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from env import AgriMarketEnv as GymEnv


# ── Pydantic models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task: Optional[str] = None   # override AGRI_TASK env var per-request


class StepRequest(BaseModel):
    action:   int   = Field(default=0, ge=0, le=3,
                            description="0=Hold, 1=SellWheat, 2=SellCorn, 3=SellTomatoes")
    quantity: float = Field(default=1.0, ge=0.0, le=1.0,
                            description="Fraction of inventory to sell (0.0–1.0). Default 1.0 = sell all.")


class Observation(BaseModel):
    episode_id:                str
    done:                      bool
    reward:                    float
    state:                     List[float]
    day:                       int
    total_profit:              float
    rot_events:                int
    crash_warnings_heeded:     int
    crash_warnings_received:   int
    task:                      str
    info:                      Dict[str, Any]
    metadata:                  Dict[str, Any]


class StateSnapshot(BaseModel):
    episode_id:    Optional[str]
    task:          str
    day:           int
    step_count:    int
    inventory:     Dict[str, int]
    freshness:     Dict[str, int]
    market_prices: Dict[str, float]
    news_feed:     int
    total_profit:  float
    rot_events:    int


class HealthResponse(BaseModel):
    status: str
    env_id: str
    task:   str


# ── Environment singleton ──────────────────────────────────────────────────────

class _EnvManager:
    """Holds one environment instance per server process."""

    def __init__(self):
        self.task: str = os.environ.get("AGRI_TASK", "task1")
        self._gym: GymEnv = GymEnv(task=self.task)
        self._episode_id: str = str(uuid.uuid4())

    def reset(self, seed: Optional[int] = None, task: Optional[str] = None) -> Observation:
        if task and task != self.task:
            self.task = task
            self._gym = GymEnv(task=self.task)
        self._episode_id = str(uuid.uuid4())
        obs_arr, _ = self._gym.reset(seed=seed)
        return self._make_obs(obs_arr, reward=0.0, done=False, info={})

    def step(self, action: int, quantity: float = 1.0) -> Observation:
        act = max(0, min(action, 3))
        obs_arr, reward, done, truncated, info = self._gym.step(act, quantity=quantity)
        return self._make_obs(obs_arr, float(reward), done or truncated, info)

    def get_state(self) -> StateSnapshot:
        return StateSnapshot(
            episode_id    = self._episode_id,
            task          = self.task,
            day           = self._gym.day,
            step_count    = self._gym.day,
            inventory     = dict(self._gym.inventory),
            freshness     = dict(self._gym.freshness),
            market_prices = dict(self._gym.market_prices),
            news_feed     = int(self._gym.news_feed),
            total_profit  = float(self._gym.total_profit),
            rot_events    = int(self._gym.rot_events),
        )

    def _make_obs(self, obs_arr: np.ndarray, reward: float,
                  done: bool, info: Dict[str, Any]) -> Observation:
        return Observation(
            episode_id               = self._episode_id,
            done                     = done,
            reward                   = reward,
            state                    = obs_arr.tolist(),
            day                      = int(self._gym.day),
            total_profit             = float(self._gym.total_profit),
            rot_events               = int(self._gym.rot_events),
            crash_warnings_heeded    = int(self._gym.crash_warnings_heeded),
            crash_warnings_received  = int(self._gym.crash_warnings_received),
            task                     = self.task,
            info                     = {
                "rot":          info.get("rot", []),
                "crash_warned": bool(info.get("crash_warned", False)),
                "sold":         str(info.get("sold", "")),
            },
            metadata     = {
                "action_meanings": ["Hold", "SellWheat", "SellCorn", "SellTomatoes"],
                "state_keys": [
                    "wheat_qty", "corn_qty", "tomato_qty",
                    "wheat_freshness", "corn_freshness", "tomato_freshness",
                    "wheat_price", "corn_price", "tomato_price",
                    "news_feed",
                ],
            },
        )


_env = _EnvManager()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AgriMarket Optimizer v1",
    description = "OpenEnv-compatible RL environment for agricultural market decisions — Team 404",
    version     = "1.0.0",
)


# ── Landing page ───────────────────────────────────────────────────────────────

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AgriMarket Optimizer v1 — Team 404</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
      --green:  #00ff88;
      --gold:   #ffc32b;
      --red:    #ff4444;
      --blue:   #4dabf7;
      --purple: #cc5de8;
      --bg:     #07090f;
      --card:   #0d1017;
      --border: rgba(255,255,255,0.07);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: #e2e8f0;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ── animated grain overlay ── */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
      pointer-events: none; z-index: 0;
    }

    canvas#bg { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; opacity: 0.35; }

    .wrap { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 0 20px 80px; }

    /* ── HERO ── */
    .hero {
      text-align: center;
      padding: 72px 0 48px;
    }
    .hero-eyebrow {
      font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase;
      color: var(--green); margin-bottom: 16px; font-weight: 600;
    }
    .hero h1 {
      font-size: clamp(2.4rem, 6vw, 4rem);
      font-weight: 900; line-height: 1.1;
      background: linear-gradient(135deg, #fff 30%, var(--green) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 16px;
    }
    .hero-sub {
      font-size: 1.05rem; color: #7a8a9a; max-width: 560px; margin: 0 auto 32px;
      line-height: 1.6;
    }
    .badges { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 40px; }
    .badge {
      padding: 5px 14px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
      letter-spacing: 0.05em; border: 1px solid transparent;
    }
    .badge-green  { background: rgba(0,255,136,0.12); border-color: rgba(0,255,136,0.3); color: var(--green); }
    .badge-blue   { background: rgba(77,171,247,0.12); border-color: rgba(77,171,247,0.3); color: var(--blue); }
    .badge-gold   { background: rgba(255,195,43,0.12); border-color: rgba(255,195,43,0.3); color: var(--gold); }
    .badge-purple { background: rgba(204,93,232,0.12); border-color: rgba(204,93,232,0.3); color: var(--purple); }

    /* ── SCORE STRIP ── */
    .score-strip {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px;
    }
    .score-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px; padding: 24px 20px;
      text-align: center;
      position: relative; overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }
    .score-card:hover { transform: translateY(-3px); border-color: var(--green); }
    .score-card::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, var(--green), transparent);
    }
    .score-card .task-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #4a5568; margin-bottom: 8px; }
    .score-card .score-num  { font-size: 2.4rem; font-weight: 900; color: var(--green); line-height: 1; margin-bottom: 4px; }
    .score-card .score-desc { font-size: 0.78rem; color: #64748b; }
    .score-card .profit-val { font-size: 1.1rem; font-weight: 700; color: var(--gold); margin-top: 6px; }

    /* ── LIVE DEMO ── */
    .demo-section { margin-bottom: 32px; }
    .section-title {
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.18em;
      color: #4a5568; margin-bottom: 16px; font-weight: 600;
    }
    .demo-panel {
      background: var(--card); border: 1px solid var(--border); border-radius: 20px;
      padding: 28px; position: relative; overflow: hidden;
    }

    /* news banner */
    #news-banner {
      padding: 10px 18px; border-radius: 10px; font-size: 0.85rem; font-weight: 600;
      margin-bottom: 22px; display: flex; align-items: center; gap: 10px;
      transition: background 0.4s, color 0.4s;
    }
    .news-normal  { background: rgba(77,171,247,0.12); color: var(--blue); }
    .news-rain    { background: rgba(77,171,247,0.22); color: #74c0fc; }
    .news-crash   { background: rgba(255,68,68,0.2); color: var(--red); animation: crashPulse 0.6s infinite alternate; }
    @keyframes crashPulse { from { background: rgba(255,68,68,0.15); } to { background: rgba(255,68,68,0.35); } }

    /* crop cards */
    .crop-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 24px; }
    .crop-card {
      background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 14px; padding: 18px 16px;
      transition: border-color 0.3s, transform 0.2s;
      position: relative; overflow: hidden;
    }
    .crop-card.active-sell { border-color: var(--green); animation: sellFlash 0.5s ease; }
    .crop-card.rotting     { border-color: var(--red); }
    @keyframes sellFlash {
      0%   { background: rgba(0,255,136,0.2); }
      100% { background: rgba(255,255,255,0.03); }
    }
    .crop-icon  { font-size: 2rem; margin-bottom: 6px; }
    .crop-name  { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4a5568; margin-bottom: 2px; }
    .crop-qty   { font-size: 1.5rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
    .crop-price { font-size: 0.95rem; font-weight: 700; color: var(--gold); margin-top: 4px; }
    .fresh-bar-wrap { margin-top: 10px; }
    .fresh-label { font-size: 0.65rem; color: #4a5568; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .fresh-bar { height: 5px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
    .fresh-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease, background 0.4s; }
    .sold-overlay {
      position: absolute; inset: 0; border-radius: 14px;
      background: rgba(0,255,136,0.08);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.8rem; font-weight: 700; color: var(--green); letter-spacing: 0.12em;
      pointer-events: none;
    }

    /* stats row */
    .stats-row { display: flex; gap: 20px; margin-bottom: 22px; flex-wrap: wrap; }
    .stat-box {
      flex: 1; min-width: 100px;
      background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 14px; text-align: center;
    }
    .stat-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4a5568; margin-bottom: 4px; }
    .stat-val   { font-size: 1.4rem; font-weight: 800; }
    .stat-green { color: var(--green); }
    .stat-gold  { color: var(--gold); }
    .stat-red   { color: var(--red); }
    .stat-blue  { color: var(--blue); }

    /* price sparkline */
    .sparkline-row { display: flex; gap: 12px; margin-bottom: 22px; }
    .spark-card {
      flex: 1; background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 12px;
    }
    .spark-title { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4a5568; margin-bottom: 6px; }
    .spark-canvas { width: 100%; height: 40px; display: block; }

    /* action log */
    .log-box {
      background: rgba(0,0,0,0.4); border: 1px solid var(--border); border-radius: 10px;
      padding: 12px 14px; height: 130px; overflow-y: auto;
      font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
      scrollbar-width: thin; scrollbar-color: #2d3148 transparent;
    }
    .log-box::-webkit-scrollbar { width: 4px; }
    .log-box::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 4px; }
    .log-line { padding: 2px 0; line-height: 1.5; }
    .log-start  { color: var(--blue); }
    .log-hold   { color: #4a5568; }
    .log-sell   { color: var(--green); }
    .log-rot    { color: var(--red); }
    .log-crash  { color: var(--red); font-weight: 700; }
    .log-end    { color: var(--gold); font-weight: 600; }

    /* controls */
    .controls { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
    .btn {
      padding: 10px 22px; border-radius: 10px; font-size: 0.85rem; font-weight: 600;
      cursor: pointer; border: none; transition: all 0.2s; font-family: 'Inter', sans-serif;
      letter-spacing: 0.03em;
    }
    .btn-primary {
      background: linear-gradient(135deg, #00c864, #00ff88);
      color: #07090f;
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(0,255,136,0.3); }
    .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-ghost {
      background: rgba(255,255,255,0.05); color: #94a3b8;
      border: 1px solid var(--border);
    }
    .btn-ghost:hover { background: rgba(255,255,255,0.1); color: #e2e8f0; }
    .task-select {
      background: rgba(255,255,255,0.05); border: 1px solid var(--border);
      color: #e2e8f0; padding: 10px 14px; border-radius: 10px; font-size: 0.85rem;
      font-family: 'Inter', sans-serif; cursor: pointer; outline: none;
    }
    .task-select option { background: #0d1017; }
    .speed-select {
      background: rgba(255,255,255,0.05); border: 1px solid var(--border);
      color: #94a3b8; padding: 10px 12px; border-radius: 10px; font-size: 0.82rem;
      font-family: 'Inter', sans-serif; cursor: pointer; outline: none;
    }
    .speed-select option { background: #0d1017; }

    /* ── API ENDPOINTS ── */
    .api-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .ep {
      display: flex; align-items: center; gap: 12px;
      background: rgba(255,255,255,0.02); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 14px;
      transition: border-color 0.2s;
    }
    .ep:hover { border-color: rgba(255,255,255,0.14); }
    .method-tag {
      font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600;
      padding: 3px 8px; border-radius: 5px; min-width: 48px; text-align: center;
    }
    .m-get  { background: rgba(0,255,136,0.12); color: var(--green); }
    .m-post { background: rgba(77,171,247,0.12); color: var(--blue); }
    .ep-path { font-family: 'JetBrains Mono', monospace; color: #c4b5fd; font-size: 0.82rem; }
    .ep-desc { color: #4a5568; font-size: 0.75rem; margin-top: 2px; }

    /* ── INDIA SECTION ── */
    .india-card {
      background: linear-gradient(135deg, rgba(255,153,0,0.06), rgba(19,136,8,0.06));
      border: 1px solid rgba(255,153,0,0.18); border-radius: 20px;
      padding: 28px; margin-bottom: 32px; position: relative; overflow: hidden;
    }
    .india-card::before {
      content: '🇮🇳'; position: absolute; right: 24px; top: 20px; font-size: 2.8rem; opacity: 0.15;
    }
    .india-card h3 { font-size: 1rem; font-weight: 700; color: #ff9933; margin-bottom: 12px; }
    .india-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
    .india-item {
      background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px; padding: 12px 14px;
    }
    .india-item .ii-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4a5568; margin-bottom: 4px; }
    .india-item .ii-val   { font-size: 0.88rem; color: #e2e8f0; font-weight: 500; }

    /* ── FOOTER ── */
    .footer {
      text-align: center; color: #2d3748; font-size: 0.78rem; padding-top: 40px;
      border-top: 1px solid var(--border); margin-top: 20px;
    }
    .footer a { color: #4a5568; text-decoration: none; }
    .footer a:hover { color: var(--green); }

    /* ── DOCS BTN ── */
    .docs-btn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 12px 26px; border-radius: 10px;
      background: rgba(77,171,247,0.1); border: 1px solid rgba(77,171,247,0.3);
      color: var(--blue); font-weight: 600; font-size: 0.88rem; text-decoration: none;
      transition: all 0.2s; margin-top: 6px;
    }
    .docs-btn:hover { background: rgba(77,171,247,0.2); transform: translateY(-1px); }

    @media (max-width: 640px) {
      .score-strip  { grid-template-columns: 1fr; }
      .crop-grid    { grid-template-columns: 1fr; }
      .api-grid     { grid-template-columns: 1fr; }
      .india-grid   { grid-template-columns: 1fr; }
      .sparkline-row { flex-direction: column; }
    }
  </style>
</head>
<body>
<canvas id="bg"></canvas>

<div class="wrap">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-eyebrow">OpenEnv · Reinforcement Learning · India 🌾</div>
    <h1>AgriMarket Optimizer</h1>
    <p class="hero-sub">
      An AI agent that manages a perishable crop warehouse — balancing freshness decay,
      dynamic market prices, and real-world crash events. Built for Indian agriculture.
    </p>
    <div class="badges">
      <span class="badge badge-green">Team 404</span>
      <span class="badge badge-blue">OpenEnv v1</span>
      <span class="badge badge-gold">All Tasks ✓ 3/3</span>
      <span class="badge badge-purple">Q-Learning Agent</span>
    </div>
    <a class="docs-btn" href="/docs">📖 Swagger API Docs</a>
  </div>

  <!-- SCORES -->
  <div class="section-title">Evaluation Results</div>
  <div class="score-strip" style="margin-bottom:40px">
    <div class="score-card">
      <div class="task-label">Task 1 · No-Rot</div>
      <div class="score-num">1.0</div>
      <div class="score-desc">Target ≥ 0.90 · Zero rot events</div>
      <div class="profit-val">avg $1,140</div>
    </div>
    <div class="score-card">
      <div class="task-label">Task 2 · Profit Max</div>
      <div class="score-num">1.0</div>
      <div class="score-desc">Target ≥ 0.70 · $1000+ per episode</div>
      <div class="profit-val">avg $1,720</div>
    </div>
    <div class="score-card">
      <div class="task-label">Task 3 · Crash Mgmt</div>
      <div class="score-num">1.0</div>
      <div class="score-desc">Target ≥ 0.80 · Heed crash warnings</div>
      <div class="profit-val">avg $1,655</div>
    </div>
  </div>

  <!-- LIVE DEMO -->
  <div class="demo-section">
    <div class="section-title">Live AI Demo — Watch the Agent Trade in Real-Time</div>
    <div class="demo-panel">

      <!-- controls -->
      <div class="controls">
        <button class="btn btn-primary" id="btn-play" onclick="startEpisode()">▶ Run AI Agent</button>
        <button class="btn btn-ghost" id="btn-stop" onclick="stopEpisode()" disabled>■ Stop</button>
        <select class="task-select" id="task-sel">
          <option value="task3">Task 3 — Crash Events</option>
          <option value="task2">Task 2 — Dynamic Prices</option>
          <option value="task1">Task 1 — Fixed Prices</option>
        </select>
        <select class="speed-select" id="speed-sel">
          <option value="700">Normal</option>
          <option value="300">Fast</option>
          <option value="1400">Slow</option>
        </select>
      </div>

      <!-- news banner -->
      <div id="news-banner" class="news-normal">
        <span id="news-icon">📡</span>
        <span id="news-text">Market feed: Normal — start the agent to begin trading</span>
      </div>

      <!-- stats -->
      <div class="stats-row">
        <div class="stat-box">
          <div class="stat-label">Day</div>
          <div class="stat-val stat-blue" id="s-day">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Total Profit</div>
          <div class="stat-val stat-gold" id="s-profit">$0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Last Reward</div>
          <div class="stat-val stat-green" id="s-reward">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Rot Events</div>
          <div class="stat-val stat-red" id="s-rot">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Last Action</div>
          <div class="stat-val" id="s-action" style="font-size:0.9rem;color:#94a3b8">—</div>
        </div>
      </div>

      <!-- crop cards -->
      <div class="crop-grid">
        <div class="crop-card" id="card-wheat">
          <div class="crop-icon">🌾</div>
          <div class="crop-name">Wheat</div>
          <div class="crop-qty" id="qty-wheat">100</div>
          <div class="crop-price" id="price-wheat">$5.00/unit</div>
          <div class="fresh-bar-wrap">
            <div class="fresh-label"><span>Freshness</span><span id="fresh-wheat-label">10d</span></div>
            <div class="fresh-bar"><div class="fresh-fill" id="fresh-wheat" style="width:100%;background:var(--green)"></div></div>
          </div>
        </div>
        <div class="crop-card" id="card-corn">
          <div class="crop-icon">🌽</div>
          <div class="crop-name">Corn</div>
          <div class="crop-qty" id="qty-corn">80</div>
          <div class="crop-price" id="price-corn">$3.00/unit</div>
          <div class="fresh-bar-wrap">
            <div class="fresh-label"><span>Freshness</span><span id="fresh-corn-label">7d</span></div>
            <div class="fresh-bar"><div class="fresh-fill" id="fresh-corn" style="width:100%;background:var(--green)"></div></div>
          </div>
        </div>
        <div class="crop-card" id="card-tomatoes">
          <div class="crop-icon">🍅</div>
          <div class="crop-name">Tomatoes</div>
          <div class="crop-qty" id="qty-tomatoes">50</div>
          <div class="crop-price" id="price-tomatoes">$8.00/unit</div>
          <div class="fresh-bar-wrap">
            <div class="fresh-label"><span>Freshness</span><span id="fresh-tomatoes-label">3d</span></div>
            <div class="fresh-bar"><div class="fresh-fill" id="fresh-tomatoes" style="width:100%;background:var(--green)"></div></div>
          </div>
        </div>
      </div>

      <!-- sparklines -->
      <div class="sparkline-row">
        <div class="spark-card">
          <div class="spark-title">🌾 Wheat Price History</div>
          <canvas class="spark-canvas" id="spark-wheat"></canvas>
        </div>
        <div class="spark-card">
          <div class="spark-title">🌽 Corn Price History</div>
          <canvas class="spark-canvas" id="spark-corn"></canvas>
        </div>
        <div class="spark-card">
          <div class="spark-title">🍅 Tomato Price History</div>
          <canvas class="spark-canvas" id="spark-tomatoes"></canvas>
        </div>
      </div>

      <!-- log -->
      <div class="section-title" style="margin-bottom:8px">Agent Action Log</div>
      <div class="log-box" id="log-box">
        <div class="log-line log-hold">── Waiting for agent to start ──</div>
      </div>
    </div>
  </div>

  <!-- INDIA MECHANICS -->
  <div class="india-card">
    <h3>🇮🇳 India-Specific Mechanics</h3>
    <p style="color:#94a3b8;font-size:0.85rem;line-height:1.6">
      Inspired by the 2023 tomato price crash where wholesale tomato prices collapsed
      from ₹120/kg to ₹2/kg in 3 weeks — millions of farmers lost their entire harvest value.
      This environment models real APMC (Agricultural Produce Market Committee) mechanics.
    </p>
    <div class="india-grid">
      <div class="india-item">
        <div class="ii-label">MSP — Minimum Support Price</div>
        <div class="ii-val">Wheat ≥ ₹4.50/unit · Corn ≥ ₹2.50/unit (govt. floor)</div>
      </div>
      <div class="india-item">
        <div class="ii-label">Mandi Commission (APMC)</div>
        <div class="ii-val">2.5% fee on all crop sales — reflects real market structure</div>
      </div>
      <div class="india-item">
        <div class="ii-label">Crash Warning Signal</div>
        <div class="ii-val">news_feed=2 → prices drop 80% next step — sell immediately</div>
      </div>
      <div class="india-item">
        <div class="ii-label">Perishability</div>
        <div class="ii-val">Tomatoes: 3 days · Corn: 7 days · Wheat: 10 days</div>
      </div>
    </div>
  </div>

  <!-- API ENDPOINTS -->
  <div class="section-title">API Endpoints</div>
  <div class="api-grid" style="margin-bottom:40px">
    <div class="ep">
      <span class="method-tag m-get">GET</span>
      <div><div class="ep-path">/</div><div class="ep-desc">This dashboard</div></div>
    </div>
    <div class="ep">
      <span class="method-tag m-get">GET</span>
      <div><div class="ep-path">/health</div><div class="ep-desc">Health check — env_id, task status</div></div>
    </div>
    <div class="ep">
      <span class="method-tag m-post">POST</span>
      <div><div class="ep-path">/reset</div><div class="ep-desc">Reset env — returns initial observation</div></div>
    </div>
    <div class="ep">
      <span class="method-tag m-post">POST</span>
      <div><div class="ep-path">/step</div><div class="ep-desc">Take action 0–3, get next observation</div></div>
    </div>
    <div class="ep">
      <span class="method-tag m-get">GET</span>
      <div><div class="ep-path">/state</div><div class="ep-desc">Current env snapshot</div></div>
    </div>
    <div class="ep">
      <span class="method-tag m-get">GET</span>
      <div><div class="ep-path">/docs</div><div class="ep-desc">Interactive Swagger UI</div></div>
    </div>
  </div>

  <div class="footer">
    AgriMarket Optimizer v1 &nbsp;·&nbsp; Team 404 &nbsp;·&nbsp;
    <a href="https://huggingface.co/NSRexe" target="_blank">huggingface.co/NSRexe</a>
  </div>
</div>

<script>
/* ── Animated field background ── */
(function(){
  const c = document.getElementById('bg');
  const ctx = c.getContext('2d');
  let W, H, particles = [];

  function resize(){
    W = c.width  = window.innerWidth;
    H = c.height = window.innerHeight;
  }

  function Particle(){
    this.x = Math.random() * W;
    this.y = Math.random() * H;
    this.vy = 0.2 + Math.random() * 0.5;
    this.vx = (Math.random() - 0.5) * 0.3;
    this.size = 1 + Math.random() * 2;
    this.alpha = 0.2 + Math.random() * 0.5;
    this.color = Math.random() > 0.6 ? '#00ff88' : Math.random() > 0.5 ? '#ffc32b' : '#4dabf7';
  }

  function init(){
    resize();
    particles = Array.from({length: 120}, () => new Particle());
  }

  function tick(){
    ctx.clearRect(0,0,W,H);
    for(let p of particles){
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if(p.y > H + 5){ p.y = -5; p.x = Math.random()*W; }
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', resize);
  init(); tick();
})();

/* ── Price sparklines ── */
const priceHistory = { wheat: [], corn: [], tomatoes: [] };
const sparkColors  = { wheat: '#ffc32b', corn: '#a9e34b', tomatoes: '#ff6b6b' };

function drawSpark(id, data, color){
  const c = document.getElementById(id);
  if(!c) return;
  const ctx = c.getContext('2d');
  c.width  = c.offsetWidth  * devicePixelRatio;
  c.height = c.offsetHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = c.offsetWidth, H = c.offsetHeight;
  ctx.clearRect(0,0,W,H);
  if(data.length < 2) return;
  const mn = Math.min(...data), mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v,i)=>({
    x: (i/(data.length-1))*W,
    y: H - ((v-mn)/range)*(H-6) - 3
  }));
  // fill
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, color+'44');
  grad.addColorStop(1, color+'00');
  ctx.beginPath();
  ctx.moveTo(pts[0].x, H);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length-1].x, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
  // line
  ctx.beginPath();
  pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  // last dot
  const last = pts[pts.length-1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, 3, 0, Math.PI*2);
  ctx.fillStyle = color;
  ctx.fill();
}

/* ── Smart partial-sell agent ── */
// Returns {action, quantity} where quantity is 0.0–1.0 fraction of inventory
function ruleDecision(state){
  const [wq,cq,tq,wf,cf,tf,wp,cp,tp,news] = state;
  const ni = Math.round(news);

  // ── CRASH WARNING: dump 100% of most perishable crop NOW ──
  if(ni === 2){
    if(tq > 0) return {action:3, quantity:1.0};
    if(cq > 0) return {action:2, quantity:1.0};
    if(wq > 0) return {action:1, quantity:1.0};
  }

  // ── EXPIRY URGENCY: sell all of dying crop ──
  if(tf <= 1 && tq > 0) return {action:3, quantity:1.0};
  if(cf <= 2 && cq > 0) return {action:2, quantity:1.0};
  if(wf <= 3 && wq > 0) return {action:1, quantity:1.0};

  // ── FRESHNESS WARNING: start partial sells (50%) ──
  if(tf <= 2 && tq > 0) return {action:3, quantity:0.5};
  if(cf <= 3 && cq > 0) return {action:2, quantity:0.5};
  if(wf <= 5 && wq > 0) return {action:1, quantity:0.5};

  // ── PEAK PRICE: sell a third to lock in gains without committing fully ──
  if(tp >= 13 && tq > 0) return {action:3, quantity:0.33};
  if(wp >= 8.5 && wq > 0) return {action:1, quantity:0.33};
  if(cp >= 6.5 && cq > 0) return {action:2, quantity:0.33};

  // ── RAIN + GOOD PRICE: small 25% sell ──
  if(ni === 1){
    if(tp >= 11 && tq > 0) return {action:3, quantity:0.25};
    if(wp >= 7.5 && wq > 0) return {action:1, quantity:0.25};
  }

  // ── LATE GAME: price ok + running out of time → sell 25% ──
  const daysLeft = 15 - (wf + cf + tf) / 3; // rough estimate
  if(daysLeft > 10){
    if(tq > 30 && tp >= 9)  return {action:3, quantity:0.25};
    if(wq > 50 && wp >= 6)  return {action:1, quantity:0.25};
    if(cq > 40 && cp >= 5)  return {action:2, quantity:0.25};
  }

  return {action:0, quantity:0};
}

const ACTION_NAMES = ['HOLD','SELL WHEAT','SELL CORN','SELL TOMATOES'];
const CROP_KEYS    = ['wheat','corn','tomatoes'];
const FRESH_MAX    = {wheat:10, corn:7, tomatoes:3};
let running = false, stopFlag = false;

function addLog(text, cls){
  const box = document.getElementById('log-box');
  const d = document.createElement('div');
  d.className = 'log-line ' + cls;
  const ts = new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  d.textContent = '[' + ts + '] ' + text;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
  if(box.childElementCount > 200) box.removeChild(box.firstElementChild);
}

function updateCropCard(crop, qty, freshness, price, action, actionCrop){
  const card  = document.getElementById('card-'+crop);
  const qtyEl = document.getElementById('qty-'+crop);
  const prEl  = document.getElementById('price-'+crop);
  const fbEl  = document.getElementById('fresh-'+crop);
  const flEl  = document.getElementById('fresh-'+crop+'-label');

  qtyEl.textContent = qty;
  prEl.textContent  = '$' + price.toFixed(2) + '/unit';
  flEl.textContent  = freshness + 'd';

  const maxF = FRESH_MAX[crop];
  const pct  = Math.max(0, (freshness / maxF) * 100);
  const col  = freshness <= 1 ? 'var(--red)' : freshness <= 2 ? 'var(--gold)' : 'var(--green)';
  fbEl.style.width      = pct + '%';
  fbEl.style.background = col;

  card.classList.remove('active-sell','rotting');
  // remove old sold overlay
  const old = card.querySelector('.sold-overlay');
  if(old) old.remove();

  if(qty === 0){
    const ov = document.createElement('div');
    ov.className = 'sold-overlay';
    ov.textContent = 'SOLD OUT';
    card.appendChild(ov);
  }
  if(freshness <= 1 && qty > 0) card.classList.add('rotting');
  if(action > 0 && CROP_KEYS[action-1] === crop) card.classList.add('active-sell');
}

function updateNews(newsVal){
  const banner = document.getElementById('news-banner');
  const icon   = document.getElementById('news-icon');
  const text   = document.getElementById('news-text');
  const ni = Math.round(newsVal);
  banner.className = '';
  if(ni === 2){
    banner.classList.add('news-crash');
    icon.textContent = '🚨';
    text.textContent = 'CRASH WARNING — Sell highest-value crop immediately! Prices drop 80% next step!';
  } else if(ni === 1){
    banner.classList.add('news-rain');
    icon.textContent = '🌧️';
    text.textContent = 'Rain signal — prices may rise. Consider holding if freshness allows.';
  } else {
    banner.classList.add('news-normal');
    icon.textContent = '📡';
    text.textContent = 'Market feed: Normal conditions.';
  }
}

function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

async function startEpisode(){
  if(running) return;
  running  = true;
  stopFlag = false;
  document.getElementById('btn-play').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  document.getElementById('log-box').innerHTML = '';
  priceHistory.wheat = []; priceHistory.corn = []; priceHistory.tomatoes = [];

  const task  = document.getElementById('task-sel').value;
  const speed = parseInt(document.getElementById('speed-sel').value);

  addLog('Episode start → task=' + task, 'log-start');

  try {
    let obs = await fetch('/reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task})
    }).then(r=>r.json());

    while(!obs.done && !stopFlag){
      const state = obs.state;
      const {action, quantity} = ruleDecision(state);

      // update UI
      const news = Math.round(state[9]);
      updateNews(news);
      CROP_KEYS.forEach((crop,i)=>{
        updateCropCard(crop, state[i], state[3+i], state[6+i], action, CROP_KEYS[action-1]);
        priceHistory[crop].push(state[6+i]);
        if(priceHistory[crop].length > 20) priceHistory[crop].shift();
        drawSpark('spark-'+crop, priceHistory[crop], sparkColors[crop]);
      });

      document.getElementById('s-day').textContent    = obs.day;
      document.getElementById('s-profit').textContent = '$' + obs.total_profit.toFixed(0);
      document.getElementById('s-rot').textContent    = obs.rot_events;

      // format action label with quantity
      const pct = quantity > 0 ? ' (' + Math.round(quantity*100) + '%)' : '';
      document.getElementById('s-action').textContent = ACTION_NAMES[action] + pct;

      // take step — send quantity so server does partial sell
      obs = await fetch('/step', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action, quantity})
      }).then(r=>r.json());

      document.getElementById('s-reward').textContent = obs.reward > 0
        ? '+$' + obs.reward.toFixed(0)
        : obs.reward.toFixed(0);
      document.getElementById('s-reward').style.color = obs.reward > 0 ? 'var(--green)' : obs.reward < 0 ? 'var(--red)' : '#94a3b8';

      // log
      const qLabel = quantity > 0 && action > 0 ? ' [' + Math.round(quantity*100) + '%]' : '';
      if(news === 2)
        addLog('🚨 CRASH! ' + ACTION_NAMES[action] + qLabel + ' · reward=' + obs.reward.toFixed(0), 'log-crash');
      else if(action > 0)
        addLog('Day ' + obs.day + ' → ' + ACTION_NAMES[action] + qLabel + ' · +$' + obs.reward.toFixed(0) + ' · total=$' + obs.total_profit.toFixed(0), 'log-sell');
      else
        addLog('Day ' + obs.day + ' → HOLD · watching prices · total=$' + obs.total_profit.toFixed(0), 'log-hold');

      if(obs.rot_events > 0)
        addLog('🔴 ROT EVENT! Crop lost to spoilage', 'log-rot');

      await sleep(speed);
    }

    addLog('═══ Episode done · Profit: $' + obs.total_profit.toFixed(2) + ' · Rot: ' + obs.rot_events, 'log-end');
    document.getElementById('s-profit').textContent = '$' + obs.total_profit.toFixed(0);

  } catch(e){
    addLog('Error: ' + e.message, 'log-rot');
  }

  running = false;
  document.getElementById('btn-play').disabled = false;
  document.getElementById('btn-stop').disabled = true;
}

function stopEpisode(){
  stopFlag = true;
  addLog('── Stopped by user ──', 'log-hold');
  document.getElementById('btn-stop').disabled = true;
}
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing():
    return HTMLResponse(content=_LANDING_HTML, status_code=200)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", env_id="agrimarket-optimizer-v1", task=_env.task)


@app.post("/reset", response_model=Observation)
async def reset(req: ResetRequest = ResetRequest()):
    return _env.reset(seed=req.seed, task=req.task)


@app.post("/step", response_model=Observation)
async def step(req: StepRequest):
    return _env.step(req.action, quantity=req.quantity)


@app.get("/state", response_model=StateSnapshot)
async def state():
    return _env.get_state()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
