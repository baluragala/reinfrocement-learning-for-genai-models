"""Tiny reinforcement-learning worlds for Notebook 01. Pure NumPy, runs in seconds.

* `Bandit`: a recommender choosing which of five banners to show. One state, several actions, noisy reward.
  This is where exploration versus exploitation shows up.
* `GridWorld`: a robot on a grid with walls, a small nearby reward and a big distant one. It has states,
  actions, delayed reward and discounting.
* `q_learning`: the trial-and-error loop (observe, act, receive reward, improve) as twenty lines of code.
* `behaviour_cloning`: supervised learning on demonstrations, for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ bandit (recommendations)

BANNERS = ["tents", "rain gear", "footwear", "headlamps", "backpacks"]
CLICK_RATES = [0.04, 0.06, 0.05, 0.11, 0.08]     # hidden from the agent


@dataclass
class Bandit:
    rates: list = field(default_factory=lambda: list(CLICK_RATES))
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.rates[arm])


def run_bandit(epsilon: float, steps: int = 3000, seed: int = 0, rates=None) -> dict:
    """ε-greedy: explore a random banner with probability ε, otherwise show the best one so far."""
    env = Bandit(rates or list(CLICK_RATES), seed)
    rng = np.random.default_rng(seed + 1000)
    k = len(env.rates)
    counts, values = np.zeros(k), np.zeros(k)
    rewards, arms = np.zeros(steps), np.zeros(steps, dtype=int)
    for t in range(steps):
        arm = int(rng.integers(k)) if rng.random() < epsilon else int(np.argmax(values))
        r = env.pull(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]          # running average of observed reward
        rewards[t], arms[t] = r, arm
    best = int(np.argmax(env.rates))
    return {"epsilon": epsilon, "rewards": rewards, "arms": arms, "estimates": values, "counts": counts,
            "best_arm": best, "found_best": int(np.argmax(values)) == best,
            "regret": np.cumsum(max(env.rates) - np.array(env.rates)[arms])}


def bandit_sweep(epsilons=(0.0, 0.01, 0.1, 0.5), steps=3000, seeds=range(30)) -> dict:
    """Average several seeds, since one bandit run is mostly luck."""
    out = {}
    for eps in epsilons:
        runs = [run_bandit(eps, steps, s) for s in seeds]
        out[eps] = {"avg_reward_curve": np.mean([np.cumsum(r["rewards"]) / (np.arange(steps) + 1) for r in runs], 0),
                    "regret_curve": np.mean([r["regret"] for r in runs], 0),
                    "found_best": np.mean([r["found_best"] for r in runs]),
                    "clicks": np.mean([r["rewards"].sum() for r in runs])}
    return out


# ------------------------------------------------------------------ grid world (robot navigation)

ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}   # up, right, down, left
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

# S start · G big goal (+10, ends) · c small coin (+2, ends) · T turbo pad (+1 every visit, doesn't end) · # wall
NAV_MAP = ["S.......",
           ".######.",
           ".#......",
           "c#.###.G"]
RACE_MAP = ["S.......G",
            ".T.......",
            "........."]
TILE_REWARD = {"G": 10.0, "c": 2.0, "T": 1.0}
TERMINAL = {"G", "c"}


class GridWorld:
    """Deterministic grid. step() returns (next_state, reward, done). A state is a (row, col) pair."""

    def __init__(self, grid=None, step_cost=-0.1, max_steps=80, rewards=None):
        self.grid = [list(r) for r in (grid or NAV_MAP)]
        self.h, self.w = len(self.grid), len(self.grid[0])
        self.step_cost, self.max_steps = step_cost, max_steps
        self.rewards = {**TILE_REWARD, **(rewards or {})}
        self.start = next((r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] == "S")

    @property
    def n_states(self):
        return self.h * self.w

    def index(self, s):
        return s[0] * self.w + s[1]

    def tile(self, s):
        return self.grid[s[0]][s[1]]

    def reset(self):
        self.s, self.t = self.start, 0
        return self.s

    def step(self, a):
        dr, dc = ACTIONS[a]
        r, c = self.s[0] + dr, self.s[1] + dc
        if 0 <= r < self.h and 0 <= c < self.w and self.grid[r][c] != "#":
            self.s = (r, c)
        self.t += 1
        tile = self.tile(self.s)
        reward = self.step_cost + self.rewards.get(tile, 0.0)
        done = tile in TERMINAL or self.t >= self.max_steps
        return self.s, reward, done


def q_learning(env, episodes=2000, gamma=0.95, alpha=0.8, epsilon="decay", seed=0, log_first=0):
    """The RL loop. Observe the state, act (ε-greedy), receive a reward, and nudge Q(s, a) toward
    reward + γ · (best value of the next state). Returns the Q-table and per-episode returns.

    epsilon="decay" explores a lot at first (ε = 1) and less later (down to 0.05); a number keeps ε fixed."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, len(ACTIONS)))
    returns, lines = [], []
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (0.7 * episodes)) if epsilon == "decay" else epsilon
        s, done, total = env.reset(), False, 0.0
        while not done:
            i = env.index(s)
            a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(np.argmax(Q[i]))
            s2, r, done = env.step(a)
            target = r + (0.0 if done and env.tile(s2) in TERMINAL else gamma * Q[env.index(s2)].max())
            if ep < log_first:
                lines.append(f"ep {ep} · observe {s} · act {ARROWS[a]} · reward {r:+.1f} · "
                             f"Q{ARROWS[a]} {Q[i, a]:+.2f} → {Q[i, a] + alpha * (target - Q[i, a]):+.2f}")
            Q[i, a] += alpha * (target - Q[i, a])
            s, total = s2, total + r
        returns.append(total)
    return {"Q": Q, "returns": np.array(returns), "gamma": gamma, "log": lines}


def rollout(env, Q=None, policy=None, max_steps=None):
    """Follow a policy greedily and report where it ended up."""
    s, done, path, total, tiles = env.reset(), False, [env.start], 0.0, []
    steps = 0
    while not done and steps < (max_steps or env.max_steps):
        a = int(np.argmax(Q[env.index(s)])) if Q is not None else policy(s)
        s, r, done = env.step(a)
        path.append(s); total += r; steps += 1
        tiles.append(env.tile(s))
    end = env.tile(s)
    return {"return": round(total, 2), "steps": steps, "path": path, "ended_on": end,
            "reached_goal": end == "G", "turbo_visits": tiles.count("T")}


def policy_grid(env, Q) -> list[str]:
    """The greedy action in each cell, as arrows."""
    rows = []
    for r in range(env.h):
        row = ""
        for c in range(env.w):
            t = env.grid[r][c]
            row += t if t in "#GcT" else ARROWS[int(np.argmax(Q[env.index((r, c))]))]
        rows.append(row)
    return rows


# ------------------------------------------------------------------ supervised: imitation of demonstrations

def demonstrations(env, n=20, coin_share=0.7, seed=0):
    """A human operator's recorded drives. Most of the time they take the easy coin."""
    rng = np.random.default_rng(seed)
    to_coin = [2, 2]                                   # down, down → coin at (2, 0)
    to_goal = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2]        # an actual route to G is found below
    to_goal = _shortest_route(env, "G") or to_goal
    demos = []
    for _ in range(n):
        route = to_coin if rng.random() < coin_share else to_goal
        s = env.reset()
        traj = []
        for a in route:
            traj.append((s, a))
            s, _, done = env.step(a)
            if done:
                break
        demos.append(traj)
    return demos


def _shortest_route(env, target):
    from collections import deque
    start = env.start
    q, seen = deque([(start, [])]), {start}
    while q:
        s, path = q.popleft()
        if env.tile(s) == target:
            return path
        for a, (dr, dc) in ACTIONS.items():
            r, c = s[0] + dr, s[1] + dc
            if 0 <= r < env.h and 0 <= c < env.w and env.grid[r][c] != "#" and (r, c) not in seen:
                if env.tile((r, c)) in TERMINAL and env.tile((r, c)) != target:
                    continue
                seen.add((r, c))
                q.append(((r, c), path + [a]))
    return None


def behaviour_cloning(env, demos):
    """Supervised learning: for each state seen in the demos, copy the most common action (the label)."""
    table = {}
    for traj in demos:
        for s, a in traj:
            table.setdefault(s, []).append(a)
    labels = {s: max(set(v), key=v.count) for s, v in table.items()}
    rng = np.random.default_rng(0)
    return (lambda s: labels.get(s, int(rng.integers(4)))), labels
