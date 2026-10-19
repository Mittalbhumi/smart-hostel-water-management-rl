"""
State-feature ablation: which observations does the agent actually need?

For each variant, ONE (or two) observation features are hidden (replaced by a
constant) in training AND evaluation, a fresh agent is trained, and it is
evaluated on VALIDATION scenarios. The most important variant is "No current
demand": the main model sees the current step's demand before acting, which is
optimistic; this shows how much the result depends on that assumption.

Run from the project root (about 25 s per training run):
    python -m experiments.state_ablation --seeds 0 1 2
    python -m experiments.state_ablation --plot-only
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent, QLearningConfig
from environment.config import EnvConfig, TRAIN_SEED_END, TRAIN_SEED_START, VAL_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import evaluate_policy

RESULTS = Path("results")
CSV = RESULTS / "state_ablation.csv"
OBS_NAMES = ["tank level", "time of day", "occupancy", "recent demand", "current demand"]
VARIANTS = {
    "Full state (default)": [],
    "No current demand": [4],
    "No recent demand": [3],
    "No demand info (current + recent)": [3, 4],
    "No tank level": [0],
    "No time of day": [1],
    "No occupancy": [2],
}


class MaskedEnv:
    """Wraps the environment and replaces hidden observation entries by 0."""
    def __init__(self, env, hidden):
        self.env, self.hidden = env, list(hidden)

    def _mask(self, obs):
        obs = obs.copy()
        obs[self.hidden] = 0.0
        return obs

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        return self._mask(obs), info

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return self._mask(obs), r, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def train_masked(hidden, seed, episodes):
    cfg = EnvConfig()
    env = MaskedEnv(HostelWaterEnv(cfg), hidden)
    agent = QLearningAgent(cfg.n_actions,
                           QLearningConfig(eps_decay_episodes=int(0.5 * episodes)), seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    for ep in range(episodes):
        eps = agent.epsilon(ep)
        obs, _ = env.reset(int(rng.integers(TRAIN_SEED_START, TRAIN_SEED_END)))
        done = False
        while not done:
            a = agent.act(obs, eps)
            nxt, r, done, _ = env.step(a)
            agent.update(obs, a, r, nxt, done)
            obs = nxt
    return agent, env


def run(names, seeds, episodes):
    RESULTS.mkdir(exist_ok=True)
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    for name in names:
        for seed in seeds:
            if not done.empty and ((done["variant"] == name) & (done["seed"] == seed)).any():
                continue
            agent, env = train_masked(VARIANTS[name], seed, episodes)
            df = evaluate_policy(env, agent, VAL_SEEDS)
            row = dict(variant=name, seed=seed,
                       **{k: df[k].mean() for k in ("reward", "unmet", "wasted", "delivered")})
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(CSV, index=False)
            print(f"{name:<36} seed={seed}: return={row['reward']:6.2f} "
                  f"unmet={row['unmet']:6.0f} wasted={row['wasted']:6.0f}", flush=True)


def summarize_and_plot():
    df = pd.read_csv(CSV)
    g = df.groupby("variant")
    mean, std = g[["reward", "unmet", "wasted"]].mean(), g[["reward", "unmet", "wasted"]].std(ddof=1)
    order = [v for v in VARIANTS if v in mean.index]
    print(f"\nState ablation (VALIDATION scenarios, mean over {int(g['seed'].nunique().min())} seeds):")
    print(mean.loc[order].round(1).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (m, title) in zip(axes, (("reward", "Validation return (higher is better)"),
                                     ("unmet", "Unmet demand, L/night (lower is better)"))):
        ax.barh(order, mean.loc[order, m], xerr=std.loc[order, m].fillna(0),
                color=["tab:blue"] + ["tab:orange"] * (len(order) - 1), capsize=3)
        ax.set_title(title); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Which observations does the agent need? (feature hidden in training and evaluation)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig8_state_ablation.png", dpi=150)
    print("Saved results/figures/fig8_state_ablation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=12_000)
    ap.add_argument("--variants", type=int, nargs="+", default=list(range(len(VARIANTS))),
                    help="indices into the variant list (to run in chunks)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        names = list(VARIANTS)
        run([names[i] for i in a.variants], a.seeds, a.episodes)
    summarize_and_plot()