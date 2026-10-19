"""
Reward-weight sensitivity study.

The reward is  R = (alpha*used - beta*unmet - gamma*wasted)/100 L - lambda*[unmet>0].
Litres of unmet demand therefore cost (alpha + beta) per 100 L, litres wasted cost gamma.
This script sweeps beta (alpha = gamma = 1, lambda = 0.5 fixed), trains a fresh agent
for each value, and reports the PHYSICAL outcomes (litres), which are comparable
across reward settings.  Everything is measured on VALIDATION seeds, so the
final TEST seeds stay untouched.

Run from the project root (about 1 minute per beta value):
    python -m experiments.reward_sensitivity --betas 0 1 2 4 8 --seeds 0 1
    python -m experiments.reward_sensitivity --plot-only
"""
import argparse
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent, QLearningConfig
from baselines.greedy_fit import GreedyFitPolicy
from baselines.midnight_cutoff import MidnightCutoffPolicy
from environment.config import EnvConfig, TRAIN_SEED_END, TRAIN_SEED_START, VAL_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import evaluate_policy

RESULTS = Path("results")
CSV = RESULTS / "sensitivity.csv"
PHYS = ["delivered", "unmet", "wasted", "shortage_steps", "service_level"]


def train_quick(cfg, seed, episodes):
    """Same Q-learning loop as experiments.train_q_learning, minus logging/saving."""
    env = HostelWaterEnv(cfg)
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
    return agent


def run_sweep(betas, seeds, episodes):
    RESULTS.mkdir(exist_ok=True)
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    base = EnvConfig()
    if done.empty or "reference" not in set(done["kind"]):
        env = HostelWaterEnv(base)
        refs = [("Greedy-FIT, no cutoff", GreedyFitPolicy(base)),
                ("Strict cutoff (FIT)", MidnightCutoffPolicy(base, "fit")),
                ("Cutoff + 04:00 resume (FIT)", MidnightCutoffPolicy(base, "fit", 20))]
        rows = []
        for name, pol in refs:
            df = evaluate_policy(env, pol, VAL_SEEDS)
            rows.append(dict(kind="reference", label=name, beta=np.nan, seed=-1,
                             **{k: df[k].mean() for k in PHYS}))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    for beta in betas:
        for seed in seeds:
            if not done.empty and ((done["kind"] == "rl") & (done["beta"] == beta)
                                   & (done["seed"] == seed)).any():
                continue
            cfg = dataclasses.replace(base, beta_unmet=float(beta))
            agent = train_quick(cfg, seed, episodes)
            df = evaluate_policy(HostelWaterEnv(cfg), agent, VAL_SEEDS)
            row = dict(kind="rl", label=f"RL beta={beta:g}", beta=float(beta), seed=seed,
                       **{k: df[k].mean() for k in PHYS})
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(CSV, index=False)
            print(f"beta={beta:g} seed={seed}: unmet={row['unmet']:.0f} L  "
                  f"wasted={row['wasted']:.0f} L  delivered={row['delivered']:.0f} L", flush=True)
    done.to_csv(CSV, index=False)


def make_plot():
    df = pd.read_csv(CSV)
    rl = df[df["kind"] == "rl"].groupby("beta")[PHYS].mean().sort_index()
    print("\nRL, mean over training seeds (VALIDATION scenarios):")
    print(rl.round(1).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(rl["unmet"], rl["wasted"], "-o", color="tab:blue", label="RL (different reward weights)")
    for b, r in rl.iterrows():
        ax.annotate(f"β={b:g}" + (" (default)" if b == 2 else ""), (r["unmet"], r["wasted"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    marks = {"Greedy-FIT, no cutoff": ("s", "tab:gray"), "Strict cutoff (FIT)": ("X", "tab:red"),
             "Cutoff + 04:00 resume (FIT)": ("^", "tab:green")}
    for _, r in df[df["kind"] == "reference"].iterrows():
        mk, c = marks[r["label"]]
        ax.scatter(r["unmet"], r["wasted"], marker=mk, color=c, s=70, label=r["label"], zorder=3)
    ax.set_xlabel("Unmet demand per night (L)  - lower is better")
    ax.set_ylabel("Wasted water per night (L)  - lower is better")
    ax.set_title("Shortage vs waste trade-off (validation scenarios)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig6_reward_sensitivity.png", dpi=150)
    print("Saved results/figures/fig6_reward_sensitivity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=12_000)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run_sweep(a.betas, a.seeds, a.episodes)
    make_plot()