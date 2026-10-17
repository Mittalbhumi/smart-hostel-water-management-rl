"""
Train the tabular Q-learning agent.

Run from the project root:
    python -m experiments.train_q_learning --seed 0 --episodes 30000

Outputs (in results/):
    q_table_seed<k>_final.npz   Q-table after the last episode
    q_table_seed<k>_best.npz    Q-table with the best mean VALIDATION return
    training_log_seed<k>.csv    per-episode training metrics
    validation_log_seed<k>.csv  periodic greedy evaluation on validation seeds
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent, QLearningConfig
from environment.config import (EnvConfig, TRAIN_SEED_END, TRAIN_SEED_START,
                                VAL_SEEDS)
from environment.hostel_env import HostelWaterEnv

RESULTS = Path("results")


def evaluate_greedy(agent, env, seeds):
    """Mean metrics of the greedy policy over the given scenario seeds."""
    rows = []
    for s in seeds:
        obs, _ = env.reset(s)
        done = False
        while not done:
            obs, _, done, _ = env.step(agent.act(obs, epsilon=0.0))
        rows.append(env.episode_totals())
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in
            ("reward", "used", "unmet", "wasted", "delivered", "shortage_steps")}


def train(seed=0, episodes=30_000, eval_every=1_000, qcfg=None, verbose=True):
    cfg = EnvConfig()
    env = HostelWaterEnv(cfg)
    qcfg = qcfg or QLearningConfig(eps_decay_episodes=int(0.5 * episodes))
    agent = QLearningAgent(cfg.n_actions, qcfg, seed=seed)
    scenario_rng = np.random.default_rng(10_000 + seed)   # picks TRAIN scenarios only

    train_rows, val_rows = [], []
    best_val, best_q, best_visits = -np.inf, agent.q.copy(), agent.visits.copy()
    t0 = time.time()

    for ep in range(episodes):
        eps = agent.epsilon(ep)
        scenario_seed = int(scenario_rng.integers(TRAIN_SEED_START, TRAIN_SEED_END))
        obs, _ = env.reset(scenario_seed)
        done = False
        while not done:
            a = agent.act(obs, eps)
            nxt, r, done, _ = env.step(a)
            agent.update(obs, a, r, nxt, done)
            obs = nxt
        tot = env.episode_totals()
        train_rows.append(dict(episode=ep, epsilon=eps, reward=tot["reward"],
                               demand=tot["demand"], used=tot["used"],
                               unmet=tot["unmet"], wasted=tot["wasted"],
                               delivered=tot["delivered"],
                               shortage_steps=tot["shortage_steps"]))

        if (ep + 1) % eval_every == 0:
            v = evaluate_greedy(agent, env, VAL_SEEDS)
            val_rows.append(dict(episode=ep + 1, **v))
            if v["reward"] > best_val:
                best_val, best_q = v["reward"], agent.q.copy()
                best_visits = agent.visits.copy()
            if verbose:
                print(f"ep {ep + 1:>6} | eps {eps:.2f} | val return {v['reward']:7.2f} "
                      f"| val unmet {v['unmet']:7.1f} L | val wasted {v['wasted']:7.1f} L "
                      f"| {time.time() - t0:5.1f}s")

    RESULTS.mkdir(exist_ok=True)
    agent.save(RESULTS / f"q_table_seed{seed}_final.npz")
    best_agent = QLearningAgent(cfg.n_actions, qcfg, seed=seed)
    best_agent.q, best_agent.visits = best_q, best_visits
    best_agent.save(RESULTS / f"q_table_seed{seed}_best.npz")
    pd.DataFrame(train_rows).to_csv(RESULTS / f"training_log_seed{seed}.csv", index=False)
    pd.DataFrame(val_rows).to_csv(RESULTS / f"validation_log_seed{seed}.csv", index=False)
    if verbose:
        print(f"\nBest validation return: {best_val:.2f}")
        print(f"State coverage: {agent.coverage():.1%} of 1080 states visited")
        print(f"Saved Q-tables and logs to {RESULTS.resolve()}")
    return agent, best_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=30_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    a = p.parse_args()
    train(a.seed, a.episodes, a.eval_every)