"""
Evaluate the trained RL agents and the midnight-cutoff baselines on UNSEEN
test scenarios (seeds 200000-200499), which were never used for training,
validation or tuning.

Run from the project root (after training):
    python -m experiments.evaluate --seeds 0 1 2 3 4

Baseline variants are ranked on the VALIDATION seeds; only then is everything
evaluated once on the test seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent
from baselines.midnight_cutoff import all_variants
from environment.config import EnvConfig, TEST_SEEDS, VAL_SEEDS
from environment.hostel_env import HostelWaterEnv

RESULTS = Path("results")
METRICS = ["reward", "delivered", "used", "unmet", "wasted", "overflow",
           "shortage_steps", "service_level"]


def rollout(env, seed, policy, trace=False):
    """Run one episode of `policy` on scenario `seed`. Returns metrics (+ trace)."""
    obs, info = env.reset(seed)
    done, steps = False, []
    while not done:
        a = policy.act(obs, 0.0)
        obs, _, done, step_info = env.step(a)
        step_info["action"] = a
        steps.append(step_info)
    tot = env.episode_totals()
    mid = env.cfg.midnight_step
    row = {k: tot[k] for k in ("reward", "delivered", "used", "unmet", "wasted",
                               "overflow", "shortage_steps", "demand")}
    row["service_level"] = tot["used"] / max(tot["demand"], 1e-9)
    row["unmet_after_midnight"] = sum(s["unmet"] for s in steps[mid:])
    row["exam_night"] = info["exam_night"]
    row["scenario_seed"] = seed
    return (row, steps) if trace else row


def evaluate_policy(env, policy, seeds):
    return pd.DataFrame([rollout(env, s, policy) for s in seeds])


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(seeds, resume_step=20):
    cfg = EnvConfig()
    env = HostelWaterEnv(cfg)
    RESULTS.mkdir(exist_ok=True)

    # ---- 1. rank baseline variants on VALIDATION seeds -----------------------
    variants = all_variants(cfg, resume_step)
    sel = []
    for p in variants:
        df = evaluate_policy(env, p, VAL_SEEDS)
        sel.append(dict(policy=p.name, resume=p.resume_step is not None,
                        val_return=df["reward"].mean(), val_unmet=df["unmet"].mean(),
                        val_wasted=df["wasted"].mean()))
    sel = pd.DataFrame(sel).sort_values("val_return", ascending=False)
    sel.to_csv(RESULTS / "baseline_selection_validation.csv", index=False)
    print("Baseline variants on VALIDATION seeds:\n", sel.round(2).to_string(index=False))

    strict = sel[~sel["resume"]].iloc[0]["policy"]        # as specified: off 00:00 -> 06:00
    strongest = sel.iloc[0]["policy"]                      # best fixed schedule overall
    by_name = {p.name: p for p in variants}

    # ---- 2. evaluate on TEST seeds -------------------------------------------
    frames = []
    for p in variants:
        df = evaluate_policy(env, p, TEST_SEEDS)
        df["policy"], df["agent_seed"] = p.name, -1
        frames.append(df)
    for k in seeds:
        agent = QLearningAgent.load(RESULTS / f"q_table_seed{k}_best.npz")
        df = evaluate_policy(env, agent, TEST_SEEDS)
        df["policy"], df["agent_seed"] = "RL (Q-learning)", k
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(RESULTS / "test_episodes.csv", index=False)

    # ---- 3. summary table (mean over scenarios, 95% bootstrap CI) -------------
    rl = allf[allf["policy"] == "RL (Q-learning)"]
    rl_per_scn = rl.groupby("scenario_seed")[METRICS + ["unmet_after_midnight"]].mean()
    rl_exam = rl.groupby("scenario_seed")["exam_night"].first()

    rows = []
    for name in ["RL (Q-learning)"] + [p.name for p in variants]:
        if name == "RL (Q-learning)":
            per = rl_per_scn
            seed_std = rl.groupby("agent_seed")["reward"].mean().std(ddof=1)
        else:
            per = allf[allf["policy"] == name].set_index("scenario_seed")[METRICS + ["unmet_after_midnight"]]
            seed_std = np.nan
        row = {"policy": name}
        for m in METRICS:
            lo, hi = bootstrap_ci(per[m])
            row[m] = per[m].mean()
            row[m + "_ci95"] = (hi - lo) / 2
        row["reward_std_across_train_seeds"] = seed_std
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "test_summary.csv", index=False)
    show = ["policy", "reward", "reward_ci95", "delivered", "unmet", "wasted",
            "shortage_steps", "service_level"]
    print(f"\nTEST results ({len(TEST_SEEDS)} unseen scenarios; RL = mean of {len(seeds)} training seeds):")
    print(summary[show].round(2).to_string(index=False))
    print("RL return std across training seeds:",
          round(summary.loc[0, "reward_std_across_train_seeds"], 3))

    # ---- 4. paired comparison vs the validation-selected baselines -------------
    comp = []
    for label, bname in (("strict midnight cutoff", strict), ("strongest fixed schedule", strongest)):
        base = allf[allf["policy"] == bname].set_index("scenario_seed")
        for subset, mask in (("all nights", rl_per_scn.index == rl_per_scn.index),
                             ("exam nights", rl_exam.values),
                             ("normal nights", ~rl_exam.values)):
            idx = rl_per_scn.index[mask]
            for m, better in (("reward", "higher"), ("unmet", "lower"), ("wasted", "lower")):
                d = rl_per_scn.loc[idx, m].values - base.loc[idx, m].values
                lo, hi = bootstrap_ci(d)
                win = np.mean(d > 0) if better == "higher" else np.mean(d < 0)
                comp.append(dict(baseline=label, baseline_policy=bname, subset=subset,
                                 metric=m, rl_minus_baseline=d.mean(), ci95_low=lo,
                                 ci95_high=hi, rl_better_in_frac_of_scenarios=win))
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS / "paired_comparison.csv", index=False)
    print("\nPaired comparison (RL minus baseline, per scenario):")
    print(comp.drop(columns="baseline_policy").round(2).to_string(index=False))
    print(f"\nstrict baseline = {strict}\nstrongest baseline = {strongest}")
    return summary, comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--resume-step", type=int, default=20,
                    help="step at which the 'resume' baseline variants restart supply (20 = 04:00)")
    a = ap.parse_args()
    main(a.seeds, a.resume_step)