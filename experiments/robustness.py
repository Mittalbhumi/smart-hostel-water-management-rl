"""
Robustness / distribution-shift evaluation.

The agents were trained ONLY on the default simulator settings. Here the same
saved agents (no retraining, no re-tuning) and the same fixed baselines are run
on simulators whose parameters were changed, to see whether the learned policy
still behaves sensibly when conditions differ from training.

Run after experiments.evaluate:
    python -m experiments.robustness --seeds 0 1 2 3 4
"""
import argparse
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent
from baselines.greedy_fit import GreedyFitPolicy
from baselines.midnight_cutoff import MidnightCutoffPolicy
from environment.config import EnvConfig, TEST_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import bootstrap_ci, evaluate_policy

RESULTS = Path("results")
N_SCENARIOS = 300      # first 300 test seeds per shifted setting


def shifted_configs(base: EnvConfig):
    scale = lambda t, k: tuple(v * k for v in t)
    return {
        "Training setting": base,
        "Low inflow (-30%)": dataclasses.replace(base, inflow_l_per_step=scale(base.inflow_l_per_step, 0.7)),
        "High demand (+30%)": dataclasses.replace(base, demand_per_student_l=scale(base.demand_per_student_l, 1.3)),
        "Frequent exam nights (80%)": dataclasses.replace(base, late_night_event_prob=0.8),
        "Larger hostel (120 students)": dataclasses.replace(base, n_residents=120),
        "Small tank (3000 L)": dataclasses.replace(base, tank_capacity_l=3000.0),
        "Combined stress": dataclasses.replace(
            base, inflow_l_per_step=scale(base.inflow_l_per_step, 0.8),
            demand_per_student_l=scale(base.demand_per_student_l, 1.2),
            late_night_event_prob=0.6),
    }


def main(seeds):
    base = EnvConfig()
    agents = [QLearningAgent.load(RESULTS / f"q_table_seed{k}_best.npz") for k in seeds]
    seed_list = TEST_SEEDS[:N_SCENARIOS]
    rows = []
    for setting, cfg in shifted_configs(base).items():
        env = HostelWaterEnv(cfg)
        pols = {"Strict cutoff": [MidnightCutoffPolicy(cfg, "fit")],
                "Strongest fixed": [MidnightCutoffPolicy(cfg, "fit", 20)],
                "Greedy-FIT (no cutoff)": [GreedyFitPolicy(cfg)],
                "RL agent": agents}
        for name, plist in pols.items():
            per = []
            for p in plist:
                df = evaluate_policy(env, p, seed_list).set_index("scenario_seed")
                per.append(df)
            avg = sum(d[["reward", "unmet", "wasted", "delivered", "service_level"]]
                      for d in per) / len(per)       # mean over agent seeds, per scenario
            row = dict(setting=setting, policy=name)
            for m in ("reward", "unmet", "wasted", "delivered", "service_level"):
                lo, hi = bootstrap_ci(avg[m])
                row[m] = avg[m].mean(); row[m + "_ci95"] = (hi - lo) / 2
            rows.append(row)
        print(f"done: {setting}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "robustness.csv", index=False)
    show = out.pivot(index="setting", columns="policy", values="reward")
    print("\nMean return per setting (higher is better):")
    print(show[["RL agent", "Greedy-FIT (no cutoff)", "Strongest fixed", "Strict cutoff"]].round(1).to_string())
    show = out.pivot(index="setting", columns="policy", values="unmet")
    print("\nMean unmet demand L/night (lower is better):")
    print(show[["RL agent", "Greedy-FIT (no cutoff)", "Strongest fixed", "Strict cutoff"]].round(0).to_string())
    make_plot(out)


def make_plot(out):
    order = list(out["setting"].unique())
    pols = [("RL agent", "tab:blue"), ("Greedy-FIT (no cutoff)", "tab:gray"),
            ("Strongest fixed", "tab:green"), ("Strict cutoff", "tab:red")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (m, title) in zip(axes, (("reward", "Mean return (higher is better)"),
                                     ("unmet", "Unmet demand, L/night (lower is better)"))):
        w = 0.2
        for i, (p, c) in enumerate(pols):
            sub = out[out["policy"] == p].set_index("setting").loc[order]
            ax.bar(np.arange(len(order)) + (i - 1.5) * w, sub[m], w, yerr=sub[m + "_ci95"],
                   color=c, label=p, capsize=2)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([s.replace(" (", "\n(") for s in order], fontsize=7)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Robustness: agents trained on the default simulator, evaluated on shifted simulators "
                 f"(n={N_SCENARIOS} unseen scenarios each)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig7_robustness.png", dpi=150)
    print("Saved results/figures/fig7_robustness.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    main(ap.parse_args().seeds)