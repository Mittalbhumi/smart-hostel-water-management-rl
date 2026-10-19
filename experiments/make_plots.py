"""
Create all report figures in results/figures/.

Run from the project root (after train_all and evaluate):
    python -m experiments.make_plots --seeds 0 1 2 3 4
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.q_learning_agent import QLearningAgent
from baselines.midnight_cutoff import all_variants
from environment.config import EnvConfig, TEST_SEEDS, VAL_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import evaluate_policy, rollout

RESULTS = Path("results")
FIG = RESULTS / "figures"
WINDOW = 500


def smooth(x, w=WINDOW):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def mean_band(ax, curves, x, label, color=None):
    arr = np.vstack(curves)
    m, s = arr.mean(axis=0), arr.std(axis=0)
    line, = ax.plot(x, m, label=label, color=color, lw=1.6)
    ax.fill_between(x, m - s, m + s, alpha=0.2, color=line.get_color())


def fig_training_curves(seeds, strict_val, best_val):
    logs = [pd.read_csv(RESULTS / f"training_log_seed{k}.csv") for k in seeds]
    ep = logs[0]["episode"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("reward", "Episode return (reward units)"),
              ("delivered", "Water delivered per night (L)"),
              ("unmet", "Unmet demand per night (L)"),
              ("wasted", "Wasted water per night (L)")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        mean_band(ax, [smooth(l[col]) for l in logs], ep, "RL during training (exploring)")
        ax.set_title(title); ax.set_xlabel("Training episode"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Training curves: mean ± std over {len(seeds)} training seeds "
                 f"({WINDOW}-episode moving average)")
    fig.tight_layout(); fig.savefig(FIG / "fig1_training_curves.png", dpi=150); plt.close(fig)

    vlogs = [pd.read_csv(RESULTS / f"validation_log_seed{k}.csv") for k in seeds]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    mean_band(ax, [v["reward"].to_numpy() for v in vlogs], vlogs[0]["episode"].to_numpy(),
              "RL greedy policy (validation)")
    ax.axhline(strict_val, color="tab:red", ls="--", label="Strict midnight cutoff (validation)")
    ax.axhline(best_val, color="tab:green", ls=":", label="Strongest fixed schedule (validation)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Mean validation return")
    ax.set_title("Validation return during training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_validation_return.png", dpi=150); plt.close(fig)


def fig_test_bars(summary, strict, strongest):
    names = ["RL (Q-learning)", strict, strongest]
    labels = ["RL agent", "Strict midnight\ncutoff", "Strongest fixed\nschedule"]
    cols = ["tab:blue", "tab:red", "tab:green"]
    panels = [("reward", "Return"), ("delivered", "Water delivered (L)"),
              ("unmet", "Unmet demand (L)"), ("wasted", "Wasted water (L)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (m, title) in zip(axes, panels):
        rows = [summary[summary["policy"] == n].iloc[0] for n in names]
        ax.bar(labels, [r[m] for r in rows], yerr=[r[m + "_ci95"] for r in rows],
               color=cols, capsize=4)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Unseen test scenarios (n={len(TEST_SEEDS)}); error bars = 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(FIG / "fig3_test_comparison.png", dpi=150); plt.close(fig)


def fig_exam_vs_normal(episodes, strict, strongest, rl_name="RL (Q-learning)"):
    rl = episodes[episodes["policy"] == rl_name].groupby("scenario_seed").mean(numeric_only=True)
    exam = episodes[episodes["policy"] == rl_name].groupby("scenario_seed")["exam_night"].first()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (flag, title) in zip(axes, ((False, "Normal nights"), (True, "Exam nights (late-night surge)"))):
        vals, labs, cols = [], [], []
        idx = exam.index[exam.values == flag]
        vals.append(rl.loc[idx, "unmet"].mean()); labs.append("RL agent"); cols.append("tab:blue")
        for n, lab, c in ((strict, "Strict cutoff", "tab:red"), (strongest, "Strongest fixed", "tab:green")):
            b = episodes[episodes["policy"] == n].set_index("scenario_seed")
            vals.append(b.loc[idx, "unmet"].mean()); labs.append(lab); cols.append(c)
        ax.bar(labs, vals, color=cols); ax.set_title(f"{title} (n={len(idx)})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean unmet demand per night (L)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_exam_vs_normal.png", dpi=150); plt.close(fig)


def fig_timeline(env, rl_agent, strict_pol, strongest_pol):
    seed = next(s for s in TEST_SEEDS if (env.reset(s), env.exam_night)[1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    hours = [env.cfg.clock_hour(t) for t in range(env.cfg.n_steps)]
    labels = [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours]
    xs = np.arange(env.cfg.n_steps)
    for pol, name, c in ((rl_agent, "RL agent", "tab:blue"), (strict_pol, "Strict cutoff", "tab:red"),
                         (strongest_pol, "Strongest fixed", "tab:green")):
        _, steps = rollout(env, seed, pol, trace=True)
        axes[0].plot(xs, [s["level_after"] for s in steps], label=name, color=c, lw=1.8)
        axes[1].plot(xs, [s["unmet"] for s in steps], label=name, color=c, lw=1.8, marker="o", ms=3)
    demand = [s["demand"] for s in steps]
    axes[1].plot(xs, demand, color="gray", ls="--", label="Demand")
    axes[0].set_ylabel("Tank level (L)"); axes[1].set_ylabel("Unmet demand per step (L)")
    axes[0].axvline(env.cfg.midnight_step, color="k", ls=":"); axes[1].axvline(env.cfg.midnight_step, color="k", ls=":")
    axes[1].set_xticks(xs[::2]); axes[1].set_xticklabels(labels[::2], rotation=45)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    axes[0].set_title(f"One unseen exam-night scenario (seed {seed}); dotted line = midnight")
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig5_example_night.png", dpi=150); plt.close(fig)


def main(seeds):
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(); env = HostelWaterEnv(cfg)
    sel = pd.read_csv(RESULTS / "baseline_selection_validation.csv")
    strict = sel[~sel["resume"]].iloc[0]; strongest = sel.iloc[0]
    summary = pd.read_csv(RESULTS / "test_summary.csv")
    episodes = pd.read_csv(RESULTS / "test_episodes.csv")

    fig_training_curves(seeds, strict["val_return"], strongest["val_return"])
    fig_test_bars(summary, strict["policy"], strongest["policy"])
    fig_exam_vs_normal(episodes, strict["policy"], strongest["policy"])
    by_name = {p.name: p for p in all_variants(cfg)}
    rl = QLearningAgent.load(RESULTS / f"q_table_seed{seeds[0]}_best.npz")
    fig_timeline(env, rl, by_name[strict["policy"]], by_name[strongest["policy"]])
    print(f"Saved 5 figures to {FIG.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    main(ap.parse_args().seeds)