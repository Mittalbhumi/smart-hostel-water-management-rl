"""
Ablation: is the RL gain just 'no midnight cutoff', or does learning add value?

Compares, on the same 500 unseen TEST scenarios:
  - the trained RL agents (read from results/test_episodes.csv)
  - Greedy-FIT with NO cutoff (a fixed, non-learned rule; no parameters to tune)

Run after experiments.evaluate:
    python -m experiments.ablation_no_cutoff
"""
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.greedy_fit import GreedyFitPolicy
from environment.config import EnvConfig, TEST_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import bootstrap_ci, evaluate_policy

RESULTS = Path("results")


def main():
    cfg = EnvConfig()
    env = HostelWaterEnv(cfg)
    greedy = evaluate_policy(env, GreedyFitPolicy(cfg), TEST_SEEDS).set_index("scenario_seed")
    eps = pd.read_csv(RESULTS / "test_episodes.csv")
    rl = eps[eps["policy"] == "RL (Q-learning)"]
    rl_scn = rl.groupby("scenario_seed").mean(numeric_only=True)
    exam = rl.groupby("scenario_seed")["exam_night"].first()

    rows = []
    for subset, mask in (("all nights", np.ones(len(exam), bool)),
                         ("exam nights", exam.values), ("normal nights", ~exam.values)):
        idx = exam.index[mask]
        for m in ("reward", "delivered", "unmet", "wasted"):
            d = rl_scn.loc[idx, m].values - greedy.loc[idx, m].values
            lo, hi = bootstrap_ci(d)
            rows.append(dict(subset=subset, metric=m, rl=rl_scn.loc[idx, m].mean(),
                             greedy_fit_no_cutoff=greedy.loc[idx, m].mean(),
                             rl_minus_greedy=d.mean(), ci95_low=lo, ci95_high=hi))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ablation_no_cutoff.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()