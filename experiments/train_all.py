"""
Train several independent Q-learning agents (different random seeds).

    python -m experiments.train_all --seeds 0 1 2 3 4 --episodes 30000
"""
import argparse

from experiments.train_q_learning import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=30_000)
    a = ap.parse_args()
    for s in a.seeds:
        print(f"\n===== training seed {s} =====")
        _, best = train(seed=s, episodes=a.episodes, eval_every=1_000, verbose=False)
        print(f"seed {s}: best validation return = {best:.2f}")