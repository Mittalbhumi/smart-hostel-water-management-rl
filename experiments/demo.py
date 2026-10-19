"""
Live demo: replay ONE simulated night with the trained RL agent and a fixed
midnight-cutoff baseline, step by step, side by side.

    python -m experiments.demo                    # first unseen EXAM night
    python -m experiments.demo --seed 200010      # a specific scenario seed
    python -m experiments.demo --normal           # first unseen NORMAL night
    python -m experiments.demo --agent-seed 2     # use another trained agent

All numbers are from the simulator (assumed parameters), not a real hostel.
"""
import argparse
from pathlib import Path

from agent.q_learning_agent import QLearningAgent
from baselines.midnight_cutoff import MidnightCutoffPolicy
from environment.config import ACTION_NAMES, EnvConfig, TEST_SEEDS
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import rollout

RESULTS = Path("results")


def pick_seed(env, want_exam):
    for s in TEST_SEEDS:
        env.reset(s)
        if env.exam_night == want_exam:
            return s
    raise RuntimeError("no matching scenario found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--normal", action="store_true", help="pick a normal (non-exam) night")
    ap.add_argument("--agent-seed", type=int, default=0)
    a = ap.parse_args()

    cfg = EnvConfig()
    env = HostelWaterEnv(cfg)
    seed = a.seed if a.seed is not None else pick_seed(env, not a.normal)
    agent = QLearningAgent.load(RESULTS / f"q_table_seed{a.agent_seed}_best.npz")
    base = MidnightCutoffPolicy(cfg, "fit", 20)       # strongest fixed schedule

    rl_row, rl = rollout(env, seed, agent, trace=True)
    env.reset(seed)
    kind = "EXAM night (late-night surge)" if env.exam_night else "normal night"
    b_row, bl = rollout(env, seed, base, trace=True)

    print(f"\nScenario seed {seed}: {kind}; starting tank level {env.initial_level:.0f} L "
          f"of {cfg.tank_capacity_l:.0f} L")
    print(f"Baseline = {base.name}\n")
    head = f"{'time':>5} {'demand':>7} | {'RL act':>6} {'tank':>6} {'unmet':>6} {'waste':>6} | " \
           f"{'BL act':>6} {'tank':>6} {'unmet':>6} {'waste':>6}"
    print(head)
    print("-" * len(head))
    for r, b in zip(rl, bl):
        h = r["clock_hour"]
        print(f"{int(h):02d}:{int((h % 1) * 60):02d} {r['demand']:7.0f} | "
              f"{ACTION_NAMES[r['action']]:>6} {r['level_after']:6.0f} {r['unmet']:6.0f} {r['wasted']:6.0f} | "
              f"{ACTION_NAMES[b['action']]:>6} {b['level_after']:6.0f} {b['unmet']:6.0f} {b['wasted']:6.0f}")
    print("-" * len(head))
    print(f"{'TOTAL':>5} {rl_row['demand']:7.0f} | {'':>6} {'':>6} {rl_row['unmet']:6.0f} {rl_row['wasted']:6.0f} | "
          f"{'':>6} {'':>6} {b_row['unmet']:6.0f} {b_row['wasted']:6.0f}")
    print(f"\nReturn:  RL agent {rl_row['reward']:.2f}   baseline {b_row['reward']:.2f}")
    print(f"Demand served: RL {rl_row['service_level']:.0%}   baseline {b_row['service_level']:.0%}")


if __name__ == "__main__":
    main()