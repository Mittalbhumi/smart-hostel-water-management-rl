"""Validation checks for the hostel water environment.
Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from environment.config import EnvConfig, TEST_SEEDS, VAL_SEEDS
from environment.hostel_env import HostelWaterEnv

CFG = EnvConfig()
LONG_VOL = CFG.action_volumes_l.max()


def run_episode(env, seed, policy):
    obs, _ = env.reset(seed)
    done, steps = False, []
    while not done:
        obs, r, done, info = env.step(policy(obs))
        steps.append(info)
    return env.episode_totals(), steps


def always(a):
    return lambda obs: a


def fit_demand(obs):
    """Test-only sanity policy: smallest action whose volume covers current
    demand. Used ONLY to check the reward is not hackable; it is not the agent."""
    need = obs[4] * LONG_VOL
    for a, vol in enumerate(CFG.action_volumes_l):
        if vol >= need:
            return a
    return CFG.n_actions - 1


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = HostelWaterEnv(CFG)

    def test_config_and_midnight(self):
        self.assertEqual(CFG.midnight_step, 12)
        self.assertEqual(CFG.clock_hour(12), 0.0)
        self.assertEqual(list(CFG.action_volumes_l), [0.0, 150.0, 450.0])

    def test_observation_shape_and_range(self):
        obs, _ = self.env.reset(1)
        self.assertEqual(obs.shape, (HostelWaterEnv.OBS_DIM,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 2.0))

    def test_water_balance_every_step_and_episode(self):
        for seed in range(30):
            for pol in (always(0), always(1), always(2), fit_demand):
                tot, steps = run_episode(self.env, seed, pol)
                for s in steps:
                    lhs = s["level_before"] + s["inflow"]
                    rhs = s["level_after"] + s["delivered"] + s["overflow"]
                    self.assertAlmostEqual(lhs, rhs, places=6)
                    self.assertAlmostEqual(s["demand"], s["used"] + s["unmet"], places=6)
                    self.assertAlmostEqual(s["delivered"], s["used"] + s["wasted"], places=6)
                self.assertAlmostEqual(
                    tot["initial_level"] + tot["inflow"],
                    tot["final_level"] + tot["delivered"] + tot["overflow"], places=5)

    def test_tank_limits(self):
        rng = np.random.default_rng(0)
        for seed in range(50):
            _, steps = run_episode(self.env, seed, lambda o: int(rng.integers(3)))
            for s in steps:
                self.assertGreaterEqual(s["level_after"], -1e-9)
                self.assertLessEqual(s["level_after"], CFG.tank_capacity_l + 1e-9)

    def test_action_handling(self):
        self.env.reset(3)
        _, _, _, info = self.env.step(0)
        self.assertEqual(info["delivered"], 0.0)
        self.env.reset(3)
        _, _, _, info = self.env.step(1)
        self.assertLessEqual(info["delivered"], 150.0 + 1e-9)
        self.env.reset(3)
        _, _, _, info = self.env.step(2)
        self.assertLessEqual(info["delivered"], 450.0 + 1e-9)
        for bad in (-1, 3, 1.5, "x"):
            self.env.reset(3)
            with self.assertRaises(ValueError):
                self.env.step(bad)

    def test_cannot_deliver_more_than_tank_holds(self):
        self.env.reset(4)
        self.env._level = 50.0
        self.env._demand[0] = 300.0
        _, _, _, info = self.env.step(2)
        self.assertAlmostEqual(info["delivered"], 50.0)
        self.assertAlmostEqual(info["unmet"], 250.0)

    def test_reward_values(self):
        c = CFG
        # Over-supply: demand 300, LONG (450) -> used 300, wasted 150
        self.env.reset(5)
        self.env._level, self.env._demand[0] = 1000.0, 300.0
        _, r, _, _ = self.env.step(2)
        exp = (c.alpha_met * 300 - c.gamma_waste * 150) / c.reward_ref_l
        self.assertAlmostEqual(r, exp)
        # Under-supply: demand 300, SHORT (150) -> used 150, unmet 150
        self.env.reset(5)
        self.env._level, self.env._demand[0] = 1000.0, 300.0
        _, r, _, _ = self.env.step(1)
        exp = (c.alpha_met * 150 - c.beta_unmet * 150) / c.reward_ref_l - c.shortage_event_penalty
        self.assertAlmostEqual(r, exp)
        # Perfect match: demand 0, OFF -> reward 0
        self.env.reset(5)
        self.env._demand[0] = 0.0
        _, r, _, _ = self.env.step(0)
        self.assertAlmostEqual(r, 0.0)

    def test_episode_termination(self):
        self.env.reset(6)
        for i in range(CFG.n_steps):
            self.assertFalse(self.env._done)
            _, _, done, _ = self.env.step(0)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            self.env.step(0)
        fresh = HostelWaterEnv(CFG)
        with self.assertRaises(RuntimeError):
            fresh.step(0)

    def test_reproducible_and_action_independent_scenarios(self):
        _, a = run_episode(self.env, 77, always(0))
        _, b = run_episode(self.env, 77, always(2))
        _, c = run_episode(self.env, 78, always(0))
        self.assertEqual([s["demand"] for s in a], [s["demand"] for s in b])
        self.assertEqual([s["inflow"] for s in a], [s["inflow"] for s in b])
        self.assertNotEqual([s["demand"] for s in a], [s["demand"] for s in c])

    def test_reward_cannot_be_hacked(self):
        def mean_return(pol):
            return np.mean([run_episode(self.env, s, pol)[0]["reward"] for s in VAL_SEEDS])
        r_off, r_long = mean_return(always(0)), mean_return(always(2))
        r_short, r_fit = mean_return(always(1)), mean_return(fit_demand)
        print(f"\n  mean return  OFF={r_off:.1f}  SHORT={r_short:.1f}  "
              f"LONG={r_long:.1f}  fit-demand(sanity)={r_fit:.1f}")
        self.assertGreater(r_fit, r_off)
        self.assertGreater(r_fit, r_long)
        self.assertGreater(r_fit, r_short)

    def test_late_night_demand_occurs(self):
        hits = 0
        for s in TEST_SEEDS[:100]:
            self.env.reset(s)
            hits += self.env.exam_night
        self.assertTrue(10 <= hits <= 50)   # roughly 30% of nights


if __name__ == "__main__":
    unittest.main()