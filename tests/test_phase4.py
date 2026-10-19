"""Checks for the reference policy and the sensitivity-study helpers.
Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import dataclasses
import unittest

import numpy as np

from baselines.greedy_fit import GreedyFitPolicy
from environment.config import EnvConfig
from environment.hostel_env import HostelWaterEnv
from experiments.reward_sensitivity import train_quick

CFG = EnvConfig()


class TestGreedyFit(unittest.TestCase):
    def test_picks_smallest_covering_action_and_never_cuts_off(self):
        pol = GreedyFitPolicy(CFG)
        mk = lambda litres, t=0.7: np.array([0.5, t, 0.9, 0.2, litres / 450.0], dtype=np.float32)
        self.assertEqual(pol.act(mk(0.0)), 0)
        self.assertEqual(pol.act(mk(100.0)), 1)
        self.assertEqual(pol.act(mk(149.0)), 1)   # just under SHORT volume
        self.assertEqual(pol.act(mk(151.0)), 2)   # just over -> needs LONG
        self.assertEqual(pol.act(mk(300.0)), 2)
        self.assertEqual(pol.act(mk(900.0)), 2)
        # same decision after midnight (time fraction 0.9): no clock rule
        self.assertEqual(pol.act(mk(300.0, t=0.9)), 2)


class TestRewardWeightsAreConfigurable(unittest.TestCase):
    def test_beta_changes_only_the_unmet_penalty(self):
        outs = {}
        for beta in (0.0, 4.0):
            cfg = dataclasses.replace(CFG, beta_unmet=beta)
            env = HostelWaterEnv(cfg)
            env.reset(5)
            env._level, env._demand[0] = 1000.0, 300.0
            _, r, _, _ = env.step(1)          # used 150, unmet 150
            outs[beta] = r
        self.assertAlmostEqual(outs[0.0] - outs[4.0], 4.0 * 150 / CFG.reward_ref_l)

    def test_quick_training_returns_agent_with_visited_states(self):
        agent = train_quick(CFG, seed=0, episodes=300)
        self.assertGreater(agent.coverage(), 0.05)


class TestMaskedEnv(unittest.TestCase):
    def test_hidden_features_are_zero_and_physics_unchanged(self):
        from experiments.state_ablation import MaskedEnv
        plain, masked = HostelWaterEnv(CFG), MaskedEnv(HostelWaterEnv(CFG), [3, 4])
        o1, _ = plain.reset(12)
        o2, _ = masked.reset(12)
        self.assertEqual(o2[3], 0.0)
        self.assertEqual(o2[4], 0.0)
        np.testing.assert_array_equal(o1[:3], o2[:3])
        _, r1, _, i1 = plain.step(1)
        _, r2, _, i2 = masked.step(1)
        self.assertAlmostEqual(r1, r2)
        self.assertAlmostEqual(i1["unmet"], i2["unmet"])
        self.assertEqual(masked.cfg.n_steps, CFG.n_steps)   # attribute passthrough


class TestRobustnessConfigs(unittest.TestCase):
    def test_seven_settings_and_changes_applied(self):
        from experiments.robustness import shifted_configs
        cfgs = shifted_configs(CFG)
        self.assertEqual(len(cfgs), 7)
        self.assertEqual(cfgs["Larger hostel (120 students)"].n_residents, 120)
        self.assertEqual(cfgs["Small tank (3000 L)"].tank_capacity_l, 3000.0)
        self.assertAlmostEqual(cfgs["Low inflow (-30%)"].inflow_l_per_step[0], 210.0)
        self.assertEqual(cfgs["Training setting"], CFG)


if __name__ == "__main__":
    unittest.main()