"""Checks for the state encoder and the Q-learning agent.
Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.q_learning_agent import QLearningAgent, QLearningConfig
from agent.state_encoder import DIMS, N_STATES, decode, encode
from environment.config import EnvConfig
from environment.hostel_env import HostelWaterEnv


class TestStateEncoder(unittest.TestCase):
    def test_state_count(self):
        self.assertEqual(N_STATES, 1080)
        self.assertEqual(DIMS, (5, 6, 3, 3, 4))

    def test_indices_always_in_range(self):
        env = HostelWaterEnv(EnvConfig())
        for seed in range(40):
            obs, _ = env.reset(seed)
            done = False
            while not done:
                self.assertTrue(0 <= encode(obs) < N_STATES)
                obs, _, done, _ = env.step(seed % 3)

    def test_encode_decode_roundtrip_and_bins(self):
        obs = np.array([0.9, 0.55, 0.95, 0.6, 0.2], dtype=np.float32)
        # tank 0.9 -> bin 4 | time 0.55*6=3.3 -> slot 3 | occ -> 2 | recent -> 2 | current -> 1
        self.assertEqual(decode(encode(obs)), (4, 3, 2, 2, 1))
        self.assertEqual(decode(encode(np.zeros(5, dtype=np.float32))), (0, 0, 0, 0, 0))

    def test_demand_bins_match_action_volumes(self):
        cfg = EnvConfig()
        ref = cfg.action_volumes_l.max()
        mk = lambda litres: np.array([0.5, 0.1, 0.9, 0.2, litres / ref], dtype=np.float32)
        self.assertEqual(decode(encode(mk(10)))[4], 0)     # negligible
        self.assertEqual(decode(encode(mk(150)))[4], 1)    # SHORT exactly covers
        self.assertEqual(decode(encode(mk(300)))[4], 2)    # needs LONG
        self.assertEqual(decode(encode(mk(600)))[4], 3)    # exceeds LONG


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent(3, QLearningConfig(alpha=0.5, gamma=0.9), seed=1)
        self.s = np.array([0.5, 0.1, 0.9, 0.2, 0.2], dtype=np.float32)
        self.s2 = np.array([0.4, 0.2, 0.9, 0.2, 0.5], dtype=np.float32)

    def test_td_update_matches_formula(self):
        i, j = encode(self.s), encode(self.s2)
        self.agent.q[j] = [1.0, 3.0, 2.0]
        self.agent.q[i, 1] = 2.0
        self.agent.update(self.s, 1, -1.0, self.s2, done=False)
        target = -1.0 + 0.9 * 3.0
        self.assertAlmostEqual(self.agent.q[i, 1], 2.0 + 0.5 * (target - 2.0))

    def test_terminal_update_does_not_bootstrap(self):
        i, j = encode(self.s), encode(self.s2)
        self.agent.q[j] = [100.0, 100.0, 100.0]
        self.agent.update(self.s, 0, -2.0, self.s2, done=True)
        self.assertAlmostEqual(self.agent.q[i, 0], 0.5 * (-2.0 - 0.0))

    def test_greedy_picks_best_and_epsilon_explores(self):
        self.agent.q[encode(self.s)] = [0.0, 5.0, 1.0]
        self.assertEqual(self.agent.act(self.s, 0.0), 1)
        picks = {self.agent.act(self.s, 1.0) for _ in range(200)}
        self.assertEqual(picks, {0, 1, 2})

    def test_epsilon_schedule(self):
        a = QLearningAgent(3, QLearningConfig(eps_start=1.0, eps_end=0.1,
                                              eps_decay_episodes=100))
        self.assertAlmostEqual(a.epsilon(0), 1.0)
        self.assertAlmostEqual(a.epsilon(50), 0.55)
        self.assertAlmostEqual(a.epsilon(100), 0.1)
        self.assertAlmostEqual(a.epsilon(10_000), 0.1)

    def test_save_load_roundtrip(self):
        self.agent.update(self.s, 2, -1.0, self.s2, done=False)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.npz"
            self.agent.save(path)
            loaded = QLearningAgent.load(path)
        np.testing.assert_array_equal(loaded.q, self.agent.q)
        np.testing.assert_array_equal(loaded.visits, self.agent.visits)

    def test_short_training_beats_random(self):
        from experiments.train_q_learning import evaluate_greedy, train
        from environment.config import VAL_SEEDS
        import experiments.train_q_learning as tq
        with tempfile.TemporaryDirectory() as d:
            tq.RESULTS = Path(d)
            agent, _ = train(seed=0, episodes=4000, eval_every=2000, verbose=False)
        env = HostelWaterEnv(EnvConfig())
        trained = evaluate_greedy(agent, env, VAL_SEEDS[:50])["reward"]
        rng = np.random.default_rng(0)
        rand_agent = QLearningAgent(3, seed=0)
        rand_agent.act = lambda obs, epsilon=0.0: int(rng.integers(3))
        random_ret = evaluate_greedy(rand_agent, env, VAL_SEEDS[:50])["reward"]
        self.assertGreater(trained, random_ret)


if __name__ == "__main__":
    unittest.main()