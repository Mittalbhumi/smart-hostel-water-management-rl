"""
Tabular Q-learning agent (written from scratch with NumPy).

Update rule (off-policy TD control), for a transition (s, a, r, s', done):
    target = r                               if done
    target = r + gamma * max_a' Q(s', a')    otherwise
    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
Exploration: epsilon-greedy with linear epsilon decay.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.state_encoder import N_STATES, encode


@dataclass
class QLearningConfig:
    alpha: float = 0.1            # learning rate
    gamma: float = 0.95           # discount factor
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 15_000   # linear decay length
    q_init: float = 0.0           # rewards are mostly <= 0, so 0 is optimistic


class QLearningAgent:
    def __init__(self, n_actions: int, cfg: QLearningConfig = None, seed: int = 0):
        self.cfg = cfg or QLearningConfig()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.q = np.full((N_STATES, n_actions), self.cfg.q_init, dtype=np.float64)
        self.visits = np.zeros((N_STATES, n_actions), dtype=np.int64)

    # ---------------------------------------------------------------- policy
    def epsilon(self, episode: int) -> float:
        c = self.cfg
        frac = min(1.0, episode / max(1, c.eps_decay_episodes))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs, epsilon: float = 0.0) -> int:
        """Epsilon-greedy. epsilon=0 -> greedy (ties broken by lowest action id)."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[encode(obs)]))

    # ---------------------------------------------------------------- learning
    def update(self, obs, action: int, reward: float, next_obs, done: bool) -> float:
        s, s2 = encode(obs), encode(next_obs)
        target = reward if done else reward + self.cfg.gamma * np.max(self.q[s2])
        td_error = target - self.q[s, action]
        self.q[s, action] += self.cfg.alpha * td_error
        self.visits[s, action] += 1
        return float(td_error)

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, q=self.q, visits=self.visits)

    @classmethod
    def load(cls, path, seed: int = 0) -> "QLearningAgent":
        data = np.load(path)
        agent = cls(n_actions=data["q"].shape[1], seed=seed)
        agent.q = data["q"].copy()
        agent.visits = data["visits"].copy()
        return agent

    def coverage(self) -> float:
        """Fraction of states visited at least once."""
        return float((self.visits.sum(axis=1) > 0).mean())