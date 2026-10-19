"""
Reference policy (NOT RL, NOT the assignment baseline): "greedy fit, no cutoff".

At every step it picks the smallest action whose volume covers the CURRENT
demand. It has no learning and no clock rule. It exists only to answer:
how much of the RL gain comes from simply not cutting supply at midnight,
and how much comes from learning?
"""
from environment.config import EnvConfig


class GreedyFitPolicy:
    name = "Greedy-FIT (no cutoff, not learned)"

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.volumes = cfg.action_volumes_l

    def act(self, obs, epsilon: float = 0.0) -> int:
        need = float(obs[4]) * self.volumes.max()
        for a, vol in enumerate(self.volumes):
            if vol >= need:
                return a
        return self.cfg.n_actions - 1