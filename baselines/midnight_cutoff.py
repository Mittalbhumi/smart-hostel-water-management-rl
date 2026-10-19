"""
Fixed midnight-cutoff baselines (NO learning, NO adaptation to demand level).

Rule: before the cutoff the valve follows a fixed behaviour; from the cutoff
step onward (default: 00:00 = step 12) the supply is OFF. Optionally the supply
can resume at a fixed step (e.g. 04:00 = step 20) - this is still a fixed
clock-based schedule.

Pre-cutoff behaviours:
    "short" : always the SHORT action
    "long"  : always the LONG action (the 'normal supply' rule)
    "fit"   : smallest action that covers the current demand (a strong,
              generous variant: it sees demand before midnight)

Every variant is evaluated on identical scenarios; the headline baseline is the
variant with the best VALIDATION return, so the RL agent is not compared
against a weak straw-man.
"""
import numpy as np

from environment.config import EnvConfig

PRE_ACTIONS = ("short", "long", "fit")


class MidnightCutoffPolicy:
    def __init__(self, cfg: EnvConfig, pre_action: str = "long", resume_step=None):
        if pre_action not in PRE_ACTIONS:
            raise ValueError(f"pre_action must be one of {PRE_ACTIONS}")
        self.cfg = cfg
        self.pre_action = pre_action
        self.cutoff_step = cfg.midnight_step
        self.resume_step = resume_step
        self.volumes = cfg.action_volumes_l

    @property
    def name(self) -> str:
        resume = "no resume" if self.resume_step is None else \
            f"resume {self.cfg.clock_hour(self.resume_step):04.1f}h"
        return f"Cutoff-{self.pre_action.upper()} ({resume})"

    def supply_open(self, step: int) -> bool:
        if step < self.cutoff_step:
            return True
        return self.resume_step is not None and step >= self.resume_step

    def act(self, obs, epsilon: float = 0.0) -> int:
        step = int(round(float(obs[1]) * self.cfg.n_steps))
        if not self.supply_open(step):
            return 0
        if self.pre_action == "short":
            return 1
        if self.pre_action == "long":
            return 2
        need = float(obs[4]) * self.volumes.max()
        for a, vol in enumerate(self.volumes):
            if vol >= need:
                return a
        return self.cfg.n_actions - 1


def all_variants(cfg: EnvConfig, resume_step: int = 20):
    """6 candidate baselines: 3 pre-cutoff behaviours x {no resume, resume at 04:00}."""
    return [MidnightCutoffPolicy(cfg, p, r) for r in (None, resume_step) for p in PRE_ACTIONS]