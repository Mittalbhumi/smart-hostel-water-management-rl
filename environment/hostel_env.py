"""
Simulated hostel water system (no real plumbing involved).

Per 30-minute step the order of events is:
  1. The agent observes the state (incl. this step's demand).
  2. The agent picks an action -> the distribution valve is open for 0/10/30 min.
  3. delivered = min(valve capacity, water in tank)
     used      = min(demand, delivered)
     unmet     = demand - used
     wasted    = delivered - used            (delivered but not consumed)
  4. Inflow arrives at the END of the step; anything above capacity overflows.

Water balance (exact, per step):
  level_t + inflow_t = level_{t+1} + delivered_t + overflow_t
Demand balance:
  demand_t = used_t + unmet_t ;  delivered_t = used_t + wasted_t
"""
from typing import Dict, Optional, Tuple

import numpy as np

from environment.config import EnvConfig

EPS = 1e-9


class HostelWaterEnv:
    OBS_DIM = 5   # [tank_frac, time_frac, occupancy_frac, recent_demand, current_demand]

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self._ready = False

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Build a full scenario from `seed`. The whole night (presence, demand,
        inflow, initial level) is generated up-front, so it does NOT depend on
        the agent's actions -> two policies given the same seed face exactly
        the same scenario (fair comparison).
        """
        c = self.cfg
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31))
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        presence_rate = rng.uniform(*c.presence_rate_range)
        self._present = rng.binomial(c.n_residents, presence_rate, size=c.n_steps)

        demand_scale = rng.lognormal(0.0, c.demand_scale_sigma)
        self.exam_night = bool(rng.random() < c.late_night_event_prob)

        per_student = np.empty(c.n_steps)
        inflow_base = np.empty(c.n_steps)
        for t in range(c.n_steps):
            block = min(t // c.steps_per_block, len(c.demand_per_student_l) - 1)
            rate = c.demand_per_student_l[block]
            hour = c.clock_hour(t)
            lo, hi = c.late_night_window_hours
            if self.exam_night and lo <= hour < hi:
                rate *= c.late_night_multiplier
            per_student[t] = rate
            inflow_base[t] = c.inflow_l_per_step[block]

        mean_demand = self._present * per_student * demand_scale
        shape = c.demand_gamma_shape
        self._demand = rng.gamma(shape, np.maximum(mean_demand, EPS) / shape)
        self._demand[mean_demand <= 0] = 0.0
        self._inflow = inflow_base * rng.uniform(*c.inflow_noise_range, size=c.n_steps)

        self._level = rng.uniform(*c.init_level_frac_range) * c.tank_capacity_l
        self.initial_level = self._level
        self._t = 0
        self._done = False
        self._ready = True
        self._tot = dict(demand=0.0, inflow=0.0, delivered=0.0, used=0.0,
                         unmet=0.0, wasted=0.0, overflow=0.0,
                         shortage_steps=0, reward=0.0)
        return self._obs(), {"seed": self.seed, "exam_night": self.exam_night,
                             "initial_level": self.initial_level}

    # ------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        c = self.cfg
        t = min(self._t, c.n_steps - 1)
        ref = c.action_volumes_l.max()          # normalise demand by LONG volume
        past = self._demand[max(0, t - 3):t]
        recent = past.mean() if len(past) else 0.0
        obs = np.array([
            self._level / c.tank_capacity_l,
            t / c.n_steps,
            self._present[t] / c.n_residents,
            min(recent / ref, 2.0),
            min(self._demand[t] / ref, 2.0),
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------- step
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        c = self.cfg
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode finished; call reset().")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < c.n_actions:
            raise ValueError(f"Invalid action {action!r}; must be 0..{c.n_actions - 1}.")

        t = self._t
        demand = float(self._demand[t])
        inflow = float(self._inflow[t])
        level_before = self._level
        valve_cap = float(c.action_volumes_l[action])

        delivered = min(valve_cap, level_before)
        used = min(demand, delivered)
        unmet = demand - used
        wasted = delivered - used

        after = level_before - delivered + inflow
        overflow = max(0.0, after - c.tank_capacity_l)
        self._level = after - overflow

        reward = self._reward(used, unmet, wasted)

        for k, v in (("demand", demand), ("inflow", inflow), ("delivered", delivered),
                     ("used", used), ("unmet", unmet), ("wasted", wasted),
                     ("overflow", overflow), ("reward", reward)):
            self._tot[k] += v
        self._tot["shortage_steps"] += int(unmet > EPS)

        self._t += 1
        self._done = self._t >= c.n_steps

        info = dict(step=t, clock_hour=c.clock_hour(t), demand=demand, inflow=inflow,
                    level_before=level_before, level_after=self._level,
                    delivered=delivered, used=used, unmet=unmet, wasted=wasted,
                    overflow=overflow)
        return self._obs(), reward, self._done, info

    # ----------------------------------------------------------------- reward
    def _reward(self, used: float, unmet: float, wasted: float) -> float:
        c = self.cfg
        r = (c.alpha_met * used
             - c.beta_unmet * unmet
             - c.gamma_waste * wasted) / c.reward_ref_l
        if unmet > EPS:
            r -= c.shortage_event_penalty
        return float(r)

    # ---------------------------------------------------------------- summary
    def episode_totals(self) -> Dict[str, float]:
        """Totals so far in this episode (litres, except reward / shortage_steps)."""
        out = dict(self._tot)
        out["final_level"] = self._level
        out["initial_level"] = self.initial_level
        return out

    @property
    def t(self) -> int:
        return self._t