"""
Central configuration for the Smart Hostel Water Management simulation.

EVERY number in this file is a SIMULATION ASSUMPTION, not a measured hostel
statistic. Change them freely; the rest of the code reads only from here.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Seed splits: train / validation / test use DISJOINT seed ranges, so the
# final evaluation scenarios are never seen during training or tuning.
# ---------------------------------------------------------------------------
TRAIN_SEED_START = 0            # training episodes draw seeds from 0 .. 99_999
TRAIN_SEED_END = 100_000
VAL_SEEDS = tuple(range(100_000, 100_100))    # 100 scenarios, for tuning
TEST_SEEDS = tuple(range(200_000, 200_500))   # 500 scenarios, final report only


@dataclass(frozen=True)
class EnvConfig:
    # ---- Time (assumption) -------------------------------------------------
    step_minutes: int = 30            # one decision every 30 minutes
    start_hour: int = 18              # episode starts at 18:00 (6 PM)
    n_steps: int = 24                 # 24 x 30 min = 12 h  -> 18:00 to 06:00

    # ---- Tank (assumption) -------------------------------------------------
    tank_capacity_l: float = 6000.0
    init_level_frac_range: Tuple[float, float] = (0.30, 1.00)

    # ---- Students (assumption) ---------------------------------------------
    n_residents: int = 80
    presence_rate_range: Tuple[float, float] = (0.70, 0.95)  # per-episode

    # ---- Demand (assumption) -----------------------------------------------
    # Mean litres per present student per 30-min step, for six 2-hour blocks:
    # 18-20, 20-22, 22-24, 00-02, 02-04, 04-06
    demand_per_student_l: Tuple[float, ...] = (5.0, 4.5, 3.0, 1.2, 0.3, 2.5)
    demand_scale_sigma: float = 0.15   # lognormal spread of whole-night demand
    demand_gamma_shape: float = 8.0    # per-step noise (higher = less noisy)
    # "Exam night": extra late-night demand between 00:00 and 03:00.
    late_night_event_prob: float = 0.30
    late_night_multiplier: float = 4.0
    late_night_window_hours: Tuple[float, float] = (0.0, 3.0)  # clock hours

    # ---- Inflow into the tank (assumption) ---------------------------------
    # Litres arriving per 30-min step in the same six 2-hour blocks.
    inflow_l_per_step: Tuple[float, ...] = (300.0, 300.0, 300.0, 0.0, 0.0, 200.0)
    inflow_noise_range: Tuple[float, float] = (0.8, 1.2)

    # ---- Actions (assumption) ----------------------------------------------
    # Valve is open for this many minutes within the 30-min step.
    action_minutes: Tuple[int, ...] = (0, 10, 30)    # OFF, SHORT, LONG
    valve_flow_l_per_min: float = 15.0               # -> 0, 150, 450 L max

    # ---- Reward (design choice; see README for justification) ---------------
    reward_ref_l: float = 100.0      # litres per reward unit
    alpha_met: float = 1.0           # reward per 100 L of demand met
    beta_unmet: float = 2.0          # penalty per 100 L of unmet demand
    gamma_waste: float = 1.0         # penalty per 100 L delivered but unused
    shortage_event_penalty: float = 0.5   # flat penalty on any step with unmet>0

    # ---- Derived helpers ---------------------------------------------------
    @property
    def action_volumes_l(self) -> np.ndarray:
        return np.array(self.action_minutes, dtype=float) * self.valve_flow_l_per_min

    @property
    def n_actions(self) -> int:
        return len(self.action_minutes)

    @property
    def steps_per_block(self) -> int:
        return int(120 / self.step_minutes)   # 2-hour blocks

    @property
    def midnight_step(self) -> int:
        """Index of the first step that starts at 00:00 (the cutoff moment)."""
        return int(((24 - self.start_hour) * 60) / self.step_minutes)

    def clock_hour(self, step: int) -> float:
        """Clock time (0-24) at the start of a step."""
        return (self.start_hour + step * self.step_minutes / 60.0) % 24.0


ACTION_NAMES = ("OFF", "SHORT", "LONG")