"""
Turns the 5-dimensional continuous observation from HostelWaterEnv into ONE
integer state index for the Q-table.

Observation (from environment/hostel_env.py):
  obs[0] tank_frac        tank level / capacity                    (0..1)
  obs[1] time_frac        step / n_steps                           (0..1)
  obs[2] occupancy_frac   students present / residents             (0..1)
  obs[3] recent_demand    mean demand of last 3 steps / LONG volume (0..2)
  obs[4] current_demand   this step's demand / LONG volume          (0..2)

Discretisation (bin edges are design choices, fixed BEFORE training):
  tank            5 bins  edges 0.10, 0.30, 0.50, 0.75
  time slot       6 bins  = six 2-hour blocks (18-20, 20-22, 22-24, 00-02, 02-04, 04-06)
  occupancy       3 bins  edges 0.78, 0.88
  recent demand   3 bins  edges 0.15, 0.50
  current demand  4 bins  edges 0.03, 0.34, 1.00
        bin 0: ~no demand | bin 1: SHORT (150 L) covers it
        bin 2: LONG (450 L) covers it | bin 3: demand exceeds even LONG
  -> 5 * 6 * 3 * 3 * 4 = 1080 discrete states.
"""
import numpy as np

TANK_EDGES = np.array([0.10, 0.30, 0.50, 0.75])
OCC_EDGES = np.array([0.78, 0.88])
RECENT_EDGES = np.array([0.15, 0.50])
CURRENT_EDGES = np.array([0.03, 0.34, 1.00])
N_TIME_SLOTS = 6

DIMS = (len(TANK_EDGES) + 1, N_TIME_SLOTS, len(OCC_EDGES) + 1,
        len(RECENT_EDGES) + 1, len(CURRENT_EDGES) + 1)
N_STATES = int(np.prod(DIMS))   # 1080


def encode(obs) -> int:
    tank = int(np.digitize(obs[0], TANK_EDGES))
    slot = min(int(obs[1] * N_TIME_SLOTS), N_TIME_SLOTS - 1)
    occ = int(np.digitize(obs[2], OCC_EDGES))
    recent = int(np.digitize(obs[3], RECENT_EDGES))
    current = int(np.digitize(obs[4], CURRENT_EDGES))
    return int(np.ravel_multi_index((tank, slot, occ, recent, current), DIMS))


def decode(index: int):
    """Inverse of encode(): index -> (tank, slot, occupancy, recent, current) bins."""
    return tuple(int(i) for i in np.unravel_index(index, DIMS))