# Smart Hostel Water Management using Reinforcement Learning

AI course mini-project, Assignment 2: *Real-Life Use Cases of Reinforcement Learning*.

**Authors:** Bhumi Mittal and Stuti Yadav

> **This is a simulation-based academic project.** Nothing here is connected to real hostel plumbing, and
> **every number (tank size, student count, demand, inflow, valve flow) is a simulation assumption, not a
> measured hostel statistic.** Results describe the simulator, not any real hostel.

## Problem

In a hostel in Jodhpur, Rajasthan, water supply is stopped after 12 AM to save water. A fixed cutoff can
leave students without water when they genuinely need it (for example during exam nights) and can still
waste water at other times. We train a tabular Q-learning agent in a simulated hostel to decide, every
30 minutes, whether to keep the supply OFF, give a SHORT supply, or give a LONG supply, and we compare it
against fixed midnight-cutoff policies on unseen scenarios.

## Simulation design (all values configurable in `environment/config.py`)

| Item | Value (assumption) |
|---|---|
| Episode | 18:00 to 06:00, 24 steps of 30 min |
| Tank | 6000 L, initial level 30-100% of capacity |
| Residents | 80, per-night presence rate 70-95% |
| Demand | gamma-distributed, per-student profile over six 2-hour blocks; 30% of nights are "exam nights" with 4x demand between 00:00 and 03:00 |
| Inflow | 300 L/step 18:00-24:00, 0 from 00:00-04:00, 200 L/step 04:00-06:00, with +/-20% noise |
| Actions | 0 = OFF (0 L), 1 = SHORT (10 min, up to 150 L), 2 = LONG (30 min, up to 450 L) at 15 L/min |

**Transitions** (per step): `delivered = min(valve capacity, tank level)`, `used = min(demand, delivered)`,
`unmet = demand - used`, `wasted = delivered - used`; inflow arrives at the end of the step and anything above
capacity overflows. The water balance `level + inflow = next level + delivered + overflow` holds exactly
(checked by unit tests). Scenarios are generated from a seed at `reset()` and do not depend on the agent's
actions, so every policy faces identical conditions for the same seed.

**State (5 dimensions):** tank fraction, time fraction, occupancy fraction, recent demand (mean of last 3
steps), current demand. For Q-learning it is discretised into 5 x 6 x 3 x 3 x 4 = **1080 states**
(`agent/state_encoder.py`). With 3 actions the Q-table has 3240 entries, so tabular Q-learning is suitable and
DQN/PPO are not needed.

**Reward per step** (units: reward per 100 L):

`R = (alpha*used - beta*unmet - gamma*wasted) / 100 L - lambda * 1[unmet > 0]`

with alpha = 1, beta = 2, gamma = 1, lambda = 0.5. A litre of demand that goes unserved is therefore penalised
more heavily than a litre wasted. These weights are a design choice, so we ran a sensitivity study (below).
Tests confirm the reward cannot be hacked by never supplying or always supplying water.

**Algorithm:** tabular Q-learning, alpha = 0.1, gamma = 0.95, epsilon-greedy with linear decay from 1.0 to
0.05 over the first half of training, 30,000 training episodes, Q-table initialised to 0.

**Seed splits (disjoint):** training scenarios use seeds 0-99,999; validation (baseline selection, model
selection, sensitivity study) uses 100,000-100,099; the final test uses 200,000-200,499 and was used only
for the final evaluation.

## Installation

Windows PowerShell (Python 3.12 tested):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS: `source .venv/bin/activate` instead of the activate line above.

## Reproduce everything (run from the project root)

```powershell
python -m unittest discover -s tests -t . -v                      # unit tests
python -m experiments.train_all --seeds 0 1 2 3 4 --episodes 30000 # train 5 agents (~5 min)
python -m experiments.evaluate --seeds 0 1 2 3 4                   # test evaluation vs baselines
python -m experiments.make_plots --seeds 0 1 2 3 4                 # figures 1-5
python -m experiments.ablation_no_cutoff                           # no-cutoff ablation
python -m experiments.reward_sensitivity --betas 0 1 2 4 8 --seeds 0 1   # figure 6
```

Outputs go to `results/`. The trained policies are saved as `results/q_table_seed<k>_best.npz` (best
validation return) and `results/q_table_seed<k>_final.npz`.

## Project structure

```
environment/   config.py (all assumptions), hostel_env.py (simulator)
agent/         state_encoder.py, q_learning_agent.py
baselines/     midnight_cutoff.py (fixed cutoff variants), greedy_fit.py (non-learned reference)
experiments/   train_q_learning.py, train_all.py, evaluate.py, make_plots.py,
               ablation_no_cutoff.py, reward_sensitivity.py
tests/         unit tests (water balance, tank limits, actions, reward, termination, encoder, agent, baselines)
results/       saved Q-tables, logs, CSV tables, figures/
report/        two-page report
```

## Baselines

The assignment baseline is a fixed midnight cutoff: supply on before 00:00, OFF from 00:00. Before midnight the
rule is ambiguous, so we evaluate six variants (SHORT, LONG, or demand-matched "FIT" supply, each with and
without a fixed 04:00 resume). The two headline baselines are chosen on **validation** seeds: the best strict
variant (FIT, no resume) and the strongest fixed schedule (FIT, resume 04:00). The FIT variants can see current
demand before midnight, which makes them generous baselines.

## Results (500 unseen test scenarios; RL = mean of 5 training seeds; simulated, not real-world data)

| Policy | Return (95% CI) | Unmet (L/night) | Wasted (L/night) | Delivered (L/night) | Demand served |
|---|---|---|---|---|---|
| **RL agent** | **10.6 +/- 0.7** | **359** | 2332 | 6736 | 93% |
| Strict midnight cutoff (FIT, no resume) | -20.9 +/- 1.1 | 1507 | 1664 | 4920 | 69% |
| Strongest fixed schedule (FIT, resume 04:00) | -4.4 +/- 1.0 | 838 | 2219 | 6144 | 84% |

RL return varies by only 0.30 across training seeds. Paired per-scenario differences against the strongest
fixed schedule: return +15.0 (95% CI 14.0 to 16.2), unmet -479 L (CI -518 to -442), wasted +113 L
(CI +82 to +144). RL does not reduce total water waste compared with the cutoffs; it removes most shortages at
a similar waste level.

**Ablation (is it just "no cutoff"?).** A non-learned rule with no cutoff (Greedy-FIT) reaches return 9.2.
Most of the RL gain over the cutoff baselines therefore comes from not cutting supply at midnight. Learning adds
a smaller but statistically significant improvement: +1.4 return (CI 1.2 to 1.7), 447 L less waste per night
(-16%), at the cost of 57 L more unmet demand per night (mainly on normal nights).

**Reward sensitivity (validation scenarios, mean over 5 training seeds).** Changing beta traces a smooth
trade-off between shortage and waste (`results/figures/fig6_reward_sensitivity.png`): beta=0 gives 1182 L
unmet / 877 L wasted per night, beta=1 gives 545 / 1720, the default beta=2 gives 347 / 2384, and beta>=4
saturates near 301 / 2650 (behaviour close to the non-learned no-cutoff rule). With beta=1 the agent beats
the strongest fixed schedule (902 L unmet / 2244 L wasted) on both measures.

**Robustness (`fig7_robustness.png`).** Agents trained on the default simulator were evaluated, without
retraining, on seven simulator settings (low inflow, high demand, frequent exam nights, larger hostel,
smaller tank, combined stress). The RL agent beat both midnight-cutoff baselines in all seven. Against the
non-learned no-cutoff rule it was ahead or tied in six settings and behind in the larger-hostel setting,
so its advantage over that simple rule is small and not guaranteed under large shifts.

**State ablation (`fig8_state_ablation.png`, validation scenarios, 3 seeds).** Current demand is the most
important observation (return 10.5 with it, 4.7 without it). With no demand information at all the agent
still scores 1.3, above the best cutoff schedule (-6.5). Tank level, recent demand and occupancy added no
measurable benefit in this simulator, which means most good decisions here are about matching current
demand; time of day helps a little.

## Limitations

- Simulated demand, inflow and tank are assumptions; real data would change the numbers.
- The agent observes the current step's demand before acting, which is optimistic.
- The smallest non-zero supply (150 L) is larger than much late-night demand, so some waste is structural.
- The 04:00 resume time of the stronger baseline and the reward weights are design choices.
- Tabular Q-learning with a coarse state discretisation; finer states or DQN could do better.

## Originality

All code in this repository was written by the project team for this assignment. No existing repository or
other team's code was copied.