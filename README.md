# Smart Hostel Water Management using Reinforcement Learning

AI course mini-project, Assignment 2: *Real-Life Use Cases of Reinforcement Learning*.
**Authors:** Bhumi Mittal (B23CH1014) and Stuti Yadav (B23CH1044)

> **Simulation only.** Nothing here is connected to real plumbing. Every number (tank size, students,
> demand, inflow, valve flow) is a simulation assumption, not a measured hostel statistic.

## What this project does

Our hostel in Jodhpur stops its water supply after 12 AM to save water. A fixed cutoff cannot tell a quiet
night from an exam night. We train a tabular Q-learning agent in a simulated hostel to choose, every 30
minutes, between supply **OFF**, **SHORT** or **LONG**, and compare it with fixed midnight-cutoff policies
on nights it has never seen. A small web demo lets you try different nights yourself.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m frontend.app            # then open http://127.0.0.1:5050
```

The trained agents are already saved in `results/`, so the demo works without retraining.

## Web demo (frontend)

A local Flask page that runs the **real simulator and the saved Q-table** (`results/q_table_seed0_best.npz`).
You set the conditions for one night, press **Run simulation**, and watch it play out in 30-minute steps,
with the RL agent and a midnight cutoff side by side.

![Web demo](docs/frontend_demo.png)

| You set | Meaning |
|---|---|
| Water in tank at 6 PM | starting tank level (%) |
| Residents / students present | hostel size and share at home tonight |
| Incoming water supply | lower values simulate scarcity |
| Exam night | adds the late-night demand surge (00:00 to 03:00) |
| Compare against | strict cutoff, or cutoff that restarts at 04:00 |
| Scenario number | random seed: same number gives the same night |

| You see | Meaning |
|---|---|
| Clock and tank gauges | water left in the tank for each policy, live |
| Four number cards | demand served, unmet, wasted and delivered water |
| Two charts | tank level, and student demand vs unmet demand |
| Valve strips | OFF / SHORT / LONG decision for every half hour |
| Verdict line | one-sentence summary of the night |

The agent was trained with 80 residents; other hostel sizes are outside its training range. A terminal
version prints one night as a table: `python -m experiments.demo` (add `--normal` for a normal night).

## How it works

| Part | Definition |
|---|---|
| Episode | one night, 18:00 to 06:00, 24 steps of 30 minutes |
| State | tank level, time, students present, recent demand, current demand, grouped into 5 x 6 x 3 x 3 x 4 = **1080** states |
| Actions | OFF (0 L), SHORT (10 min, up to 150 L), LONG (30 min, up to 450 L) |
| Dynamics | `delivered = min(valve, tank)`, `used = min(demand, delivered)`, `unmet = demand - used`, `wasted = delivered - used`; inflow is added at the end of the step, extra water overflows. The water balance is exact and unit-tested |
| Reward | `R = (1*used - 2*unmet - 1*wasted) / 100 L - 0.5 * 1[unmet > 0]` (a litre not served costs more than a litre wasted; the weights are our design choice) |
| Algorithm | tabular Q-learning, alpha 0.1, gamma 0.95, epsilon-greedy 1.0 to 0.05 over 15,000 of 30,000 training nights, 5 agents with different seeds |

**Assumptions** (all in `environment/config.py`): 6000 L tank starting 30-100% full; 80 residents, 70-95%
present; demand high in the evening and low after 2 AM; 30% of nights are exam nights with 4x demand from 00:00
to 03:00; inflow in the evening and early morning, none from 00:00 to 04:00.

**Fair testing:** training nights use seeds 0-99,999; validation (model and baseline selection) 100,000-100,099;
the final test uses 200,000-200,499 only. Every policy is tested on the same nights.

**Baselines:** a fixed midnight cutoff (supply before 00:00, OFF after). Six variants were built (SHORT, LONG or
demand-matched "FIT", each with and without a 04:00 restart); the best two on validation nights are the
*strict cutoff* (FIT, no restart) and the *strongest fixed schedule* (FIT, restart at 04:00).

## Reproduce everything (from the project root)

```powershell
python -m unittest discover -s tests -t . -v                              # 32 unit tests
python -m experiments.train_all --seeds 0 1 2 3 4 --episodes 30000        # train 5 agents (~5 min)
python -m experiments.evaluate --seeds 0 1 2 3 4                          # test results and tables
python -m experiments.make_plots --seeds 0 1 2 3 4                        # figures 1-5
python -m experiments.ablation_no_cutoff                                  # learning vs "no cutoff" rule
python -m experiments.reward_sensitivity --betas 0 1 2 4 8 --seeds 0 1 2 3 4   # figure 6
python -m experiments.robustness --seeds 0 1 2 3 4                        # figure 7
python -m experiments.state_ablation --seeds 0 1 2                        # figure 8
```

Outputs go to `results/` (tables as CSV, figures in `results/figures/`). Trained policies are saved as
`results/q_table_seed<k>_best.npz` and `..._final.npz`.

## Results (500 unseen nights, RL = mean of 5 agents; simulated data)

| Policy | Return (95% CI) | Unmet (L/night) | Wasted (L/night) | Delivered (L/night) | Served |
|---|---|---|---|---|---|
| **RL agent** | **10.6 +/- 0.7** | **359** | 2332 | 6736 | 93% |
| Strict midnight cutoff | -20.9 +/- 1.1 | 1507 | 1664 | 4920 | 69% |
| Strongest fixed schedule | -4.4 +/- 1.0 | 838 | 2219 | 6144 | 84% |

- **RL clearly beats both cutoffs on return and shortages**, and the gap is largest on exam nights.
  It does **not** save water: it delivers more and wastes more than the strict cutoff.
- **Most of the gain comes from not cutting supply at midnight.** A non-learned "no cutoff" rule scores 9.2;
  learning adds +1.4 return (CI 1.2 to 1.7) and 447 L less waste per night.
- **Reward weights** (`fig6`): changing the unmet-demand weight beta moves the agent along a shortage-vs-waste
  curve. At beta = 1 it beats the strongest schedule on both measures.
- **Robustness** (`fig7`): on six altered simulators (less inflow, more demand, more exam nights, 120
  residents, smaller tank, combined) it stayed above both cutoffs. Against the no-cutoff rule it was ahead in
  four, tied in one and behind with 120 residents.
- **Which inputs matter** (`fig8`): current demand is the key input. With no demand information at all the
  agent still scores 1.3, above the best cutoff schedule (-6.5 on validation nights).

## Project structure

```
environment/   config.py (all assumptions), hostel_env.py (simulator)
agent/         state_encoder.py, q_learning_agent.py
baselines/     midnight_cutoff.py, greedy_fit.py (non-learned reference)
experiments/   train, evaluate, plots, ablation, sensitivity, robustness, state ablation, demo
frontend/      app.py (Flask), templates/index.html
tests/         unit tests
results/       saved Q-tables, CSV tables, figures/
report/        two-page report
```

