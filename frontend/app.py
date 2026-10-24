"""
Local web demo for the Smart Hostel Water Management project.
Uses the REAL simulator and the REAL trained Q-table (results/q_table_seed0_best.npz).

Run from the project root:   python -m frontend.app     ->  http://127.0.0.1:5050
All numbers come from the simulator (assumed parameters), not from a real hostel.
"""
import dataclasses
import json
from pathlib import Path

from flask import Flask, Response, render_template, request

from agent.q_learning_agent import QLearningAgent
from baselines.midnight_cutoff import MidnightCutoffPolicy
from environment.config import EnvConfig
from environment.hostel_env import HostelWaterEnv
from experiments.evaluate import rollout

ROOT = Path(__file__).resolve().parent.parent
BASE = EnvConfig()
AGENT = QLearningAgent.load(ROOT / "results" / "q_table_seed0_best.npz")
app = Flask(__name__)


def clamp(x, lo, hi):
    return min(max(float(x), lo), hi)


def build_cfg(p):
    scale = lambda t, k: tuple(v * k for v in t)
    present = clamp(p.get("present_pct", 85), 50, 100) / 100
    level = clamp(p.get("tank_pct", 60), 5, 100) / 100
    return dataclasses.replace(
        BASE,
        n_residents=int(clamp(p.get("residents", 80), 20, 150)),
        presence_rate_range=(present, present),
        init_level_frac_range=(level, level),
        late_night_event_prob=1.0 if p.get("exam_night") else 0.0,
        inflow_l_per_step=scale(BASE.inflow_l_per_step, clamp(p.get("inflow_pct", 100), 30, 150) / 100),
    )


@app.get("/")
def index():
    return render_template("index.html", capacity=BASE.tank_capacity_l)


@app.post("/api/simulate")
def simulate():
    p = request.get_json(force=True)
    cfg = build_cfg(p)
    seed = int(p.get("seed", 200001))
    env = HostelWaterEnv(cfg)
    base = MidnightCutoffPolicy(cfg, "fit", 20 if p.get("baseline") == "resume" else None)
    out = {"baseline_name": base.name}
    for key, pol in (("rl", AGENT), ("baseline", base)):
        row, steps = rollout(env, seed, pol, trace=True)
        out[key] = {"totals": row, "steps": steps}
    _, info = env.reset(seed)
    out["meta"] = {"capacity": cfg.tank_capacity_l, "initial_level": info["initial_level"],
                   "exam_night": info["exam_night"], "seed": seed}
    return Response(json.dumps(out, default=float), mimetype="application/json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)