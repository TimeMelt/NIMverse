import json, os, math, numpy as np
from universe.learn_life import loss, main, extinction, freeze_mask, KEYS
from universe.leaves import LIFE_DEFAULTS

C = {**json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config3.json"))), "asteroids": 20}

def test_loss_parts_finite_with_extinction():
    f, parts, per = loss(LIFE_DEFAULTS, [4], C, 200., 10., detail=True)
    assert math.isfinite(f) and len(parts) == 4 and all(math.isfinite(p) for p in parts)
    assert 0 <= parts[3] <= 1 and "ext" in per[4]

def test_extinction_term():
    dead = dict(evolution=dict(n_alive=0, evolution_series=[dict(n_alive=40), dict(n_alive=0)]), biosphere=dict(alive=False))
    ok = dict(evolution=dict(n_alive=30, evolution_series=[dict(n_alive=40), dict(n_alive=30)]), biosphere=dict(alive=True))
    assert extinction(dead) == 1. and extinction(ok) == 0.

def test_freeze_mask():
    m = freeze_mask("tech_*,stellar_evolve"); assert m[KEYS.index("tech_rate")] == 0 and m[KEYS.index("eco_growth")] == 1

def test_one_iter_smoke(tmp_path):
    r = main(["--iters", "1", "--years", "100", "--seed-list", "4", "--out", str(tmp_path/"l.json")])
    assert math.isfinite(r["best_loss"]) and "extinction" in r["best_parts"]
