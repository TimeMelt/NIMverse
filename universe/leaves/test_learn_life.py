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

def test_impacts_frozen_and_floored():
    from universe.learn_life import DEFAULT_FREEZE, decode, encode
    from universe.leaves import LIFE_SPACE
    import numpy as np
    m = freeze_mask(DEFAULT_FREEZE); i, j = KEYS.index("impact_scale"), KEYS.index("impact_kill")
    assert m[i] == 0 and m[j] == 0 and LIFE_SPACE["impact_scale"][0] >= .25
    assert decode(np.zeros(len(KEYS)))["impact_scale"] >= .25                 # free-optimized u=0 still keeps impacts on
    u = encode(LIFE_DEFAULTS); u2 = np.clip(u - .3*np.ones(len(KEYS))*m, 0, 1)  # SPSA-style masked update
    assert decode(u2)["impact_scale"] == LIFE_DEFAULTS["impact_scale"] and decode(u2)["impact_kill"] == LIFE_DEFAULTS["impact_kill"]
