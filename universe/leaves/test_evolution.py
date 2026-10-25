"""Individuals + evolution tests. Run: PYTHONPATH=/workspace/NIM python -m pytest universe/leaves/test_evolution.py"""
import json, os, numpy as np
from . import individuals as I
from .packets import NicheOffer, ClimateStress, LifeParamBus
from .params import LIFE_SPACE, TRAITS
COLD = ClimateStress(250., .6, "cold"); MILD = ClimateStress(293., 1., "temperate"); FOOD = NicheOffer(2., 3, .5)

def run(LP=None, cl=COLD, steps=150, dt=20., seed=0, **found):
    r = np.random.default_rng(seed); pop = I.found(LP, r, **found)
    for k in range(steps): pop, p = I.step(pop, FOOD, cl, LP, r, dt, series_every=200.)
    return pop, p

def test_ids_unique_and_parent_links():
    pop, p = run(steps=20); s = pop.summary()
    ids = [d["id"] for d in s["individuals"]] + [d["id"] for d in s["deceased_sample"]]
    assert len(ids) == len(set(ids)) and s["total_births"] > 0
    born = [d for d in s["individuals"] if d["generation"] > 0]; assert born
    assert all(d["parent_id"] is not None and d["parent_id"] < d["id"] and d["birth_t_w"] > 0 for d in born)
    assert all(d["parent_id"] is None for d in s["individuals"] + s["deceased_sample"] if d["generation"] == 0)
    for d in s["individuals"]:
        for t, v in d["traits"].items(): assert LIFE_SPACE[t][0] <= v <= LIFE_SPACE[t][1]
        assert d["energy"] >= 0

def test_pop_cap():
    LP = LifeParamBus({"pop_cap": 80.}); r = np.random.default_rng(3); pop = I.found(LP, r)
    for _ in range(100): pop, p = I.step(pop, NicheOffer(2., 3, 5.), MILD, LP, r, 20.); assert len(pop) <= 80
    assert len(pop) > 40

def test_cold_selects_hardiness():
    pop, p = run(steps=250, traits=dict(hardiness=.2)); s = pop.series
    h0, h1 = s[0]["trait_means"]["hardiness"], s[-1]["trait_means"]["hardiness"]
    assert len(pop) > 0 and h1 > h0 + .3, (h0, h1)
    assert s[-1]["generations_max"] > 50 and all(k in s[-1] for k in ("t_w", "n_alive", "trait_stds", "n_births", "n_deaths"))
    mild, _ = run(cl=MILD, steps=250, traits=dict(hardiness=.2), seed=1)
    assert mild.series[-1]["trait_means"]["hardiness"] < h1 - .2      # costly hardiness not favoured without stress

def test_mutation_raises_variance():
    lo, _ = run(LifeParamBus({"evolution_rate": .2}), cl=MILD, steps=60)
    hi, _ = run(LifeParamBus({"evolution_rate": 4.}), cl=MILD, steps=60)
    v = lambda p: np.mean([p.traits[:, j].std()/(LIFE_SPACE[t][1]-LIFE_SPACE[t][0]) for j, t in enumerate(TRAITS)])
    assert v(hi) > 2*v(lo)

def test_no_food_dies_out():
    r = np.random.default_rng(2); pop = I.found(None, r)
    for _ in range(100): pop, p = I.step(pop, NicheOffer(0., 0, 0.), COLD, None, r, 20.)
    assert len(pop) == 0 and p.n == 0 and p.births == 0

def test_life_report_and_viewer_pack(tmp_path=None):
    from ..run_life import main
    from ..viewer_life import main as vmain
    d = str(tmp_path or "/tmp"); out = os.path.join(d, "life_evo_smoke.json"); rep = main(["--seed", "4", "--years", "3000", "--dt", "20", "--plant", "--out", out])
    J = json.load(open(out)); pl = [p for p in J["planets"] if p.get("individuals")]; assert pl
    pl = pl[0]; ev = pl["evolution"]; assert ev["evolution_series"] and ev["n_alive"] == len(pl["individuals"])
    html = os.path.join(d, "life_evo_smoke.html"); vmain(["--life", out, "--out", html])
    from .creature_look import creatures_from_individuals
    cr = creatures_from_individuals(pl["individuals"], J["seed"], pl["biosign"], pl["habitat"]["ptype"])
    ids = {d["id"] for d in pl["individuals"]}; assert cr and all(c["id"] in ids for c in cr) and len({c["id"] for c in cr}) == len(cr)
    s = open(html).read(); assert all(f'"id":{c["id"]}' in s for c in cr[:10])

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
