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

# ---- rough-edge fixes: speciation, mobility, drift, sexual repro + positions, impacts, culture ----
def _long(LP=None, cl=COLD, steps=1000, seed=4):
    r = np.random.default_rng(seed); pop = I.found(LP, r)
    for _ in range(steps): pop, p = I.step(pop, FOOD, cl, LP, r, 20., series_every=200.)
    return pop

def test_speciation_bounded():
    pop = _long(steps=1500)                     # 30 kyr
    sm = pop.summary(); alive = [s["n_species_alive"] for s in pop.series]
    assert len(pop) > 50 and sm["n_species_alive"] <= 25 and max(alive) <= 25, alive
    assert sm["n_species_ever"] < 200 and sm["n_species_ever"] == pop.series[-1]["n_species_ever"], sm["n_species_ever"]

def test_mobility_no_arms_race():
    pop = _long(steps=600, cl=MILD)
    m = pop.traits[:, I.MOB]; assert m.mean() < .85 and m.std() > .05, (m.mean(), m.std())
    hi = pop.traits.copy(); hi[:, I.MOB] = .95; lo = pop.traits.copy(); lo[:, I.MOB] = .5
    assert I.need_of(hi).mean() > 1.15*I.need_of(lo).mean()      # convex mobility cost

def test_selection_keeps_moving():
    pop = _long(LifeParamBus({"climate_drift": 1.}), steps=1500, seed=5)
    S = [s for s in pop.series if s["t_w"] > 3000]; h = np.array([s["trait_means"]["hardiness"] for s in S])
    off = np.array([s["env_offset"] for s in S])
    assert len(pop) > 50 and np.ptp(off) > .2 and np.ptp(h) > .1, (np.ptp(off), np.ptp(h))

def test_sexual_reproduction_and_positions():
    pop = _long(steps=60); s = pop.summary(); inds = s["individuals"]
    sex = [d for d in inds + s["deceased_sample"] if d["parent2_id"] is not None]
    assert s["total_sexual_births"] > 0 and sex and all(d["parent2_id"] != d["parent_id"] for d in sex)
    assert all(-np.pi/2 <= d["lat"] <= np.pi/2 and -np.pi <= d["lon"] <= np.pi for d in inds)
    assert np.allclose(np.linalg.norm(pop.pos, axis=1), 1.)
    asex = _long(LifeParamBus({"sex_rate": 0.}), steps=30); assert asex.n_sexual == 0 and (asex.parent2 < 0).all()
    r = np.random.default_rng(1); p0 = I.found(None, r); x0 = p0.pos.copy()     # mobility moves organisms
    p0.traits[:20, I.MOB] = 1.; p0.traits[20:, I.MOB] = 0.
    for _ in range(20): p0.pos = I._jitter(p0.pos, .04*(.05 + p0.traits[:, I.MOB]), r)
    mv = np.arccos(np.clip((p0.pos*x0).sum(1), -1, 1)); assert mv[:20].mean() > 3*mv[20:].mean()

def test_impact_refugia_and_floor():
    ok = bad = 0
    for sd in range(30):
        r = np.random.default_rng(100 + sd); pop = I.found(None, r)
        for _ in range(60): pop, _ = I.step(pop, FOOD, MILD, None, r, 20.)
        n0 = len(pop); hit = ClimateStress(293., 1., "temperate", {"impact_kill": .99, "impacts": 3})
        pop, _ = I.step(pop, FOOD, hit, None, r, 1.); n1 = pop.impact_log[-1]["n_after"]
        assert n1 >= max(3, .05*n0) - 1e-9 and n1 < n0
        ok += n1 >= 10; bad += n1 <= 4
        for _ in range(50): pop, _ = I.step(pop, FOOD, MILD, None, r, 20.)
        assert len(pop) > 50                                         # recovers
    assert ok > bad and ok >= 25, (ok, bad)
    r = np.random.default_rng(9); pop = I.found(None, r)              # lethal climate: no floor
    dead = I.impact_deaths(pop, 1., ClimateStress(150., 0., "frozen"), LifeParamBus({}), r); assert dead.sum() > .5*len(pop)

def test_culture_activates_with_individuals():
    from . import culture as C
    from .biosphere import Biosphere
    r = np.random.default_rng(0); pop = I.found(None, r); cu = C.Culture(); bio = Biosphere(P=2., alive=True, diversity=3.)
    LP = LifeParamBus({"tech_rate": 5e-3})
    for k in range(400):
        pop, pkt = I.step(pop, FOOD, MILD, LP, r, 10.); cu = C.step(cu, bio, pkt, LP, r, 10.*k, 10.)
    assert pkt.n > 0 and cu.active and cu.level > .5

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
