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
    assert mild.series[-1]["trait_means"]["hardiness"] < h1 - .15     # costly hardiness not favoured without stress

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
    assert sm["n_species_ever"] < 500 and sm["n_species_ever"] == pop.series[-1]["n_species_ever"], sm["n_species_ever"]

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

# ---- imperfection fixes: trait trade-offs, multi-niche speciation, planetary climate drift ----
def _norm(pop, j): return (pop.traits[:, j] - I.LO[j])/I.SPAN[j]

def test_tradeoffs_keep_met_rep_off_ceiling():
    for cl, sd in ((COLD, 4), (MILD, 6)):
        pop = _long(cl=cl, steps=1200, seed=sd)          # 24 kyr of selection under LIFE_DEFAULTS
        met, rep = _norm(pop, I.MET), _norm(pop, I.REP)
        assert len(pop) > 50 and met.mean() < .85 and rep.mean() < .75, (met.mean(), rep.mean())
        assert not (met.mean() > .9 and rep.mean() > .9) and met.std() > .03 and rep.std() > .03
    tr = np.tile(pop.traits.mean(0), (2, 1)); tr[0, I.MET], tr[1, I.MET] = .2, .9
    assert I.uptake(tr)[0] > 1.1*I.uptake(tr)[1]                     # rate-yield cost of efficiency
    x = np.linspace(.1, .9, 5)[:, None]*np.ones((1, len(I.TRAITS))); u = I.uptake(x); assert (np.diff(u, 2) < 0).all()   # convex cost (interior optimum)
    tr[:, I.MET] = .5; tr[0, I.REP], tr[1, I.REP] = .05, .5
    assert I.need_of(tr)[1] > 1.15*I.need_of(tr)[0]                  # breeding upkeep

def test_multi_niche_speciation():
    LP = LifeParamBus({"evolution_rate": 2.})
    r = np.random.default_rng(11); pop = I.found(LP, r, n=120)
    pop.traits[:40, I.MOB], pop.traits[40:80, I.MOB], pop.traits[80:, I.MOB] = .05, .5, .95   # sessile / forager / roamer founders
    pop.traits[:40, I.H], pop.traits[80:, I.H] = .9, .3
    for _ in range(400): pop, _ = I.step(pop, FOOD, COLD, LP, r, 20., series_every=200.)
    alive = [s["n_species_alive"] for s in pop.series[-10:]]
    assert len(pop) > 50 and np.median(alive) >= 3 and np.mean(alive) >= 3 and max(alive) <= 25, alive
    assert I.niche_crowding(pop.traits, pop.pos).mean() > 0

def test_climate_drift_moves_world_T():
    from ..run_life import main
    out = {}
    for a in (0., .8):
        p = f"/tmp/life_drift_{a}.json"; LPf = f"/tmp/life_drift_{a}_lp.json"; json.dump({"climate_drift": a, "impact_scale": 0.}, open(LPf, "w"))
        J = main(["--seed", "4", "--years", "20000", "--dt", "50", "--plant", "--no-culture", "--params", LPf, "--out", p])
        pl = [q for q in J["planets"] if "climate_history" in q][0]; out[a] = pl["climate_history"]
    assert out[.8]["T_std"] > 3*out[0.]["T_std"] + .5, (out[0.]["T_std"], out[.8]["T_std"])
    assert np.ptp([r[1] for r in out[.8]["series"]]) > 5.

# ---- cold-spell floor, species churn, metabolism smoothing, path history ----
def _demo(drift=None, years=40000):
    from ..run_life import main
    p = f"/tmp/life_floor_{drift}.json"; args = ["--seed", "4", "--years", str(years), "--dt", "20", "--out", p]
    if drift is not None: LPf = p + ".lp.json"; json.dump({"climate_drift": drift}, open(LPf, "w")); args += ["--params", LPf]
    J = main(args); return [q for q in J["planets"] if q.get("evolution")][0]

def test_cold_spell_floor_species_churn_metabolism_seed4():
    pl = _demo(); e = pl["evolution"]; s = e["evolution_series"]
    n = np.array([x["n_alive"] for x in s]); sa = np.array([x["n_species_alive"] for x in s])
    assert n[-1] > 0 and n[3:].min() >= 80, n.min()                   # cold spells no longer shrink to ~30 (founders = 40)
    assert e["n_species_ever"] <= 70 and 3 <= np.median(sa) <= 12 and sa[-10:].min() >= 3, (e["n_species_ever"], np.median(sa))
    m = np.array([x["trait_means"]["metabolism_eff"] for x in s]); h = len(m)//2
    assert m[3:].std() < .08 and m[h:].std() < .08 and np.ptp(m[h:]) < .35 and .2 < m[h:].mean() < .8, (m[:h].std(), m[h:].std(), m[h:].mean())   # was std .12, .5-.84 swings
    assert .45 < m[h:].mean() < .72 and m[h:].std() <= 1.15*m[3:h].std(), (m[h:].mean(), m[3:h].std(), m[h:].std())   # mid-range (was .39), not boom-bust
    ind = pl["individuals"]; tr = [d for d in ind if d.get("trail")]
    assert len(tr) > .8*len(ind) and all(len(d["trail"]) <= I.TRAIL_K for d in tr)
    mv = [np.arccos(np.clip(np.dot(*[[np.cos(a)*np.cos(b), np.sin(a), np.cos(a)*np.sin(b)] for a, b in (d["trail"][0], d["trail"][-1])]), -1, 1)) for d in tr]
    assert np.median(mv) > .05                                       # recorded paths actually relocate (viewer replay)

def test_strong_drift_no_extinction():
    for a in (.8, 1.):
        n = [x["n_alive"] for x in _demo(a)["evolution"]["evolution_series"]]
        assert n[-1] > 0 and min(n[3:]) >= 50, (a, min(n))

def test_food_lag_bounds_metabolism_oscillation():
    LP = LifeParamBus({"climate_drift": 1.}); r = np.random.default_rng(5); pop = I.found(LP, r)
    raw, felt = [], []
    for k in range(1300):                                             # 26 kyr = 2 food cycles at full amplitude
        pop, _ = I.step(pop, FOOD, MILD, LP, r, 20., series_every=400.)
        raw.append(I.food_mult(pop, LP)); felt.append(pop.food_buf)
    assert np.ptp(felt[300:]) < .7*np.ptp(raw[300:])                 # lagged (stored) productivity: damped swing
    m = np.array([x["trait_means"]["metabolism_eff"] for x in pop.series[10:]])
    assert len(pop) > 50 and m.max() < .86 and m.min() > .2 and np.ptp(m) < .45, (m.min(), m.max())

def test_cold_refuge_and_acclimation_unit():
    r = np.random.default_rng(2); pop = I.found(None, r, traits=dict(hardiness=.3))
    harsh = ClimateStress(240., .3, "cold")
    for _ in range(300): pop, _ = I.step(pop, FOOD, harsh, None, r, 20.)
    assert len(pop) > 20 and pop.accl > .1                             # plastic acclimation + refugia: no extinction

# ---- mid-range metabolism, RNG robustness, separate streams, brine floor ----
def test_metabolism_midrange_robust_across_seeds():
    ms = []
    for sd in range(4):
        r = np.random.default_rng(sd); LP = LifeParamBus({"climate_drift": 1.}); pop = I.found(LP, r)
        for _ in range(900): pop, _ = I.step(pop, FOOD, COLD, LP, r, 20., series_every=400.)
        n = [x["n_alive"] for x in pop.series[3:]]; assert min(n) >= 80, (sd, min(n))   # no extinction cliff on any seed
        ms.append(np.mean([x["trait_means"]["metabolism_eff"] for x in pop.series[len(pop.series)//2:]]))   # 2nd-half time mean
    assert .4 < np.mean(ms) < .65 and np.ptp(ms) < .15, ms          # interior optimum (range .1-.9), consistent across seeds

def test_organism_rng_stream_isolated():
    from ..run_life import main
    g = []
    for nf in (40, 41):                                               # different organism draws must not move world events
        import sys; L = sys.modules["universe.leaves.life_loop"]; old = L.run_habitat.__defaults__
        d = list(old); d[-1] = nf; L.run_habitat.__defaults__ = tuple(d)
        try: J = main(["--seed", "4", "--years", "6000", "--dt", "20", "--no-culture", "--out", f"/tmp/life_rng_{nf}.json"])
        finally: L.run_habitat.__defaults__ = old
        pl = [q for q in J["planets"] if "gate" in q]; g.append([(q["gate"]["t_w"], q["impacts"]["events"]) for q in pl])
    assert g[0] == g[1], g

def test_brine_cold_floor():
    from . import chemistry as C
    from .packets import ChemInventory
    ch = ChemInventory(dict(C=1, H=4, O=2, N=.5, P=.05, S=.1, organics=1.), 0.)
    class HW: uv_proxy = 0.; flare_rate = 0.
    assert C.step(ch, ClimateStress(245., .3, "cold"), HW()).polymer_ready
    assert not C.step(ch, ClimateStress(230., .3, "cold"), HW()).polymer_ready

# ---- planet gen + abiogenesis widening ----
def _scan_ready(seeds=range(20), steps=2000, dt=20.):
    import importlib
    from ..sim2 import make_system, DEFAULTS
    from . import world, chemistry
    from .system_habitat import habitats_from_system, candidate
    LL = importlib.import_module("universe.leaves.life_loop")
    C = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config3.json"))); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    out = {}
    for sd in seeds:
        S = make_system(np.random.default_rng(sd), C, DEFAULTS); hw = LL.host_weather(S); out[sd] = []
        for h in habitats_from_system(S, sd):
            if not candidate(h): continue
            cl = world.step(h, hw, None); ch = chemistry.init_chem(h)
            for _ in range(steps): ch = chemistry.step(ch, cl, hw, None, dt)
            out[sd].append((h.ptype, cl.T_K, cl.liquid_water_frac, ch.polymer_ready and ch.vesicle_ready))
    return out

def test_desert_near_hz_reaches_polymer_ready():
    from . import world, chemistry
    from .system_habitat import VOLATILES
    from .packets import HabitatState
    class HW: L_star = 1.; uv_proxy = 1.; flare_rate = 0.
    for u in (0., .5, 1.):                                            # whole volatile draw range .7-1.3 x desert
        v = VOLATILES["desert"]*(.7+.6*u)
        h = HabitatState(0, "d", 3e-6, .85, 0., 1., .3, 1., .95, 1.67, False, "desert", v, 0., 1.)
        cl = world.step(h, HW()); assert 273 < cl.T_K < 340 and cl.liquid_water_frac > .1, (cl.T_K, cl.liquid_water_frac)
        ch = chemistry.init_chem(h)
        for _ in range(2000): ch = chemistry.step(ch, cl, HW(), None, 20.)
        assert ch.polymer_ready and ch.vesicle_ready, (v, ch.species["organics"])

def test_abiogenesis_fires_when_ready_40kyr():
    from . import chemistry as C
    from .packets import ChemInventory
    ch = ChemInventory(dict(C=1, H=4, O=2, N=.5, P=.05, S=.1, organics=1.), 0., True, True)
    cl = ClimateStress(250., .1, "cold")                              # sparse brine world (seed-7 style), w just at vesicle threshold
    fired = 0
    for sd in range(200):
        r = np.random.default_rng(sd); g = None
        for k in range(2000):
            g = C.abiogenesis_roll(ch, cl, None, r, k*20., 20., g)
            if g.fired: fired += 1; break
    assert fired >= .9*200, fired                                     # analytic: 1-exp(-4e-4*.2*40000) = .96
    nr = ChemInventory(ch.species, 0., False, False); r = np.random.default_rng(0)
    assert not any(C.abiogenesis_roll(nr, cl, None, r, 0., 20.).fired for _ in range(5000))   # not ready -> never

def test_multiseed_ready_proxy():
    rd = _scan_ready()
    n = sum(any(x[3] for x in v) for v in rd.values())
    assert n >= 10, {k: v for k, v in rd.items()}                     # full 40 kyr scan: 12/20 with life (README)
    assert not any(x[3] for v in rd.values() for x in v if x[0] == "ice"), "snowballs must stay sterile"

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
