"""Run: PYTHONPATH=/workspace/NIM python -m pytest universe/leaves/test_life_leaves.py  (or python -m universe.leaves.test_life_leaves)"""
import json, os, time, numpy as np
from ..bodies import classify, lum, EARTH
from ..sim2 import make_system, DEFAULTS
from . import world, chemistry, biosphere, organisms
from .packets import HabitatState, HostWeather, ChemInventory, ClimateStress, NicheOffer, LifeParamBus
from .system_habitat import habitats_from_system
from .life_loop import life_loop, run_habitat

def hab(a, L=1., m=EARTH, ptype="terran", vol=.6, tilt=.4):
    return HabitatState(1, "p", m, a, 0., 1., tilt, L, .95*L**.5, 1.67*L**.5, .95*L**.5 <= a <= 1.67*L**.5, ptype, vol, 0., L/a**2)
SUN = HostWeather.from_star(1., 1., 5800.)
C = dict(asteroids=20)

def test_habitat_hz_matches_classify():
    for s in range(6):
        S = make_system(np.random.default_rng(s), C, DEFAULTS)
        for h in habitats_from_system(S, s):
            assert abs(h.hz_in - .95*np.sqrt(h.L_star)) < 1e-12 and abs(h.hz_out - 1.67*np.sqrt(h.L_star)) < 1e-12
            for u in (.1, .9): 
                t = classify(h.m, h.a, h.L_star, u)
                if h.ptype not in ("ocean", "terran"): assert t == h.ptype or {t, h.ptype} <= {"ocean", "terran"}
            assert h.in_hz == (h.ptype in ("ocean", "terran") or (h.hz_in <= h.a <= h.hz_out))
    for m, a, M in [(EARTH, 1., 1.), (EARTH, .3, 1.), (EARTH, 4., 1.), (EARTH, .5, .7)]:
        L = lum(M); t = classify(m, a, L, .9); assert (t in ("terran", "ocean")) == (.95*L**.5 <= a <= 1.67*L**.5)

def test_world():
    e = world.step(hab(1.), SUN); assert 255 <= e.T_K <= 300 and e.liquid_water_frac > 0
    assert world.teq(1., 1., .3) > 250 and world.teq(1., 1., .3) < 260
    assert world.step(hab(6., ptype="ice"), SUN).liquid_water_frac == 0
    assert world.step(hab(.15, ptype="lava", vol=.02), SUN).liquid_water_frac < .05

def test_chem_needs_water():
    c = ChemInventory(); c.species["organics"] = 5.
    dry = ClimateStress(300., 0., "temperate"); assert not chemistry.step(c, dry, SUN).polymer_ready
    wet = ClimateStress(300., .8, "temperate"); assert chemistry.step(c, wet, SUN).polymer_ready
    before = {k: c.species[k] + c.species["organics"]*v for k, v in chemistry.STOICH.items()}
    n = chemistry.step(c, wet, SUN, dt_w_yr=10.); after = {k: n.species[k] + n.species["organics"]*v for k, v in chemistry.STOICH.items()}
    for k in before: assert abs(before[k]-after[k]) < 1e-9          # atoms conserved

def test_abiogenesis():
    c = ChemInventory(polymer_ready=True, vesicle_ready=True); cl = ClimateStress(300., .8, "temperate"); r = np.random.default_rng(0)
    LP = LifeParamBus({"abio_p": 0.})
    assert not any(chemistry.abiogenesis_roll(c, cl, LP, r, dt_w_yr=1e4).fired for _ in range(200))
    assert chemistry.planted_gate().fired
    LP1 = LifeParamBus({"abio_p": 1e-3}); assert any(chemistry.abiogenesis_roll(c, cl, LP1, r, dt_w_yr=1e3).fired for _ in range(50))
    assert not chemistry.abiogenesis_roll(ChemInventory(), cl, LP1, r, dt_w_yr=1e6).fired     # not ready -> never

def _good():
    c = ChemInventory(polymer_ready=True, vesicle_ready=True); c.species["organics"] = 1.; return c, ClimateStress(290., .8, "temperate")

def test_biosphere():
    c, cl = _good(); b = biosphere.Biosphere()
    for _ in range(50): b, _, _ = biosphere.step(b, cl, c)
    assert b.total() == 0 and not b.alive                          # no growth before gate
    b = biosphere.seed(b, chemistry.planted_gate())
    for _ in range(200): b, niche, flux = biosphere.step(b, cl, c, dt_w_yr=1.)
    assert b.total() > .5 and flux.o2 > 0 and niche.capacity > 0
    m0 = b.total(); b, _, _ = biosphere.step(b, cl, c, disturbance=.9); assert b.total() < .2*m0
    for _ in range(300): b, _, _ = biosphere.step(b, cl, c)
    assert b.total() > .8*m0                                        # recovery
    b2 = biosphere.seed(biosphere.Biosphere(), chemistry.planted_gate())
    for _ in range(100): b2, _, _ = biosphere.step(b2, ClimateStress(200., 0., "snowball"), c)
    assert b2.total() == 0                                          # water gate

def test_organisms():
    r = np.random.default_rng(1); coh = organisms.new_cohorts(None, r); cl = ClimateStress(290., .8, "temperate")
    for _ in range(50):
        coh, p = organisms.step(coh, NicheOffer(2., 3, 1.), cl, None, r)
        assert p.energy_budget >= 0 and all(c["energy"] >= 0 for c in coh)
        for c in coh:
            for t, v in c["traits"].items(): from .params import LIFE_SPACE; assert LIFE_SPACE[t][0] <= v <= LIFE_SPACE[t][1]
    assert p.n > 0
    n0 = p.n; coh2, p2 = organisms.step(coh, NicheOffer(0., 0, 0.), cl, None, r); assert p2.births == 0
    for _ in range(100): coh, p = organisms.step(coh, NicheOffer(0., 0, 0.), cl, None, r)
    assert p.n == 0 and not coh                                     # extinction

def test_smoke_run_life(tmp_path=None):
    from ..run_life import main
    t = time.time(); out = os.path.join(str(tmp_path or "/tmp"), "life_smoke.json")
    rep = main(["--seed", "4", "--years", "500", "--dt", "10", "--plant", "--out", out])
    assert time.time()-t < 60 and json.load(open(out))["seed"] == 4 and rep["with_life"]

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
