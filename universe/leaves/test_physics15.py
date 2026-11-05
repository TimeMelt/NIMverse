"""physics15 passes 1-5. Run: PYTHONPATH=/workspace/nim-universe:/workspace/NIM python -m pytest universe/leaves/test_physics15.py"""
import numpy as np
from .packets import HabitatState, HostWeather, NicheOffer, ClimateStress, as_bus
from . import world, biogeochem as BG, individuals as I
from .life_loop import life_loop
from .orbit_climate import insolation, SecularOrbit
P = as_bus(None); SUN = HostWeather.from_star(1., 1., 5800.)
def earth(**kw):
    d = dict(idx=0, name="earth", m=3e-6, a=1., e=.0167, spin=1., tilt=.409, albedo_scale=1., volatiles=.95, density=1.67, in_hz=True,
             kind="terran", water=.5, t_form=0., insol=1.)
    d.update(kw); return HabitatState(*d.values())

def test_pass1_band_temperature_gradient():
    lc = world.LatClimate(earth(), P, SUN); c = lc.step(earth(), SUN, P, 20.); s = c.stressors
    T = np.asarray(s["lat_T"]); assert len(T) == lc.n
    assert s["T_eq_lat"] - s["T_pole"] > 25           # Earth-like equator-pole contrast (obs ~40 K)
    assert 280 < c.T_K < 296 and 270 < s["T_eq_lat"] < 310 and 50 < s["ice_lat_deg"] <= 90
    eq = np.abs(lc.lat) < 20; po = np.abs(lc.lat) > 70; assert T[eq].mean() > T[po].mean() + 20

def test_pass2_co2_raises_temperature_and_equilibrium():
    h = earth(); Ts = []
    for p in (1e-4, 2.8e-4, 1e-3, 1e-2):
        lc = world.LatClimate(h, P, SUN); lc.pCO2 = p; Ts.append(lc.step(h, SUN, P, 20.).T_K)
    assert all(b > a for a, b in zip(Ts, Ts[1:])) and Ts[-1] - Ts[0] > 8
    lc = world.LatClimate(h, P, SUN); cc = BG.CarbonCycle(h, lc, P); p = cc.equilibrate(h, SUN, lc, P)
    assert 1e-4 < p < 1e-3                             # weathering thermostat lands near pre-industrial CO2
    V, W = cc.fluxes(lc, world.P_REF); assert abs(W/V - 1) < .1

def test_pass3_mass_burn_scaling():
    r = np.random.default_rng(1); pop = I.found(None, r, n=60)
    pop.logm = np.where(np.arange(len(pop.id)) % 2 == 0, 1., -1.); pop.energy = 0.9*I.E_MAX*np.exp(pop.logm)
    e0 = dict(zip(pop.id, pop.energy)); m = dict(zip(pop.id, pop.logm)); nd = dict(zip(pop.id, I.need_of(pop.traits)*.5*(.4 - .3*pop.traits[:, I.H])))   # famine burn
    pop, _ = I.step(pop, NicheOffer(0., 3, .5), ClimateStress(288., 1., "temperate"), dict(pred_rate=0.), r, 1.)
    rat = {k: (e0[k] - e)/nd[k] for k, e in zip(pop.id, pop.energy) if k in e0}
    big = np.median([v for k, v in rat.items() if m[k] > 0]); small = np.median([v for k, v in rat.items() if m[k] < 0])
    assert abs(np.log(big/small) - .75*2) < .35         # burn ~ M^0.75 (Kleiber); slack for age-mortality + food surplus
    assert I.mass_of(pop).std() > 0

def test_pass4_eccentricity_obliquity_change_insolation():
    x = np.sin(np.radians(np.linspace(-85, 85, 18)))
    q0 = insolation(x, np.radians(23.4), 0.); qe = insolation(x, np.radians(23.4), .1); qo = insolation(x, np.radians(40.), 0.)
    assert abs(qe["mean"].mean()/q0["mean"].mean() - (1 - .01)**-.5) < 5e-3     # annual mean ~ (1-e^2)^-1/2
    assert qo["mean"][-1] > q0["mean"][-1] and qo["mean"][9] < q0["mean"][9]       # more tilt: poles gain, equator loses
    so = SecularOrbit([1., 5.2], [3e-6, 1e-3], 1., 0, .41, seed=3); a = [so.at(t) for t in np.linspace(0, 1e5, 40)]
    ob = [d["obliquity"] for d in a]; assert np.ptp(ob) > np.radians(1.)
    assert np.ptp([d["e"] for d in a]) >= 0 and all(0 <= d["e"] < .3 for d in a)

def test_pass5_carbon_conservation_and_biosign_from_pools():
    import json, os
    from ..sim2 import make_system, DEFAULTS
    C = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config3.json"))); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    S = make_system(np.random.default_rng(4), C, DEFAULTS)
    rep = life_loop(S, 4, None, 3000., 20.)
    p2 = [p for p in rep["planets"] if "skipped" not in p and p["habitat"]["name"] == "planet2"][0]; st = p2["climate"]["stressors"]
    assert p2["biosphere"]["alive"] and st["C_bio"] > .1
    assert abs(st["C_drift"]) < 1e-6*st["C_total"]      # C_inorg+carb+fossil+bio+det+sed+CH4 = initial + outgassed
    x_o2 = st["O2_pool"]/(st["O2_pool"] + BG.ATM_COL); x_ch4 = st["CH4_pool"]/(st["CH4_pool"] + BG.ATM_COL)
    assert abs(p2["biosign"]["o2"] - x_o2) < 1e-12 and p2["biosign"]["o2"] > .05
    assert abs(p2["biosign"]["ch4"] - max(x_ch4, BG.CH4_K*st["CH4_pool"])) < 1e-12
    assert 0 < st["growth_mult"] <= 1.5 and st["N_avail"] > 0 and st["P_avail"] > 0
    assert "lat_N" in st and "lat_P" in st and len(st["lat_N"]) == len(st["lat_P"]) > 1
