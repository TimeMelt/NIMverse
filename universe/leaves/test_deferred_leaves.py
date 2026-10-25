"""Tests for deferred features A-G: HostWeather flares, climate history + impacts->bio, radiation delay,
culture/tech, learn_life, system hook, cosmology / strong-field stubs."""
import os, json, copy, numpy as np
from .packets import HostWeather, LifeParamBus, ClimateStress, ChemInventory, CreaturePacket
from . import world, chemistry, biosphere, culture, cosmology, strong_field, radiation
from .life_loop import run_habitat
from .test_life_leaves import hab, SUN

def test_hostweather_flares():
    r = np.random.default_rng(0); md = HostWeather.from_star(.3, .01, 3400.); sun = HostWeather.from_star(1., 1., 5800.)
    assert md.flare_rate > sun.flare_rate
    for t in range(200): md.step(t*10., 10., r, LifeParamBus({"flare_amp": 5.}))
    s = md.summary(); assert s["n_flares"] > 0 and s["uv_max"] > md.uv_proxy and len(md.irr_hist) == md.hist_len
    assert -1e-6 < md.dM_star < 0                                   # tiny wind mass loss
    q = HostWeather.from_star(.3, .01, 3400.)
    for t in range(200): q.step(t, 10., r, LifeParamBus({"flare_rate_mult": 0.}))
    assert q.n_flares == 0 and q.uv_now == q.uv_proxy
    # World + Chemistry consume flares
    h = hab(1.); calm = copy.deepcopy(SUN); hot = copy.deepcopy(SUN); hot.uv_now, hot.flare_duty = 5., .5
    cw, hw_ = world.step(h, calm), world.step(h, hot); assert hw_.stressors["uv_kill"] > cw.stressors["uv_kill"] == 0
    c = ChemInventory(); a, b = chemistry.step(c, cw, calm, dt_w_yr=10.), chemistry.step(c, hw_, hot, dt_w_yr=10.)
    assert b.species["organics"] > a.species["organics"]            # photochemistry boost

def test_climate_history_and_impacts():
    hist = world.ClimateHistory(n=5)
    for t in range(9): hist.push(t, ClimateStress(280.+t, .5, "temperate", {"impacts": 1}))
    s = hist.summary(); assert s["n"] == 5 and s["T_min"] == 284. and s["impacts_total"] == 9
    def run(flux, seed=3):
        h = hab(1.); h.impact_flux = flux
        return run_habitat(h, SUN, LifeParamBus({"impact_scale": 20., "impact_kill": .9}), np.random.default_rng(seed), 4000, 10., plant=True, culture=False)
    ctl, hit = run(0.), run(5.)
    assert ctl["impacts"]["events"] == 0 and hit["impacts"]["events"] > 10 and hit["impacts"]["biomass_lost"] > 0
    assert np.mean(hit["biomass_series"]) < .9*np.mean(ctl["biomass_series"])   # measurable biomass drop
    assert "T_std" in hit["climate_history"] and hit["climate_history"]["T_std"] > ctl["climate_history"]["T_std"]

def test_radiation_delay():
    assert abs(radiation.light_time_yr(1.) - 3.26156) < 1e-9
    h = hab(1.); cat = radiation.ObserverCatalog([radiation.Observer("far", 1000., aperture=1e4)])  # 3262 yr
    r = run_habitat(h, SUN, None, np.random.default_rng(1), 6000, 10., plant=True, culture=False, catalog=cat)
    d = cat.first_detection("o2"); assert d is not None and d["t_arr"] - d["t_emit"] > 3261
    assert all(x["t_arr"] >= cat.obs[0].delay() for x in cat.log)  # nothing before the light-time delay
    near = radiation.ObserverCatalog([radiation.Observer("near", 1.)]); run_habitat(h, SUN, None, np.random.default_rng(1), 6000, 10., plant=True, culture=False, catalog=near)
    assert near.first_detection("o2")["t_arr"] < d["t_arr"]
    dim = radiation.ObserverCatalog([radiation.Observer("dim", 1000.)]); run_habitat(h, SUN, None, np.random.default_rng(1), 6000, 10., plant=True, culture=False, catalog=dim)
    assert dim.first_detection("o2") is None and dim.photons                    # arrives but 1/d^2 too faint
    assert radiation.redshift(1e9) > 0 and radiation.redshift(10.) == 0.

def test_culture():
    from .biosphere import Biosphere
    r = np.random.default_rng(0); LP = LifeParamBus({"tech_rate": 1e-2, "tech_threshold": .1}); big = Biosphere(1., .2, .1, True, 3.)
    c = culture.Culture()
    for t in range(500): c = culture.step(c, big, None, LP, r, t*10., 10.)
    assert not c.active and c.level == 0                            # no organisms -> no culture, ever
    pk = CreaturePacket(n=10.)
    for t in range(500): c = culture.step(c, big, pk, LP, r, t*10., 10.)
    assert c.active and c.level > 0 and c.t_start >= 1000.
    c2 = culture.Culture()
    for t in range(500): c2 = culture.step(c2, Biosphere(.01, 0, 0, True, 3.), pk, LP, r, t*10., 10.)
    assert not c2.active                                            # below biomass threshold
    h = hab(1.); rr = run_habitat(h, SUN, LP, np.random.default_rng(2), 20000, 20., plant=True)
    assert rr["tech"] > 0 and rr["biosign"]["tech"] == rr["tech"]
    assert run_habitat(h, SUN, LP, np.random.default_rng(2), 20000, 20., plant=True, culture=False)["tech"] == 0

def test_cosmology_strong_field():
    assert abs(cosmology.a_of_t(cosmology.age_gyr()) - 1) < 1e-9 and 13 < cosmology.age_gyr() < 14.5
    assert cosmology.a_of_t(1.) < cosmology.a_of_t(5.) < 1 and abs(cosmology.hubble_velocity(10.) - 677.) < 1
    cd = cosmology.CosmoDrive.at(6.); assert cd.z > 0 and cd.H > cosmology.H0
    assert 4000 < cosmology.lum_dist_mpc(1.) < 7500
    rs = strong_field.r_s(1.); assert abs(rs*1.495979e8 - 2.953) < 1e-3
    assert abs(strong_field.r_isco(1.) - 3*rs) < 1e-12 and strong_field.r_isco(1., .99) < 1.5*rs
    assert strong_field.tde_visible(1e6) and not strong_field.tde_visible(1e9)
    assert strong_field.gamma_static(2*rs, 1.) > 1.4 and strong_field.gamma_static(.5*rs, 1.) == np.inf
    assert 4e-7 < strong_field.pn_precession_rad(.387, .2056, 1.) < 6e-7   # Mercury ~5.0e-7 rad/orbit (43"/century)
    assert strong_field.newtonian_limit_ok()

def test_system_hook_and_learn_life(tmp_path=None):
    from ..run_life_from_system import main as hook
    d = str(tmp_path or "/tmp"); rep = hook(["--seed", "4", "--years", "500", "--orbit-yr", "2", "--plant", "--out", os.path.join(d, "h.json")])
    assert rep["system_hook"]["orbit"]["steps"] > 0 and rep["planets"] and "radiation" in rep
    rep2 = hook(["--seed", "4", "--years", "200", "--orbit-yr", "0", "--host-mass", ".5", "--out", os.path.join(d, "h2.json")])
    assert rep2["host"]["M_star"] == .5
    from ..learn_life import main as learn
    from .params import LIFE_SPACE
    out = os.path.join(d, "ll.json"); res = learn(["--iters", "1", "--years", "200", "--seeds", "1", "--out", out])
    j = json.load(open(out)); assert os.path.exists(out.replace(".json", ".log")) and res["best_loss"] <= res["loss0"]
    for k, v in j["params"].items(): assert LIFE_SPACE[k][0] <= v <= LIFE_SPACE[k][1]

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
