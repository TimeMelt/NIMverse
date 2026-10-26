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

def test_culture_biosphere_coupling():
    """Item 4: tech draws producers, pollutes -> warms climate (ClimateHistory), extreme knobs -> die-off + culture decay."""
    from .biosphere import Biosphere
    r = np.random.default_rng(0); pk = CreaturePacket(n=10.); big = Biosphere(1., .2, .1, True, 3.)
    c = culture.Culture(active=True, level=.8, t_start=0., org_age=2000.)
    c = culture.step(c, big, pk, LifeParamBus({"tech_draw": .05}), r, 0., 10.)
    assert c.draw > 0 and culture.apply(big, c).P < big.P and c.footprint > 0         # resource draw
    starved = culture.step(culture.Culture(active=True, level=.8, org_age=2000.), Biosphere(.01, 0, 0, True, 3.), pk, LifeParamBus({"tech_draw": .5}), r, 0., 10.)
    assert starved.supply < .5 and starved.level < .8                                  # overshoot: industry contracts
    off = culture.step(culture.Culture(active=True, level=.8, org_age=2000.), big, pk, LifeParamBus({"tech_draw": 0.}), r, 0., 10.)
    assert off.draw == 0
    base = dict(tech_rate=1e-2, tech_threshold=.1)
    def run(**k): return run_habitat(hab(1.), SUN, LifeParamBus(dict(base, **k)), np.random.default_rng(2), 20000, 20., plant=True)
    ctl, hot = run(tech_pollute=0., tech_warm_K=0., tech_draw=0.), run(tech_pollute=1., tech_warm_K=30.)
    assert ctl["tech"] > 0 and hot["culture"]["peak"]["tech_dT"] > 5
    assert hot["climate_history"]["T_max"] > ctl["climate_history"]["T_max"] + 5        # tech-driven warming
    tail = lambda r: np.mean([x[1] for x in r["climate_history"]["tail"]])
    assert tail(hot) > tail(ctl)
    # collapse: extreme toxicity + draw crash organisms / biomass, culture then decays
    ext = run(tech_pollute=2., tech_warm_K=60., tech_draw=.5)
    assert ext["culture"]["peak"]["tech_kill"] > .1
    assert min(ext["biomass_series"]) < .5*max(ctl["biomass_series"])
    assert ext["tech"] < ctl["tech"]

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

def test_stellar_ms_track_unit():
    from . import stellar
    assert stellar.t_ms_Gyr(1.5) < stellar.t_ms_Gyr(1.) < stellar.t_ms_Gyr(.3)              # higher mass evolves faster
    L1 = [stellar.track(1., 1., 5800., 4.6, a)[0] for a in (1., 4.6, 6., 8.)]
    assert all(b > a for a, b in zip(L1, L1[1:])) and .6 < L1[0] < .85 and 1.05 < L1[2] < 1.2   # Sun brightens ~8%/Gyr
    assert abs(stellar.track(.3, .01, 3400., 4.6, 8.)[0]/.01 - 1) < .02                     # M dwarf ~flat
    dLhi = stellar.track(1.3, 2., 6400., 1., 2.)[0]/2. - 1; dLsun = stellar.track(1., 1., 5800., 1., 2.)[0] - 1
    assert dLhi > dLsun > 0
    assert stellar.activity_factor(1., 4.6, 8.) < 1 < stellar.activity_factor(1., 4.6, 1.)   # Skumanich spin-down
    assert stellar.activity_factor(.3, 1., 8.) == 1.                                          # M dwarf stays active

def test_hostweather_stellar_evolve_and_fixed():
    r = np.random.default_rng(0); s = HostWeather.from_star(1., 1., 5800.)
    P = LifeParamBus({"stellar_evolve": 1., "stellar_age_rate": 1e6, "flare_rate_mult": 0.})
    for k in range(400): s.step(k*10., 10., r, P)                                          # 4 Gyr of stellar time
    sm = s.summary(); assert abs(sm["age_Gyr"] - 8.6) < 1e-6 and sm["dL_frac"] > .2 and s.uv_proxy > 1.05 and s.T_eff > 5800
    assert sm["hz_in_now"] > .95 and s.flare_rate < .05 and len(sm["L_track"]) > 2 and abs(sm["L_track"][-1][2]/s.L_star - 1) < .05
    f = HostWeather.from_star(1., 1., 5800.); F = LifeParamBus({"stellar_evolve": 0.})
    for k in range(400): f.step(k*10., 10., r, F)
    assert f.L_star == 1. and f.uv_proxy == 1. and f.T_eff == 5800. and f.summary()["dL_frac"] == 0.
    assert all(.999 <= x[1] for x in f.irr_hist)                                           # only flare wobble on L_now

def test_stellar_brightening_warms_climate():
    h = hab(1.); base = dict(climate_drift=0., impact_scale=0., flare_rate_mult=0.)
    def run(**k):
        return run_habitat(h, SUN, LifeParamBus(dict(base, **k)), np.random.default_rng(3), 4000, 10., plant=True, culture=False)
    ev, ctl = run(stellar_evolve=1., stellar_age_rate=1e6), run(stellar_evolve=0.)
    T = lambda r: np.array([x[1] for x in r["climate_history"]["series"]])
    Te, Tc = T(ev), T(ctl); q = len(Te)//4
    assert Te[-q:].mean() - Te[:q].mean() > 10. and abs(Tc[-q:].mean() - Tc[:q].mean()) < 1e-6
    assert ev["host"]["dL_frac"] > .2 and ctl["host"]["dL_frac"] == 0. and ev["climate"]["stressors"]["hz_in_now"] > ctl["climate"]["stressors"]["hz_in_now"]

def test_stellar_post_ms_track():
    from . import stellar
    tm = stellar.t_ms_Gyr(1.); Le, Te, Re = stellar.track(1., 1., 5800., 4.6, tm)
    L, T, R = stellar.track(1., 1., 5800., 4.6, 1.25*tm)
    assert L > 1.5*Le and T < Te and R > Re and stellar.phase(1., 1.25*tm) == "rgb" and stellar.phase(1., 1.05*tm) == "subgiant"
    assert stellar.track(1., 1., 5800., 4.6, 5*tm) == stellar.track(1., 1., 5800., 4.6, stellar.max_age_Gyr(1.))   # finite cap
    assert stellar.activity_factor(1., 4.6, 1.2*tm) < .5*stellar.activity_factor(1., 4.6, tm)
    s = HostWeather.from_star(1., 1., 5800.); r = np.random.default_rng(0)
    P = LifeParamBus({"stellar_evolve": 1., "stellar_age_rate": 1e6, "stellar_age_Gyr": 9.})
    for k in range(600): s.step(k*10., 10., r, P)
    sm = s.summary(); assert sm["post_ms"] and sm["phase"] == "rgb" and abs(sm["age_Gyr"] - stellar.max_age_Gyr(1.)) < .05
    assert np.isfinite(s.L_star) and s.L_star > 2 and s.R_star > 1.5 and sm["age_source"] == "knob"

def test_blackbody_uv_bands():
    from . import stellar
    assert abs(stellar.uv_surface_rel(5772.) - 1) < 1e-9
    sun, hot, md = (stellar.uv_bands(T) for T in (5772., 9000., 3200.))
    assert abs(sum(sun["frac"][k] for k in ("uvc", "uvb", "uva", "optical", "ir")) - 1) < .02
    assert hot["uv_surface_rel_sun"] > 5 > 1 > md["uv_surface_rel_sun"] and hot["uv_to_optical"] > sun["uv_to_optical"] > 5*md["uv_to_optical"]
    assert hot["frac"]["uvc"] > sun["frac"]["uvc"] > md["frac"]["uvc"]
    s = HostWeather.from_star(1., 1., 5800.); assert "nuv" in s.uv_bands["frac"] and s.summary()["uv_bands"]

def test_live_habitat_hz_sync():
    h = hab(1.); P = LifeParamBus(dict(climate_drift=0., impact_scale=0., flare_rate_mult=0., stellar_evolve=1., stellar_age_rate=1e6))
    hz0 = (h.hz_in, h.hz_out, h.irradiance)
    r = run_habitat(h, SUN, P, np.random.default_rng(3), 4000, 10., plant=True, culture=False)
    hd, st = r["habitat"], r["climate"]["stressors"]
    assert (h.hz_in, h.hz_out, h.irradiance) == hz0                                         # caller's habitat untouched
    assert hd["hz_in"] > hz0[0] and abs(hd["hz_in"] - st["hz_in_now"]) < 1e-12 and abs(hd["hz_out"] - st["hz_out_now"]) < 1e-12
    assert abs(hd["irradiance"] - r["host"]["L_star"]/h.a**2) < 1e-12 and hd["L_star"] == r["host"]["L_star"]

def test_system_age_prior():
    from . import stellar
    from .life_loop import host_weather
    a = [stellar.age_prior(1., s) for s in (1, 2, 3)]; assert len(set(a)) == 3 and a[0] == stellar.age_prior(1., 1)
    assert stellar.age_prior(2., 1) != stellar.age_prior(1., 1)
    for M in (.3, 1., 1.5, 3.):
        for sd in range(20): assert .1 <= stellar.age_prior(M, sd) <= stellar.X_START_MAX*stellar.t_ms_Gyr(M) + 1e-9
    S = dict(kind=np.array([0]), m=np.array([1.]))
    hw = host_weather(S, seed=7); r = np.random.default_rng(0); hw.step(0., 10., r, LifeParamBus({}))
    assert hw.age_source == "system" and hw.age0_Gyr == stellar.age_prior(1., 7)
    hk = host_weather(S, seed=7); hk.step(0., 10., r, LifeParamBus({"stellar_age_Gyr": 2.}))
    assert hk.age_source == "knob" and hk.age0_Gyr == 2.

def test_biosign_spectra():
    from . import biosign
    from .packets import BioSignFlux
    from types import SimpleNamespace as NS
    L = lambda f: {l["name"]: l["strength"] for l in biosign.build_lines(f)}
    assert L(BioSignFlux(.21*.9/1.9))["o2_760"] > L(BioSignFlux(.21*.1/1.1))["o2_760"]          # more producers -> deeper O2
    wet, dry = biosign.h2o_column(NS(liquid_water_frac=.8, T_K=290.)), biosign.h2o_column(NS(liquid_water_frac=.1, T_K=290.))
    assert wet > dry > 0 and biosign.h2o_column(NS(liquid_water_frac=0., T_K=290.)) == 0
    assert "tech_589" in L(BioSignFlux(tech=.5)) and "tech_589" not in L(BioSignFlux(o2=.1))
    assert biosign.build_lines(BioSignFlux()) == []                                          # dry dead world
    z = 2.
    for l in biosign.observed(biosign.build_lines(BioSignFlux(.1, 1e-3, .5, .1, .3)), z): assert abs(l["obs_nm"] - 3*l["rest_nm"]) < 1e-9
    # integration: cosmological observer at z=2 (lookback ~10 Gyr: emit at t, poll far in the future)
    cat = radiation.ObserverCatalog([radiation.Observer("hiz", 10., z=z, aperture=10.)])
    r = run_habitat(hab(1.), SUN, None, np.random.default_rng(1), 4000, 10., plant=True, culture=False, catalog=cat)
    assert not cat.log; cat.poll(1e11); s = cat.summary()
    assert "o2_760" in s["lines_detected"] and "h2o_1400" in s["lines_detected"]
    d = cat.first_detection("o2_760"); l = [x for x in d["lines"] if x["name"] == "o2_760"][0]
    assert abs(l["obs_nm"] - 760*3) < 1e-6 and any(abs(v.get("obs_nm", 0) - 2280) < 1e-6 for v in s["first"].values())
    assert r["biosign"]["h2o"] > 0 and r["biosign"]["lines"]
