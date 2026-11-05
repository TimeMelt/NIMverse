"""Tests for universe.solvers (opt-in out-of-leaf solvers).  pytest universe/solvers/test_solvers.py"""
import copy, numpy as np
from universe.leaves.test_life_leaves import hab, SUN
from universe.leaves.packets import as_bus, HostWeather
from universe.solvers import rt, climate2d, foodweb, secular_nbody as sn, gr, stellar_track as st


# ---- rt + climate2d
def test_rt_rce_earthlike():
    Ts, d = rt.rce(S_toa=340., alb_s=.3, cloud=.0, iters=400)
    assert abs(d["imbalance"]) < .5 and 255. < Ts < 320.
    lo = rt.radiation(288., pco2=2.8e-4)["OLR"]; hi = rt.radiation(288., pco2=2.8e-3)["OLR"]
    assert np.ravel(hi)[0] < np.ravel(lo)[0]                                     # CO2 greenhouse: more CO2, less OLR at fixed Ts

def test_gcm2d_laplacian_and_balance():
    h = hab(1.); g, cc = climate2d.init_gcm2d(h, SUN, None, None, rt_on=False)
    L = g.Lop; rs = np.abs(np.asarray(L.sum(1))).max(); assert rs < 1e-9 # diffusion conserves energy (zero row sums)
    c = None
    for k in range(30): c = g.step(h, SUN, None, 20., {"dT": 0.}, 20.*(k + 1))
    s = c.stressors; assert s["climate_model"] == "gcm2d" and 200. < c.T_K < 330.
    assert abs(s["toa_imbalance"]) < 5.                                  # near radiative balance after spin-up
    assert np.asarray(s["grid_T"]).shape == (g.n, g.nlon) and len(s["lat_T"]) == g.n

def test_rt_cia_co2_condensation_runaway():
    olr = float(np.ravel(rt.radiation(288., pco2=2.8e-4)["OLR"])[0]); assert 262. < olr < 275.        # Earth clear-sky OLR
    assert abs(float(rt.p_sat_co2(216.58))/5.18e5 - 1) < .02 and abs(float(rt.T_sat_co2(1e5)) - 194.7) < 2.   # triple pt, 1-bar frost pt
    assert abs(float(rt.T_sat_co2(rt.p_sat_co2(170.))) - 170.) < 1e-6
    lim = rt.runaway_limit(); assert 270. < lim < 300. and rt.runaway_limit(20.) > lim               # Simpson-Nakajima, rises with g
    v, d = rt.runaway_blend(np.array([400.]), np.array([2.]), np.array([380.]), np.array([.77]), rt.G_EARTH)
    assert v[0] < lim + 26.                                                                    # hot moist column capped
    v0, _ = rt.runaway_blend(np.array([265.]), np.array([2.]), np.array([288.]), np.array([.77]), rt.G_EARTH); assert v0[0] == 265.
    old = dict(rt.PHYS)
    try:
        rt.PHYS["cia"] = True; a = float(np.ravel(rt.radiation(280., pco2=1., ps=1., T_strat=200.)["OLR"])[0])
        rt.PHYS["cia"] = False; b = float(np.ravel(rt.radiation(280., pco2=1., ps=1., T_strat=200.)["OLR"])[0])
    finally: rt.PHYS.clear(); rt.PHYS.update(old)
    assert a < b - 2.                                                                          # CO2-CO2 CIA adds opacity (fixed skin T)

def test_gcm2d_physics_conservative_and_default():
    h = hab(1.); g0, _ = climate2d.init_gcm2d(h, SUN, None, None, rt_on=False)
    g, _ = climate2d.init_gcm2d(h, SUN, None, None, rt_on=False, physics="all")
    assert g0.physics == frozenset() and g.physics == frozenset(climate2d.GCM_PHYSICS)
    A = g._winds(g.Tg.ravel() if hasattr(g, "Tg") else np.full(g.N, 280.))
    assert np.abs(np.asarray(A.sum(1))).max() < 1e-9 and np.abs(np.asarray(A.sum(0))).max() < 1e-9   # upwind: conservative
    assert abs(np.degrees(g.phi_H) - 29.) < 2.                                                  # Held-Hou edge, Earth
    c = None
    for k in range(20): c = g.step(h, SUN, None, 20., {"dT": 0.}, 20.*(k + 1))
    s = c.stressors; assert 200. < c.T_K < 330. and s["diurnal_mean_K"] > 0 and s["clw_mean"] > 0 and abs(s["phi_H_deg"] - 29.) < 2.
    try: climate2d.parse_physics("bogus"); assert False
    except ValueError: pass

def test_individuals_grid_lookup():
    from universe.leaves.individuals import band_values
    from universe.leaves.packets import ClimateStress
    g = np.zeros((4, 8)); g[3, 0] = 7.
    c = ClimateStress(280., .5, "temperate", {"lat_T": [1., 2., 3., 4.], "grid_T": g.tolist()})
    pos = np.array([[np.cos(np.pi*7/8)*.3, .9, np.sin(np.pi*7/8)*-.3]]); pos /= np.linalg.norm(pos)
    lo = int(((np.arctan2(pos[0, 2], pos[0, 0]) + np.pi)/(2*np.pi)*8)) % 8
    assert band_values(c, pos, "lat_T")[0] == g[3, lo]
    c2 = ClimateStress(280., .5, "temperate", {"lat_T": [1., 2., 3., 4.]}); assert band_values(c2, pos, "lat_T")[0] == 4.   # leaf default


# ---- foodweb
def test_foodweb_conservation_and_pyramid():
    fw = foodweb.FoodWeb(closed=True); fw.seed(1.); M0 = fw.total()
    for _ in range(3000): fw.step(1., 1., 1.)
    assert abs(fw.total() - M0) < 1e-9*max(M0, 1)
    s = fw.summary(); assert s["n_guilds_alive"] >= 4
    assert fw.P > fw.H > fw.C                                             # trophic pyramid

def test_foodweb_dt_independent():
    a = foodweb.FoodWeb(closed=True); b = foodweb.FoodWeb(closed=True); a.seed(1.); b.seed(1.)
    for _ in range(4000): a.step(1., 1., .5)
    for _ in range(1000): b.step(1., 1., 2.)
    assert abs(a.P - b.P) < .05*a.P and abs(a.living() - b.living()) < .05*a.living()


def _clim_band():
    from universe.leaves.packets import ClimateStress
    return ClimateStress(288., .7, "temperate", {"lat_T": list(np.linspace(250, 305, 12)), "lat_water": [1.]*12})

def test_species_web_conservation_and_structure():
    w = foodweb.make("species", seed=3); w.closed = True; w.set_climate(_clim_band()); w.seed(1.); M0 = w.total()
    for _ in range(1000): w.step(1., 1., 1.)
    assert abs(w.total() - M0) < 1e-8*max(M0, 1)
    s = w.summary(); assert s["mode"] == "species" and s["n_species_alive"] >= 4 and s["max_trophic_level"] >= 2.
    assert w.P > w.H + w.C + w.X > 0                                      # producers dominate the biomass
    assert isinstance(foodweb.make("guild"), foodweb.FoodWeb)             # guild stays the default mode

def test_species_web_biosphere_step():
    from universe.leaves import biosphere
    from universe.leaves.packets import ChemInventory
    w = foodweb.make("species", seed=5); clim = _clim_band(); bio = biosphere.Biosphere(P=.1, alive=True)
    chem = ChemInventory(polymer_ready=True); chem.species["organics"] = .5
    for _ in range(300): bio, niche, flux = foodweb.step(w, bio, clim, chem, None, 1.)
    assert bio.alive and bio.P > 0 and bio.diversity >= 1.

# ---- orbits
SEJ = dict(a=[1., 5.2], m=[3.0e-6, 9.55e-4], M_star=1., tilt=[np.radians(23.4), np.radians(3.)], spin=[1., 2.4])

def test_nbody_energy_and_secular_agreement():
    nb = sn.NBodySystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], seed=1, e_init=[.0167, .048])
    E0 = nb.energy(); nb.advance(500.); assert abs(nb.energy()/E0 - 1) < 1e-6
    sc = sn.SecularSystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], seed=1, e_init=[.0167, .048])
    x, y = nb.at(0, 500.), sc.at(0, 500.)
    assert abs(x["e"] - y["e"]) < 2e-3 and abs(x["obliquity"] - y["obliquity"]) < 2e-3

def test_megno_regular():
    nb = sn.NBodySystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], seed=1, e_init=[.0167, .048])
    assert abs(nb.megno(300.) - 2.) < .1

def test_orbit_drivers_keys():
    d = sn.FixedOrbit(.02, .4); assert d.at(5.)["e"] == .02 and d.key()[0] == "solver_orbit"

def test_tides_earth_moon_and_locking():
    T = sn.Tides([1.], [sn.M_EARTH], 1., [1.], moon=(np.array([sn.MOON_M]), np.array([sn.MOON_A])))
    Nm = T.moon_torque()[0]*sn.MSUN_KG*sn.AU_M**2/sn.YR_SI**2; assert 3.5e16 < Nm < 5.5e16          # observed 3.9-4.5e16 N m
    dam = 2*T.moon_torque()*np.sqrt(T.am/(sn.G*(T.m + T.mm)))/T.mm; assert 3. < dam[0]*sn.AU_M*100 < 4.5   # 3.8 cm/yr
    s0 = np.array([[np.sin(.4), 0, np.cos(.4)]]); nrm = np.array([[0, 0, 1.]])
    T.step(100., s0, nrm, [.0167]); assert T.w[0] < T.w0[0] and abs(T.w[0]/T.w0[0] - 1) < 1e-6
    T2 = sn.Tides([.03], [sn.M_EARTH], .15, [1.]); s = s0.copy()                                 # M-dwarf HZ planet: locks
    for _ in range(100): s = T2.step(1e4, s, nrm, [0.])
    assert abs(T2.w[0]/T2.n[0] - 1) < .01 and np.arccos(s[0, 2]) < .01
    assert sn.tide_kw(None) == {} and sn.tide_kw("moon") == dict(tides=True, moon="system")

def test_tides_secular_default_unchanged_and_damping():
    kw = dict(seed=1, e_init=[.0167, .048])
    a = sn.SecularSystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], **kw)
    b = sn.SecularSystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], tides=False, **kw)
    assert a.at(0, 3000.) == b.at(0, 3000.) and a.sig == b.sig
    c = sn.SecularSystem(SEJ["a"], SEJ["m"], SEJ["M_star"], SEJ["tilt"], SEJ["spin"], tides=True, moon=True, **kw)
    assert c.sig != a.sig and abs(c.at(0, 3000.)["obliquity"] - a.at(0, 3000.)["obliquity"]) < 1e-3   # Earth: tides negligible in kyr
    d = sn.SecularSystem([.03, .06], [sn.M_EARTH]*2, .15, [.4, .2], [1., 1.], seed=2, e_init=[.05, .03], tides=True)
    assert np.all(d.Gam < 0)                                   # Hut 1981: spin omega/n > 18/11 pumps e (Gamma < 0)
    P = [np.sqrt(a**3/.15)*365.25 for a in (.03, .06)]         # synchronous spin (rev/day): e damps, LL modes decay
    s = sn.SecularSystem([.03, .06], [sn.M_EARTH]*2, .15, [.4, .2], [1/p for p in P], seed=2, e_init=[.05, .03], tides=True)
    assert s.Gam[0] > s.Gam[1] > 0 and np.all(np.imag(s.gA) > 0)
    assert d.at(0, 2e5)["obliquity"] < .05                                                     # tidally eroded obliquity

def test_cosmo_drive_bound_pair_decouples():
    from universe import sim2
    from universe.leaves import cosmology as cz
    M = 10.; r0 = 1.; x = np.array([[0., 0, 0], [r0, 0, 0]]); m = np.array([M, 0.])
    vc = np.sqrt(M/r0); v = np.array([[0., 0, 0], [0, vc, 0]]); e2 = np.full(2, 1e-6)
    fx, fv, info = sim2.leapfrog_cosmo(x, v, m, e2, .002, 3000, 30, z_start=2.)
    rr = np.linalg.norm(fx[:, 1] - fx[:, 0], axis=1); assert np.all(np.abs(rr - r0) < .05)        # bound orbit keeps its physical size
    assert info["z_end"] < 2. and info["a_end_rel"] > 1.
    k1, d, k2, bg, ae, aa = sim2.cosmo_factors(2., .002, 10); assert np.all(d < k1 + k2) and bg > 0


# ---- GR
def test_gr_closed_forms_and_geodesic():
    assert abs(gr.r_isco(0.) - 6.) < 1e-12 and abs(gr.efficiency(0.) - (1 - np.sqrt(8/9))) < 1e-9
    assert gr.r_isco(.99) < 1.5 and gr.horizon(1.) == 1.
    for a in (0., .7):
        T, _ = gr.circular_period(10., a, orbits=1); assert abs(T/(2*np.pi*(10**1.5 + a)) - 1) < 1e-6

def test_gr_periastron_and_inspiral():
    num, ex = gr.periastron_advance(50., .3), gr.periastron_advance_exact(50., .3)
    assert abs(num/ex - 1) < 1e-3 and abs(ex/(6*np.pi/50) - 1) < .2
    r = gr.inspiral(1e-3, 0., 10., 10.); assert abs(r["t_merge_yr"]/gr.t_merge_circular_yr(1e-3, 10., 10.) - 1) < .02
    f = gr.event_fields(4e6, .5, r_peri_au=1., m2_msun=1.); assert all(k.startswith("gr_") for k in f)


# ---- stellar track
def test_stellar_pms_agb_wd_z():
    d = st.solve_track(1.); ph = list(d["phase"])
    order = [p for i, p in enumerate(ph) if i == 0 or ph[i - 1] != p]
    assert order == ["pms", "ms", "subgiant", "rgb", "hb", "agb", "tp-agb", "wd"]
    assert .005 < d["t_pms"] < .06 and d["L"][0] > 2 and abs(d["Teff"][0] - 4000) < 100            # Hayashi: bright, cool
    assert 3 <= d["n_pulses"] <= 30 and .5 < d["M_wd"] < .65                                     # IFMR: ~0.55-0.6 for 1 Msun
    assert st.solve_track(3.)["M_wd"] > d["M_wd"]                                               # IFMR increases with M
    k = np.where(d["phase"] == "tp-agb")[0]; assert d["L"][k].max() > 2e3 and d["L"][k].min() < .8*d["L"][k].max()   # pulses dip
    w = np.where(d["phase"] == "wd")[0]; assert d["L"][w[-1]] < 1e-4 and np.all(np.diff(d["L"][w]) < 0) and d["R"][w[-1]] < .015
    assert st.solve_track(1., .002)["t_ms"] < d["t_ms"] < st.solve_track(1., .03)["t_ms"]       # metal-poor: shorter-lived, hotter
    assert st._interp(st.solve_track(1., .002), "Teff", 1.) > st._interp(d, "Teff", 1.)
    assert st.L_zams(1.) == st.L_zams(1., st.ZSUN) and st.X0_of(st.ZSUN) == st.X0                # Z = Zsun -> solar calibration

def test_stellar_sun():
    assert 9. < st.t_ms_Gyr(1.) < 11. and st.t_ms_Gyr(.8) > st.t_ms_Gyr(1.) > st.t_ms_Gyr(1.5)
    L, T, R = st.track(1., 1., 5777., 4.57, 4.57); assert abs(L - 1) < 1e-9
    d = st.solve_track(1.); L0 = st._interp(d, "L", d["t_pms"]); assert .6 < L0 < .8      # faint young Sun (at the ZAMS)
    assert st.phase(1., 4.6) == "ms" and st.phase(1., st.t_ms_Gyr(1.) + 1.) in ("subgiant", "rgb", "hb")
    rgb = [t for t, p in zip(d["t"], d["phase"]) if p == "rgb"]
    assert st._interp(d, "L", rgb[-1]) > 10*st._interp(d, "L", rgb[0])       # RGB climb

def test_packets_default_unchanged():
    a = HostWeather.from_star(1., 1., 5800.); b = copy.deepcopy(a); b.solver_track = "semi"
    assert a._trk().__name__.endswith("leaves.stellar") and b._trk().__name__.endswith("stellar_track")


# ---- life_loop wiring (short)
def test_habitatrun_solvers_smoke():
    from universe.leaves.life_loop import HabitatRun
    from universe.leaves.packets import LifeParamBus
    h = hab(1.); r = HabitatRun(h, copy.deepcopy(SUN), None, np.random.default_rng(0), None, 20., plant=True, culture=False,
                                climate_model="gcm2d", solvers=dict(foodweb=True))
    for _ in range(20): r.step()
    rep = r.report(); assert rep["solvers"]["climate_model"] == "gcm2d" and "foodweb" in rep["solvers"]
    assert rep["biosphere"]["alive"]


def test_events_gr_hook_logging_only():
    """events.run_events with gr_solver: same dynamics/log types as default, plus gr_* fields on TDE entries."""
    from universe import events as ev
    r = np.random.default_rng(0); n = 40
    def gx():
        x = np.vstack([np.zeros((1, 3)), r_.normal(0, .05, (n, 3))]); v = np.zeros_like(x)
        kind = np.r_[ev.SMBH, np.ones(n, int)]; m = np.r_[5., np.full(n, 1e-4)]
        return dict(x=x, v=v, m=m, e2=np.full(n+1, 1e-4), kind=kind, gid=np.zeros(n+1, int), smass=np.r_[0., np.ones(n)], R=np.zeros(n+1))
    EP = dict(ev.EDEFAULTS); EP["tde_r"] = .5
    C = dict(galaxy_dt=1e-3, galaxy_steps=10, star_soft=.01, ev_every=1)
    r_ = np.random.default_rng(1); a = ev.run_events(gx(), C, EP, 1, 2, "direct")
    r_ = np.random.default_rng(1); b = ev.run_events(gx(), dict(C, gr_solver=True), EP, 1, 2, "direct")
    la, lb = a[-1]["log"], b[-1]["log"]
    assert [e["type"] for e in la] == [e["type"] for e in lb] and len(la) > 0
    assert np.allclose(a[0], b[0])
    tde = [e for e in lb if e["type"] in ("tidal disruption", "swallowed whole")]
    assert tde and all("gr_r_isco_au" in e and "gr_captured" in e for e in tde)
    assert not any(k.startswith("gr_") for e in la for k in e)

    r_ = np.random.default_rng(1); c = ev.run_events(gx(), dict(C, gr_solver="dynamics", gr_spin=.7), EP, 1, 2, "direct")
    tc = [e for e in c[-1]["log"] if e["type"] in ("tidal disruption", "swallowed whole")]
    assert tc and all("gr_kerr_captured" in e for e in tc)
    # M = 5e8 Msun with a sun-like star: r_tde < r_mb (Kerr a=.7) -> every encounter is swallowed whole
    assert all(e["type"] == "swallowed whole" for e in tc if e["gr_kerr_captured"])


def test_gr_remnant_ringdown_and_pw():
    r = gr.remnant(1., 1.); assert abs(r["chi_f"] - .686) < .01 and .04 < r["E_rad_frac"] < .055   # GW150914-like: .69, ~5%
    q = gr.qnm_220(68., .67); assert abs(q["f_hz"] - 250.) < 10. and 3.5e-3 < q["tau_s"] < 4.5e-3    # GW150914 (detector-frame M_f~68): ~250 Hz, ~4 ms
    assert abs(gr.qnm_220(1., 0.)["Mw"] - .3737) < .01                     # Schwarzschild M omega_220 = 0.3737
    for a in (0., .5, .9):                                                  # PW circular L^2 = M r^3/(r-rs)^2 minimum at the Kerr ISCO
        rs = gr.pw_rs(a); rr = np.linspace(1.2*rs, 20., 200001); L2 = rr**3/(rr - rs)**2
        assert abs(rr[np.argmin(L2)]/gr.r_isco(a) - 1) < 1e-3
    d = np.array([[1e3, 0., 0.]]); assert abs(gr.pw_correction(d, 1., 2.)[0, 0]/(-1e-6) - (1/(1 - 2e-3)**2 - 1)) < 1e-9
    assert gr.capture_radius(0.) == 4. and abs(gr.capture_radius(1.) - 1.) < 1e-12


def test_live_checkpoint_restore_with_solvers():
    """LiveSim with gcm2d + foodweb + secular + stellar solver: checkpoints deep-copy and restore deterministically."""
    import json
    from universe.run_life_live import LiveSim
    sv = dict(rt=False, stellar_solver=True, foodweb=True, orbit_solver="secular")
    s = LiveSim(seed=4, dt=20., climate_model="gcm2d", solvers=sv, realtime_s=0, ckpt_every=2)
    s.loop(max_years=120); n = s.seq; ref = json.dumps({x["seq"]: x for x in s.hist}[n], sort_keys=True, default=str)
    s.restore(2); s.paused = False; s.done = False; s.loop(max_years=120)
    assert json.dumps({x["seq"]: x for x in s.hist}[n], sort_keys=True, default=str) == ref
