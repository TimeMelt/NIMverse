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
def test_stellar_sun():
    assert 9. < st.t_ms_Gyr(1.) < 11. and st.t_ms_Gyr(.8) > st.t_ms_Gyr(1.) > st.t_ms_Gyr(1.5)
    L, T, R = st.track(1., 1., 5777., 4.57, 4.57); assert abs(L - 1) < 1e-9
    d = st.solve_track(1.); L0 = st._interp(d, "L", 0.); assert .6 < L0 < .8      # faint young Sun
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


def test_live_checkpoint_restore_with_solvers():
    """LiveSim with gcm2d + foodweb + secular + stellar solver: checkpoints deep-copy and restore deterministically."""
    import json
    from universe.run_life_live import LiveSim
    sv = dict(rt=False, stellar_solver=True, foodweb=True, orbit_solver="secular")
    s = LiveSim(seed=4, dt=20., climate_model="gcm2d", solvers=sv, realtime_s=0, ckpt_every=2)
    s.loop(max_years=120); n = s.seq; ref = json.dumps({x["seq"]: x for x in s.hist}[n], sort_keys=True, default=str)
    s.restore(2); s.paused = False; s.done = False; s.loop(max_years=120)
    assert json.dumps({x["seq"]: x for x in s.hist}[n], sort_keys=True, default=str) == ref
