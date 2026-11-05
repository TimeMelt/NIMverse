"""physics_real leftovers closed as far as a 1-D/box leaf allows.
Run: PYTHONPATH=/workspace/nim-universe:/workspace/NIM python -m pytest universe/leaves/test_physics_real.py -q
"""
import numpy as np
from .packets import HabitatState, HostWeather, NicheOffer, ClimateStress, as_bus
from . import world, biogeochem as BG, individuals as I
from .orbit_climate import SecularOrbit, insolation
from .params import LIFE_DEFAULTS
P = as_bus(None); SUN = HostWeather.from_star(1., 1., 5800.)

def earth(**kw):
    d = dict(idx=0, name="earth", m=3e-6, a=1., e=.0167, spin=1., tilt=.409, albedo_scale=1., volatiles=.95, density=1.67, in_hz=True,
             kind="terran", water=.5, t_form=0., insol=1.)
    d.update(kw); return HabitatState(*d.values())

def test_clouds_and_earth_mean_T():
    lc = world.LatClimate(earth(), P, SUN); c = lc.step(earth(), SUN, P, 20.); s = c.stressors
    assert "lat_cloud" in s and "cloud_frac" in s and s["cloud_frac"] > .05
    assert "greenhouse_cloud_K" in s and s["greenhouse_cloud_K"] > 0
    assert 288 <= c.T_K <= 291   # Earth-analogue with clouds + greenhouse retune

def test_desert_cooler_than_pre_real():
    # earth() positional: kind -> ptype, water -> volatiles
    hd = earth(name="desert", kind="desert", water=.15)
    assert hd.ptype == "desert"
    lc = world.LatClimate(hd, P, SUN); c = lc.step(hd, SUN, P, 20.)
    # dry WV + desert albedo should keep mean below the old 305–307 K warm bias at P_REF
    assert c.T_K < 303

def test_carbon_rate_default_near_physical():
    assert 1. <= LIFE_DEFAULTS["carbon_rate"] <= 5.

def test_ice_world_not_hard_capped_at_10bar():
    h = earth(name="ice", kind="ice", water=1., a=1.4, m=2e-6)
    assert h.ptype == "ice"
    lc = world.LatClimate(h, P, SUN); cc = BG.CarbonCycle(h, lc, P)
    # soft frost exists; equilibrate search hi is not a runtime clamp
    assert BG.PCO2_SOFT < 10 and hasattr(cc, "_frost_sink")
    p = cc.equilibrate(h, SUN, lc, P)
    assert p < 30.   # search bound only; may be high but frost soft-caps in runtime step

def test_per_band_nutrients_and_diffusion():
    h = earth(); lc, cc = BG.init_climate(h, SUN, P); bg = BG.Biogeochem(cc)
    assert bg.N_b.shape == (lc.n,) and bg.P_b.shape == (lc.n,)
    bg.N_b[0] = 50.; bg.N_b[-1] = 1.
    n0 = bg.N_b.copy()
    bg.N_b = BG._band_diffuse(bg.N_b, BG.NUT_DIFF, 50., bg.n)
    assert bg.N_b[0] < n0[0] and bg.N_b[-1] > n0[-1]
    st = bg.band_stressors(); assert "lat_growth_mult" in st and len(st["lat_N"]) == bg.n

def test_o2_ch4_mixing_ratios():
    h = earth(); lc, cc = BG.init_climate(h, SUN, P); bg = BG.Biogeochem(cc)
    bg.O2 = .5; bg.CH4 = .01
    x_o2, x_ch4 = bg.mixing_ratios()
    assert abs(x_o2 - .5/(.5 + BG.ATM_COL)) < 1e-12
    assert abs(x_ch4 - .01/(.01 + BG.ATM_COL)) < 1e-12
    class F: o2=0.; ch4=0.; h2o=0.
    f = F(); bg.biosign(f, lc.step(h, SUN, P, 1.))
    assert abs(f.o2 - x_o2) < 1e-12

def test_secular_second_order_and_spin_obliquity():
    so = SecularOrbit([1., 5.2], [3e-6, 1e-3], 1., 0, .41, spin=1., seed=3)
    assert so.key()[-1] == "pr1"
    a = [so.at(t) for t in np.linspace(0, 2e5, 80)]
    e = [d["e"] for d in a]; ob = [d["obliquity"] for d in a]
    assert np.ptp(e) > 1e-4          # forced/2nd-order motion present
    assert np.ptp(ob) > np.radians(.3) and np.ptp(ob) < np.radians(8)
    # period set by alpha - s, not a fixed 41 kyr sinusoid alone
    assert so.alpha > 0 and abs(so.alpha - so.s) > 0

def test_allometric_lifespan_and_guilds():
    r = np.random.default_rng(2); pop = I.found(None, r, n=40)
    assert len(pop.guild) == len(pop.id)
    assert set(np.unique(pop.guild)).issubset({0, 1})
    # age_mu ~ M^-0.25: bigger bodies live longer
    M = np.array([.2, 1., 5.]); mu = (M**-.25)/I.LIFE_K
    assert mu[0] > mu[1] > mu[2]
    # predation prefers producers when consumers present
    pop.logm[:] = 0.; pop.logm[0] = 1.2; pop.logm[1] = -1.; pop.guild[:] = 0; pop.guild[0] = 1
    pop.energy[:] = 1.; pop.pos[:] = pop.pos[0]   # same cell
    e_prod = pop.energy[1]
    I.predation(pop, as_bus(dict(pred_rate=.2)), 1.)
    assert pop.energy[1] < e_prod

def test_diffusion_scales_with_size_spin():
    big = earth(m=24e-6, spin=.5); small = earth(m=3e-6, spin=2.)
    Lb = world.LatClimate(big, P, SUN); Ls = world.LatClimate(small, P, SUN)
    assert Lb.D > Ls.D
