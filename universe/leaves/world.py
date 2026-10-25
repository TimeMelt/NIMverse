"""World/climate leaf: zero-dimensional energy-balance box model."""
import numpy as np
from .packets import ClimateStress, as_bus

ALBEDO = {"terran": .3, "ocean": .25, "desert": .35, "ice": .6, "lava": .1, "gas giant": .34, "ice giant": .3}

def teq(L, a, A): return float(278.6*(L/a**2)**.25*(1-A)**.25)   # K, Lsun, AU; Earth A=.3 -> 255 K

def water_frac(T, volatiles, snowball, runaway):
    if T >= runaway or T <= snowball: return 0.
    if T < 273.: f = (T-snowball)/(273.-snowball)*.5          # partially frozen (equatorial belts)
    elif T <= 373.: f = 1.
    else: f = (runaway-T)/(runaway-373.)                       # hot, steam-heavy, shrinking oceans
    return float(np.clip(f*min(1., 1.5*volatiles), 0, 1))

def step(h, hw, LP=None, dt_w_yr=1., stress=None):
    """habitat h, HostWeather hw -> ClimateStress. dt unused in equilibrium v1 (kept for interface)."""
    P = as_bus(LP); A = float(np.clip(ALBEDO[h.ptype]*P["albedo_mult"], 0, .95))
    T0 = teq(hw.L_star, h.a*(1-h.e**2)**.25, A) + P["greenhouse_K"]*min(1., h.volatiles/.5)
    season = 30.*abs(np.sin(h.tilt))                            # seasonal half-amplitude proxy
    T = T0 + (stress or {}).get("dT", 0.)
    w = .5*(water_frac(T-season, h.volatiles, P["snowball_K"], P["runaway_K"]) + water_frac(T+season, h.volatiles, P["snowball_K"], P["runaway_K"]))
    cls = ("runaway" if T >= P["runaway_K"] else "snowball" if T <= P["snowball_K"] else
           "hot" if T > 323 else "cold" if T < 273 else "temperate")
    st = dict(uv=float(hw.uv_proxy), flare=float(hw.flare_rate), impact=float(h.impact_flux), season_K=float(season))
    st.update(stress or {}); return ClimateStress(float(T), w, cls, st)
