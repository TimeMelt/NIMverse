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
    T0 = teq(getattr(hw, "L_now", hw.L_star), h.a*(1-h.e**2)**.25, A) + P["greenhouse_K"]*min(1., h.volatiles/.5)
    season = 30.*abs(np.sin(h.tilt))                            # seasonal half-amplitude proxy
    T = T0 + (stress or {}).get("dT", 0.)
    w = .5*(water_frac(T-season, h.volatiles, P["snowball_K"], P["runaway_K"]) + water_frac(T+season, h.volatiles, P["snowball_K"], P["runaway_K"]))
    cls = ("runaway" if T >= P["runaway_K"] else "snowball" if T <= P["snowball_K"] else
           "hot" if T > 323 else "cold" if T < 273 else "temperate")
    uvn = float(getattr(hw, "uv_now", hw.uv_proxy)); duty = float(getattr(hw, "flare_duty", 0.))
    st = dict(uv=uvn, flare=float(hw.flare_rate), flare_duty=duty, xray=float(getattr(hw, "xray_now", 0.)),
              uv_kill=float(min(.5, .05*max(0., uvn/max(hw.uv_proxy, 1e-9) - 1.))),   # flare UV sterilises surface fraction
              impact=float(h.impact_flux), season_K=float(season))
    st.update(stress or {}); return ClimateStress(float(T), w, cls, st)

def impacts(h, LP, rng, dt_w_yr=1.):
    """Stochastic impacts: Poisson(impact_flux*impact_scale*dt/1000) events per step.  Each event -> transient
    impact-winter dT (-15 K) and a biomass kill fraction impact_kill (compounded).  Returns stress dict."""
    P = as_bus(LP); lam = h.impact_flux*P["impact_scale"]*dt_w_yr/1000.
    n = int(rng.poisson(lam)) if lam > 0 else 0
    if not n: return {"impacts": 0}
    return {"impacts": n, "dT": -15.*min(n, 3), "impact_kill": float(1 - (1 - P["impact_kill"])**n)}

class ClimateHistory:
    """Rolling (t, T, water_frac, class) history for a world (bounded length)."""
    def __init__(self, n=200): self.n, self.rows, self.impacts = n, [], 0
    def push(self, t, c):
        self.rows.append((float(t), c.T_K, c.liquid_water_frac, c.climate_class)); del self.rows[:-self.n]
        self.impacts += int(c.stressors.get("impacts", 0))
    def summary(self):
        if not self.rows: return {}
        T = np.array([r[1] for r in self.rows]); W = np.array([r[2] for r in self.rows])
        return dict(n=len(self.rows), T_mean=float(T.mean()), T_std=float(T.std()), T_min=float(T.min()), T_max=float(T.max()),
                    water_mean=float(W.mean()), water_min=float(W.min()), impacts_total=self.impacts,
                    tail=[[r[0], round(r[1], 2), round(r[2], 3), r[3]] for r in self.rows[-5:]],
                    series=[[r[0], round(r[1], 2), round(r[2], 3)] for r in self.rows[::max(1, len(self.rows)//200)]])

DRIFT_K = 35.   # K of planetary cooling per unit of organism-felt stress offset (stress uses |T-293|/80)

def drift_offset(t, phase, flip, a):
    """Shared nonstationary climate signal: two incommensurate cycles (8.3 & 21 kyr) + telegraph flip state, scaled by climate_drift a."""
    return float(a*(.22*np.sin(2*np.pi*t/8300. + phase) + .12*np.sin(2*np.pi*t/21000. + 2*phase) + .12*flip))

class ClimateDrift:
    """Planetary climate drift: same cycles/flips that organisms feel, applied to the world box as dT (and thus water_frac,
    climate class, chronic stress). step() -> stress dict {dT, drift_offset, drift_flip} to merge into world.step(stress=...)."""
    def __init__(self, a, rng): self.a, self.phase, self.flip, self.offset = float(a), float(rng.uniform(0, 2*np.pi)), 0., 0.
    def step(self, t, dt, rng):
        if rng.random() < 1 - np.exp(-dt/12000.): self.flip = -self.flip if self.flip else float(rng.choice([-1., 1.]))
        self.offset = drift_offset(t, self.phase, self.flip, self.a)
        return {"dT": -DRIFT_K*self.offset, "drift_offset": self.offset, "drift_flip": self.flip}
