"""World/climate leaf: zero-dimensional energy-balance box model."""
import numpy as np
from .packets import ClimateStress, as_bus
from .orbit_climate import insolation

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
    if getattr(hw, "hz_in_now", 0.) > 0:                         # moving HZ from the evolving host (orbit a fixed)
        Ln = float(getattr(hw, "L_now", hw.L_star)); inz = bool(hw.hz_in_now <= h.a <= hw.hz_out_now)
        st.update(L_host=Ln, irradiance_now=float(Ln/max(h.a, 1e-9)**2), hz_in_now=float(hw.hz_in_now), hz_out_now=float(hw.hz_out_now),
                  in_hz_now=inz, hz_drift=bool(inz != bool(h.in_hz)))
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

# ---------------------------------------------------------------------------------------------------------------
# Physics15 pass 1: latitudinal energy-balance climate (stateful; HabitatRun uses it, world.step above stays the
# stateless 0-D v1 box for callers that only need a quick equilibrium: learn_life filters, life_loop habitability).
# ---------------------------------------------------------------------------------------------------------------
SIGMA = 5.670e-8; S_SUN = 1361.          # W m^-2 K^-4; W m^-2 at 1 AU from 1 Lsun
N_BANDS = 24                              # equal-area bands (uniform in x = sin latitude)
C_MIX, C_LAND, C_DEEP, GAMMA = 10., 1., 100., .7   # W yr m^-2 K^-1 (mixed layer ~80 m, land, deep ocean); W m^-2 K^-1 exchange
ALB_ICE, T_ICE, ICE_W = .6, 265., 2.5     # ice albedo; ice where (T + .3 x seasonal amp) < T_ICE (smooth, K)
ALB_WARM_K = .95                          # ice-free surface albedo = .95 x v1 bond albedo (calibrated: Earth analogue -> 289 K mean, 304 K equator, 262 K poles)
ALB_P2 = .15                              # zenith-angle/cloud albedo rise toward the poles: + ALB_P2 x P2(sin lat) (area-mean zero)
S2X0 = 4.                                 # K per CO2 doubling (default co2_sens_K knob)
WV_K, WV_CAP = 2., 120.                   # water-vapour feedback K x (e_s(T)/e_s(288) - 1), capped
P_REF = 2.8e-4                            # bar CO2 at which greenhouse_K applies (pre-industrial Earth)
WV_SAT = 10.                              # water-vapour feedback saturation (es ratio - 1 scale)
H_MAX = 2.                                # yr implicit substep

def es_ratio(T): return float(np.exp(5420.*(1/288. - 1/max(T, 150.))))   # Clausius-Clapeyron saturation vapour pressure / e_s(288 K)

def band_x(n=N_BANDS): return (np.arange(n) + .5)*2./n - 1.

def band_of(sinlat, n=N_BANDS): return np.clip(((np.asarray(sinlat) + 1)/2*n).astype(int), 0, n-1)

def greenhouse(h, P, pCO2, T_mean):
    """Total greenhouse offset G (K): v1 baseline greenhouse_K at P_REF (scaled by volatiles) + co2 log forcing + water-vapour feedback."""
    fv = min(1., h.volatiles/.5); g0 = P["greenhouse_K"]*fv
    g_co2 = P["co2_sens_K"]*np.log2(max(pCO2, 1e-7)/P_REF)
    x = max(0., es_ratio(T_mean) - 1.); g_wv = min(WV_CAP, WV_K*fv*x/(1 + x/WV_SAT))   # saturating (band-saturated H2O lines)
    return float(max(0., g0 + g_co2 + g_wv)), float(g_co2), float(g_wv)

class LatClimate:
    """Latitudinal EBM (North/Budyko-Sellers style) with diffusion in x = sin(lat), two-layer heat capacity (mixed layer +
    deep ocean), ice-albedo feedback and seasonal amplitude from the orbit geometry.  Per band i:
        C_s dT/dt = S0 q_i (1 - alpha_i) - sigma (T_i - G)^4 + F + gamma (Td_i - T_i) + D d/dx[(1 - x^2) dT/dx]
        C_d dTd/dt = gamma (T_i - Td_i)
    Emission temperature T - G (greenhouse offset) reproduces the v1 box exactly for a uniform, ice-free planet.
    F = forcing from stress dT (drift, tech aerosol) converted with the local Planck slope; impact winter is a one-step
    diagnostic offset (not integrated) so 'water_chronic' is the pre-impact state as in v1.
    pCO2 comes from an attached carbon cycle (leaves.biogeochem) or stays at P_REF; orbit (e, varpi, obliquity) from
    an attached driver (leaves.orbit_climate.SecularOrbit) or the fixed habitat values."""
    def __init__(self, h, LP=None, hw=None, n=N_BANDS, orbit=None, spinup=True):
        P = as_bus(LP); self.n = n; self.x = band_x(n); self.lat = np.degrees(np.arcsin(self.x)); self.orbit = orbit
        xe = np.linspace(-1, 1, n + 1); self.w_edge = 1 - xe**2; self.dx = 2./n
        self.ocean = float(np.clip(h.volatiles, 0, 1)) if h.ptype in ("ocean", "terran", "ice") else float(np.clip(.5*h.volatiles, 0, 1))
        self.C_s = C_MIX*self.ocean + C_LAND*(1 - self.ocean); self.C_d = C_DEEP*self.ocean + 2.
        self.D = float(np.clip(P["lat_diffusion"]/max(h.spin, .3)**.5, .05, 5.))
        A = float(np.clip(ALBEDO[h.ptype]*P["albedo_mult"], 0, .95))
        self.alb_w = A if h.ptype in ("ice", "lava", "gas giant", "ice giant") else ALB_WARM_K*A
        self.alb_w = np.clip(self.alb_w + ALB_P2*(1.5*self.x**2 - .5)*(h.ptype not in ("lava",)), .02, .9)
        self.alb_ice = np.maximum(self.alb_w, ALB_ICE) if h.ptype not in ("lava",) else self.alb_w
        self.pCO2 = P_REF; self.t = 0.; self.T = None; self.Td = None; self.G = 0.; self.last = {}; self.tech_co2_mult = 1.
        self.orb = self._orbit(h, 0.)
        if spinup:
            T0 = teq(getattr(hw, "L_now", getattr(hw, "L_star", h.L_star)) if hw is not None else h.L_star, h.a, float(np.mean(self.alb_w))) + P["greenhouse_K"]*min(1., h.volatiles/.5)
            self.T = np.full(n, T0); self.Td = self.T.copy()
            for _ in range(3): self.integrate(h, hw, P, 400., 0., deep_eq=True)
    def _orbit(self, h, t):
        if self.orbit is not None: return self.orbit.at(t)
        return dict(e=float(h.e), varpi=0., obliquity=float(abs(h.tilt)))
    def S0(self, h, hw):
        L = float(getattr(hw, "L_now", getattr(hw, "L_star", h.L_star))) if hw is not None else float(h.L_star)
        return S_SUN*L/max(h.a, 1e-6)**2
    def season_amp(self, ins, S0):
        """Annual-cycle half-range (K): forced-damped response Q_amp(1-alpha)/sqrt(B^2 + (C omega)^2), ocean/land mix."""
        B = 4*SIGMA*np.maximum(self.T - self.G, 20.)**3; Q = S0*ins["amp"]*(1 - np.mean(self.alb_w))
        om = 2*np.pi; a_oc = Q/np.sqrt(B**2 + (C_MIX*om)**2); a_ld = Q/np.sqrt(B**2 + (.3*om)**2)
        return self.ocean*a_oc + (1 - self.ocean)*a_ld
    def ice(self, T, amp): return 1/(1 + np.exp((T + .3*amp - T_ICE)/ICE_W))
    def integrate(self, h, hw, P, dt, F=0., deep_eq=False):
        """Backward-Euler substeps (<= H_MAX yr) with linearised OLR; deep layer explicit (or held at equilibrium for spin-up)."""
        S0 = self.S0(h, hw); ins = insolation(self.x, self.orb["obliquity"], self.orb["e"], self.orb["varpi"])
        k = max(1, int(np.ceil(dt/H_MAX))); hh = dt/k; n = self.n; De = self.D*self.w_edge/self.dx**2
        Lap = np.zeros((n, n))
        for i in range(n):
            if i > 0: Lap[i, i-1] += De[i]; Lap[i, i] -= De[i]
            if i < n-1: Lap[i, i+1] += De[i+1]; Lap[i, i] -= De[i+1]
        for _ in range(k):
            self.G, gco2, gwv = greenhouse(h, P, self.pCO2*self.tech_co2_mult, float(self.T.mean()))
            amp = self.season_amp(ins, S0); ice = self.ice(self.T, amp)
            alb = self.alb_w + (self.alb_ice - self.alb_w)*ice
            Te = np.maximum(self.T - self.G, 20.); olr = SIGMA*Te**4; B = 4*SIGMA*Te**3
            asr = S0*ins["mean"]*(1 - alb)
            if deep_eq: self.Td = self.T.copy()
            M = np.diag(self.C_s/hh + B + GAMMA) - Lap
            rhs = self.C_s/hh*self.T + asr - olr + B*self.T + F + GAMMA*self.Td
            self.T = np.linalg.solve(M, rhs)
            if not deep_eq: self.Td = self.Td + hh*GAMMA*(self.T - self.Td)/self.C_d
        amp = self.season_amp(ins, S0); ice = self.ice(self.T, amp)
        self.last = dict(ins=ins, amp=amp, ice=ice, S0=S0, asr=float((S0*ins["mean"]*(1 - (self.alb_w + (self.alb_ice - self.alb_w)*ice))).mean()),
                         g_co2=gco2, g_wv=gwv)
        return self.T
    def step(self, h, hw, LP=None, dt_w_yr=1., stress=None, t=None):
        """Integrate dt, return a ClimateStress compatible with world.step plus per-band fields in stressors."""
        P = as_bus(LP); st0 = dict(stress or {}); t = self.t + dt_w_yr if t is None else float(t); self.t = t
        self.orb = self._orbit(h, t)
        dT_imp = float(st0.get("impact_dT", 0.)); dT_force = float(st0.get("dT", 0.)) - dT_imp
        B_mean = float((4*SIGMA*np.maximum(self.T - self.G, 20.)**3).mean())
        self.integrate(h, hw, P, dt_w_yr, B_mean*dT_force)
        return self.packet(h, hw, P, st0, dT_imp)
    def bands(self, dT=0.):
        T = self.T + dT; amp = self.last["amp"]; return T, amp
    def packet(self, h, hw, P, st0, dT_imp=0.):
        T, amp = self.bands(dT_imp); vs, sb, rk = h.volatiles, P["snowball_K"], P["runaway_K"]
        def wat(TT): return np.array([.5*(water_frac(a - s, vs, sb, rk) + water_frac(a + s, vs, sb, rk)) for a, s in zip(TT, amp)])
        w_b = wat(T); w_c = wat(self.T) if dT_imp else w_b
        ins = self.last["ins"]["mean"]; light = np.sqrt(np.clip(ins/max(ins.mean(), 1e-9), 0, 3))
        prod = w_b*light*np.exp(-((T - 295.)/30.)**2); pm = float(prod.mean())
        ice = self.ice(T, amp); Tm = float(T.mean()); w = float(w_b.mean()); ice_f = float(ice.mean())
        cls = ("runaway" if Tm >= rk else "snowball" if (Tm <= sb or ice_f > .97) else "hot" if Tm > 323 else "cold" if Tm < 273 else "temperate")
        iceline = float(np.degrees(np.arcsin(np.clip(1 - ice_f, 0., 1.))))   # equivalent symmetric ice edge: polar caps poleward of it cover 1 - sin(lat)
        uvn = float(getattr(hw, "uv_now", hw.uv_proxy)); duty = float(getattr(hw, "flare_duty", 0.))
        st = dict(uv=uvn, flare=float(hw.flare_rate), flare_duty=duty, xray=float(getattr(hw, "xray_now", 0.)),
                  uv_kill=float(min(.5, .05*max(0., uvn/max(hw.uv_proxy, 1e-9) - 1.))), impact=float(h.impact_flux), season_K=float(amp.mean()),
                  lat_deg=[round(float(v), 2) for v in self.lat], lat_T=[round(float(v), 3) for v in T], lat_water=[round(float(v), 4) for v in w_b],
                  lat_prod=[round(float(v/pm), 4) if pm > 0 else 0. for v in prod], lat_ice=[round(float(v), 4) for v in ice],
                  lat_season_K=[round(float(v), 3) for v in amp], lat_insol=[round(float(v), 5) for v in ins],
                  T_eq_lat=float(T[self.n//2 - 1:self.n//2 + 1].mean()), T_pole=float(.5*(T[0] + T[-1])), ice_frac=ice_f, ice_lat_deg=iceline,
                  greenhouse_total_K=float(self.G), greenhouse_co2_K=float(self.last.get("g_co2", 0.)), greenhouse_wv_K=float(self.last.get("g_wv", 0.)),
                  pCO2_bar=float(self.pCO2*self.tech_co2_mult), pCO2_geo_bar=float(self.pCO2), T_deep=float(self.Td.mean()), ecc=float(self.orb["e"]), obliquity_deg=float(np.degrees(self.orb["obliquity"])),
                  varpi_deg=float(np.degrees(self.orb["varpi"]) % 360.), insol_mean=float(self.last["S0"]*ins.mean()), climate_model="lat_ebm")
        if dT_imp: st["water_chronic"] = float(w_c.mean()); st["lat_water_chronic"] = [round(float(v), 4) for v in w_c]
        if getattr(hw, "hz_in_now", 0.) > 0:
            Ln = float(getattr(hw, "L_now", hw.L_star)); inz = bool(hw.hz_in_now <= h.a <= hw.hz_out_now)
            st.update(L_host=Ln, irradiance_now=float(Ln/max(h.a, 1e-9)**2), hz_in_now=float(hw.hz_in_now), hz_out_now=float(hw.hz_out_now),
                      in_hz_now=inz, hz_drift=bool(inz != bool(h.in_hz)))
        for k_, v in st0.items():
            if k_ not in st: st[k_] = v
        return ClimateStress(Tm, w, cls, st)
