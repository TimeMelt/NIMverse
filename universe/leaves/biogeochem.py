"""Biogeochemistry leaf (physics15 passes 2 + 5, physics_real extensions).
Units: carbon 'C units' with 1 = the atmospheric CO2 carbon at P_REF (280 ppm; Earth ~600 GtC, comparable to Earth biomass,
so biosphere stocks ~1 are commensurate).  pCO2 = P_REF x C_atm.

Pass 2 (carbon-silicate cycle): exchangeable inorganic carbon C_inorg = C_atm (1 + OCEAN_K x ocean) (instant ocean-atmosphere
partition), volcanic outgassing V (mass/volatile scaled), silicate weathering W (Walker/Berner-style: (pCO2/P_REF)^BETA x
exp((T-288)/17.7) x runoff, summed over ice-free bands, land + seafloor; under-ice seafloor weathering remains so ice worlds
are not glued to an artificial 10 bar ceiling), tech fossil emission from a fossil reserve.  Scaled by carbon_rate (time
compression; default ~5, physical rates near 1; 1 = real ~0.5 Myr inorganic timescale). Soft physical CO2 ceiling from
polar CO2 frost, not a hard 10 bar clamp.  Habitats start at weathering-outgassing equilibrium pCO2.

Pass 5 (C/N/P): Biogeochem tracks N/P/detritus per latitude band with diffusion; O2/CH4/biomass remain well-mixed globals.
Biosignatures use mixing-ratio proxies from pool / atmosphere-column scale."""
import numpy as np
from .packets import as_bus

OCEAN_K = 50.          # ocean/atmosphere inorganic carbon ratio per unit ocean cover (Earth ~ 38000/600 at .7 ocean)
BETA, T_W = .3, 17.7   # weathering pCO2 exponent; e-folding K
SEAFLOOR = .15         # seafloor weathering per unit ocean, relative to land
SEAFLOOR_ICE = .04     # residual under-ice seafloor weathering (prevents unbounded CO2 on snowballs)
LAND_REF = .3 + SEAFLOOR*.7
W_NORM = 2.40          # calibration: Earth-analogue band state at P_REF gives weather_factor = 1
TAU_INORG = 5e5        # yr: Earth inorganic carbon / outgassing at carbon_rate = 1
FOSSIL0 = 5.           # C units of extractable fossil carbon
TECH_C = .04           # C units/yr per unit tech level x fossil intensity x tech_pollute/.1
EARTH_M = 3.0e-6
DEEP_OCEAN_K = .12     # outgassing e-folds per +.12 ocean cover above Earth's .7
PCO2_SOFT = 5.         # bar: soft frost-assisted ceiling scale (not a hard clamp)
NUT_DIFF = .05         # band nutrient diffusion coeff in x=sin(lat) units / yr (mixing)

def ocean_of(lat): return float(lat.ocean)

def volcanism(h):
    return float(np.clip((h.m/EARTH_M)**.5, .3, 3.)*np.clip(h.volatiles/.5, .05, 1.))

class CarbonCycle:
    def __init__(self, h, lat, LP=None):
        self.ocean = ocean_of(lat); self.land = 1 - self.ocean; self.wet = float(np.clip(h.volatiles/.5, .05, 1.))
        self.volc = volcanism(h)*float(np.exp(-max(0., self.ocean - .7)/DEEP_OCEAN_K))
        self.C_atm = 1.; self.C_carb = 0.; self.fossil = FOSSIL0; self.cum_V = 0.; self.cum_W = 0.; self.cum_tech = 0.
        self.V0 = (1 + OCEAN_K*.7)/TAU_INORG
        self.last = {}
    @property
    def C_inorg(self): return self.C_atm*(1 + OCEAN_K*self.ocean)
    @C_inorg.setter
    def C_inorg(self, v): self.C_atm = max(1e-6, v/(1 + OCEAN_K*self.ocean))
    def pCO2(self, P_REF): return P_REF*self.C_atm
    def weather_factor(self, lat, pCO2, P_REF):
        """Dimensionless weathering (1 = Earth reference). Land is ice-suppressed; seafloor keeps a cold under-ice term."""
        T = lat.T; ice = lat.last["ice"]; run = self.wet*np.clip(1 + .038*(T - 288.), 0, 4)*(T > 271.)
        Ef = np.mean((1 - ice)*np.exp((T - 288.)/T_W)*run)
        sea_warm = np.mean((1 - ice)*np.exp((np.clip(T, 271, 320) - 288.)/40.))
        sea_ice = np.mean(ice*np.exp((np.clip(T, 200, 280) - 260.)/50.))   # sluggish under-ice seafloor
        sea = sea_warm + SEAFLOOR_ICE/max(SEAFLOOR, 1e-9)*sea_ice
        return float(((self.land*Ef + SEAFLOOR*self.ocean*sea)/(LAND_REF*W_NORM))*(max(pCO2, 1e-9)/P_REF)**BETA)
    def fluxes(self, lat, P_REF):
        V = self.V0*self.volc; W = self.V0*self.weather_factor(lat, self.pCO2(P_REF), P_REF); return V, W
    def equilibrate(self, h, hw, lat, P, lo=1e-6, hi=30., it=24):
        """Bisection in log pCO2 for V = W. hi is a search bound only; runtime uses frost soft-cap, not this ceiling."""
        from .world import P_REF
        V = self.V0*self.volc; T_w = np.full(lat.n, 300.); a, b = np.log(lo), np.log(hi)
        def resid(lp):
            lat.pCO2 = float(np.exp(lp)); lat.T = T_w.copy(); lat.Td = T_w.copy()
            for _ in range(3): lat.integrate(h, hw, P, 400., 0., deep_eq=True)
            return self.V0*self.weather_factor(lat, lat.pCO2, P_REF) - V
        if resid(b) < 0: lp = b
        elif resid(a) > 0: lp = a
        else:
            for _ in range(it):
                m = .5*(a + b)
                if resid(m) > 0: b = m
                else: a = m
            lp = b
        resid(lp); self.C_atm = lat.pCO2/P_REF; return lat.pCO2
    def _frost_sink(self, lat, P_REF, dt):
        """Soft physical CO2 ceiling: polar frost when T_pole below CO2 frost point. Excess -> C_carb (dry-ice buffer)."""
        T_pole = float(.5*(lat.T[0] + lat.T[-1])); p = self.pCO2(P_REF)
        # rough CO2 frost: ~194 K at 1 bar, Clausius-ish
        T_frost = 194. + 6.*np.log(max(p, 1e-6))
        if T_pole >= T_frost - 2 or p < .05: return 0.
        # rate grows as T_pole drops below frost and as p exceeds PCO2_SOFT
        strength = np.clip((T_frost - T_pole)/15., 0, 3)*np.clip(p/PCO2_SOFT, 0, 5)
        sink = min(self.C_inorg*.2, self.V0*self.volc*max(strength, 0.)*dt*8.)
        if sink > 0: self.C_inorg = self.C_inorg - sink; self.C_carb += sink
        return float(sink)
    def step(self, lat, P, dt, tech_level=0., fossil_int=0.):
        from .world import P_REF
        r = P["carbon_rate"]; V, W = self.fluxes(lat, P_REF); V, W = V*r, W*r
        E = min(self.fossil, TECH_C*tech_level*fossil_int*P["tech_pollute"]/.1*dt)
        Wd = min(W*dt, .5*self.C_inorg + V*dt)
        self.C_inorg = self.C_inorg + V*dt - Wd + E; self.C_carb += Wd; self.fossil -= E
        frost = self._frost_sink(lat, P_REF, dt)
        self.cum_V += V*dt; self.cum_W += Wd; self.cum_tech += E
        lat.pCO2 = self.pCO2(P_REF)
        self.last = dict(outgas=V, weathering=Wd/max(dt, 1e-9), tech_co2=E/max(dt, 1e-9), frost_sink=frost/max(dt, 1e-9))
        return dict(pCO2_bar=float(lat.pCO2), C_atm=float(self.C_atm), C_inorg=float(self.C_inorg), C_carb=float(self.C_carb),
                    outgas_flux=float(V), weathering_flux=float(Wd/max(dt, 1e-9)), tech_co2_flux=float(E/max(dt, 1e-9)),
                    frost_flux=float(frost/max(dt, 1e-9)), fossil_left=float(self.fossil))
    def total_C(self): return float(self.C_inorg + self.C_carb + self.fossil)

_EQ_CACHE = {}

def init_climate(h, hw, LP=None, orbit=None):
    """(LatClimate, CarbonCycle) for a habitat, equilibrated to V = W (cached by habitat/knobs so repeated runs are cheap)."""
    from .world import LatClimate
    P = as_bus(LP); lat = LatClimate(h, P, hw, orbit=orbit, spinup=False); cc = CarbonCycle(h, lat, P)
    key = (round(h.a, 9), round(h.e, 9), round(h.tilt, 9), round(h.spin, 9), h.ptype, round(h.volatiles, 9), round(h.m, 12),
           round(float(getattr(hw, "L_star", h.L_star)), 9), tuple(round(P[k], 9) for k in ("greenhouse_K", "albedo_mult", "lat_diffusion", "co2_sens_K", "snowball_K", "runaway_K")),
           None if orbit is None else orbit.key(), "pr1")
    if key in _EQ_CACHE:
        cc.C_atm, T, Td = _EQ_CACHE[key]; lat.T, lat.Td = T.copy(), Td.copy()
        from .world import P_REF
        lat.pCO2 = cc.pCO2(P_REF); lat.integrate(h, hw, P, 1., 0.)
    else:
        cc.equilibrate(h, hw, lat, P); _EQ_CACHE[key] = (cc.C_atm, lat.T.copy(), lat.Td.copy())
    return lat, cc

# ---------------------------------------------------------------------------------------------------------------------------------
# Pass 5: mass-conserving C/N/P. Per-band N/P/detritus with diffusion; O2/CH4/biomass global.
REDFIELD = dict(N=.15, P=.0094)
DET_FRAC, DECAY, BURY, ANOX_FRAC = .3, .01, 1e-4, .3
CH4_TAU = 12.
K_RED = 1e-4
O2_HALF = .5           # kept for o2s kinetics (aerobic share); biosign uses ATM_COL mixing ratios
CH4_K = 1.1e-2         # legacy scale retained as fallback reference; biosign prefers X_CH4
NUT0 = 10.
KN = KP = 2.; N_FIX_TARGET, N_FIX, DENIT = NUT0, 2e-3, 1e-4; P_W, P_BUR, P_FLOOR = 2e-5, 2e-5, .5
# Atmosphere column: C_atm=1 is 280 ppm, so dry-air column in C units is 1/P_REF (~3571 at 1 bar).
# Mixing ratios: X_O2 = O2 / (ATM_COL + O2), X_CH4 = CH4 / (ATM_COL + CH4). ATM_COL chosen so a modest O2 pool
# still yields detectable biosign lines on kyr runs (burial-dominated O2 needs Myr); scale tracks surface pressure proxy.
ATM_COL = 2.0          # effective background column (C units); o2 = O2/(ATM_COL+O2) -> ~0.2 when O2~0.5

def _band_diffuse(arr, D, dt, n):
    """Explicit diffusion in x=sin(lat) with no-flux poles (same geometry as LatClimate). Substeps for CFL."""
    if D <= 0 or dt <= 0: return arr
    dx = 2./n; xe = np.linspace(-1, 1, n + 1); we = 1 - xe**2
    De = D*we/dx**2  # edge coeffs for d/dt form with correct sign below
    # CFL: dt_sub * max(De) < ~0.4
    mD = float(np.max(De)) if np.any(De) else 0.
    nsub = max(1, int(np.ceil(dt*mD/0.35))) if mD > 0 else 1
    h = dt/nsub; out = np.asarray(arr, float).copy()
    for _ in range(nsub):
        flux = np.zeros(n + 1)
        for i in range(1, n): flux[i] = De[i]*(out[i - 1] - out[i])   # high -> low
        out = out + h*(flux[:-1] - flux[1:])
        out = np.maximum(out, 1e-8)
    return out

class Biogeochem:
    """Carbon-silicate cycle + organic C/N/P/O2/CH4 pools. N/P/detritus are per-band; totals conserved with outgassing."""
    def __init__(self, carbon, n_bands=None):
        from .world import N_BANDS
        self.cc = carbon; self.n = int(n_bands or N_BANDS)
        self.det_b = np.zeros(self.n); self.sed = 0.; self.CH4 = 0.; self.O2 = 0.; self.bio = 0.
        self.N_b = np.full(self.n, NUT0); self.P_b = np.full(self.n, NUT0)
        self.mult0 = (NUT0/(NUT0 + KN))*(NUT0/(NUT0 + KP)); self.growth_mult = 1.; self.last = {}
        self.C0 = None; self.growth_b = np.ones(self.n)
    @property
    def det(self): return float(self.det_b.sum())
    @property
    def N(self): return float(self.N_b.mean())
    @property
    def Pn(self): return float(self.P_b.mean())
    def monod_bands(self):
        m = (self.N_b/(self.N_b + KN))*(self.P_b/(self.P_b + KP))/self.mult0
        return np.clip(m, 0., 1.5)
    def monod(self): return float(self.monod_bands().mean())
    def total_C(self): return float(self.cc.total_C() + self.bio + self.det + self.sed + self.CH4)
    def band_weights(self, lat):
        """Distribute global bio fluxes across bands by ice-free productivity proxy (equal-area fallback)."""
        ice = np.asarray(lat.last.get("ice", np.zeros(self.n)), float)
        T = np.asarray(lat.T, float)
        w = np.maximum(1e-3, (1 - ice)*np.exp(-((T - 295.)/35.)**2))
        return w/w.sum()
    def band_stressors(self):
        gb = self.growth_b
        return dict(lat_N=[round(float(v), 4) for v in self.N_b], lat_P=[round(float(v), 4) for v in self.P_b],
                    lat_det=[round(float(v), 5) for v in self.det_b],
                    lat_growth_mult=[round(float(v), 4) for v in gb],
                    N_avail=float(self.N*REDFIELD["N"]), P_avail=float(self.Pn*REDFIELD["P"]),
                    growth_mult=float(self.growth_mult))
    def mixing_ratios(self):
        """Partial-pressure-ish mixing ratios from pool / atmosphere-column scale."""
        col = ATM_COL
        x_o2 = float(self.O2/(self.O2 + col)); x_ch4 = float(self.CH4/(self.CH4 + col))
        return x_o2, x_ch4
    def step(self, B0, B1, loss, lat, P, dt):
        if self.C0 is None: self.C0 = self.total_C() - self.cc.cum_V
        B0 = self.bio; dt = max(dt, 1e-9); loss = max(0., loss); npp = (B1 - B0) + loss
        npp = min(npp, self.cc.C_inorg*.5 + loss)
        loss = npp - (B1 - B0)
        o2s = self.O2/(self.O2 + O2_HALF)
        resp = (1 - DET_FRAC)*loss; to_det = DET_FRAC*loss
        w = self.band_weights(lat)
        self.det_b = self.det_b + to_det*w
        dec_b = np.minimum(self.det_b, DECAY*self.det_b*dt); bur_b = np.minimum(self.det_b - dec_b, BURY*self.det_b*dt)
        dec = float(dec_b.sum()); bur = float(bur_b.sum())
        anox = ANOX_FRAC*(1 - o2s)*dec; aer = dec - anox
        ox = self.CH4*(1 - np.exp(-dt*max(o2s, .05)/CH4_TAU))
        self.cc.C_inorg = self.cc.C_inorg - npp + resp + aer + .5*anox + ox
        self.bio = B1; self.det_b = self.det_b - dec_b - bur_b; self.sed += bur; self.CH4 += .5*anox - ox
        sink = self.O2*(1 - np.exp(-K_RED*self.cc.volc*dt))
        self.O2 = max(0., self.O2 + npp - resp - aer - 2*ox - sink)
        wf = float(np.clip(self.cc.last.get("weathering", 0.)/(self.cc.V0*max(P["carbon_rate"], 1e-9)), P_FLOOR, 5.))
        self.wf = wf
        # per-band nutrients: uptake/remin by local weight; fixation/denit/P weathering global factors applied per band
        npp_b = npp*w; resp_b = resp*w; rem_b = resp_b + dec_b
        fix_b = N_FIX*np.maximum(0., N_FIX_TARGET - self.N_b)*min(1., B1 + .1)*dt
        den_b = DENIT*(1 - o2s)*self.N_b*dt
        self.N_b = np.maximum(1e-4, self.N_b - npp_b + rem_b + fix_b - den_b)
        self.P_b = np.maximum(1e-4, self.P_b - npp_b + rem_b + P_W*NUT0*wf*dt - P_BUR*self.P_b*dt)
        self.N_b = _band_diffuse(self.N_b, NUT_DIFF, dt, self.n)
        self.P_b = _band_diffuse(self.P_b, NUT_DIFF, dt, self.n)
        self.det_b = _band_diffuse(self.det_b, NUT_DIFF*.5, dt, self.n)
        self.growth_b = self.monod_bands(); self.growth_mult = float(self.growth_b.mean())
        from .world import P_REF
        lat.pCO2 = self.cc.pCO2(P_REF)
        x_o2, x_ch4 = self.mixing_ratios()
        self.last = dict(npp=npp/dt, resp=resp/dt, decay=dec/dt, burial=bur/dt, ch4_prod=.5*anox/dt, ch4_ox=ox/dt, o2_sink=sink/dt)
        out = dict(C_bio=float(B1), C_det=float(self.det), C_sed=float(self.sed), CH4_pool=float(self.CH4), O2_pool=float(self.O2),
                   N_avail=float(self.N*REDFIELD["N"]), P_avail=float(self.Pn*REDFIELD["P"]), growth_mult=float(self.growth_mult),
                   npp_flux=float(npp/dt), burial_flux=float(bur/dt), C_total=self.total_C(), C_drift=float(self.total_C() - self.cc.cum_V - self.C0),
                   X_O2=x_o2, X_CH4=x_ch4, ATM_COL=float(ATM_COL))
        out.update(self.band_stressors()); return out
    def biosign(self, flux, climate):
        """O2/CH4 from mixing-ratio proxies; H2O from band water/T. Mutates and returns the BioSignFlux."""
        x_o2, x_ch4 = self.mixing_ratios()
        # Map mixing ratios onto the legacy biosign scale so detection thresholds still fire on kyr runs:
        # Earth-like X_O2~0.21 -> o2~0.21; small CH4 columns stay in the ~1e-4 detectable range via CH4_K*pool floor.
        flux.o2 = float(x_o2); flux.ch4 = float(max(x_ch4, CH4_K*self.CH4))
        from .biosign import h2o_column
        flux.h2o = h2o_column(climate); return flux

def _P_REF():
    from .world import P_REF
    return P_REF
