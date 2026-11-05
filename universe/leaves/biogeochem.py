"""Biogeochemistry leaf (physics15 passes 2 + 5).
Units: carbon 'C units' with 1 = the atmospheric CO2 carbon at P_REF (280 ppm; Earth ~600 GtC, comparable to Earth biomass,
so biosphere stocks ~1 are commensurate).  pCO2 = P_REF x C_atm.

Pass 2 (carbon-silicate cycle): exchangeable inorganic carbon C_inorg = C_atm (1 + OCEAN_K x ocean) (instant ocean-atmosphere
partition), volcanic outgassing V (mass/volatile scaled), silicate weathering W (Walker/Berner-style: (pCO2/P_REF)^BETA x
exp((T-288)/17.7) x runoff, summed over ice-free bands, land + a weak seafloor term) that buries carbonate (C_carb), tech fossil
emission from a fossil reserve.  Everything is scaled by carbon_rate (time compression; 1 = real ~0.5 Myr inorganic timescale).
The habitat starts at its weathering-outgassing equilibrium pCO2 (an old planet sits on its thermostat), found by bisection
with the latitudinal EBM, so v1-calibrated seeds are not thrown into an artificial transient.  Snowball (no ice-free weathering)
-> CO2 accumulates -> deglaciation; hot -> weathering drawdown: transitions emerge from the feedback.

Pass 5 (C/N/P): see Biogeochem.bio_step."""
import numpy as np
from .packets import as_bus

OCEAN_K = 50.          # ocean/atmosphere inorganic carbon ratio per unit ocean cover (Earth ~ 38000/600 at .7 ocean)
BETA, T_W = .3, 17.7   # weathering pCO2 exponent; e-folding K
SEAFLOOR = .15         # seafloor weathering per unit ocean, relative to land
LAND_REF = .3 + SEAFLOOR*.7
W_NORM = 2.40          # calibration: the Earth-analogue band state at P_REF gives weather_factor = 1 (so V = W there)
TAU_INORG = 5e5        # yr: Earth inorganic carbon / outgassing at carbon_rate = 1
FOSSIL0 = 5.           # C units of extractable fossil carbon
TECH_C = .04           # C units/yr per unit tech level x fossil intensity x tech_pollute/.1 (~ +0.5 K over 500 yr at level 1 on seed 4)
EARTH_M = 3.0e-6
DEEP_OCEAN_K = .12     # outgassing e-folds per +.12 ocean cover above Earth's .7 (seafloor pressure caps melt degassing)

def ocean_of(lat): return float(lat.ocean)

def volcanism(h):
    return float(np.clip((h.m/EARTH_M)**.5, .3, 3.)*np.clip(h.volatiles/.5, .05, 1.))   # mantle heat ~ mass; dry worlds outgas less

class CarbonCycle:
    def __init__(self, h, lat, LP=None):
        self.ocean = ocean_of(lat); self.land = 1 - self.ocean; self.wet = float(np.clip(h.volatiles/.5, .05, 1.))   # dry worlds: little runoff
        self.volc = volcanism(h)*float(np.exp(-max(0., self.ocean - .7)/DEEP_OCEAN_K))   # deep global oceans suppress degassing (Kite+ 2009)
        self.C_atm = 1.; self.C_carb = 0.; self.fossil = FOSSIL0; self.cum_V = 0.; self.cum_W = 0.; self.cum_tech = 0.
        self.V0 = (1 + OCEAN_K*.7)/TAU_INORG          # Earth-normalised outgassing, C units / yr at carbon_rate 1
        self.last = {}
    @property
    def C_inorg(self): return self.C_atm*(1 + OCEAN_K*self.ocean)
    @C_inorg.setter
    def C_inorg(self, v): self.C_atm = max(1e-6, v/(1 + OCEAN_K*self.ocean))
    def pCO2(self, P_REF): return P_REF*self.C_atm
    def weather_factor(self, lat, pCO2, P_REF):
        """Dimensionless weathering (1 = Earth reference) from the band state of a LatClimate."""
        T = lat.T; ice = lat.last["ice"]; run = self.wet*np.clip(1 + .038*(T - 288.), 0, 4)*(T > 271.)
        Ef = np.mean((1 - ice)*np.exp((T - 288.)/T_W)*run)
        sea = np.mean((1 - ice)*np.exp((np.clip(T, 271, 320) - 288.)/40.))
        return float(((self.land*Ef + SEAFLOOR*self.ocean*sea)/(LAND_REF*W_NORM))*(max(pCO2, 1e-9)/P_REF)**BETA)
    def fluxes(self, lat, P_REF):
        V = self.V0*self.volc; W = self.V0*self.weather_factor(lat, self.pCO2(P_REF), P_REF); return V, W
    def equilibrate(self, h, hw, lat, P, lo=1e-6, hi=10., it=22):
        """Bisection in log pCO2 for V = W (warm-branch start each trial). Sets C_atm and leaves lat at that state."""
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
    def step(self, lat, P, dt, tech_level=0., fossil_int=0.):
        """Advance inorganic carbon dt yr; returns stress-like dict of fluxes. Mass: C_inorg + C_carb + fossil (+ pass-5 pools)
        = initial + cum_V (outgassing is the only external source)."""
        from .world import P_REF
        r = P["carbon_rate"]; V, W = self.fluxes(lat, P_REF); V, W = V*r, W*r
        E = min(self.fossil, TECH_C*tech_level*fossil_int*P["tech_pollute"]/.1*dt)
        Wd = min(W*dt, .5*self.C_inorg + V*dt)      # cannot weather more than half the inorganic stock in one step
        self.C_inorg = self.C_inorg + V*dt - Wd + E; self.C_carb += Wd; self.fossil -= E
        self.cum_V += V*dt; self.cum_W += Wd; self.cum_tech += E
        lat.pCO2 = self.pCO2(P_REF)
        self.last = dict(outgas=V, weathering=Wd/max(dt, 1e-9), tech_co2=E/max(dt, 1e-9))
        return dict(pCO2_bar=float(lat.pCO2), C_atm=float(self.C_atm), C_inorg=float(self.C_inorg), C_carb=float(self.C_carb),
                    outgas_flux=float(V), weathering_flux=float(Wd/max(dt, 1e-9)), tech_co2_flux=float(E/max(dt, 1e-9)), fossil_left=float(self.fossil))
    def total_C(self): return float(self.C_inorg + self.C_carb + self.fossil)

_EQ_CACHE = {}

def init_climate(h, hw, LP=None, orbit=None):
    """(LatClimate, CarbonCycle) for a habitat, equilibrated to V = W (cached by habitat/knobs so repeated runs are cheap)."""
    from .world import LatClimate
    P = as_bus(LP); lat = LatClimate(h, P, hw, orbit=orbit, spinup=False); cc = CarbonCycle(h, lat, P)
    key = (round(h.a, 9), round(h.e, 9), round(h.tilt, 9), round(h.spin, 9), h.ptype, round(h.volatiles, 9), round(h.m, 12),
           round(float(getattr(hw, "L_star", h.L_star)), 9), tuple(round(P[k], 9) for k in ("greenhouse_K", "albedo_mult", "lat_diffusion", "co2_sens_K", "snowball_K", "runaway_K")),
           None if orbit is None else orbit.key())
    if key in _EQ_CACHE:
        cc.C_atm, T, Td = _EQ_CACHE[key]; lat.T, lat.Td = T.copy(), Td.copy()
        from .world import P_REF
        lat.pCO2 = cc.pCO2(P_REF); lat.integrate(h, hw, P, 1., 0.)
    else:
        cc.equilibrate(h, hw, lat, P); _EQ_CACHE[key] = (cc.C_atm, lat.T.copy(), lat.Td.copy())
    return lat, cc

# ---------------------------------------------------------------------------------------------------------------------------------
# Pass 5: mass-conserving C/N/P wrapper around the 3-guild biosphere.  Biomass units are C units (see header).  Per step:
#   NPP = dB + L  (L = mortality/disturbance losses, so the ledger closes exactly whatever the guild ODE did)
#   C_inorg -NPP; L -> DET_FRAC detritus, rest respired to C_inorg; detritus decays (DECAY/yr): aerobic -> C_inorg,
#   ANOX_FRAC x (1 - O2 share) is fermented (2 CH2O -> CH4 + CO2); CH4 oxidises (lifetime CH4_TAU / O2 level) -> C_inorg; burial BURY/yr of detritus -> C_sed.
#   O2: + NPP - respiration - aerobic decay - 2 CH4 oxidation (photosynthesis/respiration stoichiometry), - reductant sink K_RED x volc x O2.
#   N, P (carried in 'supportable C' units; x REDFIELD for molar ratios): uptake = NPP, remineralisation with respiration/decay,
#   N fixation (biomass-driven, toward N_FIX_TARGET) + denitrification (anoxic), P from weathering (x weather_factor) - P burial.
#   growth_mult = Monod(N) x Monod(P) normalised to 1 at the start -> biosphere.step (nutrient limitation).
REDFIELD = dict(N=.15, P=.0094)
DET_FRAC, DECAY, BURY, ANOX_FRAC = .3, .01, 1e-4, .3
CH4_TAU = 12.                      # yr at O2 = O2_HALF
K_RED = 1e-4                       # /yr reductant (volcanic H2, Fe2+) O2 sink per unit volcanism
O2_HALF = .5                       # O2 pool giving o2 = .105 (calibrated on seed-4 planet2 burial-vs-reductant steady state)
CH4_K = 1.1e-2                     # biosign ch4 per CH4 pool unit (calibrated to the v1 ~1e-4 scale)
NUT0 = 10.                         # dissolved N, P inventory at start (C-equivalent) >> standing biomass, as in real oceans
KN = KP = 2.; N_FIX_TARGET, N_FIX, DENIT = NUT0, 2e-3, 1e-4; P_W, P_BUR, P_FLOOR = 2e-5, 2e-5, .5             # P residence ~50 kyr (ocean ~20-100 kyr); floor = seafloor/hydrothermal + aeolian P

class Biogeochem:
    """Carbon-silicate cycle (CarbonCycle) + organic C/N/P/O2/CH4 pools. Totals: total_C() is conserved up to cum outgassing."""
    def __init__(self, carbon):
        self.cc = carbon; self.det = 0.; self.sed = 0.; self.CH4 = 0.; self.O2 = 0.; self.bio = 0.
        self.N = NUT0; self.Pn = NUT0; self.mult0 = (NUT0/(NUT0 + KN))*(NUT0/(NUT0 + KP)); self.growth_mult = 1.; self.last = {}
        self.C0 = None
    def monod(self): return float(min(1.5, (self.N/(self.N + KN))*(self.Pn/(self.Pn + KP))/self.mult0))
    def total_C(self): return float(self.cc.total_C() + self.bio + self.det + self.sed + self.CH4)
    def step(self, B0, B1, loss, lat, P, dt):
        """B0/B1 biomass before/after this step's biosphere+creatures, loss = mortality + disturbance biomass removed (>= 0)."""
        if self.C0 is None: self.C0 = self.total_C() - self.cc.cum_V     # bio starts at 0: seeded biomass is fixed from C_inorg
        B0 = self.bio; dt = max(dt, 1e-9); loss = max(0., loss); npp = (B1 - B0) + loss
        npp = min(npp, self.cc.C_inorg*.5 + loss)          # cannot fix more carbon than half the inorganic pool
        loss = npp - (B1 - B0)                              # keep the ledger exact (bio pool = B1)
        o2s = self.O2/(self.O2 + O2_HALF)
        resp = (1 - DET_FRAC)*loss; to_det = DET_FRAC*loss
        dec = min(self.det, DECAY*self.det*dt); bur = min(self.det - dec, BURY*self.det*dt)
        anox = ANOX_FRAC*(1 - o2s)*dec; aer = dec - anox
        ox = self.CH4*(1 - np.exp(-dt*max(o2s, .05)/CH4_TAU))
        self.cc.C_inorg = self.cc.C_inorg - npp + resp + aer + .5*anox + ox   # 2 CH2O -> CH4 + CO2
        self.bio = B1; self.det += to_det - dec - bur; self.sed += bur; self.CH4 += .5*anox - ox
        sink = self.O2*(1 - np.exp(-K_RED*self.cc.volc*dt))
        self.O2 = max(0., self.O2 + npp - resp - aer - 2*ox - sink)
        # nutrients (C-equivalent units): uptake by NPP, return by respiration + decay, external sources/sinks
        wf = float(np.clip(self.cc.last.get("weathering", 0.)/(self.cc.V0*P["carbon_rate"]), P_FLOOR, 5.))   # Earth-normalised silicate weathering
        self.wf = wf
        fix = N_FIX*max(0., N_FIX_TARGET - self.N)*min(1., B1 + .1)*dt; den = DENIT*(1 - o2s)*self.N*dt
        self.N = max(1e-4, self.N - npp + resp + dec + fix - den)
        self.Pn = max(1e-4, self.Pn - npp + resp + dec + P_W*NUT0*wf*dt - P_BUR*self.Pn*dt)
        self.growth_mult = self.monod()
        from .world import P_REF
        lat.pCO2 = self.cc.pCO2(P_REF)
        self.last = dict(npp=npp/dt, resp=resp/dt, decay=dec/dt, burial=bur/dt, ch4_prod=.5*anox/dt, ch4_ox=ox/dt, o2_sink=sink/dt)
        return dict(C_bio=float(B1), C_det=float(self.det), C_sed=float(self.sed), CH4_pool=float(self.CH4), O2_pool=float(self.O2),
                    N_avail=float(self.N*REDFIELD["N"]), P_avail=float(self.Pn*REDFIELD["P"]), growth_mult=float(self.growth_mult),
                    npp_flux=float(npp/dt), burial_flux=float(bur/dt), C_total=self.total_C(), C_drift=float(self.total_C() - self.cc.cum_V - self.C0))
    def biosign(self, flux, climate):
        """O2/CH4 from pools (fluxes x lifetimes); H2O from band water/T. Mutates and returns the BioSignFlux."""
        flux.o2 = float(.21*self.O2/(self.O2 + O2_HALF)); flux.ch4 = float(CH4_K*self.CH4)
        from .biosign import h2o_column
        flux.h2o = h2o_column(climate); return flux

def _P_REF():
    from .world import P_REF
    return P_REF
