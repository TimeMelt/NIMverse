"""solvers.climate2d -- lat x lon energy-balance "mini-GCM" (out-of-leaf solver, climate_model="gcm2d").

Grid: n equal-area latitude bands (uniform in x = sin lat, same as leaves.world.LatClimate) x nlon longitudes.
Per cell k (surface layer T, ocean cells also a deep layer Td):
    C_k dT/dt = ASR_k - OLR_k + F + gamma_k (Td - T) + div(D grad T)_k
    C_d dTd/dt = gamma_k (T - Td)                       (ocean cells only)
  * Horizontal transport: diffusion on the unit sphere in (x, lon):
        d/dx[(1-x^2) D dT/dx] + D_z/(1-x^2) d2T/dlon2
    atmosphere everywhere (ATM_D share of LatClimate's D, zonally enhanced by ZONAL_K for jets/stationary eddies) plus an
    ocean share (OCEAN_D) only across ocean-ocean faces, so continents block ocean heat transport.  The discrete operator is
    symmetric with zero row sums: it conserves energy exactly (tested).  Backward Euler with a sparse LU per substep.
  * Continents: deterministic mask seeded by (planet_id, a): random spherical caps + low-order noise, thresholded to the
    land fraction 1 - ocean.  Land C = C_LAND2 (no deep layer), ocean C = C_MIX + deep C_DEEP.
  * Seasons: solved, not parameterised.  The annual cycle of insolation at each latitude (Kepler-timed, eccentric orbit,
    true orbital period P = sqrt(a^3/M)) is Fourier-decomposed (harmonics 1, 2) and the linearised system
    (i k w C + B + gamma - L) T_k = Q_k is solved in the frequency domain on the full grid: land/ocean contrast and its
    transport-coupled damping come out of the operator.  Seasonal half-range per cell feeds ice and liquid water.
  * Hydrology: evaporation from open ocean ~ e_s(T), moisture transported by a steady diffusion-decay solve
    (residence ~ 9 days, length ~ 1500 km scaled by planet radius); land moisture index m = P/P_ref gives
    desert / wet land (albedo, runoff, weathering, RH for radiation, liquid water for life).
  * Radiation: rt=False -> LatClimate's greenhouse-offset emission (sigma (T-G)^4) and albedo (+ land/desert terms);
    rt=True -> solvers.rt band-model lookup tables per cell: OLR(T, pCO2, RH, cloud) and
    alpha_p(T, pCO2, mu0, surface albedo, cloud), with insolation-weighted mu0 per latitude, O3 from O2, CH4 from bgc.
Interface mirrors LatClimate (T/Td zonal-mean properties, last["ice"/"amp"/"ins"], ocean, pCO2, tech_co2_mult,
integrate(), step(), packet()) so leaves.biogeochem.CarbonCycle/Biogeochem work unchanged; GCMCarbon weathers land
cells only.  Extra stressors: grid_* (per-cell T, water, ice, land), grid_lon_deg, climate_model="gcm2d".

Fidelity: an EBM on a 2-D grid -- no dynamics (no Hadley cell, no winds, no explicit clouds); transport is diffusive.
Good for land/ocean contrast, continentality, regional ice/water, seasonality; not for circulation regimes, tidally
locked dayside/nightside jets (n/a: insolation is diurnally averaged), or precipitation patterns beyond diffusion-decay.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..leaves import world
from ..leaves.world import (SIGMA, S_SUN, C_MIX, C_DEEP, GAMMA, T_ICE, ICE_W, H_MAX, P_REF, CLOUD_ALB, ALB_ICE, ALBEDO,
                            ALB_WARM_K, ALB_P2, DESERT_ALB, D_SIZE, band_x, greenhouse, water_frac, es_ratio, teq)
from ..leaves.packets import ClimateStress, as_bus
from ..leaves import biogeochem as bgc_mod
from . import rt

N_LON = 36
C_LAND2 = .3          # W yr m^-2 K^-1 (~3 m of soil + atmosphere column)
ATM_D, OCEAN_D = .7, .3   # shares of LatClimate D (aquaplanet total = D)
ZONAL_K = 3.          # zonal / meridional atmospheric diffusivity ratio
LAND_ALB_ADD = .06    # bare land brighter than ocean (non-RT path)
MOIST_TAU = .025      # yr water-vapour residence (~9 d)
MOIST_L = .25         # rad diffusion-decay length at Earth radius (~1600 km)
MOIST_REF = .5        # precip / ocean evaporation at which land counts as fully wet
SURF_ALB = dict(ocean=.07, land=.20, desert=.32, seaice=.55, snow=.70)   # RT surface albedos
H_MAX_EQ = 50.        # yr substep for deep-equilibrium spin-up


def continents(planet_id, a, n, nlon, land_frac):
    """Boolean land mask (n x nlon); deterministic in (planet_id, a)."""
    land_frac = float(np.clip(land_frac, 0, 1)); N = n*nlon
    if land_frac*N < .5: return np.zeros((n, nlon), bool)
    if land_frac*N > N - .5: return np.ones((n, nlon), bool)
    r = np.random.default_rng([int(planet_id) % 2**31, int(abs(a)*1e6) % 2**31, 2207])
    x = band_x(n); lon = (np.arange(nlon) + .5)*2*np.pi/nlon - np.pi
    X, LON = np.meshgrid(x, lon, indexing="ij"); cl = np.sqrt(1 - X**2)
    v = np.stack([cl*np.cos(LON), cl*np.sin(LON), X], -1)
    f = np.zeros((n, nlon))
    for _ in range(4 + int(r.integers(0, 5))):
        z = r.normal(size=3); z /= np.linalg.norm(z)
        ang = np.arccos(np.clip(v @ z, -1, 1)); f += r.uniform(.5, 1.5)*np.exp(-(ang/r.uniform(.3, .9))**2)
    for _ in range(6):
        z = r.normal(size=3); z /= np.linalg.norm(z); f += .15*np.cos(r.integers(2, 5)*np.arccos(np.clip(v @ z, -1, 1)) + r.uniform(0, 6.3))
    return f >= np.quantile(f, 1 - land_frac)


def seasonal_insolation(x, obliquity, e=0., varpi=0., nt=48):
    """Daily-mean insolation / S0 on a uniform-in-time grid over one orbit (Kepler).  Returns
    mean (n,), harmonics (n, 2) complex, mu0 (insolation-weighted daily cos zenith, n), q(n, nt)."""
    phi = np.arcsin(np.clip(np.asarray(x, float), -1, 1)); e = float(min(max(e, 0.), .95))
    M = (np.arange(nt) + .5)*2*np.pi/nt; E = M.copy()
    for _ in range(30): E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    nu = 2*np.arctan2(np.sqrt(1 + e)*np.sin(E/2), np.sqrt(1 - e)*np.cos(E/2))
    lam = nu + varpi; dec = np.arcsin(np.sin(abs(obliquity))*np.sin(lam))
    inv_r2 = ((1 + e*np.cos(nu))/(1 - e*e))**2
    P_, D_ = np.meshgrid(phi, dec, indexing="ij")
    h0 = np.arccos(np.clip(-np.tan(P_)*np.tan(D_), -1, 1)); A = np.sin(P_)*np.sin(D_); B = np.cos(P_)*np.cos(D_)
    m1 = A*h0 + B*np.sin(h0); m2 = A*A*h0 + 2*A*B*np.sin(h0) + B*B*(h0/2 + np.sin(2*h0)/4)
    q = m1/np.pi*inv_r2[None, :]
    mu = np.where(m1.sum(1) > 1e-12, (m2*inv_r2).sum(1)/np.maximum((m1*inv_r2).sum(1), 1e-12), .2)
    F = np.fft.rfft(q, axis=1)/nt
    return dict(mean=q.mean(1), harm=np.stack([2*F[:, 1], 2*F[:, 2]], 1), mu0=np.clip(mu, .1, 1.), q=q,
                amp=.5*(q.max(1) - q.min(1)))


class GCM2D:
    """lat x lon EBM; drop-in for leaves.world.LatClimate (see module doc)."""
    def __init__(self, h, LP=None, hw=None, n=24, nlon=N_LON, orbit=None, rt_on=False, spinup=True):
        P = as_bus(LP); self.n, self.nlon = n, nlon; self.x = band_x(n); self.lat = np.degrees(np.arcsin(self.x)); self.orbit = orbit
        self.lon = np.degrees((np.arange(nlon) + .5)*2*np.pi/nlon - np.pi); self.rt = bool(rt_on); self.N = n*nlon
        oc = float(np.clip(h.volatiles, 0, 1)) if h.ptype in ("ocean", "terran", "ice") else float(np.clip(.5*h.volatiles, 0, 1))
        self.landm = continents(getattr(h, "planet_id", 0), h.a, n, nlon, 1 - oc); self.oc_m = ~self.landm
        self.ocean = float(self.oc_m.mean())
        lf = self.landm.ravel()
        self.C = np.where(lf, C_LAND2, C_MIX).astype(float); self.gam = np.where(lf, 0., GAMMA); self.Cd = C_DEEP
        Rrel = float(np.clip((h.m/3e-6)**(1./3.), .3, 4.)); self.Rrel = Rrel
        self.D = float(np.clip(P["lat_diffusion"]*Rrel**D_SIZE/max(h.spin, .3)**.5, .05, 5.))
        self.Lop = self._laplacian()
        A = float(np.clip(ALBEDO[h.ptype]*P["albedo_mult"], 0, .95))
        aw = A if h.ptype in ("ice", "lava", "gas giant", "ice giant") else ALB_WARM_K*A
        if h.ptype == "desert": aw = float(np.clip(aw + DESERT_ALB, .02, .9))
        self.alb_band = np.clip(aw + ALB_P2*(1.5*self.x**2 - .5), .02, .9)
        self.alb_ice_band = np.maximum(self.alb_band, ALB_ICE)
        self.g = float(round(9.81*np.clip((h.m/3e-6)**.46, .2, 6.), 1)); self.ps = 1.
        self.x_o2 = 0.; self.x_ch4 = 1e-7
        self.pCO2 = P_REF; self.t = 0.; self.Tg = None; self.Tdg = None; self.G = 0.; self.last = {}; self.tech_co2_mult = 1.; self.limit_cycle = None
        self.moist = np.where(lf, .5, 1.); self.orb = self._orbit(h, 0.)
        M_star = float(getattr(hw, "M_star", getattr(hw, "M", 1.)) or 1.) if hw is not None else 1.
        self.P_orb = float(np.sqrt(max(h.a, 1e-3)**3/max(M_star, .05)))
        if spinup:
            T0 = teq(self._L(h, hw), h.a, float(self.alb_band.mean())) + P["greenhouse_K"]*min(1., h.volatiles/.5)
            self.T = np.full(n, T0); self.Td = self.T.copy()
            for _ in range(3): self.integrate(h, hw, P, 400., 0., deep_eq=True)

    # ---- zonal-mean views (LatClimate compatibility) ----
    @property
    def T(self): return None if self.Tg is None else self.Tg.mean(1)
    @T.setter
    def T(self, v):
        v = np.asarray(v, float); self.Tg = np.repeat(v[:, None], self.nlon, 1) if v.ndim == 1 else v.copy()
    @property
    def Td(self): return None if self.Tdg is None else self.Tdg.mean(1)
    @Td.setter
    def Td(self, v):
        v = np.asarray(v, float); self.Tdg = np.repeat(v[:, None], self.nlon, 1) if v.ndim == 1 else v.copy()

    def _laplacian(self):
        n, m = self.n, self.nlon; dx = 2./n; dl = 2*np.pi/m; idx = np.arange(self.N).reshape(n, m); oc = self.oc_m
        rows, cols, vals = [], [], []
        def face(a, b, c):
            rows.extend([a, b]); cols.extend([b, a]); vals.extend([c, c])
        xe = np.linspace(-1, 1, n + 1)
        for i in range(n):
            for j in range(m):
                k = idx[i, j]
                if i < n - 1:
                    w = (1 - xe[i + 1]**2)/dx**2
                    face(k, idx[i + 1, j], w*self.D*(ATM_D + OCEAN_D*(oc[i, j] and oc[i + 1, j])))
                jn = (j + 1) % m
                w = 1/(1 - self.x[i]**2)/dl**2
                face(k, idx[i, jn], w*self.D*(ATM_D*ZONAL_K + OCEAN_D*(oc[i, j] and oc[i, jn])))
        L = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        return (L - sp.diags(np.asarray(L.sum(1)).ravel())).tocsc()

    def _orbit(self, h, t):
        if self.orbit is not None: return self.orbit.at(t)
        return dict(e=float(h.e), varpi=0., obliquity=float(abs(h.tilt)))
    def _L(self, h, hw):
        return float(getattr(hw, "L_now", getattr(hw, "L_star", h.L_star))) if hw is not None else float(h.L_star)
    def S0(self, h, hw): return S_SUN*self._L(h, hw)/max(h.a, 1e-6)**2
    def _teff(self, hw): return float(getattr(hw, "T_now", getattr(hw, "T_eff", 5772.)) or 5772.) if hw is not None else 5772.

    def set_trace(self, x_o2=None, x_ch4=None):
        """Trace gases for the RT path (quantised so the cached tables are reused)."""
        if x_o2 is not None: self.x_o2 = float(x_o2)
        if x_ch4 is not None: self.x_ch4 = float(x_ch4)
    def _tables(self, hw):
        du = rt.o3_du_from_o2(self.x_o2); du = 0. if du < 3 else float(round(10**(round(np.log10(du)*4)/4), 1))
        ch4 = float(10**(round(np.log10(max(self.x_ch4, 1e-7))*2)/2))
        te = float(round(self._teff(hw), -2))
        return rt.climate_tables(self.g, te, ch4, du, self.ps)

    # ---- physics pieces (cell arrays, flattened) ----
    def ice_of(self, T, amp): return 1/(1 + np.exp((T + .3*amp - T_ICE)/ICE_W))
    def ice(self, T, amp): return self.ice_of(T, amp)
    def cloud_of(self, T, ice):
        if self.ocean < 1e-3 and self.moist.mean() < 1e-3: return np.zeros_like(T)
        wet = (np.where(self.landm.ravel(), .6*self.moist.ravel(), 1.))*(1 - .85*ice)
        wet = wet*min(1., .3 + self.ocean)
        cold = np.clip((295. - T)/45., 0., 1.5)
        f = wet*(world.CLOUD_WET*.4 + world.CLOUD_COLD*cold/(1. + .25*es_ratio(float(np.mean(T)))))
        return np.clip(f, 0., .85)
    def _moisture(self, T, ice, h):
        """Steady diffusion-decay moisture transport -> land moisture index (n, nlon)."""
        E = np.where(self.oc_m.ravel(), (1 - ice)*np.minimum(np.exp(5420.*(1/288. - 1/np.maximum(T, 150.))), 8.), 0.)
        if E.sum() <= 0: return np.where(self.landm, float(np.clip(h.volatiles, 0, 1))*.3, 0.)
        Dq = (MOIST_L*self.Rrel)**2/MOIST_TAU
        Lq = self._lap_unit
        q = spla.spsolve((sp.identity(self.N, format="csc")/MOIST_TAU - Dq*Lq).tocsc(), E)
        Pr = q/MOIST_TAU; ref = MOIST_REF*max(float(E[self.oc_m.ravel()].mean()), 1e-9)
        return np.clip(Pr/ref, 0, 1.).reshape(self.n, self.nlon)
    @property
    def _lap_unit(self):
        if getattr(self, "_lu", None) is None:
            n, m = self.n, self.nlon; dx = 2./n; dl = 2*np.pi/m; idx = np.arange(self.N).reshape(n, m); xe = np.linspace(-1, 1, n + 1)
            r, c, v = [], [], []
            for i in range(n):
                for j in range(m):
                    k = idx[i, j]
                    if i < n - 1: w = (1 - xe[i + 1]**2)/dx**2; r += [k, idx[i + 1, j]]; c += [idx[i + 1, j], k]; v += [w, w]
                    jn = (j + 1) % m; w = 1/(1 - self.x[i]**2)/dl**2; r += [k, idx[i, jn]]; c += [idx[i, jn], k]; v += [w, w]
            L = sp.csr_matrix((v, (r, c)), shape=(self.N, self.N)); self._lu = (L - sp.diags(np.asarray(L.sum(1)).ravel())).tocsc()
        return self._lu

    def _albedo_surface(self, ice):
        """Non-RT cell albedo (before clouds) and RT surface albedo."""
        lf = self.landm.ravel(); m = self.moist.ravel()
        ab = np.repeat(self.alb_band, self.nlon); ai = np.repeat(self.alb_ice_band, self.nlon)
        ab = np.where(lf, np.clip(ab + LAND_ALB_ADD + DESERT_ALB*(1 - m), .02, .9), ab)
        a_nr = ab + (ai - ab)*ice
        s = SURF_ALB; land = s["land"] + (s["desert"] - s["land"])*(1 - m)
        a_rt = np.where(lf, land + (s["snow"] - land)*ice, s["ocean"] + (s["seaice"] - s["ocean"])*ice)
        return a_nr, a_rt

    def _radiation(self, h, hw, P, T, ice, cf, ins, S0):
        """ASR, OLR, dOLR/dT per cell; plus diagnostics."""
        Q = S0*np.repeat(ins["mean"], self.nlon)
        a_nr, a_rt = self._albedo_surface(ice)
        pc = self.pCO2*self.tech_co2_mult
        if not self.rt:
            self.G, gco2, gwv, gcl = greenhouse(h, P, pc, float(T.mean()), self.ocean, float(cf.mean()))
            alb = a_nr + CLOUD_ALB*cf*(1 - a_nr)
            Te = np.maximum(T - self.G, 20.); return Q*(1 - alb), SIGMA*Te**4, 4*SIGMA*Te**3, dict(g_co2=gco2, g_wv=gwv, g_cloud=gcl, alb=alb)
        tab = self._tables(hw)
        fv = min(1., h.volatiles/.5)
        rh = np.where(self.landm.ravel(), .35 + .42*self.moist.ravel(), .77)*fv
        rh = np.clip(rh, RH_MIN, .77)
        pce = P_REF*(max(pc, 1e-9)/P_REF)**(P["co2_sens_K"]/4.)       # co2_sens_K knob rescales the band-model CO2 sensitivity
        cf = np.clip(RT_CLOUD_K*cf, 0., .9)                            # band-model cloud = overcast deck (tau 10); cover calibrated to Earth CRE
        olr, dolr = rt.olr_lookup(tab, T, pce, rh, cf)
        dG = P["greenhouse_K"] - 33.                                   # greenhouse_K knob: shift of the OLR curve (0 at default)
        if dG: olr = olr - dolr*dG*fv
        mu = np.repeat(ins["mu0"], self.nlon)
        ap, sa, aa = rt.sw_lookup(tab, T, pce, mu, a_rt, cf)
        Tg = max(float(np.mean(T)), 150.); self.G = float(Tg - (max(float(np.mean(olr)), 1.)/SIGMA)**.25)
        return Q*(1 - ap), olr, dolr, dict(g_co2=0., g_wv=0., g_cloud=0., alb=ap, surf_abs=float((Q*sa).mean()), atm_abs=float((Q*aa).mean()))

    def _season(self, B, alb, S0, insr):
        """Frequency-domain seasonal response: half-range per cell (flattened)."""
        om = 2*np.pi/max(self.P_orb, 1e-3); Tser = 0.
        nt = 24; ph = (np.arange(nt) + .5)*2*np.pi/nt; out = np.zeros((self.N, nt))
        for kk in (1, 2):
            Qk = S0*np.repeat(insr["harm"][:, kk - 1], self.nlon)*(1 - alb)
            Mk = (sp.diags(1j*kk*om*self.C + B + self.gam*(1 - 1/(1 + 1j*kk*om*self.Cd/np.maximum(self.gam, 1e-9)))) - self.Lop).tocsc()
            Tk = spla.spsolve(Mk, Qk.astype(complex))
            out += np.real(Tk[:, None]*np.exp(1j*kk*ph[None, :]))
        return .5*(out.max(1) - out.min(1))

    def integrate(self, h, hw, P, dt, F=0., deep_eq=False):
        S0 = self.S0(h, hw); o = self.orb
        insr = seasonal_insolation(self.x, o["obliquity"], o["e"], o["varpi"])
        hm = H_MAX_EQ if deep_eq else H_MAX
        k = max(1, int(np.ceil(dt/hm))); hh = dt/k
        T = self.Tg.ravel().copy(); Td = self.Tdg.ravel().copy()
        amp = self.last.get("amp_g"); amp = np.zeros(self.N) if amp is None or deep_eq and False else amp
        if amp is None or len(amp) != self.N: amp = np.zeros(self.N)
        ocf = self.oc_m.ravel()
        for it in range(k):
            ice = self.ice_of(T, amp); cf = self.cloud_of(T, ice)
            asr, olr, B, dg = self._radiation(h, hw, P, T, ice, cf, insr, S0)
            if it == 0 or deep_eq:
                amp = self._season(B, dg["alb"], S0, insr); ice = self.ice_of(T, amp)
                self.moist = np.where(self.landm, self._moisture(T, ice, h), 1.)
            if deep_eq: Td = T.copy()
            M = (sp.diags(self.C/hh + B + self.gam) - self.Lop).tocsc()
            rhs = self.C/hh*T + asr - olr + B*T + F + self.gam*Td
            T = spla.spsolve(M, rhs)
            if not deep_eq: Td = np.where(ocf, Td + hh*self.gam*(T - Td)/self.Cd, T)
        if deep_eq: Td = T.copy()
        self.Tg = T.reshape(self.n, self.nlon); self.Tdg = Td.reshape(self.n, self.nlon)
        ice = self.ice_of(T, amp); cf = self.cloud_of(T, ice)
        asr, olr, B, dg = self._radiation(h, hw, P, T, ice, cf, insr, S0)
        Z = lambda a: a.reshape(self.n, self.nlon).mean(1)
        self.last = dict(ins=dict(mean=insr["mean"], amp=insr["amp"], mu0=insr["mu0"]), amp=Z(amp), ice=Z(ice), cloud=Z(cf), S0=S0,
                         asr=float(asr.mean()), olr=float(olr.mean()), toa_imbalance=float((asr - olr).mean()),
                         g_co2=dg["g_co2"], g_wv=dg["g_wv"], g_cloud=dg["g_cloud"], cloud_mean=float(cf.mean()),
                         amp_g=amp, ice_g=ice, cloud_g=cf, B_g=B, alb_g=dg["alb"],
                         surf_abs=dg.get("surf_abs"), atm_abs=dg.get("atm_abs"))
        return self.T

    def step(self, h, hw, LP=None, dt_w_yr=1., stress=None, t=None):
        P = as_bus(LP); st0 = dict(stress or {}); t = self.t + dt_w_yr if t is None else float(t); self.t = t
        self.orb = self._orbit(h, t)
        dT_imp = float(st0.get("impact_dT", 0.)); dT_force = float(st0.get("dT", 0.)) - dT_imp
        B_mean = float(np.mean(self.last.get("B_g", 4*SIGMA*np.maximum(self.Tg.ravel() - self.G, 20.)**3)))
        self.integrate(h, hw, P, dt_w_yr, B_mean*dT_force)
        return self.packet(h, hw, P, st0, dT_imp)

    def bands(self, dT=0.):
        return self.T + dT, self.last["amp"]

    def packet(self, h, hw, P, st0, dT_imp=0.):
        """LatClimate.packet on zonal means, then grid fields and land-aware water/productivity."""
        cs = world.LatClimate.packet(self, h, hw, P, dict(st0), dT_imp)
        st = cs.stressors; vs, sb, rk = h.volatiles, P["snowball_K"], P["runaway_K"]
        Tg = self.Tg.ravel() + dT_imp; amp = self.last["amp_g"]
        wf = np.vectorize(lambda T: water_frac(T, vs, sb, rk))
        wcell = .5*(wf(Tg - amp) + wf(Tg + amp))
        wcell = np.where(self.landm.ravel(), wcell*self.moist.ravel(), wcell)
        ice = self.ice_of(Tg, amp)
        insc = np.repeat(self.last["ins"]["mean"], self.nlon); light = np.sqrt(np.clip(insc/max(insc.mean(), 1e-9), 0, 3))
        prod = wcell*light*np.exp(-((Tg - 295.)/30.)**2); pm = float(prod.mean())
        R = lambda a, d: [[round(float(v), d) for v in row] for row in np.asarray(a).reshape(self.n, self.nlon)]
        wz = wcell.reshape(self.n, self.nlon).mean(1); pz = prod.reshape(self.n, self.nlon).mean(1)
        st.update(lat_water=[round(float(v), 4) for v in wz], lat_prod=[round(float(v/pm), 4) if pm > 0 else 0. for v in pz],
                  grid_lon_deg=[round(float(v), 2) for v in self.lon], grid_T=R(Tg, 2), grid_water=R(wcell, 3), grid_ice=R(ice, 3),
                  grid_prod=R(prod/pm if pm > 0 else prod*0, 3), grid_land=[[int(v) for v in row] for row in self.landm],
                  land_frac=float(self.landm.mean()), land_moisture=float(self.moist[self.landm].mean()) if self.landm.any() else 0.,
                  T_land=float(Tg[self.landm.ravel()].mean()) if self.landm.any() else None,
                  T_ocean=float(Tg[self.oc_m.ravel()].mean()) if self.oc_m.any() else None,
                  season_land_K=float(amp[self.landm.ravel()].mean()) if self.landm.any() else 0.,
                  toa_imbalance=float(self.last.get("toa_imbalance", 0.)), climate_model="gcm2d", rt=bool(self.rt),
                  co2_limit_cycle=self.limit_cycle)
        if self.rt: st.update(rt_surf_abs=self.last.get("surf_abs"), rt_atm_abs=self.last.get("atm_abs"), planetary_albedo=float(np.mean(self.last["alb_g"])))
        if dT_imp:
            Tc = self.Tg.ravel(); wc = .5*(wf(Tc - amp) + wf(Tc + amp)); wc = np.where(self.landm.ravel(), wc*self.moist.ravel(), wc)
            st["water_chronic"] = float(wc.mean()); st["lat_water_chronic"] = [round(float(v), 4) for v in wc.reshape(self.n, self.nlon).mean(1)]
        return ClimateStress(cs.T_K, float(wcell.mean()), cs.climate_class, st)

RH_MIN = .05
RT_CLOUD_K = 2.0


class GCMCarbon(bgc_mod.CarbonCycle):
    """CarbonCycle with silicate weathering on land cells (runoff from the moisture solve) and seafloor under ocean cells."""
    def weather_factor(self, lat, pCO2, P_REF_):
        if not isinstance(lat, GCM2D): return super().weather_factor(lat, pCO2, P_REF_)
        T = lat.Tg.ravel(); ice = lat.last.get("ice_g", np.zeros(lat.N)); lf = lat.landm.ravel()
        run = self.wet*np.clip(1 + .038*(T - 288.), 0, 4)*(T > 271.)*np.clip(lat.moist.ravel()/.6, 0, 1.5)
        Ef = float(np.mean(((1 - ice)*np.exp((T - 288.)/bgc_mod.T_W)*run)[lf])) if lf.any() else 0.
        o = ~lf
        if o.any():
            sea_warm = np.mean(((1 - ice)*np.exp((np.clip(T, 271, 320) - 288.)/40.))[o])
            sea_ice = np.mean((ice*np.exp((np.clip(T, 200, 280) - 260.)/50.))[o])
        else: sea_warm = sea_ice = 0.
        sea = sea_warm + bgc_mod.SEAFLOOR_ICE/max(bgc_mod.SEAFLOOR, 1e-9)*sea_ice
        return float(((self.land*Ef + bgc_mod.SEAFLOOR*self.ocean*sea)/(bgc_mod.LAND_REF*bgc_mod.W_NORM))*(max(pCO2, 1e-9)/P_REF_)**bgc_mod.BETA)


_EQ_CACHE = {}

def init_gcm2d(h, hw, LP=None, orbit=None, rt_on=False, nlon=N_LON, warm_branch=True):
    """(GCM2D, GCMCarbon) at the carbon-silicate equilibrium (cached)."""
    P = as_bus(LP); g = GCM2D(h, P, hw, nlon=nlon, orbit=orbit, rt_on=rt_on, spinup=False); cc = GCMCarbon(h, g, P)
    key = (round(h.a, 9), round(h.e, 9), round(h.tilt, 9), round(h.spin, 9), h.ptype, round(h.volatiles, 9), round(h.m, 12), getattr(h, "planet_id", 0),
           round(float(getattr(hw, "L_star", h.L_star)), 9), tuple(round(P[k], 9) for k in ("greenhouse_K", "albedo_mult", "lat_diffusion", "co2_sens_K", "snowball_K", "runaway_K")),
           None if orbit is None else orbit.key(), bool(rt_on), nlon, bool(warm_branch), "g2")
    if key in _EQ_CACHE:
        cc.C_atm, T, Td, mo, g.limit_cycle = _EQ_CACHE[key]; g.Tg, g.Tdg, g.moist = T.copy(), Td.copy(), mo.copy()
        g.pCO2 = cc.pCO2(P_REF); g.integrate(h, hw, P, 1., 0.)
    else:
        cc.equilibrate(h, hw, g, P); g.integrate(h, hw, P, 1., 0.)
        if warm_branch and float(np.mean(g.last["ice"])) > SNOW_ICE: _warm_branch(h, hw, P, g, cc)
        g.integrate(h, hw, P, 1., 0.)
        _EQ_CACHE[key] = (cc.C_atm, g.Tg.copy(), g.Tdg.copy(), g.moist.copy(), g.limit_cycle)
    return g, cc

SNOW_ICE = .9

def _settle(h, hw, P, g, passes=12, tol=.05):
    """deep_eq passes until the mean T moves < tol K per pass (a warm start can need many passes to fall to the snowball:
    3 passes were not enough -- a 'warm' state at 3 bar on seed-4 planet2 under RT was transient).  False if a warm state
    is still cooling at the end, or the run lands on the snowball."""
    T0 = float(g.T.mean())
    for _ in range(passes):
        g.integrate(h, hw, P, 400., 0., deep_eq=True); T1 = float(g.T.mean())
        if abs(T1 - T0) < tol: return True
        T0 = T1
    return False

def _warm_branch(h, hw, P, g, cc):
    """V = W fell on the snowball side of a bistable climate: low-insolation planets then run a CO2-snowball limit cycle
    (Menou 2015; Haqq-Misra et al. 2016): CO2 builds under ice until deglaciation, then weathering (W > V) draws it down
    over ~TAU_INORG/(W/V).  Start at the warm phase: the lowest pCO2 (<= 30 bar) with a warm (ice < SNOW_ICE) equilibrium
    from a warm start.  g.limit_cycle records it; the carbon cycle then evolves freely."""
    p0 = g.pCO2
    for pc in np.geomspace(max(p0, 1e-5), 30., 14)[1:]:
        g.T = np.full(g.n, 300.); g.Td = g.T.copy(); g.pCO2 = float(pc)
        if _settle(h, hw, P, g) and float(np.mean(g.last["ice"])) < SNOW_ICE:
            cc.C_atm = float(pc)/P_REF; g.limit_cycle = dict(snowball_pCO2_bar=float(p0), warm_pCO2_bar=float(pc)); return True
    g.T = np.full(g.n, 300.); g.Td = g.T.copy(); g.pCO2 = p0
    for _ in range(3): g.integrate(h, hw, P, 400., 0., deep_eq=True)
    return False
