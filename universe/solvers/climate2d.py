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

  * RT-path thermodynamics (solvers.rt PHYS flags): H2-H2/N2-N2/CO2-CO2 CIA in the tables, CO2 collapse onto the cold
    trap (p_atm <= p_sat,CO2(T_coldest winter band); excess kept as surface CO2 ice, stressor co2_ice_bar), CO2-ice cloud
    opacity above the CO2 condensation level, Simpson-Nakajima runaway (OLR capped at rt.runaway_limit(g); stressor
    runaway when ASR exceeds it with an ocean), and heat transport scaled with surface pressure (Williams & Kasting 1997
    D ~ p cp/m^2, clipped to [1, D_P_MAX]).
  * Optional physics (physics=... / --gcm-physics clouds,diurnal,winds|all; default none = bit-identical to before):
      clouds  -- prognostic condensate path W (source from the precipitation field, ~4 d lifetime) -> cover and tau.
      diurnal -- first daily harmonic of insolation on a land/ocean diurnal skin (day = 1/spin); its amplitude adds to the
                 seasonal range for ice / liquid water and makes OLR nonlinear (<T^4> correction).
      winds   -- Held-Hou Hadley edge (Earth-relative transport boost inside phi_H) + friction-limited thermal wind U(lat)
                 with conservative upwind zonal advection of heat (boundary-layer share) and moisture.
      dynamics -- baroclinic-eddy closure: meridional atmospheric diffusivity per face D_k = D (|dT/dy|_k / |dT/dy|_Earth)^DYN_N
                 (Stone 1972 / Held 1999 diffusive closure, DYN_N = 1, clipped [DYN_MIN, DYN_MAX]) recomputed from the zonal-mean
                 T every integrate call; flux form (symmetric, zero row sums) so heat is conserved exactly (tested).  The eddy
                 flux therefore responds to the planet's own temperature gradient instead of a fixed Earth-calibrated D.
      frost   -- seasonal CO2 frost cycle (RT path): condensing caps are pinned at T_sat by latent heat (L_s dM_sat/dT >> C_seas),
                 so the atmosphere is limited by the annual-mean cold trap, not the winter minimum; the seasonal frost-cap mass
                 exchange (kg m^-2 and the resulting p_atm seasonal range) is diagnosed from the winter energy deficit / L_s.
  * Background-gas buffer (buffer=None|"n2"|"h2"|"auto", CLI --rt-buffer; RT path only): "n2" N2_BUFFER_BAR of N2 (pressure
    broadening, N2-N2 CIA, transport ~ p); "h2" + X_H2_BUFFER of H2 (N2-H2 CIA, Wordsworth & Pierrehumbert 2013); "auto"
    starts from the leaf 1-bar N2 column and adds n2, then n2+h2 only while the equilibrium is a snowball (last["buffer"]).

Fidelity: still an EBM on a 2-D grid, not a primitive-equation GCM: no prognostic momentum / eddies / stationary waves,
meridional transport stays diffusive (Hadley only rescales it), clouds are one condensate path with no vertical
structure, no tidally-locked day/night (insolation is diurnally averaged; the diurnal harmonic is a perturbation).
Cold CO2-rich planets: the collapse caps p_atm, so planets the leaf keeps warm by an extrapolated sigma (T-G)^4
greenhouse (e.g. seed 4 planet3, leaf ~246 K) sit near 180 K here -- the residual is the physics (the
maximum-greenhouse limit), not a bug.
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
    def __init__(self, h, LP=None, hw=None, n=24, nlon=N_LON, orbit=None, rt_on=False, spinup=True, physics=(), buffer=None):
        P = as_bus(LP); self.n, self.nlon = n, nlon; self.x = band_x(n); self.lat = np.degrees(np.arcsin(self.x)); self.orbit = orbit
        self.lon = np.degrees((np.arange(nlon) + .5)*2*np.pi/nlon - np.pi); self.rt = bool(rt_on); self.N = n*nlon
        oc = float(np.clip(h.volatiles, 0, 1)) if h.ptype in ("ocean", "terran", "ice") else float(np.clip(.5*h.volatiles, 0, 1))
        self.landm = continents(getattr(h, "planet_id", 0), h.a, n, nlon, 1 - oc); self.oc_m = ~self.landm
        self.ocean = float(self.oc_m.mean())
        lf = self.landm.ravel()
        self.C = np.where(lf, C_LAND2, C_MIX).astype(float); self.gam = np.where(lf, 0., GAMMA); self.Cd = C_DEEP
        Rrel = float(np.clip((h.m/3e-6)**(1./3.), .3, 4.)); self.Rrel = Rrel
        self.D = float(np.clip(P["lat_diffusion"]*Rrel**D_SIZE/max(h.spin, .3)**.5, .05, 5.))
        self.p_atm = None; self.co2_ice = 0.; self.f_D = 1.
        self.physics = parse_physics(physics); self.spin = float(max(h.spin, 1e-3)); self.clw = None; self.U = np.zeros(n); self.dyn = None
        self.hm = self._hadley() if "winds" in self.physics else None
        self.Lop = self._Ldiff = self._laplacian(); self._Adv = None
        A = float(np.clip(ALBEDO[h.ptype]*P["albedo_mult"], 0, .95))
        aw = A if h.ptype in ("ice", "lava", "gas giant", "ice giant") else ALB_WARM_K*A
        if h.ptype == "desert": aw = float(np.clip(aw + DESERT_ALB, .02, .9))
        self.alb_band = np.clip(aw + ALB_P2*(1.5*self.x**2 - .5), .02, .9)
        self.alb_ice_band = np.maximum(self.alb_band, ALB_ICE)
        self.g = float(round(9.81*np.clip((h.m/3e-6)**.46, .2, 6.), 1)); self.ps = 1.; self.x_h2 = 0.; self.buffer = None
        if buffer: self.set_buffer(buffer)
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

    def _laplacian(self, wa=ATM_D, wo=OCEAN_D):
        n, m = self.n, self.nlon; hm = np.ones(n - 1) if getattr(self, "hm", None) is None else self.hm; dx = 2./n
        if getattr(self, "dyn", None) is not None: hm = hm*self.dyn
        dl = 2*np.pi/m; idx = np.arange(self.N).reshape(n, m); oc = self.oc_m
        rows, cols, vals = [], [], []
        def face(a, b, c):
            rows.extend([a, b]); cols.extend([b, a]); vals.extend([c, c])
        xe = np.linspace(-1, 1, n + 1)
        for i in range(n):
            for j in range(m):
                k = idx[i, j]
                if i < n - 1:
                    w = (1 - xe[i + 1]**2)/dx**2
                    face(k, idx[i + 1, j], w*self.D*(wa*hm[i] + wo*(oc[i, j] and oc[i + 1, j])))
                jn = (j + 1) % m
                w = 1/(1 - self.x[i]**2)/dl**2
                face(k, idx[i, jn], w*self.D*(wa*ZONAL_K + wo*(oc[i, j] and oc[i, jn])))
        L = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        return (L - sp.diags(np.asarray(L.sum(1)).ravel())).tocsc()

    def _transport(self, p_atm):
        """RT path: atmospheric heat transport scales with column heat capacity / mass (Williams & Kasting 1997:
        D ~ p cp / m^2): f_D = p_tot cp/cp_air (m_air/m)^2, clipped to [1, D_P_MAX].  Rebuilds Lop when f moves > 3%."""
        x = p_atm/(self.ps + p_atm); m = x*44. + (1 - x)*28.97; cp = (x*rt.CP_CO2_MOL + (1 - x)*29100.)/m
        f = float(np.clip((self.ps + p_atm)*cp/1004.*(28.97/m)**2, 1., D_P_MAX))
        if abs(f/self.f_D - 1) > .03:
            if getattr(self, "_La", None) is None: self._La, self._Lo = self._laplacian(1., 0.), self._laplacian(0., 1.)
            self.f_D = f; self._Ldiff = (ATM_D*f*self._La + OCEAN_D*self._Lo).tocsc(); self._set_L()
        return self.f_D
    def set_buffer(self, mode):
        """Background gas (RT path): None/"n2"/"h2" (see module doc); "auto" is resolved by init_gcm2d."""
        self.buffer = None if mode in (None, "none") else str(mode); self.ps, self.x_h2 = parse_buffer(self.buffer)
        if getattr(self, "_La", None) is not None: self.f_D = 1.; self._transport(self.p_atm or 0.)
    def _dynamics(self, T):
        """'dynamics': baroclinic eddy diffusivity factor per meridional face from |dT/dy| (see module doc).  Rebuilds the
        flux-form operator when any face moves > 5 %."""
        Tz = T.reshape(self.n, self.nlon).mean(1); xe = np.linspace(-1, 1, self.n + 1)[1:-1]; a = 6.371e6*self.Rrel
        dphi = np.diff(np.arcsin(self.x)); grad = np.abs(np.diff(Tz))/np.maximum(dphi*a, 1.)
        f = np.clip((grad/DYN_GRAD_EARTH)**DYN_N, DYN_MIN, DYN_MAX)
        if self.dyn is None or np.max(np.abs(f/self.dyn - 1)) > .05:
            self.dyn = f; self._La = None; self._Ldiff = self._laplacian()
            if self.rt and self.f_D != 1.: fD = self.f_D; self.f_D = 1.; self._transport_set(fD)
            self._set_L()
        return self.dyn
    def _transport_set(self, f):
        self._La, self._Lo = self._laplacian(1., 0.), self._laplacian(0., 1.); self.f_D = f
        self._Ldiff = (ATM_D*f*self._La + OCEAN_D*self._Lo).tocsc()
    def _set_L(self):
        self.Lop = self._Ldiff if self._Adv is None else (self._Ldiff + self._Adv).tocsc()

    # ---- optional dynamics / sub-grid physics (GCM2D(physics=...), all off by default) ----
    def _hadley(self):
        """'winds': Hadley-cell edge from Held & Hou (1980), phi_H = sqrt(5/3 g H dh / (Omega a)^2) (H 10 km, dh 1/3):
        ~29 deg on Earth, global on slow rotators.  Meridional atmospheric faces inside phi_H get HADLEY_K x more transport
        (the thermally direct overturning) relative to Earth's cell: the leaf diffusivity is Earth-calibrated, so an
        Earth-sized Earth-rotator is unchanged; slow rotators / small planets (wide cells) carry heat poleward far more
        efficiently, fast rotators less (Kaspi & Showman 2015 trend)."""
        Om = 7.292e-5*self.spin; a = 6.371e6*self.Rrel; gg = 9.81*np.clip((self.Rrel**3*1.)**.46, .2, 6.)
        phiH = float(np.clip(np.sqrt(5./3.*gg*1e4/3./(Om*a)**2), .05, np.pi/2)); self.phi_H = phiH
        xe = np.linspace(-1, 1, self.n + 1)[1:-1]; ph = np.abs(np.arcsin(xe))
        w = lambda pH: .5*(1 - np.tanh((ph - pH)/.12))
        return np.clip(1 + HADLEY_K*(w(phiH) - w(PHI_H_EARTH)), .3, None)     # exactly 1 for Earth's rotation and size
    def _winds(self, T):
        """'winds': friction-limited thermal wind U(lat) = -(g H/T0) dT/dy f/(f^2 + r^2) (r = 1/RAYLEIGH_D days), from the
        zonal-mean T gradient; geostrophic flow does not advect T across isotherms, so U only advects zonally (land/sea
        contrast, downwind moisture).  Upwind flux form, periodic in longitude: rows and columns sum to zero (conservative).
        Heat uses the boundary-layer share WIND_EFF of the column; moisture uses WIND_Q."""
        Tz = T.reshape(self.n, self.nlon).mean(1); phi = np.arcsin(self.x); a = 6.371e6*self.Rrel
        dTdy = np.gradient(Tz, phi)/a; f = 2*7.292e-5*self.spin*np.sin(phi); r = 1/(RAYLEIGH_D*86400.)
        U = -(self.g*5e3/max(float(Tz.mean()), 100.))*dTdy*f/(f*f + r*r); U = np.clip(U, -60., 60.); self.U = U
        rate = np.abs(U)*3.156e7/(a*np.sqrt(1 - self.x**2)*2*np.pi/self.nlon)       # 1/yr per band
        n, m = self.n, self.nlon; idx = np.arange(self.N).reshape(n, m)
        up = np.where(U[:, None] >= 0, np.roll(idx, 1, 1), np.roll(idx, -1, 1))
        rr = np.repeat(rate, m)
        A = sp.csr_matrix((rr, (idx.ravel(), up.ravel())), shape=(self.N, self.N)) - sp.diags(rr)
        return A.tocsc()
    def _diurnal(self, Q, alb, B, T):
        """'diurnal': first daily harmonic of insolation, amplitude (pi/2) Q (1 - alb) cos(lat), on a diurnal skin
        (C_DIURNAL land / ocean; day = 1/spin days): dT_d = Q1/|i w C + B|, capped at .3 T.  Returns dT_d (cells)."""
        om = 2*np.pi*365.25*self.spin; Cd = np.where(self.landm.ravel(), C_DIURNAL[0], C_DIURNAL[1])
        cl = np.repeat(np.sqrt(1 - self.x**2), self.nlon)
        Q1 = .5*np.pi*Q*(1 - alb)*cl
        return np.minimum(Q1/np.abs(1j*om*Cd + np.maximum(B, .3)), .3*T)
    def _clouds(self, T, ice, hh):
        """'clouds': prognostic cloud condensate path W (kg m^-2), dW/dt = S - W/CLW_TAU with source S from the
        precipitation field of the moisture solve (moisture supply) x (warm convective + cold stratiform lifting),
        reduced over ice.  Cover f = 1 - exp(-W/CLW_F); in-cloud optical depth tau = 1.5 W/(rho_w r_e f) (r_e 10 um).
        Effective deck fraction for the tau-10 tables / non-RT albedo: f (1 - exp(-tau/10))/(1 - e^-1), so cover and
        opacity (albedo and OLR) both follow the condensate.  Relaxes on ~4 d: at >= monthly steps it sits on its
        source/sink balance (the difference from the diagnostic scheme is the physics of that balance, not memory)."""
        pr = getattr(self, "_pr", None); pr = np.ones(self.N) if pr is None else pr
        cold = np.clip((295. - T)/45., 0., 1.5)
        Weq = CLW_REF*np.clip(pr, 0, 2.)*(.45 + .55*cold/(1. + .25*es_ratio(float(np.mean(T)))))*(1 - .85*ice)
        if self.clw is None or len(self.clw) != self.N: self.clw = Weq.copy()
        self.clw = Weq + (self.clw - Weq)*np.exp(-hh/CLW_TAU)
        W = self.clw; fc = 1 - np.exp(-W/CLW_F); tau = 1.5*W/(1e3*1e-5*np.maximum(fc, .02))
        return np.clip(fc*(1 - np.exp(-tau/10.))/(1 - np.exp(-1.)), 0., .9)/RT_CLOUD_K     # same units as cloud_of
    def _co2_atm(self, pc, T, amp):
        """RT path: CO2 collapse.  The atmosphere cannot hold more CO2 than the vapour pressure of CO2 ice at the cold
        trap (coldest zonal band in winter, T - amp/2): the rest sits as surface CO2 ice (self.co2_ice, bar).  Smooth min
        (p^-4 + p_sat^-4)^-1/4 so the solve stays differentiable.  The carbon cycle keeps the total (C_atm) inventory."""
        if not rt.PHYS.get("co2_cond", True): return pc
        if "frost" in self.physics: return self._co2_frost(pc, T, amp)
        Tz = (T - .5*amp).reshape(self.n, self.nlon).mean(1); psat = float(rt.p_sat_co2(float(Tz.min())))/1e5
        pa = float((max(pc, 1e-12)**-4 + max(psat, 1e-12)**-4)**-.25)
        self.p_atm, self.co2_ice = pa, max(pc - pa, 0.)
        return pa
    def _co2_frost(self, pc, T, amp):
        """'frost': annual-mean cold trap (latent-heat-pinned caps) + seasonal frost-exchange diagnostic."""
        Tz = T.reshape(self.n, self.nlon).mean(1); az = amp.reshape(self.n, self.nlon).mean(1)
        psat = float(rt.p_sat_co2(float(Tz.min())))/1e5
        pa = float((max(pc, 1e-12)**-4 + max(psat, 1e-12)**-4)**-.25)
        self.p_atm, self.co2_ice = pa, max(pc - pa, 0.)
        # winter-half energy deficit where the band dips below T_sat(p_atm): mass condensed per m^2 per season (kg m^-2)
        Ts = float(rt.T_sat_co2(pa*1e5)); Cs = float(np.mean(self.C)); om = 2*np.pi/max(self.P_orb, 1e-3)
        dT = np.clip(Ts - (Tz - az), 0., None); B = 4*SIGMA*np.maximum(Tz, 100.)**3
        m_frost = (B + Cs*om)*dT*(self.P_orb*3.156e7/np.pi)/L_CO2_SUB        # J m^-2 per season / L_s
        w = np.full(self.n, 1./self.n); frac = float((w*(dT > 0)).sum())
        self.frost = dict(cap_kg_m2=float(m_frost.max()), cap_frac=frac, T_sat=Ts,
                          p_season_bar=float(min((w*m_frost).sum()*self.g/1e5, pa)))
        return pa
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
        return rt.climate_tables(self.g, te, ch4, du, self.ps, self.x_h2) if self.x_h2 else rt.climate_tables(self.g, te, ch4, du, self.ps)

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
        Mq = sp.identity(self.N, format="csc")/MOIST_TAU - Dq*Lq
        if "winds" in self.physics and self._Adv is not None: Mq = Mq - WIND_Q/WIND_EFF/C_ATM*self._Adv
        q = spla.spsolve(Mq.tocsc(), E)
        Pr = q/MOIST_TAU; ref = MOIST_REF*max(float(E[self.oc_m.ravel()].mean()), 1e-9)
        self._pr = Pr/max(float(Pr.mean()), 1e-12)
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
        pc = self._co2_atm(pc, T, getattr(self, "_amp_cur", np.zeros(self.N)))
        self._transport(pc)
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
        ocf = self.oc_m.ravel(); ph = self.physics
        if "dynamics" in ph: self._dynamics(T)
        if "winds" in ph: self._Adv = WIND_EFF*C_ATM*self._winds(T); self._set_L()
        ad = self.last.get("diurnal_g"); ad = np.zeros(self.N) if ad is None or len(ad) != self.N else ad
        for it in range(k):
            self._amp_cur = amp
            at = np.sqrt(amp**2 + ad**2) if "diurnal" in ph else amp
            ice = self.ice_of(T, at); cf = self._clouds(T, ice, hh) if "clouds" in ph else self.cloud_of(T, ice)
            asr, olr, B, dg = self._radiation(h, hw, P, T, ice, cf, insr, S0)
            if "diurnal" in ph:
                ad = self._diurnal(S0*np.repeat(insr["mean"], self.nlon), dg["alb"], B, T)
                fx = 1 + 3*(ad/np.maximum(T, 50.))**2; olr = olr*fx; B = B*fx
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
        self._amp_cur = amp
        at = np.sqrt(amp**2 + ad**2) if "diurnal" in ph else amp
        ice = self.ice_of(T, at); cf = self._clouds(T, ice, 0.) if "clouds" in ph else self.cloud_of(T, ice)
        asr, olr, B, dg = self._radiation(h, hw, P, T, ice, cf, insr, S0)
        if "diurnal" in ph:
            ad = self._diurnal(S0*np.repeat(insr["mean"], self.nlon), dg["alb"], B, T); fx = 1 + 3*(ad/np.maximum(T, 50.))**2; olr = olr*fx; B = B*fx
        amp_seas = amp; amp = at
        Z = lambda a: a.reshape(self.n, self.nlon).mean(1)
        self.last = dict(ins=dict(mean=insr["mean"], amp=insr["amp"], mu0=insr["mu0"]), amp=Z(amp), ice=Z(ice), cloud=Z(cf), S0=S0,
                         asr=float(asr.mean()), olr=float(olr.mean()), toa_imbalance=float((asr - olr).mean()),
                         g_co2=dg["g_co2"], g_wv=dg["g_wv"], g_cloud=dg["g_cloud"], cloud_mean=float(cf.mean()),
                         amp_g=amp, ice_g=ice, cloud_g=cf, B_g=B, alb_g=dg["alb"],
                         surf_abs=dg.get("surf_abs"), atm_abs=dg.get("atm_abs"))
        if "diurnal" in ph: self.last.update(diurnal_g=ad, amp_season_g=amp_seas, diurnal_mean_K=float(ad.mean()))
        if "clouds" in ph: self.last.update(clw_mean=float(self.clw.mean()))
        if "winds" in ph: self.last.update(U_band=self.U.copy(), phi_H_deg=float(np.degrees(self.phi_H)))
        if "dynamics" in ph: self.last.update(dyn_D_mean=float(np.mean(self.dyn)), dyn_D_minmax=(float(self.dyn.min()), float(self.dyn.max())))
        if "frost" in ph and getattr(self, "frost", None): self.last.update(frost=dict(self.frost))
        if self.buffer or getattr(self, "buffer_auto", False):
            self.last["buffer"] = dict(mode=self.buffer or "none", ps_n2_bar=self.ps, x_h2=self.x_h2, auto=bool(getattr(self, "buffer_auto", False)))
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
        if self.rt:
            asr = float(self.last.get("asr", 0.)); lim = rt.runaway_limit(self.g)
            st.update(rt_surf_abs=self.last.get("surf_abs"), rt_atm_abs=self.last.get("atm_abs"), planetary_albedo=float(np.mean(self.last["alb_g"])),
                      pCO2_atm_bar=None if self.p_atm is None else float(self.p_atm), co2_ice_bar=float(self.co2_ice), transport_f=float(self.f_D),
                      runaway=bool(asr > lim and self.ocean > .05), runaway_limit=lim, moist_greenhouse=bool(float(Tg.max()) > rt.T_MOIST_GH))
        if self.physics:                                      # optional-physics diagnostics (absent by default -> packets unchanged)
            st.update({k: self.last[k] for k in ("diurnal_mean_K", "clw_mean", "phi_H_deg", "dyn_D_mean", "frost") if k in self.last},
                      gcm_physics=sorted(self.physics))
        if "buffer" in self.last: st.update(rt_buffer=dict(self.last["buffer"]))
        if dT_imp:
            Tc = self.Tg.ravel(); wc = .5*(wf(Tc - amp) + wf(Tc + amp)); wc = np.where(self.landm.ravel(), wc*self.moist.ravel(), wc)
            st["water_chronic"] = float(wc.mean()); st["lat_water_chronic"] = [round(float(v), 4) for v in wc.reshape(self.n, self.nlon).mean(1)]
        return ClimateStress(cs.T_K, float(wcell.mean()), cs.climate_class, st)

RH_MIN = .05
# optional GCM2D physics (physics=("clouds", "diurnal", "winds") / --gcm-physics); off by default
GCM_PHYSICS = ("clouds", "diurnal", "winds", "dynamics", "frost")
DYN_N, DYN_MIN, DYN_MAX = 1., .3, 3.
DYN_GRAD_EARTH = 40./(np.pi/2*6.371e6)        # K m^-1, Earth's equator-pole zonal-mean gradient (~40 K over 10^4 km)
L_CO2_SUB = 5.9e5                              # J kg^-1 CO2 sublimation
N2_BUFFER_BAR, X_H2_BUFFER = 3., .1            # --rt-buffer n2: N2 column (bar); h2: + 10 % H2
CLW_TAU, CLW_REF, CLW_F = .011, .05, .06      # yr condensate lifetime (~4 d); kg m^-2 source scale; cover e-fold path
C_DIURNAL = (.016, .16)                        # W yr m^-2 K^-1 diurnal skin: land (~10 K swing on Earth), ocean (~1 K)
HADLEY_K = 1.5; RAYLEIGH_D = 5.; PHI_H_EARTH = .5047               # Hadley transport boost inside phi_H; Rayleigh friction time (days)
C_ATM, WIND_EFF, WIND_Q = .32, .1, .3          # column heat capacity (W yr m^-2 K^-1); BL share for heat; moisture share
def parse_buffer(spec):
    """'n2' | 'n2:<bar>' | 'h2' | 'h2:<bar>:<x_h2>' | None -> (ps background bar, x_h2)."""
    if not spec: return 1., 0.
    p = str(spec).split(":"); k = p[0]
    try:
        if k == "n2" and len(p) <= 2: return (float(p[1]) if len(p) > 1 else N2_BUFFER_BAR), 0.
        if k == "h2" and len(p) <= 3: return (float(p[1]) if len(p) > 1 else N2_BUFFER_BAR), (float(p[2]) if len(p) > 2 else X_H2_BUFFER)
    except ValueError: pass
    raise ValueError(f"buffer must be n2[:bar] | h2[:bar[:x_h2]] | auto, got {spec!r}")

def parse_physics(p):
    if not p: return frozenset()
    if isinstance(p, str): p = [x.strip() for x in p.split(",") if x.strip()]
    p = set(GCM_PHYSICS[:3]) | (set(p) - {"all"}) if "all" in p else set(p)    # 'all' = the original three (+ any named extra)
    bad = p - set(GCM_PHYSICS)
    if bad: raise ValueError(f"unknown gcm2d physics {sorted(bad)}; choose from {GCM_PHYSICS} or 'all'")
    return frozenset(p)
AUTO_BUFFERS = (None, "n2", "h2", "h2:10:0.3", "h2:30:0.5")   # --rt-buffer auto ladder (cached equilibria)
D_P_MAX = 10.         # cap on the pressure scaling of atmospheric heat transport (RT path)
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

def init_gcm2d(h, hw, LP=None, orbit=None, rt_on=False, nlon=N_LON, warm_branch=True, physics=(), buffer=None):
    """(GCM2D, GCMCarbon) at the carbon-silicate equilibrium (cached).  buffer (RT only): None|"n2[:bar]"|"h2[:bar[:x_h2]]"|"auto"."""
    if buffer == "auto" and rt_on:
        best = None
        for b in AUTO_BUFFERS:                       # first non-snowball rung, else the warmest (mean T) rung
            g, cc = init_gcm2d(h, hw, LP, orbit, rt_on, nlon, warm_branch, physics, b)
            if float(np.mean(g.last["ice"])) <= SNOW_ICE: best = None; break
            if best is None or float(np.mean(g.Tg)) > best[0]: best = (float(np.mean(g.Tg)), b)
        if best is not None and best[1] != b: g, cc = init_gcm2d(h, hw, LP, orbit, rt_on, nlon, warm_branch, physics, best[1])
        g.buffer_auto = True; g.last["buffer"] = dict(mode=g.buffer or "none", ps_n2_bar=g.ps, x_h2=g.x_h2, auto=True)
        return g, cc
    if buffer and not rt_on: raise ValueError("buffer needs rt_on=True")
    P = as_bus(LP); g = GCM2D(h, P, hw, nlon=nlon, orbit=orbit, rt_on=rt_on, spinup=False, physics=physics, buffer=buffer); cc = GCMCarbon(h, g, P)
    key = (round(h.a, 9), round(h.e, 9), round(h.tilt, 9), round(h.spin, 9), h.ptype, round(h.volatiles, 9), round(h.m, 12), getattr(h, "planet_id", 0),
           round(float(getattr(hw, "L_star", h.L_star)), 9), tuple(round(P[k], 9) for k in ("greenhouse_K", "albedo_mult", "lat_diffusion", "co2_sens_K", "snowball_K", "runaway_K")),
           None if orbit is None else orbit.key(), bool(rt_on), nlon, bool(warm_branch), "g3", rt.phys_key() if rt_on else None, tuple(sorted(g.physics))) + ((buffer,) if buffer else ())
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
