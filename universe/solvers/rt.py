"""solvers.rt -- plane-parallel multi-layer two-stream radiative transfer (out-of-leaf solver).

What it is
  * Longwave: 11 bands 10-3000 cm^-1, absorption-only two-stream (diffusivity 1.66) with a
    linear-in-tau Planck source per layer (no isothermal-layer surface jump).  Gas opacity is a
    synthetic k-distribution per band (log-normal spread of line strengths, Gauss-like
    equal-weight g-points), pressure-broadened k ~ p/p0, random overlap between the two major
    gases of a band (product g-points); minor gases enter at their band-mean k.
    Absorbers: H2O lines + self continuum (window, e/p0 and exp(1800(1/T-1/296))), CO2
    (15 um + wings, 9.4/10.4 um hot bands, 4.3 um), CH4 (7.7 um), O3 (9.6 um).
  * Shortwave: 4 bands (UV / vis / NIR1 / NIR2) weighted by the host star's Planck spectrum,
    delta-Eddington two-stream per layer (Meador & Weaver 1980 coefficients) combined by an
    exact adding recursion (direct beam + diffuse, Lambertian surface).  Rayleigh (N2/CO2),
    O3 Hartley/Chappuis, H2O and CO2 NIR, one cloud layer (tau 10, g .85).
  * Atmosphere: moist pseudo-adiabat from Ts (Manabe-Wetherald RH profile, cold-trapped water
    above the tropopause), isothermal stratosphere at the skin temperature (OLR/2sigma)^1/4.
  * Outputs: up/down/net fluxes at every level, heating rates (K/day), OLR, surface
    downwelling LW/SW, planetary albedo; radiative-convective equilibrium (energy-conserving
    moist convective adjustment); cached OLR / albedo lookup tables for climate coupling.

Fidelity (documented, tested in test_solvers.py)
  * Calibrated to Earth-like column within ~10 W m^-2 (clear-sky OLR ~ 265 W m^-2 at Ts=288,
    RH .77, 280 ppm).  2xCO2 instantaneous forcing a few W m^-2, ~logarithmic in pCO2;
    clear-sky dOLR/dTs at fixed RH ~ 1.5-2.2 W m^-2 K^-1 (water vapour feedback present).
  * NOT line-by-line, no CIA (high-pCO2 / H2-rich atmospheres under-absorb), no runaway steam
    atmosphere (q capped), no 3-D clouds; temperature dependence of line strengths ignored.
    It is an honest band model, not a GCM radiation package.
"""
from functools import lru_cache
import numpy as np

SIGMA = 5.670374e-8; G_EARTH = 9.81; CP = 1004.; RD = 287.; RV = 461.5; LV = 2.5e6; EPS = .622; P0 = 1e5
M_AIR, M_CO2, M_CH4, M_O3 = 28.0, 44.0, 16.0, 48.0
C1 = 1.191042e-8; C2 = 1.4387769            # W m^-2 sr^-1 (cm^-1)^-4 ; cm K
DIFF = 1.66; NG = 4; N_LAYERS = 36
_GZ = np.array([-1.5, -.5, .5, 1.5]) / 1.5   # equal-weight g-points spanning +-1 spread

LW_BANDS = np.array([[10, 350], [350, 500], [500, 630], [630, 700], [700, 820], [820, 980], [980, 1100],
                     [1100, 1400], [1400, 1800], [1800, 2200], [2200, 3000]], float)
# per band: (k_ref m^2/kg at p0, spread in decades) for H2O lines and CO2; band-mean k for CH4, O3
K_H2O = [(25., 2.6), (2.0, 2.4), (.25, 2.0), (.07, 1.6), (.025, 1.6), (.006, 1.2), (.005, 1.2),
         (.6, 2.6), (6., 2.6), (.4, 2.4), (.03, 2.0)]
K_CO2 = [(0., 0.), (0., 0.), (.01, 4.0), (12., 2.6), (.017, 4.0), (2e-4, 1.), (6e-4, 1.),
         (0., 0.), (0., 0.), (0., 0.), (6., 3.0)]
K_CH4 = [0., 0., 0., 0., 0., 0., 0., 2.0, .2, 0., .5]
K_O3 = [0., 0., 0., 0., 0., 0., 120., 0., 0., 0., 0.]
CONT_BANDS = [4, 5, 6, 7]                    # H2O self continuum (strongest in the window)
CONT_K = [.0096, .0288, .0288, .0096]

SW_BANDS = np.array([[.2, .32], [.32, .7], [.7, 1.3], [1.3, 5.]])  # um
SW_RAY = np.array([.9, .12, .012, .001])     # Rayleigh tau of 1 bar N2 at g=9.81
SW_O3 = np.array([600., 2.5, 0., 0.])        # m^2/kg band mean
SW_H2O = [(0., 0.), (0., 0.), (.006, 2.4), (.6, 2.8)]
SW_CO2 = [(0., 0.), (0., 0.), (0., 0.), (.02, 2.5)]
CLOUD_TAU_SW, CLOUD_G, CLOUD_W = 10., .85, np.array([.9999, .9999, .998, .97])
CLOUD_TAU_LW, CLOUD_SIG = 8., .5             # cloud deck in the layer nearest p = .5 ps (ISCCP mean cloud-top ~ 500-600 hPa)


# ----------------------------------------------------------------------------- thermodynamics
def e_sat(T):
    T = np.asarray(T, float)
    lv = np.where(T >= 273.15, LV, LV + 3.34e5)
    return 611.2 * np.exp(lv / RV * (1. / 273.15 - 1. / np.maximum(T, 100.)))

def q_sat(T, p):
    e = np.minimum(e_sat(T), .5 * p)          # cap: no steam atmosphere
    return EPS * e / (p - (1. - EPS) * e)

def levels(ps, n=N_LAYERS, top=1e-4):
    """Layer edges (n+1, surface last) log-spaced from ps*top to ps; top edge at 0."""
    ps = np.asarray(ps, float)[..., None]
    e = ps * np.logspace(np.log10(top), 0., n)
    pe = np.concatenate([np.zeros_like(ps), e], -1)
    pm = np.concatenate([.5 * e[..., :1], np.sqrt(e[..., :-1] * e[..., 1:])], -1)
    return pe, pm

def moist_profile(Ts, p, ps, T_strat=200., rh_s=.77, lapse_mix=1.):
    """Moist pseudo-adiabat from (Ts, ps) evaluated at pressures p (..., k) (any order),
    floored at T_strat; lapse_mix<1 blends toward dry adiabat (more stable)."""
    Ts = np.asarray(Ts, float); ps = np.asarray(ps, float)
    lnp = np.log(np.maximum(p, 1e-3)); lnps = np.log(ps)[..., None]
    grid = np.linspace(0., 1., 61)
    lo = np.log(np.maximum(p.min(-1, keepdims=True), 1e-3))
    L = lnps + (lo - lnps) * grid                    # integration path in ln p (surface -> top)
    T = np.empty(L.shape); T[..., 0] = Ts
    def dTdlnp(T, lp):
        pp = np.exp(lp); rs = q_sat(T, pp) * rh_s
        moist = (RD * T / CP) * (1 + LV * rs / (RD * T)) / (1 + LV * LV * rs * EPS / (CP * RD * T * T))
        return lapse_mix * moist + (1 - lapse_mix) * RD * T / CP
    for i in range(1, L.shape[-1]):
        h = L[..., i] - L[..., i - 1]; t0 = T[..., i - 1]
        k1 = dTdlnp(t0, L[..., i - 1]); k2 = dTdlnp(t0 + .5 * h * k1, L[..., i - 1] + .5 * h)
        T[..., i] = t0 + h * k2
    Ts_ = np.asarray(T_strat, float)
    T = np.maximum(T, Ts_[..., None] if Ts_.ndim else Ts_)
    out = np.empty(p.shape)
    flat_L, flat_T, flat_q = L.reshape(-1, L.shape[-1]), T.reshape(-1, T.shape[-1]), lnp.reshape(-1, p.shape[-1])
    o = out.reshape(-1, p.shape[-1])
    for j in range(flat_L.shape[0]):
        o[j] = np.interp(-flat_q[j], -flat_L[j], flat_T[j])
    return out

def humidity(T, p, ps, rh_s=.77):
    """Specific humidity: Manabe-Wetherald RH(p) below the cold trap, constant above it."""
    sig = p / ps[..., None]
    rh = rh_s * np.clip((sig - .02) / .98, 0., 1.)
    q = rh * q_sat(T, p)
    # cold trap: above the temperature minimum keep the minimum q
    it = np.argmin(np.where(sig > .02, T, np.inf), -1)
    qmin = np.take_along_axis(q, it[..., None], -1)
    above = np.arange(T.shape[-1]) < it[..., None]
    return np.where(above, np.minimum(q, qmin), q)


# ----------------------------------------------------------------------------- Planck
_NU = None
def planck_bands(T, bands=LW_BANDS, n_sub=40):
    """pi*B integrated over each band (W m^-2), residual of sigma T^4 folded into end bands."""
    T = np.asarray(T, float)[..., None, None]
    nu = np.linspace(0, 1, n_sub)[None, :] * (bands[:, 1:] - bands[:, :1]) + bands[:, :1]  # (nb, n_sub)
    x = np.minimum(C2 * nu / T, 700.)
    f = np.pi * C1 * nu ** 3 / np.expm1(np.maximum(x, 1e-9))
    out = np.trapezoid(f, nu, axis=-1) if hasattr(np, "trapezoid") else np.trapz(f, nu, axis=-1)
    tot = SIGMA * T[..., 0, 0] ** 4
    res = tot - out.sum(-1)
    out[..., -1] += np.maximum(res, 0.) * .5; out[..., 0] += np.maximum(res, 0.) * .5
    return out

def stellar_sw_fractions(T_eff):
    """Fraction of a T_eff blackbody's flux in each SW band (beyond 5 um folded into NIR2)."""
    lam = np.linspace(.05, 30., 6000)  # um
    x = np.minimum(14387.77 / (lam * T_eff), 700.)
    B = 1. / (lam ** 5 * np.expm1(x))
    edges = SW_BANDS.copy(); edges[0, 0] = .05; edges[-1, 1] = 30.
    fr = np.array([B[(lam >= a) & (lam < b)].sum() for a, b in edges])
    return fr / fr.sum()


# ----------------------------------------------------------------------------- column state
class Column:
    """A set of atmospheric columns (leading batch shape) with layers ordered top -> surface."""
    def __init__(self, Ts, ps=1., pco2=2.8e-4, rh_s=.77, x_ch4=7e-7, o3_du=0., g=G_EARTH, T_strat=None,
                 n=N_LAYERS, cloud=0., T_profile=None):
        Ts = np.atleast_1d(np.asarray(Ts, float))
        shp = Ts.shape
        b = lambda v: np.broadcast_to(np.asarray(v, float), shp).astype(float)
        self.Ts, self.pco2, self.g = Ts, b(pco2), b(g)
        self.ps = (b(ps) + self.pco2) * 1e5                  # N2 background + CO2 (bar -> Pa)
        self.pe, self.pm = levels(self.ps, n)
        if T_strat is None: T_strat = np.clip(.62 * Ts, 150., 230.)          # first guess; radiation() refines to skin T
        self.T_strat = b(T_strat)
        both = np.concatenate([self.pm, self.pe], -1)
        Tb = moist_profile(Ts, both, self.ps, self.T_strat, float(np.mean(rh_s)))
        self.Tm = Tb[..., :n] if T_profile is None else T_profile
        self.Te = Tb[..., n:]; self.Te[..., -1] = Ts
        self.rh_s = b(rh_s)
        self.q = humidity(self.Tm, self.pm, self.ps, self.rh_s[..., None])
        x_co2 = self.pco2 * 1e5 / self.ps
        mbar = x_co2 * M_CO2 + (1 - x_co2) * M_AIR
        self.mbar = mbar
        self.q_co2 = (x_co2 * M_CO2 / mbar)[..., None] * np.ones_like(self.Tm)
        self.q_ch4 = (b(x_ch4) * M_CH4 / mbar)[..., None] * np.ones_like(self.Tm)
        # ozone: column (DU -> kg m^-2) spread as a Gaussian in ln p peaking at ~20 hPa (scaled to ps)
        col = b(o3_du) * 2.1415e-5
        lnp = np.log(np.maximum(self.pm, 1e-3) / (2e3 * self.ps[..., None] / 1e5))
        dm = np.diff(self.pe, axis=-1) / self.g[..., None]
        w = np.exp(-.5 * (lnp / .9) ** 2) * dm
        self.q_o3 = col[..., None] * w / np.maximum(w.sum(-1, keepdims=True), 1e-30) / np.maximum(dm, 1e-30)
        self.dm = dm
        self.cloud = b(cloud)
        self.icld = np.argmin(np.abs(self.pm / self.ps[..., None] - CLOUD_SIG), -1)

    def precipitable_water(self):
        return (self.q * self.dm).sum(-1)


def _kpoints(kref, spread):
    if kref <= 0: return np.zeros(NG)
    return kref * 10. ** (.5 * spread * _GZ)


def lw_optical_depth(c):
    """d tau (..., nlay, nbg) and per-g weights (nbg,) and band index (nbg,)."""
    pfac = np.maximum(c.pm / P0, 1e-3)
    e = c.q * c.pm / EPS
    taus, wts, bidx = [], [], []
    for b in range(len(LW_BANDS)):
        kh, kc = _kpoints(*K_H2O[b]), _kpoints(*K_CO2[b])
        grey = (K_CH4[b] * c.q_ch4 + K_O3[b] * c.q_o3) * c.dm
        if b in CONT_BANDS:
            kcont = CONT_K[CONT_BANDS.index(b)] * (e / 1.5e3) * np.exp(1800. * (1. / c.Tm - 1. / 296.))
            grey = grey + kcont * c.q * c.dm
        th = (c.q * c.dm * pfac)[..., None] * kh            # (..., nlay, NG)
        if K_CO2[b][0] > 0:
            tc = (c.q_co2 * c.dm * pfac)[..., None] * kc
            t = (th[..., :, None] + tc[..., None, :]).reshape(th.shape[:-1] + (NG * NG,))
            w = np.full(NG * NG, 1. / (NG * NG))
        else:
            t = th; w = np.full(NG, 1. / NG)
        taus.append(t + grey[..., None]); wts.append(w); bidx.append(np.full(len(w), b))
    return np.concatenate(taus, -1), np.concatenate(wts), np.concatenate(bidx)


def lw(c, overcast=False, eps_s=1.):
    tau, w, bidx = lw_optical_depth(c)
    if overcast:
        sel = np.zeros(tau.shape[:-1], bool); np.put_along_axis(sel, c.icld[..., None], True, -1)
        tau = tau + np.where(sel[..., None], CLOUD_TAU_LW, 0.)
    Bm = planck_bands(c.Te)[..., bidx] * w
    Bs = planck_bands(c.Ts)[..., bidx] * w
    D = DIFF * tau; t = np.exp(-D)
    lin = np.where(D < 1e-4, .5 * D, (1. - t * (1. + D)) / np.maximum(D, 1e-30))
    nl = tau.shape[-2]
    up = np.empty(Bm.shape); dn = np.empty(Bm.shape); dn[..., 0, :] = 0.
    for i in range(nl):
        dn[..., i + 1, :] = dn[..., i, :] * t[..., i, :] + Bm[..., i + 1, :] * (1 - t[..., i, :]) + (Bm[..., i, :] - Bm[..., i + 1, :]) * lin[..., i, :]
    up[..., nl, :] = eps_s * Bs + (1 - eps_s) * dn[..., nl, :]
    for i in range(nl - 1, -1, -1):
        up[..., i, :] = up[..., i + 1, :] * t[..., i, :] + Bm[..., i, :] * (1 - t[..., i, :]) + (Bm[..., i + 1, :] - Bm[..., i, :]) * lin[..., i, :]
    return up.sum(-1), dn.sum(-1)


# ----------------------------------------------------------------------------- shortwave
def two_stream_layer(tau, w, g, mu0):
    """Delta-Eddington layer reflectance/transmittance (Meador & Weaver 1980).
    Returns R_dif, T_dif, R_dir, T_dir (diffuse part of transmitted direct), Tdb (direct)."""
    f = g * g
    tau = (1 - w * f) * tau; w = (1 - f) * w / (1 - w * f); g = g / (1 + g)
    w = np.minimum(w, 1 - 1e-7)
    g1 = (7 - w * (4 + 3 * g)) / 4.; g2 = -(1 - w * (4 - 3 * g)) / 4.
    g3 = (2 - 3 * g * mu0) / 4.; g4 = 1 - g3
    lam = np.sqrt(np.maximum(g1 * g1 - g2 * g2, 1e-12))
    mu = np.where(np.abs(lam * mu0 - 1) < 1e-3, mu0 * (1 + 2e-3), mu0)
    tau = np.minimum(tau, 200.)
    e1 = np.exp(-lam * tau); e2 = e1 * e1; ed = np.exp(-tau / mu)
    den_d = (lam + g1) + (lam - g1) * e2
    R_dif = g2 * (1 - e2) / den_d
    T_dif = 2 * lam * e1 / den_d
    a1 = g1 * g4 + g2 * g3; a2 = g1 * g3 + g2 * g4
    # exp(+lam tau) factored out: den = (lam+g1) + (lam-g1) e2  (times e^{lam tau})
    pre = w / ((1 - lam * lam * mu * mu) * den_d)
    R_dir = pre * ((1 - lam * mu) * (a2 + lam * g3) - (1 + lam * mu) * (a2 - lam * g3) * e2
                   - 2 * lam * (g3 - a2 * mu) * ed * e1)
    # direct-to-diffuse transmission (stable form): multiply MW expression by e^{-lam tau}
    T_dir = pre * (ed * ((1 + lam * mu) * (a1 + lam * g4) - (1 - lam * mu) * (a1 - lam * g4) * e2)
                   - 2 * lam * (g4 + a1 * mu) * e1)
    T_dir = -T_dir
    return np.clip(R_dif, 0, 1), np.clip(T_dif, 0, 1), np.clip(R_dir, 0, 1), np.clip(T_dir, 0, 1), ed


def adding(Rd, Td, Rr, Tr, Tdb, alb, mu0, F0=1.):
    """Adding recursion. Layer arrays (..., nlay); returns U, D(diffuse), S(direct, horizontal) at edges."""
    nl = Rd.shape[-1]
    S = np.empty(Rd.shape[:-1] + (nl + 1,)); S[..., 0] = mu0 * F0
    for i in range(nl): S[..., i + 1] = S[..., i] * Tdb[..., i]
    A = np.empty_like(S); B = np.empty_like(S)
    A[..., nl] = alb; B[..., nl] = alb * S[..., nl]
    den = np.empty(Rd.shape)
    for i in range(nl - 1, -1, -1):
        den[..., i] = 1 - Rd[..., i] * A[..., i + 1]
        A[..., i] = Rd[..., i] + Td[..., i] ** 2 * A[..., i + 1] / den[..., i]
        B[..., i] = Td[..., i] * (A[..., i + 1] * (Rd[..., i] * B[..., i + 1] + Tr[..., i] * S[..., i]) / den[..., i] + B[..., i + 1]) + Rr[..., i] * S[..., i]
    D = np.empty_like(S); U = np.empty_like(S); D[..., 0] = 0.
    for i in range(nl):
        D[..., i + 1] = (Td[..., i] * D[..., i] + Rd[..., i] * B[..., i + 1] + Tr[..., i] * S[..., i]) / den[..., i]
    U[...] = A * D + B
    return U, D, S


def sw(c, mu0=.5, alb_s=.1, T_eff=5772., overcast=False):
    """Shortwave fluxes for unit TOA normal flux F0=1 (multiply by S). Returns U, Dtot, net(...,nlay+1)."""
    mu0 = np.broadcast_to(np.asarray(mu0, float), c.Ts.shape)[..., None]
    alb = np.broadcast_to(np.asarray(alb_s, float), c.Ts.shape)
    fr = stellar_sw_fractions(T_eff)
    colfac = (c.dm / (1e5 / G_EARTH)) * (29. / c.mbar[..., None]) * (1 + 1.5 * c.q_co2 * c.mbar[..., None] / M_CO2)
    pfac = np.maximum(c.pm / P0, 1e-3)
    Utot = 0.; Dtot = 0.; Nt = 0.
    for b in range(len(SW_BANDS)):
        tr = SW_RAY[b] * colfac
        grey = SW_O3[b] * c.q_o3 * c.dm
        kh, kc = _kpoints(*SW_H2O[b]), _kpoints(*SW_CO2[b])
        hs = (c.q * c.dm * pfac)[..., None] * kh if SW_H2O[b][0] > 0 else np.zeros(c.Tm.shape + (1,))
        cs = (c.q_co2 * c.dm * pfac)[..., None] * kc if SW_CO2[b][0] > 0 else np.zeros(c.Tm.shape + (1,))
        ta = (hs[..., :, None] + cs[..., None, :]).reshape(c.Tm.shape + (-1,)) + grey[..., None]
        ng = ta.shape[-1]
        tR = tr[..., None] * np.ones(ng)
        tc = np.zeros_like(tR); wc = 0.
        if overcast:
            sel = np.zeros(c.Tm.shape, bool); np.put_along_axis(sel, c.icld[..., None], True, -1)
            tc = np.where(sel[..., None], CLOUD_TAU_SW, 0.) * np.ones(ng)
        tt = ta + tR + tc
        w = (tR + CLOUD_W[b] * tc) / np.maximum(tt, 1e-12)
        gg = CLOUD_G * CLOUD_W[b] * tc / np.maximum(tR + CLOUD_W[b] * tc, 1e-12)
        mm = mu0[..., None]
        Rd, Td, Rr, Tr, Tdb = two_stream_layer(tt, w, gg, mm)
        mv = lambda a: np.moveaxis(a, -1, 0)
        U, D, S = adding(mv(Rd), mv(Td), mv(Rr), mv(Tr), mv(Tdb), alb[None, ...], mu0[..., 0][None, ...])
        U, D = U.mean(0) * fr[b], (D + S).mean(0) * fr[b]
        Utot = Utot + U; Dtot = Dtot + D
    return Utot, Dtot, Dtot - Utot


# ----------------------------------------------------------------------------- column API
def radiation(Ts, S_toa=340., mu0=.5, alb_s=.1, pco2=2.8e-4, rh_s=.77, x_ch4=7e-7, o3_du=0., ps=1., g=G_EARTH,
              T_eff=5772., cloud=0., T_strat=None, n=N_LAYERS, column=None):
    """Full column diagnostics: OLR, ASR, fluxes at edges and heating rates (K/day).
    S_toa = mean incident flux (W m^-2) on the horizontal at the top (S0/4 for global mean)."""
    c = column or Column(Ts, ps, pco2, rh_s, x_ch4, o3_du, g, T_strat, n)
    if T_strat is None and column is None:
        U, _ = lw(c); Tsk = np.clip((U[..., 0] / (2 * SIGMA)) ** .25, 120., 260.)
        c = Column(Ts, ps, pco2, rh_s, x_ch4, o3_du, g, Tsk, n)
    Ucl, Dcl = lw(c); Uov, Dov = lw(c, overcast=True)
    cf = np.asarray(cloud, float)[..., None] if np.ndim(cloud) else cloud
    Ul = (1 - cf) * Ucl + cf * Uov; Dl = (1 - cf) * Dcl + cf * Dov
    sU, sD, _ = sw(c, mu0, alb_s, T_eff)
    if np.any(np.asarray(cloud) > 0):
        oU, oD, _ = sw(c, mu0, alb_s, T_eff, overcast=True)
        sU = (1 - cf) * sU + cf * oU; sD = (1 - cf) * sD + cf * oD
    F0 = S_toa / np.broadcast_to(np.asarray(mu0, float), c.Ts.shape)[..., None]
    sU, sD = sU * F0, sD * F0
    net = (sD - sU) - (Ul - Dl)                        # downward positive at each edge
    heat = (net[..., :-1] - net[..., 1:]) * c.g[..., None] / (CP * np.diff(c.pe, axis=-1)) * 86400.
    return dict(OLR=Ul[..., 0], ASR=(sD - sU)[..., 0], albedo=sU[..., 0] / np.maximum(sD[..., 0], 1e-12),
                lw_up=Ul, lw_down=Dl, sw_up=sU, sw_down=sD, net=net, heating_K_day=heat,
                surf_lw_down=Dl[..., -1], surf_sw_down=sD[..., -1], surf_net=net[..., -1],
                T_layers=c.Tm, p_layers=c.pm, p_edges=c.pe, q=c.q, pw_kg_m2=c.precipitable_water(), column=c)


def rce(S_toa=340., Ts0=288., mu0=.5, alb_s=.12, pco2=2.8e-4, rh_s=.77, x_ch4=7e-7, o3_du=300., ps=1., g=G_EARTH,
        T_eff=5772., cloud=0., n=N_LAYERS, iters=600, tol=.05, dt_strat=1.0):
    """Radiative-convective equilibrium of one column.  Stratosphere layers march with their
    radiative heating; surface + convective troposphere form one enthalpy reservoir constrained
    to the moist adiabat (energy-conserving convective adjustment).  Returns (Ts, diag)."""
    Ts = float(Ts0); c = Column(Ts, ps, pco2, rh_s, x_ch4, o3_du, g, 200., n)
    T = c.Tm[0].copy(); C_s = 4.2e6 * 1.0                                   # 1 m water reservoir (accelerated)
    dp = np.diff(c.pe[0]); cpm = CP * dp / g
    for it in range(iters):
        c.Tm = T[None]; c.Te = np.concatenate([[T[0]], .5 * (T[:-1] + T[1:]), [Ts]])[None]
        c.Ts = np.array([Ts]); c.q = humidity(c.Tm, c.pm, c.ps, c.rh_s[..., None])
        d = radiation(Ts, S_toa, mu0, alb_s, pco2, rh_s, x_ch4, o3_du, ps, g, T_eff, cloud, column=c)
        h = d["heating_K_day"][0]
        Tn = T + dt_strat * h
        ad = moist_profile(np.array([Ts]), c.pm, c.ps, np.array([1.]), float(rh_s))[0]
        conv = ad >= Tn - 1e-9
        conv[-1] = True
        # enthalpy budget of surface + convective layers over the step
        dE = d["surf_net"][0] * dt_strat * 86400. + (cpm[conv] * (Tn[conv] - T[conv])).sum()
        dadT = (moist_profile(np.array([Ts + .5]), c.pm, c.ps, np.array([1.]), float(rh_s))[0] - ad) / .5
        dTs = dE / (C_s + (cpm[conv] * dadT[conv]).sum())
        Ts_new = Ts + dTs
        adn = moist_profile(np.array([Ts_new]), c.pm, c.ps, np.array([1.]), float(rh_s))[0]
        T = np.where(conv, adn, Tn)
        T = np.maximum(T, 100.)
        imb = d["ASR"][0] - d["OLR"][0]
        if it > 50 and abs(Ts_new - Ts) < 1e-4 and abs(imb) < tol and np.abs(h[~conv]).max(initial=0) < 2e-3: Ts = Ts_new; break
        Ts = Ts_new
    return Ts, dict(d, iters=it + 1, imbalance=float(imb), conv_top_hPa=float(c.pm[0][conv].min() / 100.))


# ----------------------------------------------------------------------------- climate lookup tables
TS_GRID = np.arange(160., 391., 10.)
LCO2_GRID = np.linspace(-6., np.log10(30.), 10)
MU_GRID = np.array([.15, .3, .45, .6, .8])
RH_GRID = np.array([.35, .77])

@lru_cache(maxsize=16)
def climate_tables(g=G_EARTH, T_eff=5772., x_ch4=7e-7, o3_du=0., ps=1.):
    """OLR[clear/overcast](RH, Ts, log pCO2) and SW atmosphere coefficients
    (R, X, R*, Tt) [clear/overcast](Ts, log pCO2, mu0) so that for surface albedo a:
        alpha_p = R + a X / (1 - a R*),   surface absorbed = Tt (1-a)/(1-a R*)."""
    Ts, L = np.meshgrid(TS_GRID, LCO2_GRID, indexing="ij")
    olr = np.empty((2, len(RH_GRID)) + Ts.shape)
    for k, rh in enumerate(RH_GRID):
        c = Column(Ts, ps, 10 ** L, rh, x_ch4, o3_du, g, 200.)
        U, _ = lw(c); Tsk = np.clip((U[..., 0] / (2 * SIGMA)) ** .25, 120., 260.)
        c = Column(Ts, ps, 10 ** L, rh, x_ch4, o3_du, g, Tsk)
        olr[0, k] = lw(c)[0][..., 0]; olr[1, k] = lw(c, overcast=True)[0][..., 0]
    TT, LL, MM = np.meshgrid(TS_GRID, LCO2_GRID, MU_GRID, indexing="ij")
    c = Column(TT, ps, 10 ** LL, .77, x_ch4, o3_du, g, 200.)
    swc = np.empty((2, 4) + TT.shape)
    for ov in (0, 1):
        p = {}
        for a in (0., .5, 1.):
            U, D, _ = sw(c, MM, a, T_eff, overcast=bool(ov))
            p[a] = (U[..., 0] / MM, (D[..., -1] - U[..., -1]) / MM)
        R = p[0.][0]; y1 = p[1.][0] - R; y5 = p[.5][0] - R
        Rs = np.clip((y1 - 2 * y5) / np.where(np.abs(y1 - y5) < 1e-9, 1e-9, y1 - y5), 0., .99)
        X = y1 * (1 - Rs); Tt = p[0.][1]
        swc[ov] = np.stack([R, X, Rs, Tt])
    return dict(olr=olr, sw=swc)


def _interp_idx(grid, v):
    v = np.clip(v, grid[0], grid[-1]); i = np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2)
    return i, (v - grid[i]) / (grid[i + 1] - grid[i])

def olr_lookup(tab, Ts, pco2, rh=.77, cloud=0.):
    """OLR (W m^-2) and dOLR/dTs for arrays Ts, pco2 (bar), rh, cloud fraction."""
    Ts = np.asarray(Ts, float)
    i, fi = _interp_idx(TS_GRID, Ts); j, fj = _interp_idx(LCO2_GRID, np.log10(np.maximum(pco2, 1e-8)))
    k, fk = _interp_idx(RH_GRID, rh)
    o = tab["olr"]
    def at(ov):
        v = 0.; d = 0.
        for kk, wk in ((k, 1 - fk), (k + 1, fk)):
            lo = o[ov, kk, i, j] * (1 - fj) + o[ov, kk, i, j + 1] * fj; hi = o[ov, kk, i + 1, j] * (1 - fj) + o[ov, kk, i + 1, j + 1] * fj
            v = v + wk * (lo * (1 - fi) + hi * fi); d = d + wk * (hi - lo) / (TS_GRID[1] - TS_GRID[0])
        return v, d
    v0, d0 = at(0); v1, d1 = at(1)
    # beyond the table edges continue with a Planck-like slope
    ext = SIGMA * 4 * np.maximum(Ts, 150.) ** 3 * .45
    lo = Ts < TS_GRID[0]; hi = Ts > TS_GRID[-1]
    v = (1 - cloud) * v0 + cloud * v1; d = (1 - cloud) * d0 + cloud * d1
    v = np.where(lo, v + ext * (Ts - TS_GRID[0]), np.where(hi, v + ext * (Ts - TS_GRID[-1]) * .3, v))
    return v, np.maximum(d, .3)

def sw_lookup(tab, Ts, pco2, mu0, alb_s, cloud=0.):
    """Planetary albedo and fraction of TOA flux absorbed by the surface / atmosphere."""
    i, fi = _interp_idx(TS_GRID, Ts); j, fj = _interp_idx(LCO2_GRID, np.log10(np.maximum(pco2, 1e-8)))
    k, fk = _interp_idx(MU_GRID, mu0)
    s = tab["sw"]
    def tri(a):
        out = 0.
        for ii, wi in ((i, 1 - fi), (i + 1, fi)):
            for jj, wj in ((j, 1 - fj), (j + 1, fj)):
                for kk, wk in ((k, 1 - fk), (k + 1, fk)):
                    out = out + wi * wj * wk * a[ii, jj, kk]
        return out
    res = []
    for ov in (0, 1):
        R, X, Rs, Tt = (tri(s[ov, m]) for m in range(4))
        ap = R + alb_s * X / (1 - alb_s * Rs); sa = Tt * (1 - alb_s) / (1 - alb_s * Rs)
        res.append((ap, sa))
    ap = (1 - cloud) * res[0][0] + cloud * res[1][0]; sa = (1 - cloud) * res[0][1] + cloud * res[1][1]
    return ap, sa, 1 - ap - sa


# ----------------------------------------------------------------------------- biosignature hook
# grey band strengths (m^2/kg) for slant-path line depths of the leaves/biosign LINES species
LINE_K = {"O2": 3e-4, "O3": 400., "CH4": 2.0, "H2O": .15, "CO2": .5}
def o3_du_from_o2(x_o2):
    """Photochemical ozone column vs O2 (Segura et al. 2003-like, 300 DU at 0.21)."""
    x = max(float(x_o2), 0.)
    return 0. if x < 1e-6 else float(300. * (x / .21) ** .4)

def line_depths(x_o2, x_ch4, pco2, pw, ps=1., g=G_EARTH, cloud=.5, airmass=2.):
    """Band-centre depths 1-exp(-tau) along a two-way slant path, cloud-truncated."""
    colm = (ps + pco2) * 1e5 / g
    x_co2 = pco2 / (ps + pco2)
    cols = {"O2": x_o2 * 32. / 29. * colm, "CH4": x_ch4 * 16. / 29. * colm, "CO2": x_co2 * 44. / 29. * colm,
            "H2O": pw, "O3": o3_du_from_o2(x_o2) * 2.1415e-5}
    vis = 1. - .6 * cloud                      # clouds hide the lower half of the column
    return {k: float(1 - np.exp(-LINE_K[k] * v * airmass * vis)) for k, v in cols.items()}

def surface_contrast(pigments, pw, cloud, T_eff=5772.):
    """Continuum contrast of a surface pigment edge seen through the atmosphere (two-way Rayleigh+cloud)."""
    tr = np.exp(-2 * (.12 + .002 * pw)) * (1 - .8 * cloud)
    return float(pigments * .3 * tr)
