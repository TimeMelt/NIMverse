"""solvers.secular_nbody -- orbit + spin-axis drivers for the climate (out-of-leaf; life_loop(orbit_solver=...)).

Units: AU, yr, M_sun (G = 4 pi^2), planet masses in M_sun (Earth = 3e-6).  Every driver has the leaf API of
leaves.orbit_climate.SecularOrbit: key() and at(t) -> dict(e, varpi, obliquity) with varpi the longitude of perihelion
measured from the moving vernal equinox (climatic precession) and obliquity in radians.

orbit_solver="nbody"   NBodySystem: Wisdom-Holman symplectic map in democratic-heliocentric coordinates
    (Duncan, Levison & Lee 1998; kick-drift-kepler-drift-kick, 2nd order), universal-free elliptic f/g Kepler solver,
    1PN apsidal precession via the Nobili-Roxburgh potential U = -3(GM/c)^2/r^2.  Integrated lazily as at(t) asks
    (state is kept, so checkpoints resume); beyond `horizon` years the driver hands over to the secular solution.
    MEGNO chaos indicator (Cincotta & Simo 2000) from a renormalised shadow trajectory: <Y> -> 2 quasi-periodic, grows ~lambda t/2 chaotic.
orbit_solver="secular" SecularSystem: full Laplace-Lagrange theory in BOTH eccentricity (h,k; matrix A) and inclination
    (p,q; matrix B) with GR apsidal terms, eigen-decomposed (Murray & Dermott ch. 7).  Unlike the leaf proxy it needs no
    constant forced inclination (I_ORB) or ad-hoc 2nd-order coupling: the orbit normal n(t) comes from the nodal modes.
Both: spin axis integrated from Colombo's top  ds/dt = alpha (s.n)(s x n)  with alpha = (2 pi/25.8 kyr) spin (M*/Msun)(AU/a)^3
    (alpha = 3 n^2 J2/(2 omega C) with J2 ~ omega^2 -> alpha ~ n^2 omega ~ spin M*/a^3; Earth-Moon calibrated, i.e. it
    assumes an Earth-like satellite boost; the leaf omits the M*/a^3 factor) against the actual orbit normal -> obliquity = acos(s.n) including spin-orbit resonances,
    and climatic precession from the equinox direction (n x s).
orbit_solver="off"     FixedOrbit: e, tilt from the habitat, varpi 0 (no secular forcing).
tides=True (opt-in; solvers["orbit_tides"] = "on" | "moon"): constant-time-lag equilibrium tides (Hut 1981; Leconte et al.
    2010 eqs. with the planet's spin AM << orbital AM): K = 3/2 k2 dt_lag G M*^2/R (R/a)^6 n^2,
      d omega/dt = -K/(C m R^2) [(1 + cos^2 eps) Omega(e) omega/n - 2 cos eps N(e)]          (spin-down -> pseudo-synchronous)
      d eps/dt   =  K sin eps/(C m R^2 omega) [cos eps Omega(e) omega/n - 2 N(e)]              (obliquity)
      de/dt      =  11 a e K/(2 G M* m) [Omega_e(e) cos eps omega/n - 18/11 N_e(e)]          (eccentricity)
    k2 = 0.3, dt_lag = 600 s (Earth's present lunar-tide lag), C = 0.33, R = R_E (m/M_E)^0.27.  The spin rate feeds the
    precession constant (alpha ~ omega).  Secular: e damping enters LL as an imaginary diagonal (A + i Gamma, rate frozen
    at the initial spin/obliquity, so the theory stays linear); spin/obliquity integrated on the dt_spin grid.  N-body:
    spin/obliquity tides only (e-damping times >> the 20 kyr N-body horizon; handled by the secular hand-over).
    moon=True adds a satellite (the system's largest moon of that planet, else an Earth-like Moon, M_E/81.3 at 60.3 R):
    constant-lag torque T = 3 k2 G M_m^2 R^5 (omega - n_m) dt_lag / a_m^6 spins the planet down and moves the moon
    (da_m/dt = 2 T sqrt(a_m/(G(m+M_m)))/M_m; Earth-Moon today: 4.5e16 N m, 3.8 cm/yr), and the satellite share of the
    spin-axis precession scales as M_m/a_m^3 (lunar/solar = 2.17 for Earth).  Without moon=True alpha keeps the leaf's
    Earth-Moon calibration (tides then only scale it with omega).
    N-body + moon (default for orbit_tides="moon"; "moon-ring" keeps the ring model): each satellite is a massive
    particle.  The WH particle is the planet+moon barycentre (mass m + M_m); the planet-moon relative vector rho is
    integrated with its own Kepler drift (mu = G(m+M_m)) sub-cycled at P_moon/MOON_SUB inside every planetary step, with
    the star's differential (tidal) pull  -G M* (r_m/|r_m|^3 - r_p/|r_p|^3)  as the kick (barycentre linearly
    interpolated across the step).  The spin then precesses about the planet's orbit normal (solar share alpha/(1+lam0)
    times omega/omega0) AND about the moon's own instantaneous orbit normal (lunar share x lam(a_m)), so the moon's
    nodal regression (18.6 yr for Earth) and inclination enter the obliquity directly.  Tidal migration of a_m (Tides)
    rescales rho, v_rho (shape preserved).  Not included: planet-planet forces on the moon, moon back-reaction on the
    barycentre beyond the point-mass sum, planet J2 on the moon orbit (Laplace plane).
    Secular + tides: Gamma_e is re-evaluated from the evolving spin rate and obliquity every GAM_DT yr (piecewise
    eigen-solution with z continuous at each re-evaluation), no longer frozen at t = 0.
    horizon (solvers["nbody_kyr"], CLI --nbody-kyr): N-body window before the LL hand-over (default 20 kyr) and the
    MEGNO window reported for the system (megno(t_end) on a copy).
LL-linear vs N-body chaos: SecularSystem is quasi-periodic by construction (a sum of eigenmodes; no resonances, no
    diffusion, MEGNO would be exactly 2); NBodySystem captures mean-motion resonances, secular chaos and close encounters
    (MEGNO > 2) but only up to `horizon` years, after which every driver continues on the LL solution.  Long-term chaotic
    diffusion of e/i (e.g. Mercury-like Laskar 1994 behaviour) and obliquity chaos from overlapping spin-orbit resonances
    beyond the horizon are therefore NOT represented; the obliquity integration itself is nonlinear and does show
    spin-orbit resonance capture against the LL nodal modes.
Fidelity / gaps: point masses (no planet J2 beyond alpha; tides are equilibrium constant-lag only, no ocean resonance /
    thermal atmospheric tides / Cassini-state damping); WH accuracy ~ (m/M)(dt/P)^2, dt = P_inner/20; close encounters are
    not regularised (MEGNO flags them); LL theory is linear in e, i (good for e,i < ~0.1)."""
import numpy as np

def prec_const(a, M_star, spin):
    """Spin-axis precession constant alpha (rad/yr) = 2 pi/P_PREC0 * spin * M*/a^3 (Earth: 25.8 kyr)."""
    sp = np.ones(len(a)) if spin is None else np.maximum(np.asarray(spin, float), .05)
    return 2*np.pi/P_PREC0*sp*float(M_star)/np.asarray(a, float)**3

G = 4*np.pi**2
C_AUYR = 63241.077          # speed of light, AU/yr
P_PREC0 = 25800.            # yr, spin-axis precession at Earth spin (same calibration as the leaf)
SIGMA_I = np.radians(1.5)   # Rayleigh scale of initial mutual inclinations
HORIZON = 20000.            # yr of N-body integration before handing over to the secular driver
MOON_SUB = 12               # moon sub-steps per moon period (N-body moon particles)
GAM_DT = 500.               # yr between tidal Gamma_e re-evaluations (secular + tides)
REC_DT = .5                 # yr between stored samples


def _rng(seed): return np.random.default_rng([int(seed), 3307])


# ---------------------------------------------------------------- Kepler / elements
def elements_to_xv(mu, a, e, inc, Om, w, M):
    E = M.copy()
    for _ in range(30): E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    cE, sE = np.cos(E), np.sin(E); b = np.sqrt(1 - e*e)
    xo, yo = a*(cE - e), a*b*sE; n = np.sqrt(mu/a**3); den = 1 - e*cE
    vxo, vyo = -a*n*sE/den, a*n*b*cE/den
    cO, sO, cw, sw, ci, si = np.cos(Om), np.sin(Om), np.cos(w), np.sin(w), np.cos(inc), np.sin(inc)
    R = np.array([[cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci], [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci], [sw*si, cw*si]])  # 3x2xN
    x = np.stack([R[k, 0]*xo + R[k, 1]*yo for k in range(3)], -1); v = np.stack([R[k, 0]*vxo + R[k, 1]*vyo for k in range(3)], -1)
    return x, v


def xv_to_evec(mu, x, v):
    """Eccentricity vector, unit orbit normal, semi-major axis."""
    r = np.linalg.norm(x, axis=-1)[..., None]; v2 = (v*v).sum(-1)[..., None]; xv = (x*v).sum(-1)[..., None]
    ev = ((v2 - mu[..., None]/r)*x - xv*v)/mu[..., None]; h = np.cross(x, v)
    nrm = h/np.linalg.norm(h, axis=-1)[..., None]; a = 1/(2/r[..., 0] - v2[..., 0]/mu)
    return ev, nrm, a


def kepler_drift(mu, x, v, dt):
    """Advance bound two-body orbits by dt (vectorised f/g functions in eccentric-anomaly difference)."""
    r0 = np.linalg.norm(x, axis=1); v2 = (v*v).sum(1); u = (x*v).sum(1)
    a = 1/(2/r0 - v2/mu); n = np.sqrt(mu/a**3); sa = np.sqrt(a)
    ec, es = 1 - r0/a, u/np.sqrt(mu)/sa           # e cosE0, e sinE0
    dM = n*dt; dE = dM.copy()
    for _ in range(12):
        s, c = np.sin(dE), np.cos(dE)
        f = dE - ec*s + es*(1 - c) - dM; fp = 1 - ec*c + es*s
        dE = dE - f/fp
        if np.all(np.abs(f) < 1e-14): break
    s, c = np.sin(dE), np.cos(dE)
    r = a*(1 - ec*c + es*s)
    f = 1 - a/r0*(1 - c); g = dt + (s - dE)/n
    fd = -a*a*n*s/(r*r0); gd = 1 - a/r*(1 - c)
    return f[:, None]*x + g[:, None]*v, fd[:, None]*x + gd[:, None]*v


def _spin_rotate(s, n, ang):
    """Rotate unit vectors s about unit axes n by angles ang (Rodrigues)."""
    c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return s*c + np.cross(n, s)*sn + n*(n*s).sum(1)[:, None]*(1 - c)


def _climatic(s, nrm, ev):
    """(obliquity, varpi from vernal equinox) for spin s, orbit normal nrm, eccentricity vector ev (arrays N x 3)."""
    cosobl = np.clip((s*nrm).sum(1), -1, 1); obl = np.arccos(cosobl)
    gam = np.cross(nrm, s); gn = np.linalg.norm(gam, axis=1)
    ref = np.where(gn[:, None] > 1e-9, gam/np.maximum(gn, 1e-12)[:, None], np.array([1., 0, 0]))
    e = np.linalg.norm(ev, axis=1); eh = ev/np.maximum(e, 1e-12)[:, None]
    var = np.arctan2((nrm*np.cross(ref, eh)).sum(1), (ref*eh).sum(1)) % (2*np.pi)
    return obl, var, e


def _initial(a, m, M_star, tilt, seed, e_init, sigma_e=.02):
    r = _rng(seed); N = len(a)
    e0 = r.rayleigh(sigma_e, N)
    if e_init is not None: e0 = np.maximum(e0, np.asarray(e_init, float))
    e0 = np.minimum(e0, .6)
    inc = r.rayleigh(SIGMA_I, N); Om, w, M = r.uniform(0, 2*np.pi, (3, N))
    tilt = np.abs(np.asarray(tilt, float)); phs = r.uniform(0, 2*np.pi, N)
    return e0, inc, Om, w, M, tilt, phs


def _spin0(nrm, tilt, phs):
    """Unit spin vectors at angle tilt from nrm, azimuth phs."""
    ref = np.array([0., 0., 1.]); out = []
    for k in range(len(tilt)):
        n = nrm[k]; u = np.cross(n, ref if abs(n[2]) < .9 else np.array([1., 0, 0])); u /= np.linalg.norm(u); w = np.cross(n, u)
        out.append(np.cos(tilt[k])*n + np.sin(tilt[k])*(np.cos(phs[k])*u + np.sin(phs[k])*w))
    return np.array(out)


# ---------------------------------------------------------------- tides (constant time lag)
AU_M = 1.495978707e11; MSUN_KG = 1.989e30; YR_SI = 3.15576e7
M_EARTH = 3.003e-6; R_EARTH = 6.371e6/AU_M; DAY_YR = 86164.1/YR_SI       # sidereal day in yr
TIDE_K2, TIDE_DT, TIDE_C = .3, 600./YR_SI, .33
MOON_M, MOON_A = M_EARTH/81.3, 60.27*R_EARTH

def _f(e):
    e2 = e*e; b = 1 - e2
    f1 = 1 + 31/2*e2 + 255/8*e2**2 + 185/16*e2**3 + 25/64*e2**4; f2 = 1 + 15/2*e2 + 45/8*e2**2 + 5/16*e2**3
    f3 = 1 + 15/4*e2 + 15/8*e2**2 + 5/64*e2**3; f4 = 1 + 3/2*e2 + 1/8*e2**2; f5 = 1 + 3*e2 + 3/8*e2**2
    return dict(Na=f1/b**7.5, N=f2/b**6, Ne=f3/b**6.5, Oe=f4/b**5, O=f5/b**4.5)

class Tides:
    """Per-system tidal state: spin rates omega (rad/yr), optional satellites (mass, semi-major axis)."""
    def __init__(self, a, m, M_star, spin, moon=None, k2=TIDE_K2, dt_lag=TIDE_DT, C=TIDE_C):
        self.a = np.asarray(a, float); self.m = np.asarray(m, float); self.M = float(M_star); N = len(self.a)
        self.R = R_EARTH*(np.maximum(self.m, 1e-9)/M_EARTH)**.27; self.k2, self.dtl, self.C = k2, dt_lag, C
        sp = np.ones(N) if spin is None else np.maximum(np.asarray(spin, float), .05)
        self.w0 = 2*np.pi/DAY_YR*sp; self.w = self.w0.copy(); self.n = 2*np.pi*np.sqrt((self.M + self.m)/self.a**3)
        self.has_moon = moon is not None
        mm = np.zeros(N) if moon is None else np.asarray(moon[0], float); am = np.ones(N) if moon is None else np.asarray(moon[1], float)
        self.mm, self.am = mm, am.astype(float).copy()
        self.lam0 = 2.17                                                # Earth: lunar/solar precession torque ratio
    def K(self): return 1.5*self.k2*self.dtl*G*self.M**2/self.R*(self.R/self.a)**6*self.n**2
    def lam(self): return np.where(self.mm > 0, (self.mm/self.am**3)/(self.M/self.a**3), 0.)
    def alpha_fac(self):
        f = self.w/self.w0
        return f*(1 + self.lam())/(1 + self.lam0) if self.has_moon else f
    def gamma_e(self, cos_eps):
        """Linear e-damping rate (1/yr, >0 damps) at the current spin and obliquity (f(e) -> 1)."""
        return -11*self.a*self.K()/(2*G*self.M*self.m)*(cos_eps*self.w/self.n - 18/11)
    def moon_torque(self):
        nm = np.where(self.mm > 0, np.sqrt(G*(self.m + self.mm)/np.maximum(self.am, 1e-12)**3), 0.)
        return np.where(self.mm > 0, 3*self.k2*G*self.mm**2*self.R**5*(self.w - nm)*self.dtl/np.maximum(self.am, 1e-12)**6, 0.)
    def step(self, h, s, nrm, e):
        """Advance omega (and moons) by h yr; returns s rotated by the tidal obliquity change.  Stiff-safe: omega relaxes
        exactly (exponential) to its instantaneous equilibrium and eps follows tan(eps/2) *= exp(g dt) (exact for
        d eps/dt = g sin eps); the coupling is operator-split over <= 8 substeps (|g| dt, a dt <= 0.5 when resolvable)."""
        f = _f(np.minimum(np.asarray(e, float), .9)); ce = np.clip((s*nrm).sum(1), -1, 1); eps = np.arccos(ce)
        K = self.K(); I = self.C*self.m*self.R**2; c = np.cos(eps); a_s = K/(I*self.n)*(1 + c*c)*f["O"]
        g = K/(I*self.w)*(c*f["O"]*self.w/self.n - 2*f["N"])
        nsub = int(np.clip(np.ceil(np.max(np.maximum(np.abs(g), a_s))*h/.5), 1, 8)); hs = h/nsub   # exact sub-solutions: stable for any hs
        for _ in range(nsub):
            c = np.cos(eps); a_s = K/(I*self.n)*(1 + c*c)*f["O"]
            if self.has_moon:
                nm = np.where(self.mm > 0, np.sqrt(G*(self.m + self.mm)/np.maximum(self.am, 1e-12)**3), 0.)
                bm = np.where(self.mm > 0, 3*self.k2*G*self.mm**2*self.R**5*self.dtl/np.maximum(self.am, 1e-12)**6/I, 0.)
                T = bm*I*(self.w - nm)
                am = self.am + hs*2*T*np.sqrt(self.am/(G*(self.m + self.mm)))/np.maximum(self.mm, 1e-30)
                self.am = np.where(self.mm > 0, np.maximum(am, 3*self.R), self.am)
            else: nm = bm = np.zeros_like(self.w)
            atot = a_s + bm; weq = (K/I*2*c*f["N"] + bm*nm)/np.maximum(atot, 1e-300)
            self.w = np.maximum(weq + (self.w - weq)*np.exp(-atot*hs), .05*self.n)
            g = K/(I*self.w)*(c*f["O"]*self.w/self.n - 2*f["N"])
            eps = 2*np.arctan(np.tan(np.clip(eps, 1e-12, np.pi - 1e-12)/2)*np.exp(np.clip(g*hs, -50, 50)))
        u = s - nrm*ce[:, None]; un = np.linalg.norm(u, axis=1); ok = un > 1e-12
        u = np.where(ok[:, None], u/np.maximum(un, 1e-12)[:, None], 0.)
        s2 = nrm*np.cos(eps)[:, None] + u*np.sin(eps)[:, None]
        return np.where(ok[:, None], s2, s)
    def state(self): return dict(omega_rel=(self.w/self.w0).tolist(), a_moon=self.am.tolist(), m_moon=self.mm.tolist())


def _moon_arg(moon, N):
    if moon is None or moon is False: return None
    if moon is True: return (np.full(N, MOON_M), np.full(N, MOON_A))
    return moon                                                        # explicit (masses, semi-major axes)


# ---------------------------------------------------------------- N-body (WH, DH coordinates)
class NBodySystem:
    """Shared integrator for one stellar system; drivers(...) gives per-planet driver objects."""
    def __init__(self, a, m, M_star, tilt, spin=None, seed=0, e_init=None, gr=True, dt=None, horizon=HORIZON, rec_dt=REC_DT, tides=False, moon=None,
                 moon_bodies=False):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.N = N; self.M = float(M_star); m_pl = m
        self.sig = (N, round(float(M_star), 9), tuple(np.round(a, 9)), tuple(np.round(np.abs(tilt), 9)), int(seed))
        moon = _moon_arg(moon, N); self.tides = Tides(a, m, M_star, spin, moon) if tides else None
        if tides: self.sig = self.sig + ("tides", moon is not None)
        self.mb = bool(moon_bodies and moon is not None)
        if self.mb:
            if not tides: raise ValueError("moon_bodies needs tides=True (moon state lives in Tides)")
            self.sig = self.sig + ("moonp",); m = m + np.asarray(moon[0], float)
        if horizon != HORIZON: self.sig = self.sig + (("hz", float(horizon)),)
        self.m = m
        e0, inc, Om, w, M0, tilt, phs = _initial(a, m, M_star, tilt, seed, e_init)
        self.mu = G*self.M*np.ones(N)                    # DH Kepler part uses the star's mass only
        x, vh = elements_to_xv(G*(self.M + m), a, e0, inc, Om, w, M0)
        Mt = self.M + m.sum(); vb = vh - (m[:, None]*vh).sum(0)/Mt
        self.x, self.v = x, vb
        self.dt = float(dt or (2*np.pi*np.sqrt(a.min()**3/(G*self.M)))/20.)
        self.gr = bool(gr); self.t = 0.; self.horizon = float(horizon); self.rec_dt = float(rec_dt)
        self.alpha = prec_const(a, M_star, spin)
        ev, nrm, _ = xv_to_evec(G*(self.M + m), x, vh); self.s = _spin0(nrm, tilt, phs)
        if self.mb: self._moon_init(m_pl, moon, nrm, seed)
        self.rec_t = [0.]; self.rec = [self._sample()]
        self.fallback = SecularSystem(a, m_pl, M_star, tilt, spin, seed, e_init, tides=tides, moon=moon)
        self.E0 = self.energy()

    def vhelio(self): return self.v + (self.m[:, None]*self.v).sum(0)/self.M

    # ---- moon particles (relative planet-moon orbit; see module doc)
    def _moon_init(self, m_pl, moon, nrm, seed):
        self.m_pl = np.asarray(m_pl, float); self.mm = np.asarray(moon[0], float); self.mu_m = G*(self.m_pl + self.mm)
        rng = np.random.default_rng([int(seed), 4411]); N = self.N
        inc = np.radians(5.)*np.ones(N); Om = rng.uniform(0, 2*np.pi, N); w = rng.uniform(0, 2*np.pi, N); M0 = rng.uniform(0, 2*np.pi, N)
        rho, u = elements_to_xv(self.mu_m, self.tides.am, np.full(N, .05), inc, Om, w, M0)
        # rotate from the ecliptic frame into each planet's orbit frame (z -> orbit normal)
        R = [self._frame(n) for n in nrm]
        self.rho = np.array([R[i] @ rho[i] for i in range(N)]); self.u = np.array([R[i] @ u[i] for i in range(N)])
        self.am_ref = self.tides.am.copy()
    @staticmethod
    def _frame(n):
        n = n/np.linalg.norm(n); t = np.cross([0., 0., 1.], n); st = np.linalg.norm(t)
        if st < 1e-12: return np.eye(3)
        t /= st; ct = n[2]; K = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        return np.eye(3) + st*K + (1 - ct)*K @ K
    def _moon_acc(self, X, rho):
        fm = (self.mm/(self.m_pl + self.mm))[:, None]; fp = 1 - fm
        rp = X - fm*rho; rm = X + fp*rho
        return -G*self.M*(rm/((rm*rm).sum(1)**1.5)[:, None] - rp/((rp*rp).sum(1)**1.5)[:, None])
    def _moon_step(self, X0, X1, h):
        Pm = np.min(2*np.pi*np.sqrt(self.tides.am**3/self.mu_m)); k = int(max(np.ceil(h/(Pm/MOON_SUB)), 1)); hs = h/k   # shortest moon period sets the sub-step
        rho, u = self.rho, self.u
        for j in range(k):
            Xa = X0 + (X1 - X0)*(j + .5)/k
            u = u + .5*hs*self._moon_acc(Xa, rho); rho, u = kepler_drift(self.mu_m, rho, u, hs); u = u + .5*hs*self._moon_acc(Xa, rho)
        self.rho, self.u = rho, u
    def _moon_rescale(self):
        f = self.tides.am/self.am_ref
        if np.any(np.abs(f - 1) > 1e-9): self.rho = self.rho*f[:, None]; self.u = self.u/np.sqrt(f)[:, None]; self.am_ref = self.tides.am.copy()
    def moon_normal(self):
        L = np.cross(self.rho, self.u); return L/np.linalg.norm(L, axis=1)[:, None]
    def moon_state(self):
        """Osculating planetocentric moon elements (a in AU, e, inclination to the planet's orbit plane in deg)."""
        ev, nm, _ = xv_to_evec(self.mu_m, self.rho, self.u); _, nrm, _ = xv_to_evec(G*(self.M + self.m), self.x, self.vhelio())
        r = np.linalg.norm(self.rho, axis=1); v2 = (self.u*self.u).sum(1); a = 1/(2/r - v2/self.mu_m)
        return dict(a=a, e=np.linalg.norm(ev, axis=1), inc_deg=np.degrees(np.arccos(np.clip((nm*nrm).sum(1), -1, 1))))

    def _sample(self):
        ev, nrm, _ = xv_to_evec(G*(self.M + self.m), self.x, self.vhelio())
        obl, var, e = _climatic(self.s, nrm, ev); return np.stack([e, var, obl])

    def accel(self, x):
        d = x[None, :, :] - x[:, None, :]; r2 = (d*d).sum(-1); np.fill_diagonal(r2, 1.)
        inv3 = r2**-1.5; np.fill_diagonal(inv3, 0.)
        acc = G*(inv3[:, :, None]*d*self.m[None, :, None]).sum(1)
        if self.gr:
            r2s = (x*x).sum(1); acc -= 6*(G*self.M)**2/C_AUYR**2*x/(r2s**2)[:, None]
        return acc

    def energy(self):
        vh = self.vhelio(); r = np.linalg.norm(self.x, axis=1)
        E = (.5*self.m*(self.v*self.v).sum(1)).sum() + .5*((self.m[:, None]*self.v).sum(0)**2).sum()/self.M - (G*self.M*self.m/r).sum()
        d = self.x[None] - self.x[:, None]; rr = np.sqrt((d*d).sum(-1)); iu = np.triu_indices(self.N, 1)
        return float(E - (G*self.m[iu[0]]*self.m[iu[1]]/rr[iu]).sum())

    def _step(self, h):
        x, v = self.x, self.v
        v = v + .5*h*self.accel(x)
        x = x + .5*h*(self.m[:, None]*v).sum(0)/self.M
        x, v = kepler_drift(self.mu, x, v, h)
        x = x + .5*h*(self.m[:, None]*v).sum(0)/self.M
        v = v + .5*h*self.accel(x)
        self.x, self.v = x, v

    def advance(self, t_to, spin=True):
        t_to = min(float(t_to), self.horizon); nxt = self.rec_t[-1] + self.rec_dt
        while self.t < t_to - 1e-12:
            h = self.dt; X0 = self.x; self._step(h); self.t += h
            if self.mb: self._moon_step(X0, self.x, h)
            if spin:
                ev, nrm, _ = xv_to_evec(G*(self.M + self.m), self.x, self.vhelio())
                if self.mb:                                                  # solar share about n, lunar share about the moon's own normal
                    T = self.tides; a_sun = self.alpha/(1 + T.lam0)*(T.w/T.w0)
                    self.s = _spin_rotate(self.s, nrm, -a_sun*(self.s*nrm).sum(1)*h)
                    nmn = self.moon_normal(); self.s = _spin_rotate(self.s, nmn, -a_sun*T.lam()*(self.s*nmn).sum(1)*h)
                else:
                    al = self.alpha if self.tides is None else self.alpha*self.tides.alpha_fac()
                    self.s = _spin_rotate(self.s, nrm, -al*(self.s*nrm).sum(1)*h)
                if self.tides is not None: self.s = self.tides.step(h, self.s, nrm, np.linalg.norm(ev, axis=1))
                if self.mb: self._moon_rescale()
                self.s /= np.linalg.norm(self.s, axis=1)[:, None]
            if self.t >= nxt: self.rec_t.append(self.t); self.rec.append(self._sample()); nxt += self.rec_dt

    def at(self, idx, t):
        if t > self.horizon: return self.fallback.at(idx, t)
        if t > self.rec_t[-1]: self.advance(t + self.rec_dt)
        k = int(np.clip(np.searchsorted(self.rec_t, t), 0, len(self.rec_t) - 1)); e, var, obl = self.rec[k][:, idx]
        return dict(e=float(min(e, .9)), varpi=float(var), obliquity=float(obl))

    def drivers(self, ids): return {pid: _Driver(self, i, "nb", pid) for i, pid in enumerate(ids)}

    def megno(self, t_end, delta0=1e-8, renorm=10):
        """MEGNO <Y> over t_end yr for this system (from its current state, on a copy). ~2: regular; >>2: chaotic."""
        import copy
        A = copy.deepcopy(self); B = copy.deepcopy(self); A.horizon = B.horizon = np.inf
        rng = np.random.default_rng(1); dx = rng.normal(size=A.x.shape); dv = rng.normal(size=A.v.shape)
        sc = np.sqrt((dx*dx).sum() + (dv*dv).sum()); B.x = A.x + delta0*dx/sc; B.v = A.v + delta0*dv/sc
        def dist(): return np.sqrt(((B.x - A.x)**2).sum() + ((B.v - A.v)**2).sum())
        n = max(int(t_end/A.dt), 10); h = A.dt; Ysum = 0.; Ys = []; ln_prev = np.log(delta0); t = 0.
        for k in range(n):
            A._step(h); B._step(h); t += h; d = dist(); ln = np.log(d)
            Ysum += t*(ln - ln_prev)                      # int t d(ln delta)
            ln_prev = ln
            if (k + 1) % renorm == 0:                      # renormalise the shadow
                f = delta0/d; B.x = A.x + (B.x - A.x)*f; B.v = A.v + (B.v - A.v)*f; ln_prev = np.log(delta0)
            Ys.append(2*Ysum/t)
        return float(np.mean(Ys))


# ---------------------------------------------------------------- Laplace-Lagrange secular (e and i)
def _laplace_b(s, j, alpha, n=256):
    psi = (np.arange(n) + .5)*2*np.pi/n
    return float(2*np.mean(np.cos(j*psi)/(1 - 2*alpha*np.cos(psi) + alpha*alpha)**s))


class SecularSystem:
    def __init__(self, a, m, M_star, tilt, spin=None, seed=0, e_init=None, gr=True, dt_spin=20., tides=False, moon=None):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.N = N
        self.sig = (N, round(float(M_star), 9), tuple(np.round(a, 9)), tuple(np.round(np.abs(np.asarray(tilt, float)), 9)), int(seed))
        moon = _moon_arg(moon, N); self.tides = Tides(a, m, M_star, spin, moon) if tides else None
        if tides: self.sig = self.sig + ("tides", moon is not None)
        e0, inc, Om, w, M0, tilt, phs = _initial(a, m, M_star, tilt, seed, e_init)
        nm = 2*np.pi*np.sqrt((M_star + m)/a**3); A = np.zeros((N, N)); B = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                if j == k: continue
                al = min(a[j], a[k])/max(a[j], a[k]); ab = al if a[k] > a[j] else 1.
                c = nm[j]/4*m[k]/(M_star + m[j])*al*ab; b1 = _laplace_b(1.5, 1, al)
                A[j, j] += c*b1; A[j, k] = -c*_laplace_b(1.5, 2, al); B[j, j] -= c*b1; B[j, k] = c*b1
            if gr: A[j, j] += 3*(G*M_star)**1.5/(C_AUYR**2*a[j]**2.5*(1 - e0[j]**2))
        self.A, self.B = A, B
        self.A0 = A
        if self.tides is not None:                                        # tidal e-damping: dz/dt = i A z - Gamma z
            self.Gam = self.tides.gamma_e(np.cos(np.abs(np.asarray(tilt, float)))); A = A + 1j*np.diag(self.Gam)
        self.gA, self.VA = np.linalg.eig(A); self.gB, self.VB = np.linalg.eig(B)
        z0 = e0*np.exp(1j*(Om + w)); zeta0 = np.sin(inc)*np.exp(1j*Om)
        self.cA = np.linalg.solve(self.VA, z0); self.cB = np.linalg.solve(self.VB, zeta0)
        self.segA = [(0., self.gA, self.VA, self.cA, self.Gam.copy())] if self.tides is not None else None   # piecewise e-modes
        self.alpha = prec_const(a, M_star, spin)
        self.dt_spin = float(dt_spin); self.ts = [0.]; self.s = [_spin0(self._normal(0.), tilt, phs)]; self.al = [self.alpha]
        self.g = np.real(self.gA)

    def _ez(self, t):
        if self.segA is None: return (self.VA*(self.cA*np.exp(1j*self.gA*t))).sum(1)
        k = max(int(np.searchsorted([s[0] for s in self.segA], t, side="right")) - 1, 0); t0, gA, VA, cA, _ = self.segA[k]
        return (VA*(cA*np.exp(1j*gA*(t - t0)))).sum(1)
    def _regam(self, t, s, nrm):
        """Re-evaluate Gamma_e from the current spin/obliquity; start a new eigen-segment if it moved > 1 %."""
        G_new = self.tides.gamma_e(np.clip((s*nrm).sum(1), -1, 1)); G_old = self.segA[-1][4]
        if np.max(np.abs(G_new - G_old)/np.maximum(np.abs(G_old), 1e-30)) < .01: return
        z = self._ez(t); gA, VA = np.linalg.eig(self.A0 + 1j*np.diag(G_new))
        self.segA.append((float(t), gA, VA, np.linalg.solve(VA, z), G_new.copy())); self.Gam = G_new
    def _normal(self, t):
        zeta = (self.VB*(self.cB*np.exp(1j*self.gB*t))).sum(1); si = np.clip(np.abs(zeta), 0, .999); Om = np.angle(zeta); ci = np.sqrt(1 - si*si)
        return np.stack([si*np.sin(Om), -si*np.cos(Om), ci], 1)
    def _evec(self, t):
        z = self._ez(t); n = self._normal(t); e = np.minimum(np.abs(z), .9); pv = np.angle(z)
        v = np.stack([np.cos(pv), np.sin(pv), np.zeros_like(pv)], 1)
        v = v - n*(n*v).sum(1)[:, None]; v /= np.linalg.norm(v, axis=1)[:, None]          # project into the orbit plane
        return e[:, None]*v, n

    def _spin_to(self, t):
        while self.ts[-1] < t:
            t0 = self.ts[-1]; h = self.dt_spin; s = self.s[-1]; al = self.al[-1]
            nm = self._normal(t0 + .5*h)                                   # midpoint normal (2nd order)
            s = _spin_rotate(s, nm, -al*(s*nm).sum(1)*h); s /= np.linalg.norm(s, axis=1)[:, None]
            if self.tides is not None:
                s = self.tides.step(h, s, nm, np.abs(self._ez(t0 + .5*h))); al = self.alpha*self.tides.alpha_fac()
                if (t0 + h) // GAM_DT > t0 // GAM_DT: self._regam(t0 + h, s, self._normal(t0 + h))
            self.ts.append(t0 + h); self.s.append(s); self.al.append(al)
        k = int(np.searchsorted(self.ts, t)); k = min(max(k, 0), len(self.ts) - 1)
        if k > 0 and self.ts[k] > t:                                      # partial step from the previous sample
            s = self.s[k - 1]; h = t - self.ts[k - 1]; nm = self._normal(self.ts[k - 1] + .5*h)
            return _spin_rotate(s, nm, -self.al[k - 1]*(s*nm).sum(1)*h)
        return self.s[k]

    def at(self, idx, t):
        t = max(float(t), 0.); s = self._spin_to(t); ev, n = self._evec(t)     # spin first: it advances the tidal e-segments
        obl, var, e = _climatic(s, n, ev)
        return dict(e=float(e[idx]), varpi=float(var[idx]), obliquity=float(obl[idx]))

    def drivers(self, ids): return {pid: _Driver(self, i, "sec", pid) for i, pid in enumerate(ids)}


class _Driver:
    def __init__(self, sysm, idx, tag, pid): self.sys, self.idx, self.tag, self.pid = sysm, idx, tag, pid
    def key(self): return ("solver_orbit", self.tag, self.idx) + self.sys.sig + ("so2",)
    def at(self, t): return self.sys.at(self.idx, float(t))


class FixedOrbit:
    def __init__(self, e, tilt): self.e, self.tilt = float(e), float(abs(tilt))
    def key(self): return ("solver_orbit", "off", round(self.e, 9), round(self.tilt, 9))
    def at(self, t): return dict(e=self.e, varpi=0., obliquity=self.tilt)


def tide_kw(mode):
    """solvers["orbit_tides"] -> drivers_from_system kwargs: None/False -> {} (default, keys unchanged); "on"/True -> tides;
    "moon" -> tides + each planet's largest system moon (Earth-like Moon if it has none)."""
    if not mode: return {}
    if mode in (True, "on", "tides"): return dict(tides=True)
    if mode == "moon": return dict(tides=True, moon="system")              # N-body: moon particles (moon_bodies default)
    if mode == "moon-ring": return dict(tides=True, moon="system", moon_bodies=False)
    raise ValueError(f"orbit_tides must be on|moon|moon-ring, got {mode!r}")


def drivers_from_system(S, habs, seed, solver="secular", **kw):
    """{planet_id: driver} like leaves.orbit_climate.drivers_from_system, for solver in {"nbody", "secular", "off"}."""
    kind = np.asarray(S["kind"]); Ms = float(np.asarray(S["m"])[kind == 0].sum())
    ids = [h.planet_id for h in habs]
    if solver == "off": return {h.planet_id: FixedOrbit(h.e, h.tilt) for h in habs}
    if not habs: return {}
    a = [h.a for h in habs]; m = [h.m for h in habs]; tilt = [h.tilt for h in habs]; spin = [h.spin for h in habs]; e = [h.e for h in habs]
    if kw.get("moon") == "system": kw["moon"] = system_moons(S, habs)
    if solver == "nbody":
        if kw.get("moon") is not None: kw.setdefault("moon_bodies", True)
        return NBodySystem(a, m, Ms, tilt, spin, seed=seed*1009, e_init=e, **kw).drivers(ids)
    kw.pop("moon_bodies", None); kw.pop("horizon", None)
    if solver == "secular": return SecularSystem(a, m, Ms, tilt, spin, seed=seed*1009, e_init=e, **kw).drivers(ids)
    raise ValueError(f"orbit_solver must be nbody|secular|off, got {solver!r}")


def system_moons(S, habs):
    """(masses, semi-major axes) of each habitat's largest moon in S ('moon{p}.k' for planet 'planet{p}', habitat
    planet_id = its index in S); an Earth-like Moon scaled to the planet (M_m/m = 1/81.3, a_m ~ m^1/3) if it has none."""
    names = [str(x) for x in S.get("name", [])]; x = np.asarray(S["x"]); mm = np.asarray(S["m"]); out_m, out_a = [], []
    for h in habs:
        pi = int(h.planet_id); pn = names[pi] if pi < len(names) else ""
        ks = [i for i, nme in enumerate(names) if pn.startswith("planet") and nme.startswith("moon" + pn[6:] + ".")]
        if not ks: out_m.append(MOON_M*h.m/M_EARTH); out_a.append(MOON_A*(h.m/M_EARTH)**(1/3)); continue
        k = max(ks, key=lambda i: mm[i]); out_m.append(float(mm[k])); out_a.append(float(np.linalg.norm(x[k] - x[pi])))
    return (np.array(out_m), np.array(out_a))
