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
Fidelity / gaps: point masses (no tides, no J2 of the planet beyond alpha, no satellites); WH accuracy ~ (m/M)(dt/P)^2,
dt = P_inner/20; close encounters are not regularised (MEGNO flags them); LL theory is linear in e, i (good for e,i < ~0.1)."""
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


# ---------------------------------------------------------------- N-body (WH, DH coordinates)
class NBodySystem:
    """Shared integrator for one stellar system; drivers(...) gives per-planet driver objects."""
    def __init__(self, a, m, M_star, tilt, spin=None, seed=0, e_init=None, gr=True, dt=None, horizon=HORIZON, rec_dt=REC_DT):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.N = N; self.M = float(M_star); self.m = m
        self.sig = (N, round(float(M_star), 9), tuple(np.round(a, 9)), tuple(np.round(np.abs(tilt), 9)), int(seed))
        e0, inc, Om, w, M0, tilt, phs = _initial(a, m, M_star, tilt, seed, e_init)
        self.mu = G*self.M*np.ones(N)                    # DH Kepler part uses the star's mass only
        x, vh = elements_to_xv(G*(self.M + m), a, e0, inc, Om, w, M0)
        Mt = self.M + m.sum(); vb = vh - (m[:, None]*vh).sum(0)/Mt
        self.x, self.v = x, vb
        self.dt = float(dt or (2*np.pi*np.sqrt(a.min()**3/(G*self.M)))/20.)
        self.gr = bool(gr); self.t = 0.; self.horizon = float(horizon); self.rec_dt = float(rec_dt)
        self.alpha = prec_const(a, M_star, spin)
        ev, nrm, _ = xv_to_evec(G*(self.M + m), x, vh); self.s = _spin0(nrm, tilt, phs)
        self.rec_t = [0.]; self.rec = [self._sample()]
        self.fallback = SecularSystem(a, m, M_star, tilt, spin, seed, e_init)
        self.E0 = self.energy()

    def vhelio(self): return self.v + (self.m[:, None]*self.v).sum(0)/self.M

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
            h = self.dt; self._step(h); self.t += h
            if spin:
                _, nrm, _ = xv_to_evec(G*(self.M + self.m), self.x, self.vhelio())
                self.s = _spin_rotate(self.s, nrm, -self.alpha*(self.s*nrm).sum(1)*h)
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
    def __init__(self, a, m, M_star, tilt, spin=None, seed=0, e_init=None, gr=True, dt_spin=20.):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.N = N
        self.sig = (N, round(float(M_star), 9), tuple(np.round(a, 9)), tuple(np.round(np.abs(np.asarray(tilt, float)), 9)), int(seed))
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
        self.gA, self.VA = np.linalg.eig(A); self.gB, self.VB = np.linalg.eig(B)
        z0 = e0*np.exp(1j*(Om + w)); zeta0 = np.sin(inc)*np.exp(1j*Om)
        self.cA = np.linalg.solve(self.VA, z0); self.cB = np.linalg.solve(self.VB, zeta0)
        self.alpha = prec_const(a, M_star, spin)
        self.dt_spin = float(dt_spin); self.ts = [0.]; self.s = [_spin0(self._normal(0.), tilt, phs)]
        self.g = np.real(self.gA)

    def _ez(self, t): return (self.VA*(self.cA*np.exp(1j*self.gA*t))).sum(1)
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
            t0 = self.ts[-1]; h = self.dt_spin; s = self.s[-1]
            nm = self._normal(t0 + .5*h)                                   # midpoint normal (2nd order)
            s = _spin_rotate(s, nm, -self.alpha*(s*nm).sum(1)*h); s /= np.linalg.norm(s, axis=1)[:, None]
            self.ts.append(t0 + h); self.s.append(s)
        k = int(np.searchsorted(self.ts, t)); k = min(max(k, 0), len(self.ts) - 1)
        if k > 0 and self.ts[k] > t:                                      # partial step from the previous sample
            s = self.s[k - 1]; h = t - self.ts[k - 1]; nm = self._normal(self.ts[k - 1] + .5*h)
            return _spin_rotate(s, nm, -self.alpha*(s*nm).sum(1)*h)
        return self.s[k]

    def at(self, idx, t):
        t = max(float(t), 0.); ev, n = self._evec(t); s = self._spin_to(t)
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


def drivers_from_system(S, habs, seed, solver="secular", **kw):
    """{planet_id: driver} like leaves.orbit_climate.drivers_from_system, for solver in {"nbody", "secular", "off"}."""
    kind = np.asarray(S["kind"]); Ms = float(np.asarray(S["m"])[kind == 0].sum())
    ids = [h.planet_id for h in habs]
    if solver == "off": return {h.planet_id: FixedOrbit(h.e, h.tilt) for h in habs}
    if not habs: return {}
    a = [h.a for h in habs]; m = [h.m for h in habs]; tilt = [h.tilt for h in habs]; spin = [h.spin for h in habs]; e = [h.e for h in habs]
    if solver == "nbody": return NBodySystem(a, m, Ms, tilt, spin, seed=seed*1009, e_init=e, **kw).drivers(ids)
    if solver == "secular": return SecularSystem(a, m, Ms, tilt, spin, seed=seed*1009, e_init=e, **kw).drivers(ids)
    raise ValueError(f"orbit_solver must be nbody|secular|off, got {solver!r}")
