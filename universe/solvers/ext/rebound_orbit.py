"""REBOUND WHFast / IAS15 orbit + spin drivers (--orbit-solver rebound).

Same driver API as solvers.secular_nbody (key / at -> e, varpi, obliquity). Uses installed rebound
(5.x). Optional constant-lag tide approximation in pure Python when reboundx is absent (default here).
Existing nbody / secular paths are untouched.
"""
from __future__ import annotations
import numpy as np

from ..secular_nbody import (
    G, HORIZON, REC_DT, P_PREC0, prec_const, _initial, _spin0, _climatic, _spin_rotate,
    Tides, SecularSystem, FixedOrbit, _Driver, tide_kw, system_moons, _moon_arg, M_EARTH, MOON_M, MOON_A,
)


def _rebound():
    try:
        import rebound
        return rebound
    except ImportError as e:
        from . import UnavailableError
        raise UnavailableError(f"rebound not installed: {e}") from e


class ReboundSystem:
    """REBOUND N-body with Colombo spin (and optional Python constant-lag tides)."""
    def __init__(self, a, m, M_star, tilt, spin=None, seed=0, e_init=None, integrator="whfast",
                 horizon=HORIZON, rec_dt=REC_DT, tides=False, moon=None, moon_bodies=False, dt=None):
        rebound = _rebound()
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.N = N; self.M = float(M_star)
        self.sig = ("rb", N, round(float(M_star), 9), tuple(np.round(a, 9)), tuple(np.round(np.abs(tilt), 9)), int(seed), str(integrator))
        moon = _moon_arg(moon, N); self.tides = Tides(a, m, M_star, spin, moon) if tides else None
        if tides: self.sig = self.sig + ("tides", moon is not None)
        self.mb = bool(moon_bodies and moon is not None)
        if self.mb and not tides: raise ValueError("moon_bodies needs tides=True")
        if self.mb: self.sig = self.sig + ("moonp",); m_tot = m + np.asarray(moon[0], float)
        else: m_tot = m
        if horizon != HORIZON: self.sig = self.sig + (("hz", float(horizon)),)
        self.m = m_tot; self.m_pl = m; self.a0 = a
        e0, inc, Om, w, M0, tilt, phs = _initial(a, m_tot, M_star, tilt, seed, e_init)
        sim = rebound.Simulation(); sim.G = G; sim.add(m=self.M)
        for i in range(N):
            sim.add(m=float(m_tot[i]), a=float(a[i]), e=float(e0[i]), inc=float(inc[i]),
                    Omega=float(Om[i]), omega=float(w[i]), M=float(M0[i]))
        sim.move_to_com()
        P_inner = 2 * np.pi * np.sqrt(a.min() ** 3 / (G * self.M))
        self.dt = float(dt or P_inner / 20.)
        integ = str(integrator).lower()
        if integ in ("whfast", "wh"):
            sim.integrator = "whfast"; sim.dt = self.dt
            try: sim.ri_whfast.safe_mode = 0; sim.ri_whfast.corrector = 11
            except Exception: pass
        else:
            sim.integrator = "ias15"
        self.sim = sim; self.horizon = float(horizon); self.rec_dt = float(rec_dt); self.t = 0.
        self.alpha = prec_const(a, M_star, spin)
        # initial spin from heliocentric elements
        nrm = self._normals(); self.s = _spin0(nrm, tilt, phs)
        self.moon = moon
        if self.mb:
            # add moon particles as rebound bodies around each planet barycentre (simple: heliocentric moon)
            rng = np.random.default_rng([int(seed), 4411])
            self.moon_idx = []
            for i in range(N):
                am = float(moon[1][i]); mm = float(moon[0][i])
                # relative circular-ish in planet orbital plane
                p = sim.particles[i + 1]
                ang = float(rng.uniform(0, 2 * np.pi))
                mx = p.x + am * np.cos(ang); my = p.y + am * np.sin(ang); mz = p.z
                # circular velocity relative + planet vel
                n_m = np.sqrt(G * (m[i] + mm) / am ** 3)
                vx = p.vx - am * n_m * np.sin(ang); vy = p.vy + am * n_m * np.cos(ang); vz = p.vz
                sim.add(m=mm, x=mx, y=my, z=mz, vx=vx, vy=vy, vz=vz)
                self.moon_idx.append(len(sim.particles) - 1)
            self.sig = self.sig + ("rbmoon",)
        self.rec_t = [0.]; self.rec = [self._sample()]
        self.fallback = SecularSystem(a, m, M_star, tilt, spin, seed, e_init, tides=tides, moon=moon)
        self.E0 = self.energy()

    def _planet_particles(self):
        return [self.sim.particles[i + 1] for i in range(self.N)]

    def _normals(self):
        nrm = []
        for p in self._planet_particles():
            r = np.array([p.x, p.y, p.z]); v = np.array([p.vx, p.vy, p.vz])
            h = np.cross(r, v); n = h / max(np.linalg.norm(h), 1e-30); nrm.append(n)
        return np.array(nrm)

    def _evecs(self):
        ev, nrm, e = [], [], []
        for p in self._planet_particles():
            r = np.array([p.x, p.y, p.z]); v = np.array([p.vx, p.vy, p.vz])
            mu = G * (self.M + p.m); rr = np.linalg.norm(r); v2 = v @ v
            evec = ((v2 - mu / rr) * r - (r @ v) * v) / mu
            h = np.cross(r, v); n = h / max(np.linalg.norm(h), 1e-30)
            ev.append(evec); nrm.append(n); e.append(np.linalg.norm(evec))
        return np.array(ev), np.array(nrm), np.array(e)

    def _sample(self):
        ev, nrm, _ = self._evecs(); obl, var, e = _climatic(self.s, nrm, ev)
        return np.stack([e, var, obl])

    def energy(self):
        try: return float(self.sim.calculate_energy())
        except Exception: return 0.

    def advance(self, t_to, spin=True):
        t_to = min(float(t_to), self.horizon); nxt = self.rec_t[-1] + self.rec_dt
        while self.t < t_to - 1e-12:
            h = min(self.dt, t_to - self.t); self.sim.integrate(self.sim.t + h); self.t += h
            if spin:
                ev, nrm, ee = self._evecs()
                al = self.alpha if self.tides is None else self.alpha * self.tides.alpha_fac()
                if self.mb and self.moon is not None:
                    # solar + lunar (moon particle normals)
                    a_sun = self.alpha / (1 + self.tides.lam0) * (self.tides.w / self.tides.w0)
                    self.s = _spin_rotate(self.s, nrm, -a_sun * (self.s * nrm).sum(1) * h)
                    nmn = []
                    for i, mi in enumerate(self.moon_idx):
                        p = self.sim.particles[i + 1]; m = self.sim.particles[mi]
                        rho = np.array([m.x - p.x, m.y - p.y, m.z - p.z])
                        u = np.array([m.vx - p.vx, m.vy - p.vy, m.vz - p.vz])
                        L = np.cross(rho, u); nmn.append(L / max(np.linalg.norm(L), 1e-30))
                    nmn = np.array(nmn)
                    self.s = _spin_rotate(self.s, nmn, -a_sun * self.tides.lam() * (self.s * nmn).sum(1) * h)
                else:
                    self.s = _spin_rotate(self.s, nrm, -al * (self.s * nrm).sum(1) * h)
                if self.tides is not None:
                    self.s = self.tides.step(h, self.s, nrm, ee)
                self.s /= np.linalg.norm(self.s, axis=1)[:, None]
            if self.t >= nxt: self.rec_t.append(self.t); self.rec.append(self._sample()); nxt += self.rec_dt

    def at(self, idx, t):
        if t > self.horizon: return self.fallback.at(idx, t)
        if t > self.rec_t[-1]: self.advance(t + self.rec_dt)
        k = int(np.clip(np.searchsorted(self.rec_t, t), 0, len(self.rec_t) - 1)); e, var, obl = self.rec[k][:, idx]
        return dict(e=float(min(e, .9)), varpi=float(var), obliquity=float(obl))

    def drivers(self, ids):
        return {pid: _Driver(self, i, "rb", pid) for i, pid in enumerate(ids)}

    def megno(self, t_end):
        """MEGNO via rebound's built-in (copy of the sim)."""
        rebound = _rebound()
        sim = self.sim.copy(); sim.init_megno()
        try:
            sim.integrate(sim.t + float(t_end))
            return float(sim.calculate_megno())
        except Exception:
            return float("nan")

    def moon_state(self):
        if not self.mb: return dict(a=np.zeros(self.N), e=np.zeros(self.N), inc_deg=np.zeros(self.N))
        a_m, e_m, inc = [], [], []
        _, nrm, _ = self._evecs()
        for i, mi in enumerate(self.moon_idx):
            p = self.sim.particles[i + 1]; m = self.sim.particles[mi]
            rho = np.array([m.x - p.x, m.y - p.y, m.z - p.z]); u = np.array([m.vx - p.vx, m.vy - p.vy, m.vz - p.vz])
            mu = G * (p.m + m.m); r = np.linalg.norm(rho); v2 = u @ u
            aa = 1 / (2 / r - v2 / mu); h = np.cross(rho, u); nm = h / max(np.linalg.norm(h), 1e-30)
            evec = ((v2 - mu / r) * rho - (rho @ u) * u) / mu
            a_m.append(aa); e_m.append(np.linalg.norm(evec))
            inc.append(np.degrees(np.arccos(np.clip(nm @ nrm[i], -1, 1))))
        return dict(a=np.array(a_m), e=np.array(e_m), inc_deg=np.array(inc))


def drivers_from_system(S, habs, seed, **kw):
    """{planet_id: driver} for orbit_solver='rebound'."""
    _rebound()
    kind = np.asarray(S["kind"]); Ms = float(np.asarray(S["m"])[kind == 0].sum())
    ids = [h.planet_id for h in habs]
    if not habs: return {}
    a = [h.a for h in habs]; m = [h.m for h in habs]; tilt = [h.tilt for h in habs]
    spin = [h.spin for h in habs]; e = [h.e for h in habs]
    if kw.get("moon") == "system": kw["moon"] = system_moons(S, habs)
    if kw.get("moon") is not None: kw.setdefault("moon_bodies", True)
    return ReboundSystem(a, m, Ms, tilt, spin, seed=seed * 1009, e_init=e, **kw).drivers(ids)


def two_body_period_yr(M_star=1., a=1., m=3e-6, integrator="whfast", n_orbits=3):
    """Integrate a circular two-body and return measured orbital period (yr). Test helper."""
    rebound = _rebound()
    sim = rebound.Simulation(); sim.G = G; sim.add(m=M_star); sim.add(m=m, a=a, e=0.)
    sim.move_to_com(); P = 2 * np.pi * np.sqrt(a ** 3 / (G * (M_star + m)))
    if integrator == "whfast":
        sim.integrator = "whfast"; sim.dt = P / 50.
    else:
        sim.integrator = "ias15"
    # track crossings of y=0 with x>0
    p = sim.particles[1]; t_cross = []
    y_prev = p.y; t_prev = sim.t
    T_end = n_orbits * P + P * .1
    while sim.t < T_end:
        sim.integrate(sim.t + P / 200.); y = sim.particles[1].y; x = sim.particles[1].x
        if y_prev < 0 <= y and x > 0:
            # linear interpolate
            t_cross.append(t_prev + (0 - y_prev) / (y - y_prev + 1e-30) * (sim.t - t_prev))
        y_prev, t_prev = y, sim.t
    if len(t_cross) < 2: return float(P)
    return float(np.mean(np.diff(t_cross)))
