"""Orbit -> insolation geometry for the latitudinal climate (leaves.world.LatClimate).
insolation(x, obliquity, e, varpi): exact daily-mean top-of-atmosphere insolation (Berger-style geometry) on a grid of
solar longitudes, time-weighted by Kepler's 2nd law (dt ~ r^2 dlambda), returning per-band annual mean and seasonal
half-range in units of the planet's mean-distance flux S0 = L/a^2 (global annual mean = 1/(4 sqrt(1-e^2))).

physics_real: SecularOrbit adds a weak 2nd-order / near-resonant eccentricity forcing on top of first-order
Laplace-Lagrange, and obliquity from a spin-axis precession model (torque ~ cos ε sin ε) with period from spin/C21,
not a fixed ±1.2° sinusoid. life_loop(secular=False) still freezes the orbit for bit-stable tests."""
import numpy as np

NLAM = 72   # solar-longitude samples per orbit

def daily_mean(phi, delta):
    """Daily-mean insolation / S (S = flux at the current distance) for latitudes phi[...,None] and declinations delta[None,...]."""
    phi, delta = np.broadcast_arrays(phi[:, None], delta[None, :])
    c = np.clip(-np.tan(phi)*np.tan(delta), -1., 1.); h0 = np.arccos(c)
    return (h0*np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.sin(h0))/np.pi

def insolation(x, obliquity, e=0., varpi=0., nlam=NLAM):
    """x = sin(latitude) band centres. obliquity (rad), e, varpi = longitude of perihelion measured from the vernal equinox (rad).
    Returns dict(mean=annual mean q_i, amp=seasonal half-range, qmax, qmin) in units of S0 = L/a^2 (Lsun/AU^2 -> x1361 W/m^2)."""
    x = np.asarray(x, float); phi = np.arcsin(np.clip(x, -1, 1)); e = float(min(max(e, 0.), .95))
    lam = (np.arange(nlam) + .5)*2*np.pi/nlam; delta = np.arcsin(np.sin(abs(obliquity))*np.sin(lam))
    inv_r = (1 + e*np.cos(lam - varpi))/(1 - e*e)
    q = daily_mean(phi, delta)*inv_r[None, :]**2
    w = inv_r**-2; w = w/w.sum()
    return dict(mean=(q*w).sum(1), amp=.5*(q.max(1) - q.min(1)), qmax=q.max(1), qmin=q.min(1))

def laplace_b(s, j, alpha, n=256):
    """Laplace coefficient b_s^(j)(alpha) = (1/pi) int_0^{2pi} cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s dpsi."""
    psi = (np.arange(n) + .5)*2*np.pi/n
    return float(2*np.mean(np.cos(j*psi)/(1 - 2*alpha*np.cos(psi) + alpha*alpha)**s))

P_PREC0 = 25800.   # yr at Earth spin; precession constant alpha ~ 2pi/(P_PREC0/spin) (C21 absorbed into calibration)
I_ORB = 0.02       # forced orbital inclination proxy (rad) for spin-axis torque
E2_COUP = 0.12     # amplitude of 2nd-order e_i e_j coupling relative to |c_i c_j|
RES_W = 0.06       # near-resonant forced-e width in period-ratio space

class SecularOrbit:
    """Secular e/varpi (Laplace-Lagrange + weak 2nd-order / near-resonant forcing) + spin-axis obliquity.
    at(t) -> dict(e, varpi, obliquity). life_loop(secular=False) skips this driver entirely."""
    def __init__(self, a, m, M_star, idx, tilt, spin=1., seed=0, sigma_e=.02, e_init=None):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.idx = int(idx)
        r = np.random.default_rng([int(seed), 1504])
        e0 = r.rayleigh(sigma_e, N); w0 = r.uniform(0, 2*np.pi, N)
        if e_init is not None: e0 = np.maximum(e0, np.asarray(e_init, float))
        nmot = 2*np.pi*np.sqrt(M_star/np.maximum(a, 1e-9)**3); A = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                if j == k: continue
                al = min(a[j], a[k])/max(a[j], a[k]); ab = al if a[k] > a[j] else 1.
                c = nmot[j]/4*m[k]/(M_star + m[j])*al*ab
                A[j, j] += c*laplace_b(1.5, 1, al); A[j, k] = -c*laplace_b(1.5, 2, al)
        g, V = np.linalg.eig(A) if N else (np.zeros(0), np.zeros((0, 0)))
        z0 = e0*np.exp(1j*w0); self.g = np.real(g); self.V = np.real(V)
        self.c = np.linalg.solve(self.V, z0) if N else z0
        self.tilt = float(abs(tilt)); sp = max(float(spin), .05)
        # Spin-axis precession constant alpha (rad/yr); psi_dot = alpha cos ε. C21 folded into P_PREC0.
        self.alpha = 2*np.pi/(P_PREC0/sp)
        # Dominant secular frequency as nodal proxy (mean |g|); torque frequency ~ alpha - s
        self.s = float(np.mean(np.abs(self.g))) if N else 2*np.pi/41000.*sp
        self.ph = r.uniform(0, 2*np.pi, 4)
        # Near-resonant forced eccentricity from adjacent planets (period ratio near p:p+1)
        self.res = []
        for k in range(N):
            if k == self.idx: continue
            ratio = nmot[self.idx]/max(nmot[k], 1e-18)
            if ratio < 1: ratio = 1/ratio; n_in, n_out = nmot[self.idx], nmot[k]
            else: n_in, n_out = nmot[k], nmot[self.idx]
            # nearest first-order MMR p:p+1
            p = max(1, int(round(n_in/(n_out - n_in)))) if n_out != n_in else 1
            target = (p + 1)/p; delta = abs(n_in/n_out - p/(p + 1.))
            if delta < RES_W:
                amp = 0.015*(m[k]/(M_star + m[self.idx]))/max(delta, .005)*min(a[self.idx], a[k])/max(a[self.idx], a[k])
                amp = float(min(amp, .08)); om = (p + 1)*n_out - p*n_in
                self.res.append((amp, om, float(r.uniform(0, 2*np.pi))))
        self.a = a; self.m = m; self.nmot = nmot
        self._key = (round(float(a[idx]), 9), int(seed), round(self.tilt, 9), round(sp, 9), N, "pr1")
    def key(self): return self._key
    def e_varpi(self, t):
        if not len(self.g): return 0., 0.
        z = (self.V[self.idx]*self.c*np.exp(1j*self.g*t)).sum()
        # 2nd-order coupling among mode amplitudes at g_i ± g_j
        for i in range(len(self.g)):
            for j in range(i + 1, len(self.g)):
                coup = E2_COUP*self.c[i]*self.c[j]
                z += coup*np.exp(1j*((self.g[i] + self.g[j])*t + self.ph[2]))
                z += .5*coup*np.exp(1j*((self.g[i] - self.g[j])*t + self.ph[3]))
        for amp, om, ph in self.res:
            z += amp*np.exp(1j*(om*t + ph))
        return float(min(abs(z), .9)), float(np.angle(z))
    def obliquity(self, t):
        """Spin-axis precession: torque ~ alpha cosε sinε against orbital plane (forced i = I_ORB).
        Linearised: ε = ε0 + (alpha/(alpha - s))*I_ORB*sin((alpha - s)t + φ); amp uses |sinε cosε| torque factor."""
        eps0 = self.tilt
        denom = self.alpha - self.s
        if abs(denom) < 1e-10: denom = 1e-10*np.sign(denom or 1.)
        torque = max(abs(np.sin(eps0)*np.cos(eps0))*2, .15)   # ~|sin 2ε|; floor so near-equatorial spins still wobble
        amp = min(abs(self.alpha/denom)*I_ORB*torque, np.radians(3.5))
        obl = eps0 + amp*np.sin(denom*t + self.ph[1])
        return float(max(0., min(obl, np.radians(60.))))
    def at(self, t):
        e, w = self.e_varpi(t)
        # climatic precession: perihelion relative to moving equinox (psi_dot = alpha cos ε)
        eps = self.obliquity(t)
        psi = self.alpha*np.cos(eps)*t + self.ph[0]
        return dict(e=e, varpi=float((w + psi) % (2*np.pi)), obliquity=eps)

def drivers_from_system(S, habs, seed):
    """{planet_id: SecularOrbit} for the habitat list of a make_system dict (all planets perturb each other)."""
    kind = np.asarray(S["kind"]); Ms = float(np.asarray(S["m"])[kind == 0].sum())
    a = [h.a for h in habs]; m = [h.m for h in habs]
    return {h.planet_id: SecularOrbit(a, m, Ms, i, h.tilt, h.spin, seed=seed*1009 + i, e_init=[x.e for x in habs]) for i, h in enumerate(habs)}
