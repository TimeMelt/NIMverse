"""Orbit -> insolation geometry for the latitudinal climate (leaves.world.LatClimate).
insolation(x, obliquity, e, varpi): exact daily-mean top-of-atmosphere insolation (Berger-style geometry) on a grid of
solar longitudes, time-weighted by Kepler's 2nd law (dt ~ r^2 dlambda), returning per-band annual mean and seasonal
half-range in units of the planet's mean-distance flux S0 = L/a^2 (global annual mean = 1/(4 sqrt(1-e^2)))."""
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
    inv_r = (1 + e*np.cos(lam - varpi))/(1 - e*e)            # a/r
    q = daily_mean(phi, delta)*inv_r[None, :]**2             # (bands, lambda)
    w = inv_r**-2; w = w/w.sum()                             # time weight ~ r^2
    return dict(mean=(q*w).sum(1), amp=.5*(q.max(1) - q.min(1)), qmax=q.max(1), qmin=q.min(1))

# ---------------------------------------------------------------------------------------------------------------------------------
# physics15 pass 4: cheap secular orbit driver (Milankovitch-style forcing for LatClimate).  This is NOT the N-body leapfrog
# (sim2 / demos): it is first-order Laplace-Lagrange secular theory on the habitat list, evaluated analytically at any t.

def laplace_b(s, j, alpha, n=256):
    """Laplace coefficient b_s^(j)(alpha) = (1/pi) int_0^{2pi} cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s dpsi."""
    psi = (np.arange(n) + .5)*2*np.pi/n
    return float(2*np.mean(np.cos(j*psi)/(1 - 2*alpha*np.cos(psi) + alpha*alpha)**s))

OBL_AMP = np.radians(1.2)     # obliquity oscillation amplitude (Earth ~ 1.2 deg)
P_OBL0, P_PREC0 = 41000., 25800.   # yr at Earth spin; both scale ~ 1/spin (precession constant ~ spin rate)

class SecularOrbit:
    """Secular e / varpi (Laplace-Lagrange eccentricity-vector modes) + obliquity cycle + axial precession for one habitat.
    at(t) -> dict(e, varpi (perihelion from the vernal equinox, rad), obliquity (rad)) for LatClimate(orbit=...).
    a [AU], m [Msun] of all planets, M_star [Msun]; e0 seeds the free eccentricities (make_system orbits are ~circular, so
    a Rayleigh(sigma_e) draw stands in for primordial excitation); mode frequencies g [rad/yr] from the A matrix."""
    def __init__(self, a, m, M_star, idx, tilt, spin=1., seed=0, sigma_e=.02, e_init=None):
        a = np.asarray(a, float); m = np.asarray(m, float); N = len(a); self.idx = int(idx); r = np.random.default_rng([int(seed), 1504])
        e0 = r.rayleigh(sigma_e, N); w0 = r.uniform(0, 2*np.pi, N)
        if e_init is not None: e0 = np.maximum(e0, np.asarray(e_init, float))     # measured osculating e wins when larger
        nmot = 2*np.pi*np.sqrt(M_star/a**3); A = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                if j == k: continue
                al = min(a[j], a[k])/max(a[j], a[k]); ab = al if a[k] > a[j] else 1.
                c = nmot[j]/4*m[k]/(M_star + m[j])*al*ab
                A[j, j] += c*laplace_b(1.5, 1, al); A[j, k] = -c*laplace_b(1.5, 2, al)
        g, V = np.linalg.eig(A) if N else (np.zeros(0), np.zeros((0, 0)))
        z0 = e0*np.exp(1j*w0); self.g = np.real(g); self.V = np.real(V); self.c = np.linalg.solve(self.V, z0) if N else z0
        self.tilt = float(abs(tilt)); sp = max(float(spin), .05); self.P_obl = P_OBL0/sp; self.P_prec = P_PREC0/sp
        self.ph = r.uniform(0, 2*np.pi, 2); self._key = (round(float(a[idx]), 9), int(seed), round(self.tilt, 9), round(sp, 9), N)
    def key(self): return self._key
    def e_varpi(self, t):
        if not len(self.g): return 0., 0.
        z = (self.V[self.idx]*self.c*np.exp(1j*self.g*t)).sum(); return float(min(abs(z), .9)), float(np.angle(z))
    def at(self, t):
        e, w = self.e_varpi(t); psi = 2*np.pi*t/self.P_prec + self.ph[0]
        obl = self.tilt + OBL_AMP*np.sin(2*np.pi*t/self.P_obl + self.ph[1])
        return dict(e=e, varpi=float((w + psi) % (2*np.pi)), obliquity=float(max(0., obl)))

def drivers_from_system(S, habs, seed):
    """{planet_id: SecularOrbit} for the habitat list of a make_system dict (all planets perturb each other)."""
    kind = np.asarray(S["kind"]); Ms = float(np.asarray(S["m"])[kind == 0].sum())
    a = [h.a for h in habs]; m = [h.m for h in habs]
    return {h.planet_id: SecularOrbit(a, m, Ms, i, h.tilt, h.spin, seed=seed*1009 + i, e_init=[x.e for x in habs]) for i, h in enumerate(habs)}
