"""solvers.gr -- strong-field gravity solvers (out-of-leaf; events(gr_solver=True) / C["gr_solver"]).

Geometrized units G = c = 1 with lengths/times in units of the hole mass M unless a function says otherwise
(M_SUN_AU = GM_sun/c^2 in AU, M_SUN_S = GM_sun/c^3 in s).
  * Kerr metric in Boyer-Lindquist coordinates; timelike / null geodesics integrated from Hamilton's equations
    H = g^{mu nu} p_mu p_nu / 2 (p_t = -E, p_phi = L conserved; r, theta, p_r, p_theta evolved with scipy DOP853,
    rtol 1e-11).  The Hamiltonian constraint H = -mu^2/2 is monitored (returned as `H_drift`).
  * Closed forms (Bardeen, Press & Teukolsky 1972): horizon, ergosphere, prograde/retrograde ISCO, photon orbit,
    circular-orbit E, L, Omega = 1/(r^1.5 + a); radiative efficiency eta = 1 - E_isco (0.057 Schwarzschild, 0.42 extremal).
  * Peters (1964) orbit-averaged GW inspiral: da/dt, de/dt integrated (solve_ivp, stiff-safe) to coalescence; circular
    closed form T = 5 a^4 / (256 m1 m2 (m1+m2)) used as a test; chirp mass, GW frequency.
  * event_fields(): extra strong-field fields for leaves/events TDE / AGN / SMBH-merger log entries.
Fidelity / gaps: test-particle geodesics (no self-force, no spin-curvature coupling); vacuum Kerr (no accretion-disk
radiative transfer); Peters is leading-order (0PN radiation reaction, no ringdown/NR merger)."""
import numpy as np
from scipy.integrate import solve_ivp

M_SUN_AU = 1476.625/1.495978707e11     # GM_sun/c^2 in AU
M_SUN_S = 4.925491e-6                  # GM_sun/c^3 in s
YR_S = 3.15576e7


# ----------------------------------------------------------------- closed forms
def horizon(a): return float(1 + np.sqrt(1 - a*a))
def ergosphere(a, theta=np.pi/2): return 1 + np.sqrt(1 - a*a*np.cos(theta)**2)

def r_isco(a, prograde=True):
    z1 = 1 + (1 - a*a)**(1/3)*((1 + a)**(1/3) + (1 - a)**(1/3)); z2 = np.sqrt(3*a*a + z1*z1)
    s = -1 if prograde else 1
    return float(3 + z2 + s*np.sqrt((3 - z1)*(3 + z1 + 2*z2)))

def r_photon(a, prograde=True):
    return float(2*(1 + np.cos(2/3*np.arccos(-a if prograde else a))))

def circular(r, a=0., prograde=True):
    """Equatorial circular geodesic: dict(E, L, Omega) per unit rest mass (BPT72)."""
    s = 1 if prograde else -1; a = s*a
    den = r**.75*np.sqrt(r**1.5 - 3*np.sqrt(r) + 2*a)
    E = (r**1.5 - 2*np.sqrt(r) + a)/den; L = s*(r*r - 2*a*np.sqrt(r) + a*a)/den
    return dict(E=float(E), L=float(L), Omega=float(s/(r**1.5 + a)))

def efficiency(a, prograde=True): return float(1 - circular(r_isco(a, prograde), a, prograde)["E"])


# ----------------------------------------------------------------- Kerr geodesics (Hamiltonian, BL)
def ginv(r, th, a):
    """Inverse Kerr metric components (tt, tphi, phiphi, rr, thth)."""
    c, s = np.cos(th), np.sin(th); S = r*r + a*a*c*c; D = r*r - 2*r + a*a; A = (r*r + a*a)**2 - a*a*D*s*s
    return -A/(S*D), -2*a*r/(S*D), (D - a*a*s*s)/(S*D*s*s), D/S, 1/S

def _H(r, th, pr, pth, E, L, a):
    tt, tp, pp, rr, hh = ginv(r, th, a)
    return .5*(tt*E*E - 2*tp*E*L + pp*L*L + rr*pr*pr + hh*pth*pth)

def _rhs(lam, y, E, L, a):
    t, r, th, ph, pr, pth = y; tt, tp, pp, rr, hh = ginv(r, th, a)
    dr, dth = 1e-6*max(r, 1.), 1e-6
    dHr = (_H(r + dr, th, pr, pth, E, L, a) - _H(r - dr, th, pr, pth, E, L, a))/(2*dr)
    dHt = (_H(r, th + dth, pr, pth, E, L, a) - _H(r, th - dth, pr, pth, E, L, a))/(2*dth)
    return [-tt*E + tp*L, rr*pr, hh*pth, -tp*E + pp*L, -dHr, -dHt]   # dx/dlam = g^{mu nu} p_nu with p_t=-E, p_phi=L

def geodesic(r0, E, L, a=0., pr0=None, th0=np.pi/2, pth0=0., mu=1., lam_end=1e4, max_step=np.inf, n_out=2000, sign_pr=1.):
    """Integrate a Kerr geodesic in affine parameter (proper time for mu=1).  If pr0 is None it is solved from the
    mass-shell constraint.  Returns dict(lam, t, r, theta, phi, H_drift)."""
    if pr0 is None:
        tt, tp, pp, rr, hh = ginv(r0, th0, a); rest = tt*E*E - 2*tp*E*L + pp*L*L + hh*pth0*pth0 + mu*mu
        pr0 = sign_pr*np.sqrt(max(-rest/rr, 0.))
    y0 = [0., r0, th0, 0., pr0, pth0]; H0 = _H(r0, th0, pr0, pth0, E, L, a)
    hor = horizon(a)
    def plunge(lam, y, *args): return y[1] - hor*1.01
    plunge.terminal = True
    sol = solve_ivp(_rhs, (0, lam_end), y0, args=(E, L, a), method="DOP853", rtol=1e-11, atol=1e-12, dense_output=False,
                    t_eval=np.linspace(0, lam_end, n_out), events=plunge, max_step=max_step)
    t, r, th, ph, pr, pth = sol.y
    Hd = float(np.max(np.abs(_H(r, th, pr, pth, E, L, a) - H0))) if len(r) else 0.
    return dict(lam=sol.t, t=t, r=r, theta=th, phi=ph, H_drift=Hd, plunged=bool(sol.status == 1))

def circular_period(r, a=0., prograde=True, orbits=3):
    """Coordinate-time period of a numerically integrated circular orbit (compare 2 pi (r^1.5 + a))."""
    c = circular(r, a, prograde); tau = orbits*2*np.pi*(r**1.5 + a)*1.5
    g = geodesic(r, c["E"], c["L"], a, pr0=0., lam_end=tau, n_out=20000)
    ph = np.unwrap(g["phi"]); k = np.searchsorted(np.abs(ph), 2*np.pi*orbits)
    return float(np.interp(2*np.pi*orbits, np.abs(ph[:k + 2]), g["t"][:k + 2])/orbits), g

def periastron_advance_exact(p, e):
    """Exact Schwarzschild periastron advance per radial period: 4 sqrt(p/(p-6+2e)) K(k) - 2 pi, k^2 = 4e/(p-6+2e)."""
    from scipy.special import ellipk
    m = 4*e/(p - 6 + 2*e); return float(4*np.sqrt(p/(p - 6 + 2*e))*ellipk(m) - 2*np.pi)

def periastron_advance(p, e, a=0., n_orb=2):
    """Numerical periastron advance per radial period (rad) for an equatorial orbit of semi-latus rectum p, eccentricity e
    (Schwarzschild E, L from p, e).  1PN: 6 pi / p."""
    E = np.sqrt(((p - 2)**2 - 4*e*e)/(p*(p - 3 - e*e))); L = p/np.sqrt(p - 3 - e*e)
    rp = p/(1 + e); g = geodesic(rp*(1 + 1e-10), E, L, 0., pr0=None, lam_end=n_orb*2*np.pi*(p/(1 - e*e))**1.5*1.4, n_out=200000)
    r = g["r"]; ph = g["phi"]; i = np.where((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:]))[0] + 1
    if len(i) < 1: return np.nan
    j = i[0] + np.arange(-3, 4); j = j[(j >= 0) & (j < len(r))]
    c = np.polyfit(ph[j] - ph[i[0]], r[j], 2); phmin = ph[i[0]] - c[1]/(2*c[0])
    return float(phmin - 2*np.pi)


# ----------------------------------------------------------------- Peters inspiral (physical units in, out)
def chirp_mass(m1, m2): return (m1*m2)**.6/(m1 + m2)**.2

def peters_rates(a, e, m1, m2):
    """da/dt, de/dt (geometrized: a in total-mass-free units of M_sun, t in GM_sun/c^3)."""
    M = m1 + m2; f = m1*m2*M
    da = -64/5*f/(a**3*(1 - e*e)**3.5)*(1 + 73/24*e*e + 37/96*e**4)
    de = -304/15*e*f/(a**4*(1 - e*e)**2.5)*(1 + 121/304*e*e)
    return da, de

def t_merge_circular_yr(a_au, m1, m2):
    a = a_au/M_SUN_AU; return float(5/256*a**4/(m1*m2*(m1 + m2))*M_SUN_S/YR_S)

def inspiral(a_au, e0, m1, m2, a_stop_M=6.):
    """Integrate Peters to a = a_stop_M (m1+m2) (ISCO-ish).  Returns dict(t_yr, a_au, e, t_merge_yr)."""
    a0 = a_au/M_SUN_AU; M = m1 + m2; ast = a_stop_M*M
    if a0 <= ast: return dict(t_yr=np.zeros(1), a_au=np.array([a_au]), e=np.array([e0]), t_merge_yr=0.)
    # integrate in x = ln a (smooth) with t as output
    def rhs(x, y):
        a = np.exp(x); e = min(max(y[1], 0.), .999999); da, de = peters_rates(a, e, m1, m2)
        return [a/da, de*a/da]
    def done(x, y): return y[1] - .999999
    sol = solve_ivp(rhs, (np.log(a0), np.log(ast)), [0., e0], method="DOP853", rtol=1e-10, atol=1e-14)
    t = sol.y[0]*M_SUN_S/YR_S
    return dict(t_yr=t, a_au=np.exp(sol.t)*M_SUN_AU, e=sol.y[1], t_merge_yr=float(t[-1]))

def f_gw_isco_hz(M_sun): return float(1/(np.pi*6**1.5*M_sun*M_SUN_S))


# ----------------------------------------------------------------- events hook
def event_fields(M_bh_msun, chi=0., r_peri_au=None, m2_msun=None, a_bin_au=None, e_bin=0.):
    """Extra GR fields for an event-log entry: Kerr ISCO/photon orbit/horizon (AU), efficiency, ISCO GW frequency;
    for a pericentre passage r_peri_au -> plunge/capture flag from the marginally bound orbit r_mb = 2 - a + 2 sqrt(1 - a);
    for a binary (m2, a_bin, e_bin) -> Peters merger time."""
    M = float(M_bh_msun); rg = M*M_SUN_AU; a = float(np.clip(chi, -.998, .998))
    d = dict(gr_r_isco_au=r_isco(abs(a), a >= 0)*rg, gr_r_photon_au=r_photon(abs(a), a >= 0)*rg, gr_r_horizon_au=horizon(a)*rg,
             gr_eta=efficiency(abs(a), a >= 0), gr_f_isco_hz=f_gw_isco_hz(M), gr_spin=a)
    if r_peri_au is not None:
        rmb = (2 - abs(a) + 2*np.sqrt(1 - abs(a)))*rg; d.update(gr_r_mb_au=float(rmb), gr_captured=bool(r_peri_au < rmb))
    if m2_msun is not None and a_bin_au is not None:
        d["gr_t_merge_yr"] = inspiral(a_bin_au, e_bin, M, float(m2_msun))["t_merge_yr"]
    return d
