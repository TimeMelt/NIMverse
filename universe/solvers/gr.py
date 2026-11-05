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
  * Merger remnant (Rezzolla et al. 2008 aligned-spin final spin; Barausse et al. 2012 radiated energy) and the l=m=2, n=0
    Kerr quasi-normal ringdown (Berti, Cardoso & Will 2006 fits: f_220, damping time, quality factor).
  * Optional dynamics (events gr_solver="dynamics"): Kerr-matched Paczynski-Wiita pseudo-Newtonian pull
    a = -M/(r - r_isco/3)^2 (exact Keplerian far field; ISCO at the Kerr prograde r_isco) on stars near SMBHs, and a Kerr
    marginally-bound capture gate (r_mb = 2 - a + 2 sqrt(1-a)) that swallows stars whole (no flare / AGN fuel).
    At galactic resolution (softening >> r_g) the PW term is numerically negligible; the capture gate is what changes runs.
Fidelity / gaps: test-particle geodesics (no self-force, no spin-curvature coupling); vacuum Kerr (no accretion-disk
radiative transfer); Peters is leading-order (0PN radiation reaction); remnant/ringdown are NR fits, no merger waveform.
Opt-in upgrades (default outputs unchanged):
  * pn_rates / inspiral_pn(order 0..3.5): quasi-circular TaylorT4 energy-flux series through 3.5PN (incl. the 3PN log and
    the 1.5/2.5/3.5PN tail terms; Blanchet 2014) times the Peters eccentricity enhancement -- exact Peters at order 0,
    exact quasi-circular PN as e -> 0; no eccentric PN enhancement functions, no spin-orbit/spin-spin flux terms.
    pn_cycles(): cycles / time in band.  Events: C["gr_pn"] = order adds gr_t_merge_pn_yr and the IMR tags.
  * imr_tags(): PhenomA (Ajith+ 2008) merger / ringdown / cut-off frequencies + NR-fit remnant QNM + in-band cycles.
    Tags only: no h(t)/h(f) waveform is synthesised, no NR surrogate, spins only via the remnant.
  * mpd_orbit(): Mathisson-Papapetrou-Dixon spinning test body in Schwarzschild (Tulczyjew SSC, linear in spin,
    aligned spin, equatorial) with Dixon-Killing E, L_z monitored (conserved to ~1e-11).  Still a test body: no
    self-force, no Kerr background for MPD, no spin precession for misaligned spins.  None of this is SpEC/NR."""
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


# ----------------------------------------------------------------- merger remnant + ringdown
def remnant(m1, m2, chi1=0., chi2=0.):
    """Final mass/spin of a BBH merger with aligned spins.  chi_f: Rezzolla et al. (2008); E_rad: Barausse et al. (2012)."""
    M = m1 + m2; q = min(m1, m2)/max(m1, m2); nu = m1*m2/M**2
    c1, c2 = (chi1, chi2) if m1 >= m2 else (chi2, chi1); at = (c1 + c2*q*q)/(1 + q*q)
    s4, s5, t0, t2, t3 = -.1229, .4537, -2.8904, -3.5171, 2.5763
    chi = float(np.clip(at + at*nu*(s4*at + s5*nu + t0) + nu*(2*np.sqrt(3) + t2*nu + t3*nu*nu), -.998, .998))
    Ei = lambda a: circular(r_isco(abs(a), a >= 0), abs(a), a >= 0)["E"]
    p0, p1 = .04827, .01707
    Erad = nu*(1 - Ei(at)) + 4*nu*nu*(4*p0 + 16*p1*at*(at + 1) + Ei(at) - 1)
    return dict(M_f=float(M*(1 - Erad)), chi_f=chi, E_rad_frac=float(Erad))

def qnm_220(M_msun, chi):
    """Fundamental l=m=2 Kerr ringdown (Berti+06 fits): f [Hz], damping time tau [s], quality factor Q."""
    j = min(abs(float(chi)), .998)
    w = 1.5251 - 1.1568*(1 - j)**.1292; Q = .7 + 1.4187*(1 - j)**-.4990
    f = w/(2*np.pi*M_msun*M_SUN_S); return dict(f_hz=float(f), tau_s=float(Q/(np.pi*f)), Q=float(Q), Mw=float(w))

def ringdown_fields(m1_msun, m2_msun, chi1=0., chi2=0.):
    r = remnant(m1_msun, m2_msun, chi1, chi2); q = qnm_220(r["M_f"], r["chi_f"])
    return dict(gr_M_final_msun=r["M_f"], gr_chi_final=r["chi_f"], gr_E_rad_frac=r["E_rad_frac"],
                gr_ringdown_f_hz=q["f_hz"], gr_ringdown_tau_s=q["tau_s"], gr_ringdown_Q=q["Q"])


# ----------------------------------------------------------------- optional dynamics (pseudo-Newtonian, Kerr-matched)
def pw_rs(a=0.):
    """PW 'Schwarzschild radius' (units of r_g) whose ISCO (= 3 r_s) matches the Kerr prograde ISCO."""
    return r_isco(abs(a), a >= 0)/3.

def pw_correction(dx, M, rs, soft2=0.):
    """Extra acceleration a_PW - a_Newton on bodies at separation dx (N,3) from a hole of mass M (G=1) with PW radius rs
    (same length units as dx).  Zero beyond 1e6 rs (and inside the softening the Newtonian softened force is kept)."""
    r = np.sqrt((dx*dx).sum(1) + soft2); out = np.zeros_like(dx)
    k = (r < 1e6*rs) & (r > 1.05*rs)
    if k.any():
        rk = r[k]; out[k] = -(M*(1/(rk - rs)**2 - 1/rk**2)/rk)[:, None]*dx[k]
    return out

def capture_radius(a=0.):
    """Kerr marginally-bound (parabolic) capture pericentre in r_g (prograde equatorial): 2 - a + 2 sqrt(1 - a)."""
    a = abs(float(a)); return float(2 - a + 2*np.sqrt(1 - a))


# ----------------------------------------------------------------- events hook
def event_fields(M_bh_msun, chi=0., r_peri_au=None, m2_msun=None, a_bin_au=None, e_bin=0., pn_order=0.):
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
        if pn_order: d["gr_t_merge_pn_yr"] = inspiral_pn(a_bin_au, e_bin, M, float(m2_msun), pn_order)["t_merge_yr"]
    return d


# ----------------------------------------------------------------- higher-PN inspiral (opt-in: inspiral(..., pn_order=...))
GAMMA_E = 0.5772156649015329

def _pn_flux_coeffs(nu):
    """TaylorT4 coefficients of dx/dt = 64 nu/5 x^5 sum_k c_k x^(k/2), x = (M omega)^(2/3), quasi-circular, non-spinning
    (Blanchet 2014 Living Rev. Rel. 17, 2, eq. 317 / Buonanno+ 2009 Table; the 3PN log term is handled in _pn_series)."""
    pi = np.pi
    return {0: 1., 2: -743/336 - 11/4*nu, 3: 4*pi, 4: 34103/18144 + 13661/2016*nu + 59/18*nu*nu,
            5: -(4159/672 + 189/8*nu)*pi,
            6: (16447322263/139708800 - 1712/105*GAMMA_E + 16/3*pi*pi + (-56198689/217728 + 451/48*pi*pi)*nu
                + 541/896*nu*nu - 5605/2592*nu**3),
            7: (-4415/4032 + 358675/6048*nu + 91495/1512*nu*nu)*pi}

def _pn_series(x, nu, order):
    c = _pn_flux_coeffs(nu); kmax = int(round(2*order)); s = 0.
    for k, ck in c.items():
        if k <= kmax: s = s + ck*x**(k/2)
    if kmax >= 6: s = s - 856/105*np.log(16*x)*x**3
    return s

def pn_rates(a, e, m1, m2, order=3.5):
    """Orbit-averaged da/dt, de/dt (geometrized, like peters_rates) at PN order 0..3.5 (half-integers).  Quasi-circular
    flux through `order` (TaylorT4 series in x = M/a) times the Peters eccentricity enhancement; de/dt keeps the Peters
    de/da ratio.  order=0 is exactly peters_rates.  Eccentric PN corrections (Arun+ 2009 enhancement functions) are NOT
    included (exact as e -> 0; leading-order in e otherwise); a is the PN parameter M/x, not a gauge-invariant separation."""
    da, de = peters_rates(a, e, m1, m2)
    if order <= 0: return da, de
    M = m1 + m2; nu = m1*m2/M**2; x = M/a; f = _pn_series(x, nu, order)
    return da*f, de*f

def inspiral_pn(a_au, e0, m1, m2, order=3.5, a_stop_M=6.):
    """Like inspiral() with pn_rates at `order`; stops at a = a_stop_M M or where the PN flux turns over."""
    a0 = a_au/M_SUN_AU; M = m1 + m2; ast = a_stop_M*M
    if a0 <= ast: return dict(t_yr=np.zeros(1), a_au=np.array([a_au]), e=np.array([e0]), t_merge_yr=0., pn_order=order)
    def rhs(xl, y):
        a = np.exp(xl); e = min(max(y[1], 0.), .999999); da, de = pn_rates(a, e, m1, m2, order)
        da = min(da, -1e-300); return [a/da, de*a/da]
    sol = solve_ivp(rhs, (np.log(a0), np.log(ast)), [0., e0], method="DOP853", rtol=1e-10, atol=1e-14)
    t = sol.y[0]*M_SUN_S/YR_S
    return dict(t_yr=t, a_au=np.exp(sol.t)*M_SUN_AU, e=sol.y[1], t_merge_yr=float(t[-1]), pn_order=order)

def pn_cycles(f_lo_hz, m1, m2, order=3.5, f_hi_hz=None):
    """GW cycles and time (s) from f_lo to f_hi (default f_ISCO) on the quasi-circular TaylorT4 track."""
    M = m1 + m2; nu = m1*m2/M**2; f_hi = f_hi_hz or f_gw_isco_hz(M)
    xl = (np.pi*M*M_SUN_S*f_lo_hz)**(2/3); xh = (np.pi*M*M_SUN_S*f_hi)**(2/3)
    xs = np.linspace(xl, xh, 20001); dxdt = 64*nu/5*xs**5*_pn_series(xs, nu, order)/(M*M_SUN_S)
    dt = 1/np.maximum(dxdt, 1e-300); om = xs**1.5/(M*M_SUN_S)
    T = float(np.trapezoid(dt, xs)); N = float(np.trapezoid(om/np.pi*dt, xs))
    return dict(t_s=T, n_cycles=N)


# ----------------------------------------------------------------- phenomenological IMR tags (PhenomA, Ajith+ 2008)
_PHENOM_A = dict(f_merg=(2.9740e-1, 4.4810e-2, 9.5560e-2), f_ring=(5.9411e-1, 8.9794e-2, 1.9111e-1),
                 sigma=(5.0801e-1, 7.7515e-2, 2.2369e-2), f_cut=(8.4845e-1, 1.2848e-1, 2.7299e-1))

def imr_tags(m1_msun, m2_msun, chi1=0., chi2=0., order=3.5, f_lo_hz=10.):
    """Phenomenological inspiral-merger-ringdown tags: PhenomA transition frequencies (Ajith et al. 2008 PRD 77 104017,
    non-spinning: f = (a nu^2 + b nu + c)/(pi M)), the Kerr QNM ringdown of the NR-fit remnant, and TaylorT4 cycles /
    time in band from f_lo.  Tags only -- no waveform h(t) is synthesised and spins enter only through the remnant."""
    M = float(m1_msun + m2_msun); nu = m1_msun*m2_msun/M**2; tM = np.pi*M*M_SUN_S
    ph = {k: float((a*nu*nu + b*nu + c)/tM) for k, (a, b, c) in _PHENOM_A.items()}
    cyc = pn_cycles(f_lo_hz, m1_msun, m2_msun, order, ph["f_merg"]) if f_lo_hz < ph["f_merg"] else dict(t_s=0., n_cycles=0.)
    rd = ringdown_fields(m1_msun, m2_msun, chi1, chi2)
    return dict(gr_imr_model="PhenomA-tags", gr_imr_f_merge_hz=ph["f_merg"], gr_imr_f_ring_hz=ph["f_ring"],
                gr_imr_sigma_hz=ph["sigma"], gr_imr_f_cut_hz=ph["f_cut"], gr_imr_f_qnm_hz=rd["gr_ringdown_f_hz"],
                gr_imr_t_inband_s=cyc["t_s"], gr_imr_cycles_inband=cyc["n_cycles"], gr_imr_f_lo_hz=float(f_lo_hz), gr_pn_order=float(order))


# ----------------------------------------------------------------- MPD spinning test body (Schwarzschild, opt-in)
def _schw_g(r, th):
    f = 1 - 2/r; s2 = np.sin(th)**2
    return np.diag([-f, 1/f, r*r, r*r*s2])

def _schw_gamma(r, th):
    """Christoffel symbols Gamma^a_{bc} of Schwarzschild (M = 1)."""
    f = 1 - 2/r; G = np.zeros((4, 4, 4)); s, c = np.sin(th), np.cos(th)
    G[0, 0, 1] = G[0, 1, 0] = 1/(r*r*f); G[1, 0, 0] = f/(r*r); G[1, 1, 1] = -1/(r*r*f)
    G[1, 2, 2] = -r*f; G[1, 3, 3] = -r*f*s*s; G[2, 1, 2] = G[2, 2, 1] = 1/r; G[2, 3, 3] = -s*c
    G[3, 1, 3] = G[3, 3, 1] = 1/r; G[3, 2, 3] = G[3, 3, 2] = c/s
    return G

def _schw_riemann(r, th, h=1e-5):
    """R^a_{bcd} = d_c G^a_{db} - d_d G^a_{cb} + G^a_{ce} G^e_{db} - G^a_{de} G^e_{cb} (finite-differenced in r, theta)."""
    G = _schw_gamma(r, th); dG = np.zeros((4, 4, 4, 4))           # dG[c, a, b, d] = d_c G^a_{bd}
    hr = h*r; dG[1] = (_schw_gamma(r + hr, th) - _schw_gamma(r - hr, th))/(2*hr)
    dG[2] = (_schw_gamma(r, th + h) - _schw_gamma(r, th - h))/(2*h)
    R = (np.einsum("cadb->abcd", dG) - np.einsum("dacb->abcd", dG)
         + np.einsum("ace,edb->abcd", G, G) - np.einsum("ade,ecb->abcd", G, G))
    return R

def _levi_lower(r, th):
    """epsilon_{abcd} with epsilon_{t r th ph} = sqrt(-g) = r^2 sin th."""
    import itertools
    E = np.zeros((4, 4, 4, 4)); sg = r*r*np.sin(th)
    for p in itertools.permutations(range(4)):
        sgn = np.linalg.det(np.eye(4)[list(p)]); E[p] = sgn*sg
    return E

def _mpd_rhs(tau, y):
    x = y[:4]; u = y[4:8]; S = np.zeros((4, 4)); iu = np.triu_indices(4, 1); S[iu] = y[8:]; S = S - S.T
    r, th = x[1], x[2]; G = _schw_gamma(r, th); R = _schw_riemann(r, th)
    du = -np.einsum("abc,b,c->a", G, u, u) - .5*np.einsum("abcd,b,cd->a", R, u, S)
    dS = -np.einsum("abc,b,cd->ad", G, u, S) - np.einsum("dbc,b,ac->ad", G, u, S)
    return np.concatenate([u, du, dS[iu]])

def mpd_killing(y):
    """Dixon-Killing conserved energy and z-angular momentum  Q = xi.u + 1/2 S^{ab} nabla_a xi_b  (per unit mass)."""
    x = y[:4]; u = y[4:8]; S = np.zeros((4, 4)); iu = np.triu_indices(4, 1); S[iu] = y[8:]; S = S - S.T
    r, th = x[1], x[2]; g = _schw_g(r, th); G = _schw_gamma(r, th); out = []
    for k, sgn in ((0, -1.), (3, 1.)):
        xi_l = g[:, k]; h = 1e-6*r; dxi = np.zeros((4, 4))           # d_a xi_b
        dxi[1] = (_schw_g(r + h, th)[:, k] - _schw_g(r - h, th)[:, k])/(2*h)
        dxi[2] = (_schw_g(r, th + 1e-6)[:, k] - _schw_g(r, th - 1e-6)[:, k])/2e-6
        nab = dxi - np.einsum("lab,l->ab", G, xi_l)
        out.append(sgn*(xi_l @ u + .5*np.einsum("ab,ab->", S, nab)))
    return dict(E=out[0], Lz=out[1])

def mpd_orbit(r0, s=0., n_orb=3, n_out=2000):
    """Mathisson-Papapetrou-Dixon spinning test body in Schwarzschild (M = 1), spin s (units mu M) aligned with the
    orbital angular momentum, Tulczyjew SSC, linear in spin (p = mu u, DS/dtau = 0 at O(S)).  Starts on the circular
    geodesic velocity at r0; the spin-curvature force -1/2 R^a_{bcd} u^b S^{cd} then perturbs it.  Returns
    dict(tau, t, r, phi, Omega (mean dphi/dt), dE, dLz (Dixon-Killing drifts), r_range)."""
    th = np.pi/2; f = 1 - 2/r0; Om = r0**-1.5; ut = 1/np.sqrt(f - r0*r0*Om*Om); u = np.array([ut, 0., 0., Om*ut])
    g = _schw_g(r0, th); E = _levi_lower(r0, th); Sv = np.array([0., 0., -s/r0, 0.])           # spin vector along +z
    S_l = np.einsum("abcd,c,d->ab", E, u, Sv)                                                 # S_{ab} = eps_{abcd} u^c S^d
    gi = np.linalg.inv(g); S_u = gi @ S_l @ gi.T; iu = np.triu_indices(4, 1)
    y0 = np.concatenate([[0., r0, th, 0.], u, S_u[iu]]); T = n_orb*2*np.pi*r0**1.5/ut
    sol = solve_ivp(_mpd_rhs, (0, T), y0, method="DOP853", rtol=1e-11, atol=1e-12, t_eval=np.linspace(0, T, n_out))
    q0 = mpd_killing(sol.y[:, 0]); q1 = mpd_killing(sol.y[:, -1])
    t, r, ph = sol.y[0], sol.y[1], sol.y[3]
    return dict(tau=sol.t, t=t, r=r, phi=ph, Omega=float(ph[-1]/t[-1]), dE=float(q1["E"] - q0["E"]), dLz=float(q1["Lz"] - q0["Lz"]),
                r_range=(float(r.min()), float(r.max())), E=q0["E"], Lz=q0["Lz"])
