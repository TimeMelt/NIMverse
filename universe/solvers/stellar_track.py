"""solvers.stellar_track -- semi-analytic stellar-evolution track solver (out-of-leaf; HostWeather.solver_track="semi").

Replaces the leaf's toy power laws (leaves.stellar.track) with ODE-integrated physics, same call signature:
  MS     core hydrogen burning: dX_c/dt = -L / (Q_H q_c M X0-normalised), Q_H = 0.0071 c^2 (6.4e18 erg/g), burning core
         mass fraction q_c(M); homology L = L_ZAMS(M) (mu/mu_0)^4 (electron-scattering homology, Kippenhahn & Weigert ch. 20)
         with mu = 4/(3 + 5 X_eff), X_eff = X0 - K_MIX (X0 - X_c) (envelope-weighted); R = R_ZAMS (L/L_ZAMS)^0.35.
         Calibrated: Sun L(4.57 Gyr)/L_ZAMS = 1/0.72 (faint young Sun), t_MS(1 Msun) ~ 10 Gyr.
  SG     subgiant: shell ignition, L and R interpolated (log) from the TAMS to the RGB base over a thermal-ish time
         t_SG = 0.08 t_MS (Hertzsprung-gap crossing is short; Sun ~ 0.7 Gyr).
  RGB    core-mass--luminosity relation L = 2.3e5 (M_c/Msun)^6 Lsun (Refsdal & Weigert 1970) with core growth
         dM_c/dt = L / (X_env Q_H) and the Rappaport et al. (1995) giant radius R(M_c); Reimers mass loss
         dM/dt = -4e-13 eta L R / M (eta 0.5); ends at the He flash M_c = 0.47 (L ~ 2-3e3 Lsun) for M < 2.
  HB     core-He burning clump: L ~ 50 Lsun (M_c-scaled), Teff ~ 4800-5000 K, for 0.1 Gyr.
  PMS    pre-main-sequence Kelvin-Helmholtz contraction, dR/dt = -L R^2 / (k G M^2): Hayashi track (n=1.5, k=3/7,
         Teff = T_H(M) ~ 4000 K M^0.12, L = 4 pi R^2 sigma T_H^4) until L falls to the ZAMS value, then a Henyey track
         (n=3, k=3/4) at L = L_ZAMS down to R_ZAMS.  No deuterium burning.  Ages count from birth: the ZAMS sits at
         t_pms (Sun ~ 20 Myr here vs ~40 Myr in full PMS models).
  AGB    thermal-pulse-lite (M < 8): E-AGB ramp from the HB/giant luminosity, then TP-AGB with the Paczynski (1970)
         core-mass luminosity L = 59250 (M_c - 0.495) Lsun, H-shell core growth dM_c/dt = L/(X Q_H), Bloecker (1995)
         superwind (eta_B = 0.02) on top of Reimers, R from Teff_AGB ~ 3100 K; helium-shell flashes every
         t_ip = 10^(3.05 - 4.5 (M_c - 1)) yr (Iben & Renzini) drawn as a 50 % luminosity dip -> `n_pulses` in the track.
         Ends when the envelope is gone (M - M_c < 0.01): M_WD = M_c (initial-final relation emerges, ~0.55-0.6 for 1 Msun).
  WD     Mestel cooling L = (t / (8.8 Myr (A/12)^-1 M^(5/7) (mu/2)^(-2/7)))^(-7/5) Lsun (C/O, A = 14), Eggleton
         mass-radius R = 0.0127 M^(-1/3) sqrt(1 - (M/1.44)^(4/3)) Rsun, for 12 Gyr after the AGB.
  Z      metallicity scaling (Z = ZSUN reproduces the solar calibration exactly): L_ZAMS ~ (Z/Zsun)^-0.15,
         R_ZAMS ~ (Z/Zsun)^0.1 (lower opacity -> hotter, brighter, shorter-lived), X0 shifted with dY/dZ = 1.6.
All ODEs integrated with scipy solve_ivp (RK45, rtol 1e-8) once per (mass, Z) and tabulated (log-age interpolation).
Fidelity / gaps: this is still NOT MESA -- one-zone MS (no overshoot), parametric PMS/AGB/WD (no envelope structure,
no third dredge-up/HBB, no crystallisation/Debye cooling of WDs); masses ~0.3-2 Msun are the calibrated range
(fully convective M < 0.35 burn slowly on the MS); M > 2 blue loops and M > 8 supernova endpoints are not modelled
(leaves/events handles remnants)."""
import math
import numpy as np
from scipy.integrate import solve_ivp

Q_H = 6.4e18; MSUN = 1.989e33; LSUN = 3.828e33; GYR = 3.15576e16
X0 = .70; K_MIX = .4417;   # K_MIX and q_core's .1575 fitted to the Sun: t_MS = 10.0 Gyr, L(4.57)/L_ZAMS = 1/0.72
T_SUN = 5772.; ZSUN = .0142; G_CGS = 6.674e-8; RSUN = 6.957e10; SIGMA = 5.6704e-5; YR = 3.15576e7
ETA_BLOECKER = .02; T_AGB = 3100.; T_WD_GYR = 12.
ETA_REIMERS = .5; MC_FLASH = .47; T_HB_GYR = .1
_CACHE = {}


def q_core(M):
    """Fraction of the star's mass that burns H on the MS (convective-core / radiative-core proxy)."""
    return float(np.clip(.1575*max(M, .1)**.4, .05, .4)) if M >= .35 else 1.   # fully convective low-mass stars mix everything

def L_zams(M, Z=ZSUN): return float((.72*M**4.0 if M < 1.5 else .72*1.5**4*(M/1.5)**3.5)*(Z/ZSUN)**-.15)
def R_zams(M, Z=ZSUN): return float((.89*M**.8 if M >= .35 else .89*.35**.8*(M/.35)**.95)*(Z/ZSUN)**.1)
def X0_of(Z): return float(np.clip(X0 - 2.6*(Z - ZSUN), .6, .78))     # dY/dZ = 1.6 -> dX/dZ = -2.6
def mu_of(X): return 4/(3 + 5*X)

def rgb_radius(Mc): return 3.7e3*Mc**4/(1 + Mc**3 + 1.75*Mc**4)      # Rsun (Rappaport+95)
def rgb_lum(Mc): return 2.3e5*Mc**6

def _pms(M, Z=ZSUN):
    """Kelvin-Helmholtz contraction (Hayashi then Henyey) to the ZAMS.  Returns (t [Gyr], L, R) from birth to the ZAMS."""
    Lz, Rz = L_zams(M, Z), R_zams(M, Z); TH = 4000.*max(M, .08)**.12; c = 3/7*G_CGS*(M*MSUN)**2
    def L_of(R):
        Lh = (R)**2*(TH/T_SUN)**4                                # Hayashi (Lsun, R in Rsun)
        return max(Lh, Lz)
    def rhs(t, y):                                                # y = ln R
        R = math.exp(y[0]); Lh = R*R*(TH/T_SUN)**4
        cc = c if Lh > Lz else c*7/4                                 # radiative core (Henyey): n = 3 polytrope, E = -(3/4) G M^2/R
        return [-L_of(R)*LSUN*R*RSUN/cc*GYR]
    def zams(t, y): return y[0] - math.log(Rz)
    zams.terminal = True
    R0 = max(4.*M**.6, 2.5*Rz)
    sol = solve_ivp(rhs, (0, 5.), [math.log(R0)], rtol=1e-8, atol=1e-10, events=zams, dense_output=True)
    tp = float(sol.t[-1]); ts = np.geomspace(1e-6, tp, 80); ts[-1] = tp
    R = np.exp(sol.sol(ts)[0]); L = np.array([L_of(r) for r in R]); return np.r_[0., ts], np.r_[L_of(R0), L], np.r_[R0, R]

def _ms(M, Z=ZSUN):
    qc = q_core(M); Lz = L_zams(M, Z); x0 = X0_of(Z); mu0 = mu_of(x0)
    def L_of(Xc):
        Xe = x0 - K_MIX*(x0 - Xc) if M >= .35 else Xc
        return Lz*(mu_of(Xe)/mu0)**4
    def rhs(t, y): return [-L_of(y[0])*LSUN*GYR/(Q_H*qc*M*MSUN)]
    def end(t, y): return y[0] - .01
    end.terminal = True
    sol = solve_ivp(rhs, (0, 1e3), [x0], rtol=1e-8, atol=1e-10, events=end, dense_output=True, max_step=.5)
    t_ms = float(sol.t[-1]); ts = np.linspace(0, t_ms, 400); Xc = sol.sol(ts)[0]
    L = np.array([L_of(x) for x in Xc]); R = R_zams(M, Z)*(L/Lz)**.35
    return ts, L, R, M*np.ones_like(ts), t_ms


def _post(M, t_ms, L_tams, R_tams, agb=True):
    """Subgiant -> RGB (core-mass growth + Reimers) -> HB.  Returns (t, L, R, Mstar, phase) arrays after t_ms."""
    t_sg = .08*t_ms
    L_b = 2.5*L_tams                                  # RGB base ~2.5x the TAMS luminosity (Sun: ~5 Lsun)
    Mc0 = min((L_b/2.3e5)**(1/6), .4)                 # core mass on the Refsdal-Weigert relation at the RGB base
    R_b = rgb_radius(Mc0)*max(M, .5)**.3
    tsg = np.linspace(0, t_sg, 60)[1:]; f = tsg/t_sg
    Lsg = np.exp(np.log(L_tams)*(1 - f) + np.log(max(L_b, L_tams*1.05))*f); Rsg = np.exp(np.log(R_tams)*(1 - f) + np.log(R_b)*f)
    out_t, out_L, out_R, out_M, out_p = list(t_ms + tsg), list(Lsg), list(Rsg), [M]*len(tsg), ["subgiant"]*len(tsg)
    if M < 2:
        def rhs(t, y):
            Mc, Ms = y; L = rgb_lum(Mc); R = rgb_radius(Mc)*max(Ms, .5)**.3
            return [L*LSUN*GYR/(X0*Q_H*MSUN), -4e-13*ETA_REIMERS*L*R/Ms*1e9]
        def flash(t, y): return y[0] - MC_FLASH
        flash.terminal = True
        def gone(t, y): return y[1] - y[0] - .02
        gone.terminal = True
        sol = solve_ivp(rhs, (0, 50.), [Mc0, M], rtol=1e-8, atol=1e-12, events=[flash, gone], dense_output=True)
        tr = np.geomspace(1e-4, sol.t[-1], 300); Mc, Ms = sol.sol(tr)
        t0 = out_t[-1]
        out_t += list(t0 + tr); out_L += list(rgb_lum(Mc)); out_R += list(rgb_radius(Mc)*np.maximum(Ms, .5)**.3); out_M += list(Ms); out_p += ["rgb"]*len(tr)
        Mfin = float(Ms[-1]); Lhb = 50.*(Mc[-1]/.47)**2; Rhb = math.sqrt(Lhb)*(T_SUN/4900.)**2
        th = np.linspace(0, T_HB_GYR, 20)[1:]
        out_t += list(out_t[-1] + th); out_L += [Lhb]*len(th); out_R += [Rhb]*len(th); out_M += [Mfin]*len(th); out_p += ["hb"]*len(th)
        Mc_agb = max(.5, float(Mc[-1]) + .03)                          # core after He burning (C/O core + He shell)
    else:   # intermediate mass: Hertzsprung gap then a short He-burning giant phase at ~ constant L (no blue loop)
        Lg = L_b*3; th = np.linspace(0, .1*t_ms, 40)[1:]
        out_t += list(out_t[-1] + th); out_L += [Lg]*len(th); out_R += [R_b*2]*len(th); out_M += [M]*len(th); out_p += ["giant"]*len(th)
        Mfin = M; Mc_agb = min(.36 + .1*M, 1.2)
    n_tp = 0
    if agb and M < 8:
        n_tp = _agb_wd(out_t, out_L, out_R, out_M, out_p, Mfin, Mc_agb)
    return np.array(out_t), np.array(out_L), np.array(out_R), np.array(out_M), out_p, n_tp


def _agb_wd(out_t, out_L, out_R, out_M, out_p, M, Mc0):
    """Append E-AGB, TP-AGB (pulse-lite) and WD cooling in place.  Returns the number of thermal pulses."""
    L_start = out_L[-1]; Lp = lambda Mc: 59250.*max(Mc - .495, 1e-3)
    Mc1 = max(Mc0, .52); L1 = Lp(Mc1)
    if M - Mc1 < .02:                                                     # envelope already stripped (RGB/HB mass loss): AGB-manque
        _wd(out_t, out_L, out_R, out_M, out_p, min(M, Mc1)); return 0
    t_eagb = .02 if M < 3 else .005                                      # Gyr; E-AGB ~ 10-20 Myr
    te = np.linspace(0, t_eagb, 30)[1:]; f = te/t_eagb; t0 = out_t[-1]
    Le = np.exp(np.log(L_start)*(1 - f) + np.log(L1)*f); Tagb = T_AGB*max(M, .5)**.05
    out_t += list(t0 + te); out_L += list(Le); out_R += list(np.sqrt(Le)*(T_SUN/Tagb)**2); out_M += [M]*len(te); out_p += ["agb"]*len(te)
    def rhs(t, y):
        Mc, Ms = y; Ms = max(Ms, Mc + 1e-3, .05); L = Lp(Mc); R = math.sqrt(L)*(T_SUN/Tagb)**2
        reim = 4e-13*.5*L*R/Ms; sw = 4.83e-9*Ms**-2.1*L**2.7*4e-13*ETA_BLOECKER*L*R/Ms
        return [L*LSUN*GYR/(X0*Q_H*MSUN), -(reim + sw)*1e9]
    def bare(t, y): return y[1] - y[0] - .01
    bare.terminal = True
    def chand(t, y): return y[0] - 1.38
    chand.terminal = True
    sol = solve_ivp(rhs, (0, .5), [Mc1, M], rtol=1e-8, atol=1e-12, events=[bare, chand], dense_output=True, max_step=1e-3)
    T_tp = float(sol.t[-1]); t0 = out_t[-1]
    # pulse times: interpulse period t_ip(M_c) in yr
    tps, tt = [], 0.
    while tt < T_tp and len(tps) < 400:
        Mc_ = float(sol.sol(tt)[0]); tt += 10**(3.05 - 4.5*(Mc_ - 1))*1e-9; tps.append(tt)
    tps = [x for x in tps if x < T_tp]; grid = set(np.linspace(0, T_tp, 120)[1:].tolist())
    for x in tps:                                                         # sample each flash: dip right after, recovered after 30%
        ip = 10**(3.05 - 4.5*(float(sol.sol(x)[0]) - 1))*1e-9; grid.update([x, x + .02*ip, x + .3*ip])
    tg = np.array(sorted(g for g in grid if 0 < g <= T_tp)); Mc, Ms = sol.sol(tg)
    dip = np.ones_like(tg)
    for x in tps:
        ip = 10**(3.05 - 4.5*(float(sol.sol(x)[0]) - 1))*1e-9; k = (tg > x) & (tg < x + .3*ip)
        dip[k] = np.minimum(dip[k], .5 + .5*(tg[k] - x)/(.3*ip))
    L = np.array([Lp(m) for m in Mc])*dip; R = np.sqrt(L)*(T_SUN/Tagb)**2
    out_t += list(t0 + tg); out_L += list(L); out_R += list(R); out_M += list(Ms); out_p += ["tp-agb"]*len(tg)
    _wd(out_t, out_L, out_R, out_M, out_p, float(Mc[-1])); return len(tps)


def _wd(out_t, out_L, out_R, out_M, out_p, Mwd):
    t0 = out_t[-1]; Rwd = .0127*Mwd**(-1/3)*math.sqrt(max(1 - (Mwd/1.44)**(4/3), 1e-4))
    tw = np.geomspace(1e-5, T_WD_GYR, 120)
    Lw = (tw*1e9/(8.8e6*(14/12)**-1*Mwd**(5/7)))**(-7/5)
    out_t += list(t0 + tw); out_L += list(Lw); out_R += [Rwd]*len(tw); out_M += [Mwd]*len(tw); out_p += ["wd"]*len(tw)


def solve_track(M, Z=ZSUN):
    """Tabulated track for mass M (Msun), metallicity Z: dict(t [Gyr from birth], L [Lsun], R [Rsun], Teff [K], M, phase,
    t_pms (ZAMS age), t_ms (end of core-H burning, from birth), t_end, n_pulses, M_wd)."""
    k = (round(float(M), 3), round(float(Z), 5))
    if k in _CACHE: return _CACHE[k]
    M = max(k[0], .08); Z = k[1]
    tq, Lq, Rq = _pms(M, Z); tz = float(tq[-1])
    ts, L, R, Ms, t_ms = _ms(M, Z)
    tp, Lp, Rp, Mp, pp, n_tp = _post(M, t_ms, L[-1], R[-1])
    t = np.concatenate([tq[:-1], tz + ts, tz + tp]); L = np.concatenate([Lq[:-1], L, Lp]); R = np.concatenate([Rq[:-1], R, Rp])
    Mm = np.concatenate([np.full(len(tq) - 1, M), Ms, Mp])
    ph = ["pms"]*(len(tq) - 1) + ["ms"]*len(ts) + pp; T = T_SUN*(L/R**2)**.25
    d = dict(t=t, L=L, R=R, Teff=T, M=Mm, phase=np.array(ph), t_pms=tz, t_ms=tz + t_ms, t_end=float(t[-1]), n_pulses=int(n_tp),
             M_wd=float(Mm[-1]) if pp and pp[-1] == "wd" else None, Z=Z)
    _CACHE[k] = d; return d


def _interp(d, key, age):
    a = float(np.clip(age, 0, d["t_end"])); v = d[key]
    return float(np.exp(np.interp(a, d["t"], np.log(v)))) if key != "M" else float(np.interp(a, d["t"], v))

# ---- leaf-compatible API (leaves.stellar names) ----
X_START_MAX = .9
def t_ms_Gyr(M): return solve_track(M)["t_ms"]                 # from birth (ZAMS at solve_track(M)["t_pms"])
def max_age_Gyr(M): return solve_track(M)["t_end"]
def phase(M, age_Gyr):
    d = solve_track(M); i = int(np.clip(np.searchsorted(d["t"], age_Gyr), 0, len(d["t"]) - 1)); return str(d["phase"][i])
def track(M, L0, T0, age0_Gyr, age_Gyr):
    """(L, Teff, R) at age_Gyr given the star had (L0, T0) at age0_Gyr (shape from the solved track, anchored to L0, T0)."""
    d = solve_track(M)
    L = L0*_interp(d, "L", age_Gyr)/_interp(d, "L", age0_Gyr); T = T0*_interp(d, "Teff", age_Gyr)/_interp(d, "Teff", age0_Gyr)
    return float(L), float(T), float(math.sqrt(max(L, 1e-12))*(T_SUN/T)**2)
def mass(M, age_Gyr): return _interp(solve_track(M), "M", age_Gyr)
