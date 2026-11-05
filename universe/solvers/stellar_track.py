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
  HB     core-He burning clump: L ~ 50 Lsun (M_c-scaled), Teff ~ 4800-5000 K, for 0.1 Gyr; the track is clamped after that
         (AGB / WD not modelled).
All ODEs integrated with scipy solve_ivp (RK45, rtol 1e-8) once per mass and tabulated (log-age interpolation).
Fidelity / gaps: one-zone MS (no convective-core overshoot, no metallicity dependence, no pre-MS); masses ~0.3-2 Msun
are the calibrated range (fully convective M < 0.35 just burn slowly on the MS); intermediate/massive stars' blue loops,
He-burning of M > 2 and supernova endpoints are not modelled here (leaves/events handles remnants)."""
import math
import numpy as np
from scipy.integrate import solve_ivp

Q_H = 6.4e18; MSUN = 1.989e33; LSUN = 3.828e33; GYR = 3.15576e16
X0 = .70; K_MIX = .4417;   # K_MIX and q_core's .1575 fitted to the Sun: t_MS = 10.0 Gyr, L(4.57)/L_ZAMS = 1/0.72
T_SUN = 5772.
ETA_REIMERS = .5; MC_FLASH = .47; T_HB_GYR = .1
_CACHE = {}


def q_core(M):
    """Fraction of the star's mass that burns H on the MS (convective-core / radiative-core proxy)."""
    return float(np.clip(.1575*max(M, .1)**.4, .05, .4)) if M >= .35 else 1.   # fully convective low-mass stars mix everything

def L_zams(M): return float(.72*M**4.0 if M < 1.5 else .72*1.5**4*(M/1.5)**3.5)
def R_zams(M): return float(.89*M**.8 if M >= .35 else .89*.35**.8*(M/.35)**.95)
def mu_of(X): return 4/(3 + 5*X)

def rgb_radius(Mc): return 3.7e3*Mc**4/(1 + Mc**3 + 1.75*Mc**4)      # Rsun (Rappaport+95)
def rgb_lum(Mc): return 2.3e5*Mc**6

def _ms(M):
    qc = q_core(M); Lz = L_zams(M); mu0 = mu_of(X0)
    def L_of(Xc):
        Xe = X0 - K_MIX*(X0 - Xc) if M >= .35 else Xc
        return Lz*(mu_of(Xe)/mu0)**4
    def rhs(t, y): return [-L_of(y[0])*LSUN*GYR/(Q_H*qc*M*MSUN)]
    def end(t, y): return y[0] - .01
    end.terminal = True
    sol = solve_ivp(rhs, (0, 1e3), [X0], rtol=1e-8, atol=1e-10, events=end, dense_output=True, max_step=.5)
    t_ms = float(sol.t[-1]); ts = np.linspace(0, t_ms, 400); Xc = sol.sol(ts)[0]
    L = np.array([L_of(x) for x in Xc]); R = R_zams(M)*(L/Lz)**.35
    return ts, L, R, M*np.ones_like(ts), t_ms


def _post(M, t_ms, L_tams, R_tams):
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
    else:   # intermediate mass: Hertzsprung gap then a short He-burning giant phase at ~ constant L (no blue loop)
        Lg = L_b*3; th = np.linspace(0, .1*t_ms, 40)[1:]
        out_t += list(out_t[-1] + th); out_L += [Lg]*len(th); out_R += [R_b*2]*len(th); out_M += [M]*len(th); out_p += ["giant"]*len(th)
    return np.array(out_t), np.array(out_L), np.array(out_R), np.array(out_M), out_p


def solve_track(M):
    """Tabulated track for mass M (Msun): dict(t [Gyr], L [Lsun], R [Rsun], Teff [K], M, phase, t_ms)."""
    k = round(float(M), 3)
    if k in _CACHE: return _CACHE[k]
    M = max(k, .08); ts, L, R, Ms, t_ms = _ms(M)
    tp, Lp, Rp, Mp, pp = _post(M, t_ms, L[-1], R[-1])
    t = np.concatenate([ts, tp]); L = np.concatenate([L, Lp]); R = np.concatenate([R, Rp]); Mm = np.concatenate([Ms, Mp])
    ph = ["ms"]*len(ts) + pp; T = T_SUN*(L/R**2)**.25
    d = dict(t=t, L=L, R=R, Teff=T, M=Mm, phase=np.array(ph), t_ms=t_ms, t_end=float(t[-1])); _CACHE[k] = d; return d


def _interp(d, key, age):
    a = float(np.clip(age, 0, d["t_end"])); v = d[key]
    return float(np.exp(np.interp(a, d["t"], np.log(v)))) if key != "M" else float(np.interp(a, d["t"], v))

# ---- leaf-compatible API (leaves.stellar names) ----
X_START_MAX = .9
def t_ms_Gyr(M): return solve_track(M)["t_ms"]
def max_age_Gyr(M): return solve_track(M)["t_end"]
def phase(M, age_Gyr):
    d = solve_track(M); i = int(np.clip(np.searchsorted(d["t"], age_Gyr), 0, len(d["t"]) - 1)); return str(d["phase"][i])
def track(M, L0, T0, age0_Gyr, age_Gyr):
    """(L, Teff, R) at age_Gyr given the star had (L0, T0) at age0_Gyr (shape from the solved track, anchored to L0, T0)."""
    d = solve_track(M)
    L = L0*_interp(d, "L", age_Gyr)/_interp(d, "L", age0_Gyr); T = T0*_interp(d, "Teff", age_Gyr)/_interp(d, "Teff", age0_Gyr)
    return float(L), float(T), float(math.sqrt(max(L, 1e-12))*(T_SUN/T)**2)
def mass(M, age_Gyr): return _interp(solve_track(M), "M", age_Gyr)
