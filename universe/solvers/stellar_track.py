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
(leaves/events handles remnants).

mode="mesa_lite" (HostWeather.solver_track="mesa_lite", CLI --stellar-solver [mesa_lite]; "semi" above stays the default
of solve_track and is unchanged):
  PMS    the same KH contraction plus (i) deuterium burning (X_D = 2e-5, Q_D = 5.49 MeV per D) stalling contraction once the
         n=1.5 central temperature Tc = 0.54 G M mu m_u/(k R) passes 1e6 K (deuterium main sequence, ~1 Myr for the Sun) and
         (ii) the pre-ZAMS out-of-equilibrium 12C(p,g)..14N + 3He burning in the inner ~0.4 M (F_CN calibrated) near the
         ZAMS.  Sun: t_PMS ~ 40 Myr (Baraffe et al. 2015: ~40-50 Myr).
  MS     multi-zone: N_SHELL mass shells over the burning region q_b(M) with a central-concentration weight
         exp(-m/(M_S q_b)) and eps ~ X (pp/CNO linear proxy) -> a radial X / mu profile; convective core for M > 1.1
         (q_cc = 0.08 (M-1)^0.6, receding as X_c drops) with an overshoot proxy (mixed mass x (1 + 1.5 alpha_ov),
         alpha_ov = 0.2 H_p); homology L = L_ZAMS (mu_eff/mu_0)^4 with mu_eff from the shell-averaged X.  (q_b, K_MIX, M_S)
         are fitted to the standard solar model: t_MS = 10 Gyr, L(4.57)/L_ZAMS = 1/0.72, X_c(4.57) = 0.34.
  He     degenerate cores (M < 2): He flash at M_c = 0.47 resolved as a "he-flash" phase (~1.5 Myr: tip -> sub-flashes ->
         ZAHB contraction); HB lifetime from the core-He fuel t_HB = F_HE M_c Q_He / L_He (Q_He = 7.27 MeV per 12C;
         Sun ~ 0.1 Gyr).  M >= 2: quiescent ignition ("giant" phase as before).
  AGB    TP-AGB with third dredge-up (lambda = 0 below M_c,min = 0.58 + 0.1 log10(Z/Zsun) or M_env < 0.2, rising to 0.5 for M >~ 3;
         core growth (1 - lambda) dM_H; dredged 12C (X_C,intershell = 0.20) enriches the envelope: surface C/O and Z
         tracked, carbon_star flag at C/O > 1) and hot-bottom burning for M >= 4 (C -> N keeps C/O < 1, L above the
         Paczynski relation by up to x2).
  WD     cooling from an energy equation c_eff dT_c/dt = -L(T_c), L = K M T_c^3.5 (K matched to Mestel), ion heat capacity
         liquid 3k + Debye solid 9k (T/theta_D)^3 int x^4 e^x/(e^x-1)^2, degenerate electrons, crystallisation at
         Gamma = 175 sweeping an n=1.5 polytrope density profile with latent heat 0.77 kT per ion -> the crystallisation
         cooling delay and the late Debye cooling drop (t_cryst, L_cryst in the track).
  Z      solve_track(M, Z): Z from the host composition (HostWeather.Z_star; solar Z = 0.0142 when unknown).
  grid   build_grid() writes a compact cached grid (universe/solvers/data/mesa_lite_grid.npz, generated in-process);
         solve_track(mode="mesa_lite") loads a grid track when (M, Z) is on the grid, else integrates.
  Still NOT MESA: no hydrostatic structure equations / opacities / nuclear network, parametric envelope radii and
  calibrated proxies; masses 0.5-3 Msun are the meaningful range."""
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


def solve_track(M, Z=ZSUN, mode="semi"):
    """Tabulated track for mass M (Msun), metallicity Z: dict(t [Gyr from birth], L [Lsun], R [Rsun], Teff [K], M, phase,
    t_pms (ZAMS age), t_ms (end of core-H burning, from birth), t_end, n_pulses, M_wd).  mode="mesa_lite" -> solve_track_ml."""
    if mode in ("mesa_lite", "mesa-lite", "ml"): return solve_track_ml(M, Z)
    if mode not in ("semi", True, "", None): raise ValueError(f"stellar track mode must be semi|mesa_lite, got {mode!r}")
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
def t_ms_Gyr(M, Z=ZSUN, mode="semi"): return solve_track(M, Z, mode)["t_ms"]       # from birth (ZAMS at solve_track(M)["t_pms"])
def max_age_Gyr(M, Z=ZSUN, mode="semi"): return solve_track(M, Z, mode)["t_end"]
def phase(M, age_Gyr, Z=ZSUN, mode="semi"):
    d = solve_track(M, Z, mode); i = int(np.clip(np.searchsorted(d["t"], age_Gyr), 0, len(d["t"]) - 1)); return str(d["phase"][i])
def track(M, L0, T0, age0_Gyr, age_Gyr, Z=ZSUN, mode="semi"):
    """(L, Teff, R) at age_Gyr given the star had (L0, T0) at age0_Gyr (shape from the solved track, anchored to L0, T0)."""
    d = solve_track(M, Z, mode)
    L = L0*_interp(d, "L", age_Gyr)/_interp(d, "L", age0_Gyr); T = T0*_interp(d, "Teff", age_Gyr)/_interp(d, "Teff", age0_Gyr)
    return float(L), float(T), float(math.sqrt(max(L, 1e-12))*(T_SUN/T)**2)
def mass(M, age_Gyr, Z=ZSUN, mode="semi"): return _interp(solve_track(M, Z, mode), "M", age_Gyr)


class _Api:
    """Leaf-API view bound to (mode, Z) for HostWeather (same names as leaves.stellar / this module)."""
    X_START_MAX = X_START_MAX
    def __init__(self, mode, Z): self.mode, self.Z = mode, float(Z); self.__name__ = f"{__name__}[{mode},Z={self.Z:g}]"
    def t_ms_Gyr(self, M): return t_ms_Gyr(M, self.Z, self.mode)
    def max_age_Gyr(self, M): return max_age_Gyr(M, self.Z, self.mode)
    def phase(self, M, age_Gyr): return phase(M, age_Gyr, self.Z, self.mode)
    def track(self, M, L0, T0, age0_Gyr, age_Gyr): return track(M, L0, T0, age0_Gyr, age_Gyr, self.Z, self.mode)
    def mass(self, M, age_Gyr): return mass(M, age_Gyr, self.Z, self.mode)

def api(mode="semi", Z=None):
    """Module itself for the legacy ("semi", solar) path (bit-identical); otherwise an _Api bound to (mode, Z)."""
    import sys
    if mode in ("semi", True) and (Z is None or Z < 0 or Z == ZSUN): return sys.modules[__name__]
    return _Api("mesa_lite" if mode in ("mesa_lite", "mesa-lite", "ml") else "semi", ZSUN if Z is None or Z < 0 else Z)


# ================================================================ mode="mesa_lite"
MU_U = 1.6605e-24; KB = 1.380649e-16; HBAR = 1.0546e-27; E2 = (4.8032e-10)**2; ME = 9.109e-28
X_D, Q_D = 2.0e-5, 2.63e18             # deuterium mass fraction; erg/g of D (5.49 MeV per D)
X_C12, Q_CN, F_CN = 2.3e-3, 8.8e17, .85  # solar 12C mass fraction; erg/g of 12C -> 14N; pre-ZAMS CN/3He burning-region factor (calibrated: Sun 40 Myr)
N_SHELL = 24; Q_B, K_MIX_ML, M_S = .21132, .59256, .44401   # SSM fit: t_MS 10 Gyr, L(4.57)/Lz = 1/.72, X_c(4.57) = .34
ALPHA_OV = .2
Q_HE = 5.85e17                         # erg/g, 3 alpha -> 12C (7.275 MeV)
T_FLASH_GYR = 1.5e-3
X_C_IS, X_O_SUN = .20, .0058           # intershell 12C mass fraction; solar envelope 16O
GAMMA_M, Z_ION, A_ION = 175., 7., 14.  # crystallisation coupling; C/O 50:50 mean ion (heat capacity)
Z_CR, A_CR = 8., 16.                   # O-rich centre crystallises first (phase-separation proxy)
L_TC_ANCHOR, T_C_ANCHOR = 1e-3, 9e6    # L/Lsun = 1e-3 (M/0.6) (T_c/9e6 K)^3.5 (radiative-envelope Mestel relation, calibrated)
GRID_PATH = __import__("os").path.join(__import__("os").path.dirname(__file__), "data", "mesa_lite_grid.npz")
GRID_MASSES = (.6, .8, 1., 1.2, 1.5, 2., 2.5, 3.)
GRID_VERSION = 1
_CACHE_ML = {}; _GRID = None

def _pms_ml(M, Z=ZSUN):
    """KH contraction (Hayashi n=1.5 / Henyey n=3) with deuterium burning and pre-ZAMS CN burning stalls."""
    Lz, Rz = L_zams(M, Z), R_zams(M, Z); TH = 4000.*max(M, .08)**.12; c = 3/7*G_CGS*(M*MSUN)**2
    ED0 = X_D*M*MSUN*Q_D; EC0 = X_C12*(Z/ZSUN)*F_CN*min(3*q_core(M), 1.)*M*MSUN*Q_CN
    def Tc(R): return .54*G_CGS*M*MSUN*.6*MU_U/(KB*R*RSUN)
    def parts(y):
        R = math.exp(min(max(y[0], -5.), 10.)); Lh = R*R*(TH/T_SUN)**4; L = max(Lh, Lz)
        fD = float(np.clip((Tc(R) - .8e6)/.4e6, 0, 1)*np.clip(y[1]/.05, 0, 1))
        fC = float(np.clip((1.3 - R/Rz)/.2, 0, 1)*np.clip(y[2]/.05, 0, 1))*(Lh <= Lz)
        return R, L, Lh, fD, fC
    def rhs(t, y):
        R, L, Lh, fD, fC = parts(y); cc = c if Lh > Lz else c*7/4; fn = min(fD + fC, .999); Lg = L*LSUN*GYR
        return [-(1 - fn)*L*LSUN*R*RSUN/cc*GYR, -fD*Lg/ED0, -fC*Lg/EC0]
    def zams(t, y): return y[0] - math.log(Rz)
    zams.terminal = True
    R0 = max(4.*M**.6, 2.5*Rz)
    sol = solve_ivp(rhs, (0, 5.), [math.log(R0), 1., 1.], method="LSODA", first_step=1e-9, rtol=1e-8, atol=1e-10, events=zams, dense_output=True)
    tp = float(sol.t[-1]); ts = np.geomspace(1e-6, tp, 100); ts[-1] = tp
    Y = sol.sol(ts); R = np.exp(Y[0]); L = np.array([parts(Y[:, i])[1] for i in range(len(ts))])
    t_D = float(ts[np.argmax(Y[1] < .5)]) if (Y[1] < .5).any() else None
    return np.r_[0., ts], np.r_[max(R0**2*(TH/T_SUN)**4, Lz), L], np.r_[R0, R], t_D

def _ms_ml(M, Z=ZSUN):
    """Multi-zone MS: N_SHELL burning shells (X profile), convective core + overshoot for M > 1.1."""
    Lz = L_zams(M, Z); x0 = X0_of(Z); mu0 = mu_of(x0)
    qb = float(np.clip(Q_B*max(M, .1)**.4, .05, .5)) if M >= .35 else 1.
    edges = np.linspace(0, qb, N_SHELL + 1); dm = np.diff(edges)*M*MSUN; mc = .5*(edges[1:] + edges[:-1])
    w = np.exp(-mc/(M_S*qb)); qcc = 0. if M < 1.1 else min(.08*(M - 1.)**.6, .4)
    def qmix(Xc): return 1. if M < .35 else qcc*(.7 + .3*Xc/x0)*(1 + 1.5*ALPHA_OV)
    def L_of(X):
        Xb = float((X*dm).sum()/dm.sum()); Xe = x0 - K_MIX_ML*(x0 - Xb) if M >= .35 else Xb
        return Lz*(mu_of(Xe)/mu0)**4
    def rhs(t, X):
        X = np.maximum(X, 0.); Lg = L_of(X)*LSUN*GYR/Q_H; e = w*X; r = Lg*e/max(float((e*dm).sum()), 1e-300)
        mix = mc < qmix(X[0])
        if mix.any(): r[mix] = (r[mix]*dm[mix]).sum()/dm[mix].sum()
        return -r
    def end(t, X): return X[0] - .01
    end.terminal = True
    sol = solve_ivp(rhs, (0, 1e3), np.full(N_SHELL, x0), rtol=1e-7, atol=1e-9, events=end, dense_output=True, max_step=.25)
    t_ms = float(sol.t[-1]); ts = np.linspace(0, t_ms, 400); X = sol.sol(ts)
    L = np.array([L_of(X[:, i]) for i in range(len(ts))]); R = R_zams(M, Z)*(L/Lz)**.35
    prof = dict(m_shell=mc*M, X_zams=X[:, 0], X_tams=X[:, -1], mu_tams=mu_of(np.clip(X[:, -1], 0, 1)), X_c=X[0], q_mix0=qmix(x0))
    return ts, L, R, M*np.ones_like(ts), t_ms, prof

def _post_ml(M, t_ms, L_tams, R_tams, Z):
    """SG -> RGB -> (he-flash -> HB | giant) -> AGB/TP-AGB (3DU, HBB) -> WD (Debye/crystallisation)."""
    t_sg = .08*t_ms; L_b = 2.5*L_tams; Mc0 = min((L_b/2.3e5)**(1/6), .4); R_b = rgb_radius(Mc0)*max(M, .5)**.3
    tsg = np.linspace(0, t_sg, 60)[1:]; f = tsg/t_sg
    Lsg = np.exp(np.log(L_tams)*(1 - f) + np.log(max(L_b, L_tams*1.05))*f); Rsg = np.exp(np.log(R_tams)*(1 - f) + np.log(R_b)*f)
    T, L_, R_, M_, P_ = list(t_ms + tsg), list(Lsg), list(Rsg), [M]*len(tsg), ["subgiant"]*len(tsg); info = dict(he_flash=False)
    if M < 2:
        def rhs(t, y):
            Mc, Ms = y; L = rgb_lum(Mc); R = rgb_radius(Mc)*max(Ms, .5)**.3
            return [L*LSUN*GYR/(X0*Q_H*MSUN), -4e-13*ETA_REIMERS*L*R/Ms*1e9]
        def flash(t, y): return y[0] - MC_FLASH
        flash.terminal = True
        def gone(t, y): return y[1] - y[0] - .02
        gone.terminal = True
        sol = solve_ivp(rhs, (0, 50.), [Mc0, M], rtol=1e-8, atol=1e-12, events=[flash, gone], dense_output=True)
        tr = np.geomspace(1e-4, sol.t[-1], 300); Mc, Ms = sol.sol(tr); t0 = T[-1]
        T += list(t0 + tr); L_ += list(rgb_lum(Mc)); R_ += list(rgb_radius(Mc)*np.maximum(Ms, .5)**.3); M_ += list(Ms); P_ += ["rgb"]*len(tr)
        Mfin = float(Ms[-1]); Mcf = float(Mc[-1])
        if Mfin - Mcf < .021:                                             # envelope lost on the RGB: He white dwarf, no flash
            info.update(he_wd=True); tc = _wd_ml(T, L_, R_, M_, P_, Mcf); info.update(tc); return T, L_, R_, M_, P_, 0, info
        Lhb = 50.*(Mcf/.47)**2; Rhb = math.sqrt(Lhb)*(T_SUN/4900.)**2; Ltip, Rtip = L_[-1], R_[-1]
        tf = np.linspace(0, T_FLASH_GYR, 41)[1:]; f = tf/T_FLASH_GYR       # flash: tip -> 5 sub-flashes -> ZAHB contraction
        bump = sum(.35*.6**k*np.exp(-((f - (.15 + .15*k))/.025)**2) for k in range(5))
        Lf = np.exp(np.log(Ltip)*(1 - f**.5) + np.log(Lhb)*f**.5)*(1 + bump); Rf = np.exp(np.log(Rtip)*(1 - f**.5) + np.log(Rhb)*f**.5)
        T += list(T[-1] + tf); L_ += list(Lf); R_ += list(Rf); M_ += [Mfin]*len(tf); P_ += ["he-flash"]*len(tf); info.update(he_flash=True, L_tip=float(Ltip))
        L_he = .4*Lhb; t_hb = .44*Mcf*MSUN*Q_HE/(L_he*LSUN)/GYR             # core-He fuel (44 % burnt to C/O) / He luminosity (40 % of L)
        th = np.linspace(0, t_hb, 30)[1:]; Lh = Lhb*(1 + .5*(th/t_hb)**2)
        T += list(T[-1] + th); L_ += list(Lh); R_ += list(np.sqrt(Lh)*(T_SUN/4900.)**2); M_ += [Mfin]*len(th); P_ += ["hb"]*len(th)
        info["t_hb"] = float(t_hb); Mc_agb = max(.5, Mcf + .03)
    else:
        Lg = L_b*3; th = np.linspace(0, .1*t_ms, 40)[1:]
        T += list(T[-1] + th); L_ += [Lg]*len(th); R_ += [R_b*2]*len(th); M_ += [M]*len(th); P_ += ["giant"]*len(th)
        Mfin = M; Mc_agb = min(.36 + .1*M, 1.2)
    n_tp = 0
    if M < 8: n_tp = _agb_wd_ml(T, L_, R_, M_, P_, Mfin, Mc_agb, M, Z, info)
    return T, L_, R_, M_, P_, n_tp, info

def lambda_3du(M_init, Mc, Menv, Z=ZSUN):
    """Third-dredge-up efficiency proxy: 0 below M_c,min(Z) or for thin envelopes, -> lambda_max(M) (Karakas-like)."""
    lmax = float(np.clip((M_init - 1.2)/1.8, 0, 1))*.5; mcmin = .58 + .1*math.log10(max(Z, 1e-5)/ZSUN)
    return lmax*float(np.clip((Mc - mcmin)/.05, 0, 1))*float(np.clip((Menv - .2)/.3, 0, 1))

def _agb_wd_ml(T, L_, R_, M_, P_, M, Mc0, M_init, Z, info):
    L_start = L_[-1]; Lp = lambda Mc: 59250.*max(Mc - .495, 1e-3); Mc1 = max(Mc0, .52); L1 = Lp(Mc1)
    zf = Z/ZSUN; XC0, XO = X_C12*zf, X_O_SUN*zf; hbb_on = M_init >= 4.
    if M - Mc1 < .02:
        info.update(_wd_ml(T, L_, R_, M_, P_, min(M, Mc1))); info.update(C_O=.55, carbon_star=False, Z_surface=Z, hbb=False); return 0
    t_eagb = .02 if M < 3 else .005; te = np.linspace(0, t_eagb, 30)[1:]; f = te/t_eagb; t0 = T[-1]
    Le = np.exp(np.log(L_start)*(1 - f) + np.log(L1)*f); Tagb = T_AGB*max(M, .5)**.05
    T += list(t0 + te); L_ += list(Le); R_ += list(np.sqrt(Le)*(T_SUN/Tagb)**2); M_ += [M]*len(te); P_ += ["agb"]*len(te)
    def Lof(Mc, Ms): return Lp(Mc)*(1 + (float(np.clip((M_init - 3.5)/1.5, 0, 1))*float(np.clip((Ms - Mc - .5), 0, 1)) if hbb_on else 0.))
    def rhs(t, y):
        Mc, Ms, Cm = y; Ms = max(Ms, Mc + 1e-3, .05); Me = max(Ms - Mc, 1e-4); L = Lof(Mc, Ms); R = math.sqrt(L)*(T_SUN/Tagb)**2
        reim = 4e-13*.5*L*R/Ms; sw = 4.83e-9*Ms**-2.1*L**2.7*4e-13*ETA_BLOECKER*L*R/Ms; mdot = (reim + sw)*1e9
        dH = L*LSUN*GYR/(X0*Q_H*MSUN); lam = lambda_3du(M_init, Mc, Me, Z)
        dC = lam*dH*X_C_IS - Cm/Me*(mdot + (1 - lam)*dH) - (Cm*1e4*float(np.clip(Me - 1., 0, 1)) if hbb_on else 0.)   # HBB: tau ~ 0.1 Myr
        return [(1 - lam)*dH, -mdot, dC]
    def bare(t, y): return y[1] - y[0] - .01
    bare.terminal = True
    def chand(t, y): return y[0] - 1.38
    chand.terminal = True
    sol = solve_ivp(rhs, (0, .5), [Mc1, M, XC0*(M - Mc1)], method="LSODA", rtol=1e-8, atol=1e-12, events=[bare, chand], dense_output=True, max_step=1e-3)
    T_tp = float(sol.t[-1]); t0 = T[-1]; ip_ = lambda Mc_: 10**(3.05 - 4.5*(Mc_ - 1))*1e-9
    tps, tt = [], 0.
    while tt < T_tp and len(tps) < 400: tt += ip_(float(sol.sol(tt)[0])); tps.append(tt)
    tps = [x for x in tps if x < T_tp]; grid = set(np.linspace(0, T_tp, 120)[1:].tolist())
    for x in tps: ip = ip_(float(sol.sol(x)[0])); grid.update([x, x + .02*ip, x + .3*ip])
    tg = np.array(sorted(g for g in grid if 0 < g <= T_tp)); Mc, Ms, Cm = sol.sol(tg); dip = np.ones_like(tg)
    for x in tps:
        ip = ip_(float(sol.sol(x)[0])); k = (tg > x) & (tg < x + .3*ip); dip[k] = np.minimum(dip[k], .5 + .5*(tg[k] - x)/(.3*ip))
    L = np.array([Lof(a, b) for a, b in zip(Mc, Ms)])*dip; R = np.sqrt(L)*(T_SUN/Tagb)**2
    T += list(t0 + tg); L_ += list(L); R_ += list(R); M_ += list(Ms); P_ += ["tp-agb"]*len(tg)
    Me = np.maximum(Ms - Mc, 1e-4); co = (np.maximum(Cm, 0)/Me/12)/(XO/16); ok = Me > .02
    info.update(C_O=float(co[ok][-1]) if ok.any() else float(co[0]), C_O_max=float(co[ok].max()) if ok.any() else float(co[0]),
                carbon_star=bool(ok.any() and co[ok].max() > 1.), hbb=bool(hbb_on), dMc_dredged=float(Mc1 + (sol.sol(T_tp)[0] - Mc1)),
                Z_surface=float(Z + (max(float(Cm[ok][-1]), 0.)/float(Me[ok][-1]) - XC0 if ok.any() else 0.)))
    info.update(_wd_ml(T, L_, R_, M_, P_, float(Mc[-1]))); return len(tps)

_LE = None
def _lane_emden15():
    """n=1.5 polytrope: (rho/rho_c, enclosed mass fraction) arrays from the centre out."""
    global _LE
    if _LE is None:
        def f(x, y): return [y[1], -max(y[0], 0.)**1.5 - 2*y[1]/x]
        def surf(x, y): return y[0]
        surf.terminal = True
        x0 = 1e-6; s = solve_ivp(f, (x0, 10.), [1 - x0*x0/6, -x0/3], events=surf, rtol=1e-10, atol=1e-12, dense_output=True)
        xs = np.linspace(x0, s.t[-1], 800); th, dth = s.sol(xs); m = -xs*xs*dth; _LE = (np.maximum(th, 0)**1.5, m/m[-1])
    return _LE

def wd_cooling(Mwd, t_end_Gyr=T_WD_GYR, n=120):
    """WD energy-equation cooling (see module doc).  Returns dict(t [Gyr after the AGB], L [Lsun], T_c, f_cryst, t_cryst, L_cryst, theta_D)."""
    Mg = Mwd*MSUN; Rwd = .0127*Mwd**(-1/3)*math.sqrt(max(1 - (Mwd/1.44)**(4/3), 1e-4))*RSUN
    rhoc = 5.99*3*Mg/(4*math.pi*Rwd**3); Am = A_ION*MU_U; cv0 = 3*KB/Am
    tau = 8.8e6*(14/12)**-1*Mwd**(5/7)*YR; K = L_TC_ANCHOR*LSUN/(.6*MSUN*T_C_ANCHOR**3.5)      # Mestel envelope L = K M T_c^3.5
    C_G = Z_CR**2*E2*(4*math.pi/(3*A_CR*MU_U))**(1/3)/(KB*GAMMA_M); Tcr_c = C_G*rhoc**(1/3)
    thD = .45*HBAR*math.sqrt(4*math.pi*rhoc*Z_ION**2*E2)/Am/KB
    ne = rhoc/(2*MU_U); EF = HBAR**2*(3*math.pi**2*ne)**(2/3)/(2*ME); cel = math.pi**2/2*KB/EF*KB/(2*MU_U)
    rr, mm = _lane_emden15()
    def fx(Tk): r = (Tk/C_G)**3/rhoc; return float(np.interp(-r, -rr, mm)) if r < 1 else 0.   # mass with rho > rho_cr(T)
    xg = np.linspace(1e-4, 1, 400)
    def debye(Tk):
        y = thD/Tk
        if y > 60: return 4*math.pi**4/5*(Tk/thD)**3*3/3                 # low-T limit 12 pi^4/5 (T/theta)^3 /3k units
        x = xg*y; ig = x**4*np.exp(x)/np.expm1(x)**2; return float(3*(1/y)**3*np.trapezoid(ig, x))
    T0 = (LSUN*((1e-5*GYR)/tau)**-1.4/(K*Mg))**(1/3.5)
    Tg = np.geomspace(T0, 1e4, 3000); fxs = np.array([fx(x) for x in Tg])
    cv = np.array([(1 - a)*cv0 + a*cv0*debye(x) for a, x in zip(fxs, Tg)]) + cel*Tg
    lat = .77*KB*Tg/Am*np.maximum(-np.gradient(fxs, Tg), 0.)
    dtdT = Mg*(cv + lat)/(K*Mg*Tg**3.5)
    t = 1e-5*GYR + np.r_[0., np.cumsum(.5*(dtdT[1:] + dtdT[:-1])*(Tg[:-1] - Tg[1:]))]
    Lg = K*Mg*Tg**3.5/LSUN; tw = np.geomspace(1e-5, t_end_Gyr, n)
    lt = np.interp(np.log(tw*GYR), np.log(t), np.log(Lg)); Tc = np.exp(np.interp(np.log(tw*GYR), np.log(t), np.log(Tg)))
    k = np.argmax(fxs > 1e-3); tcr = float(t[k]/GYR) if fxs.max() > 1e-3 else None
    return dict(t=tw, L=np.exp(lt), T_c=Tc, f_cryst=np.array([fx(x) for x in Tc]), t_cryst=tcr,
                L_cryst=float(Lg[k]) if tcr else None, theta_D=float(thD), T_cryst_c=float(Tcr_c), rho_c=float(rhoc))

def _wd_ml(T, L_, R_, M_, P_, Mwd):
    t0 = T[-1]; Rwd = .0127*Mwd**(-1/3)*math.sqrt(max(1 - (Mwd/1.44)**(4/3), 1e-4)); w = wd_cooling(Mwd)
    T += list(t0 + w["t"]); L_ += list(w["L"]); R_ += [Rwd]*len(w["t"]); M_ += [Mwd]*len(w["t"]); P_ += ["wd"]*len(w["t"])
    return dict(t_cryst=(t0 + w["t_cryst"]) if w["t_cryst"] else None, wd_t_cryst=w["t_cryst"], L_cryst=w["L_cryst"], theta_D=w["theta_D"])

PHASES = ("pms", "ms", "subgiant", "rgb", "he-flash", "hb", "giant", "agb", "tp-agb", "wd")

def _compute_ml(M, Z):
    tq, Lq, Rq, t_D = _pms_ml(M, Z); tz = float(tq[-1])
    ts, L, R, Ms, t_ms, prof = _ms_ml(M, Z)
    tp, Lp, Rp, Mp, pp, n_tp, info = _post_ml(M, t_ms, L[-1], R[-1], Z)
    t = np.concatenate([tq[:-1], tz + ts, tz + np.array(tp)]); L = np.concatenate([Lq[:-1], L, Lp]); R = np.concatenate([Rq[:-1], R, Rp])
    Mm = np.concatenate([np.full(len(tq) - 1, M), Ms, Mp]); ph = ["pms"]*(len(tq) - 1) + ["ms"]*len(ts) + list(pp)
    if info.get("t_cryst") is not None: info["t_cryst"] = tz + info["t_cryst"]
    d = dict(t=t, L=L, R=R, Teff=T_SUN*(L/R**2)**.25, M=Mm, phase=np.array(ph), t_pms=tz, t_D=t_D, t_ms=tz + t_ms, t_end=float(t[-1]),
             n_pulses=int(n_tp), M_wd=float(Mm[-1]) if ph[-1] == "wd" else None, Z=Z, mode="mesa_lite", profile=prof, **info)
    for s_ in _SCAL: d.setdefault(s_, None)
    for b_ in _BOOL: d[b_] = bool(d.get(b_) or False)
    return d

_SCAL = ("t_pms", "t_D", "t_ms", "t_end", "n_pulses", "M_wd", "Z", "t_hb", "L_tip", "C_O", "C_O_max", "Z_surface", "dMc_dredged",
         "t_cryst", "wd_t_cryst", "L_cryst", "theta_D")
_BOOL = ("he_flash", "he_wd", "carbon_star", "hbb")

def _grid_key(M, Z): return f"M{M:.3f}_Z{Z:.5f}"

def build_grid(path=GRID_PATH, masses=GRID_MASSES, Z=ZSUN):
    """Write the compact mesa_lite track grid (npz, generated in-process).  Returns the path."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True); out = dict(version=np.array(GRID_VERSION))
    for M in masses:
        d = _compute_ml(round(float(M), 3), round(float(Z), 5)); k = _grid_key(M, Z)
        for a in ("t", "L", "R", "M"): out[f"{k}/{a}"] = d[a]
        out[f"{k}/phase"] = np.array([PHASES.index(p) for p in d["phase"]], np.int8)
        out[f"{k}/scal"] = np.array([np.nan if d.get(s) is None else float(d[s]) for s in _SCAL])
        out[f"{k}/bool"] = np.array([bool(d.get(b, False)) for b in _BOOL])
        for a, v in d["profile"].items(): out[f"{k}/prof_{a}"] = np.atleast_1d(v)
    np.savez_compressed(path, **out); global _GRID; _GRID = None; return path

def _from_grid(M, Z):
    global _GRID
    import os
    if _GRID is None:
        _GRID = {}
        if os.path.exists(GRID_PATH):
            with np.load(GRID_PATH) as g:
                if int(g["version"]) == GRID_VERSION: _GRID = {k: g[k] for k in g.files}
    k = _grid_key(M, Z)
    if f"{k}/t" not in _GRID: return None
    G_ = lambda a: _GRID[f"{k}/{a}"]; L, R = G_("L"), G_("R")
    d = dict(t=G_("t"), L=L, R=R, Teff=T_SUN*(L/R**2)**.25, M=G_("M"), phase=np.array([PHASES[i] for i in G_("phase")]), mode="mesa_lite",
             profile={a[len(k) + 6:]: (v if v.size > 1 else float(v[0])) for a, v in _GRID.items() if a.startswith(f"{k}/prof_")}, from_grid=True)
    for s_, v in zip(_SCAL, G_("scal")): d[s_] = None if np.isnan(v) else float(v)
    d["n_pulses"] = int(d["n_pulses"]); d.update({b: bool(v) for b, v in zip(_BOOL, G_("bool"))})
    return d

def solve_track_ml(M, Z=ZSUN, use_grid=True):
    """mesa_lite track (see module doc); same keys as solve_track plus t_D, profile, he_flash, t_hb, C_O, carbon_star, hbb,
    Z_surface, t_cryst, L_cryst, theta_D."""
    k = (round(float(M), 3), round(float(Z), 5), bool(use_grid))
    if k in _CACHE_ML: return _CACHE_ML[k]
    M_, Z_ = max(k[0], .08), k[1]
    d = (_from_grid(M_, Z_) if use_grid else None) or _compute_ml(M_, Z_)
    _CACHE_ML[k] = d; return d
