"""Toy stellar evolution (NOT MESA): main sequence + a crude subgiant->RGB extension, toy blackbody spectral bands, and a
deterministic system age prior.

MS (x = age/t_MS <= 1): t_MS = 10 Gyr * M^-2.5.  Luminosity follows the Gough (1981) faint-young-Sun law generalised to
x:  L(x)/L(x_sun) = 1/(1 + 0.4(1 - x/x_sun)), x_sun = 4.57/10 (Sun brightens ~8-9%/Gyr today, ZAMS L ~0.7 Lsun).
Teff ∝ L^0.1 along the MS (Sun ~5600 K -> 5800 K), R from L = R^2 Teff^4.
Post-MS toy (x > 1): L = L_MS_end * (1 + A_POST (x-1)^B_POST) (subgiant slow climb, RGB steepening), Teff cools as
(L/L_MS_end)^-0.15 so R swells; ages are clamped at X_MAX*t_MS (L ~3x MS end) so sims stay finite.  Phases:
"ms", "subgiant" (1 < x < X_RGB), "rgb".
Activity: flare rate for M >= 0.6 spins down (Skumanich ~ age^-1/2); fully convective M < 0.6 stays active until
a ~ (8 Gyr * (0.6/M))-long saturated phase ends; post-MS it drops further by 1/(1 + 10(x-1)) (giants flare less).
Spectra: Planck band fractions (no radiative transfer, no line blanketing / chromospheric UV)."""
import math
import numpy as np
X_SUN = .457; A_POST, B_POST = 8., 1.5; X_RGB = 1.15; X_MAX = 1.4; X_START_MAX = .9
def t_ms_Gyr(M): return 10.*max(M, .08)**-2.5
def x_age(M, age_Gyr): return max(age_Gyr, 0.)/t_ms_Gyr(M)
def frac_age(M, age_Gyr): return min(x_age(M, age_Gyr), 1.)
def max_age_Gyr(M): return X_MAX*t_ms_Gyr(M)
def phase(M, age_Gyr):
    x = x_age(M, age_Gyr); return "ms" if x <= 1 else "subgiant" if x < X_RGB else "rgb"
def _post(x): return 1. + A_POST*max(min(x, X_MAX) - 1., 0.)**B_POST
def lum_factor(M, age_Gyr):
    """Relative luminosity shape (only ratios matter); MS Gough law, then post-MS toy climb."""
    x = x_age(M, age_Gyr); return 1./(1. + .4*(1. - min(x, 1.)/X_SUN))*_post(x)
def teff_factor(M, age_Gyr):
    x = x_age(M, age_Gyr); return (1./(1. + .4*(1. - min(x, 1.)/X_SUN)))**.1*_post(x)**-.15
def track(M, L0, T0, age0_Gyr, age_Gyr):
    """(L, Teff, R) at age_Gyr given the star had (L0, T0) at age0_Gyr."""
    L = L0*lum_factor(M, age_Gyr)/lum_factor(M, age0_Gyr); T = T0*teff_factor(M, age_Gyr)/teff_factor(M, age0_Gyr)
    return float(L), float(T), float(math.sqrt(max(L, 1e-12))*(5772./T)**2)
def activity_factor(M, age0_Gyr, age_Gyr):
    """Flare-rate multiplier relative to age0."""
    if M < .6:
        sat = 8.*(.6/max(M, .08)); g0 = lambda a: 1. if a < sat else (a/sat)**-.5
    else: g0 = lambda a: max(a, .05)**-.5
    g = lambda a: g0(a)/(1. + 10.*max(x_age(M, a) - 1., 0.))
    return float(g(age_Gyr)/g(age0_Gyr))
def hz(L): s = math.sqrt(max(L, 0.)); return .95*s, 1.67*s

# ---- toy blackbody bands (nm) ----
BANDS = {"uvc": (100., 280.), "uvb": (280., 315.), "uva": (315., 400.), "nuv": (200., 400.), "optical": (400., 700.), "ir": (700., 1e5)}
UV_BAND = "nuv"; T_SUN = 5772.
_HC_K = 1.4387769e7       # hc/k in nm*K
def _band_frac(T, lo, hi, n=400):
    """Fraction of sigma T^4 emitted in [lo, hi] nm (Planck, log-spaced trapezoid in x = hc/(lambda k T))."""
    x = np.linspace(_HC_K/(hi*T), _HC_K/(lo*T), n); x = np.clip(x, 1e-6, 700.)
    f = x**3/np.expm1(x); return float(np.sum(.5*(f[1:] + f[:-1])*np.diff(x))*15./math.pi**4)
def band_fractions(T):
    """{band: fraction of bolometric flux} for a blackbody at Teff T."""
    return {k: _band_frac(float(T), lo, hi) for k, (lo, hi) in BANDS.items()}
def uv_surface_rel(T):
    """Surface UV (UV_BAND) flux relative to a solar-Teff blackbody: frac_uv(T) T^4 / (frac_uv(Tsun) Tsun^4)."""
    return _band_frac(float(T), *BANDS[UV_BAND])*float(T)**4/(_band_frac(T_SUN, *BANDS[UV_BAND])*T_SUN**4)
def uv_bands(T, L=None):
    """Snapshot dict: band fractions, uv/optical ratio, surface UV vs Sun, and (if L given) band luminosities in Lsun."""
    fr = band_fractions(T); d = dict(T_eff=float(T), frac=fr, uv_to_optical=float(fr[UV_BAND]/max(fr["optical"], 1e-30)), uv_surface_rel_sun=uv_surface_rel(T))
    if L is not None: d["L_band"] = {k: float(v*L) for k, v in fr.items()}
    return d

# ---- system age prior ----
def age_prior(M, seed, lo_Gyr=.1):
    """Deterministic (seed, mass) draw, log-uniform in [lo, X_START_MAX*t_MS] (cap 12 Gyr): massive stars are younger."""
    hi = max(min(X_START_MAX*t_ms_Gyr(M), 12.), lo_Gyr*1.01)
    u = np.random.default_rng([abs(int(seed)) % 2**31, int(round(M*1e4)) % 2**31, 4242]).random()
    return float(math.exp(math.log(lo_Gyr) + u*(math.log(hi) - math.log(lo_Gyr))))
