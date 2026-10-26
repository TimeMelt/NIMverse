"""Toy main-sequence evolution track (no post-MS, no spectra).
t_MS = 10 Gyr * M^-2.5.  Luminosity follows the Gough (1981) faint-young-Sun law generalised to fractional MS age
x = age/t_MS:  L(x)/L(x_sun) = 1/(1 + 0.4(1 - x/x_sun)), x_sun = 4.57/10 (so the Sun brightens ~8-9%/Gyr today,
ZAMS L ~0.7 Lsun).  Teff ∝ L^0.1 along the track (Sun ~5600 K -> 5800 K), R from L = R^2 Teff^4.
Activity: flare rate for M >= 0.6 spins down (Skumanich ~ age^-1/2); fully convective M < 0.6 stays active until
a ~ (8 Gyr * (0.6/M))-long saturated phase ends."""
import math
X_SUN = .457
def t_ms_Gyr(M): return 10.*max(M, .08)**-2.5
def frac_age(M, age_Gyr): return min(max(age_Gyr, 0.)/t_ms_Gyr(M), 1.)
def lum_factor(M, age_Gyr):
    """Relative luminosity shape f(x) (only ratios matter)."""
    return 1./(1. + .4*(1. - frac_age(M, age_Gyr)/X_SUN))
def track(M, L0, T0, age0_Gyr, age_Gyr):
    """(L, Teff, R) at age_Gyr given the star had (L0, T0) at age0_Gyr."""
    r = lum_factor(M, age_Gyr)/lum_factor(M, age0_Gyr); L = L0*r; T = T0*r**.1
    return float(L), float(T), float(math.sqrt(max(L, 1e-12))*(5772./T)**2)
def activity_factor(M, age0_Gyr, age_Gyr):
    """Flare-rate multiplier relative to age0."""
    if M < .6:
        sat = 8.*(.6/max(M, .08)); g = lambda a: 1. if a < sat else (a/sat)**-.5
    else: g = lambda a: max(a, .05)**-.5
    return float(g(age_Gyr)/g(age0_Gyr))
def hz(L): s = math.sqrt(max(L, 0.)); return .95*s, 1.67*s
