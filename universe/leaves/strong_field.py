"""Strong-field helpers (honest stubs, not a GR solver).  Units: G=c=1 lengths in units of M (geometrized)
unless *_au variants.  Newtonian limit: every correction -> 0 / factor -> 1 as r/M -> inf (see newtonian_limit_ok)."""
import numpy as np
RS_SUN_AU = 2.953/1.495979e8

def r_s(M_sun): return RS_SUN_AU*M_sun                                          # AU

def r_isco(M_sun=1., chi=0.):
    """Kerr prograde ISCO (Bardeen 1972); chi=0 -> 6 GM/c^2 = 3 r_s.  Returns AU."""
    z1 = 1 + (1-chi**2)**(1/3)*((1+chi)**(1/3) + (1-chi)**(1/3)); z2 = np.sqrt(3*chi**2 + z1**2)
    return float((3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2)))*r_s(M_sun)/2)

def r_tde(M_bh, m_star=1., R_star_au=.00465):
    """Tidal disruption radius R* (M/m*)^(1/3) (AU); compare with r_s to decide if star is swallowed whole."""
    return float(R_star_au*(M_bh/m_star)**(1/3))
def tde_visible(M_bh, m_star=1., R_star_au=.00465): return r_tde(M_bh, m_star, R_star_au) > r_s(M_bh)

def gamma_static(r_au, M_sun):
    """Gravitational time-dilation dt/dtau for a static observer at r (Schwarzschild); inf inside horizon."""
    x = r_s(M_sun)/r_au; return float(np.inf if x >= 1 else 1/np.sqrt(1-x))
def gamma_orbit(r_au, M_sun):
    """Circular geodesic dt/dtau = 1/sqrt(1 - 1.5 r_s/r) (includes SR); valid r > 1.5 r_s (photon sphere)."""
    x = 1.5*r_s(M_sun)/r_au; return float(np.inf if x >= 1 else 1/np.sqrt(1-x))

def pn_precession_rad(a_au, e, M_sun):
    """1PN periastron advance per orbit: 6 pi G M / (c^2 a (1-e^2)) = 3 pi r_s/(a(1-e^2))."""
    return float(3*np.pi*r_s(M_sun)/(a_au*(1-e**2)))

def pn_accel_corr(r_au, v_over_c, M_sun):
    """Order-of-magnitude 1PN fractional correction to Newtonian acceleration ~ (r_s/r + v^2/c^2)."""
    return float(r_s(M_sun)/r_au + v_over_c**2)

def newtonian_limit_ok(M_sun=1., r_au=1e4, tol=1e-9):
    return abs(gamma_static(r_au, M_sun)-1) < tol and pn_precession_rad(r_au, 0., M_sun) < tol and pn_accel_corr(r_au, 0., M_sun) < tol
