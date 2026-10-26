"""Cosmology stub: flat LCDM background only (no perturbations).  a(t) analytic for matter+Lambda,
Hubble flow v = H0 d, toy redshift / distances.  Exports CosmoDrive: snapshots feed radiation observers / life reports
(item 6); it does NOT drive sim2 N-body time."""
import numpy as np
from dataclasses import dataclass, asdict

H0 = 67.7; OM = .31; OL = 1 - OM; C_KMS = 299792.458
HUBBLE_GYR = 977.8/H0                                    # 1/H0 in Gyr (~14.4)

def a_of_t(t_gyr, Om=OM, Ol=OL):
    """Flat matter+Lambda: a = (Om/Ol)^(1/3) sinh^(2/3)(1.5 sqrt(Ol) H0 t)."""
    return float((Om/Ol)**(1/3)*np.sinh(1.5*np.sqrt(Ol)*np.asarray(t_gyr)/HUBBLE_GYR)**(2/3))

def t_of_a(a, Om=OM, Ol=OL): return float(HUBBLE_GYR*2/(3*np.sqrt(Ol))*np.arcsinh((a**3*Ol/Om)**.5))
def age_gyr(): return t_of_a(1.)
def H_of_a(a, Om=OM, Ol=OL): return float(H0*np.sqrt(Om/a**3 + Ol))
def hubble_velocity(d_mpc): return float(H0*d_mpc)        # km/s
def z_hubble(d_mpc): return float(np.sqrt((1+hubble_velocity(d_mpc)/C_KMS)/(1-min(hubble_velocity(d_mpc)/C_KMS, .999))) - 1)  # toy (SR Doppler)

def comoving_mpc(z, n=512):
    zz = np.linspace(0, z, n); f = 1/np.sqrt(OM*(1+zz)**3 + OL); return float(C_KMS/H0*np.trapezoid(f, zz))
def lum_dist_mpc(z): return (1+z)*comoving_mpc(z)
def z_of_comoving(d_mpc, zmax=20.):
    """Invert comoving distance (bisection); z for a source at comoving distance d_mpc."""
    lo, hi = 0., zmax
    for _ in range(60):
        mid = .5*(lo+hi); lo, hi = (mid, hi) if comoving_mpc(mid) < d_mpc else (lo, mid)
    return .5*(lo+hi)
def lookback_gyr(z): return age_gyr() - t_of_a(1/(1+z))

@dataclass
class CosmoDrive:
    t_gyr: float = 13.8; a: float = 1.; H: float = H0; z: float = 0.
    @classmethod
    def at(cls, t_gyr): a = a_of_t(t_gyr); return cls(float(t_gyr), a, H_of_a(a), float(1/a - 1))
    @classmethod
    def at_z(cls, z): a = 1/(1+z); return cls(t_of_a(a), a, H_of_a(a), float(z))
    def to_dict(self): return asdict(self)
    def snapshot(self):
        """Emission-epoch snapshot for an observer seeing light from redshift z."""
        return dict(age_gyr=self.t_gyr, a=self.a, H=self.H, z=self.z, d_C_mpc=comoving_mpc(self.z), d_L_mpc=lum_dist_mpc(self.z), lookback_gyr=lookback_gyr(self.z))
