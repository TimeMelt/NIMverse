"""Radiation / observers leaf: BioSignFlux emitted at world time t_emit reaches an observer at t_emit + light_time.
Distance is a per-system placeholder (pc) or a cosmological z (via cosmology).  Redshift is a toy: Hubble-flow z
(or given z) and (1+z) dimming of the flux by 1/d^2 relative to 10 pc.  Observers see both legacy scalars and spectral lines (leaves.biosign) with obs_nm = rest_nm*(1+z); host spectra are toy blackbody bands (stellar.uv_bands / HostWeather.uv_bands), no radiative transfer."""
import numpy as np
from dataclasses import dataclass, field, asdict
from . import cosmology as cosmo
from . import biosign

LY_PER_PC = 3.26156

def light_time_yr(distance_pc): return float(distance_pc*LY_PER_PC)
def redshift(distance_pc=None, z=None):
    if z is not None: return float(z)
    return cosmo.z_hubble(distance_pc/1e6) if distance_pc and distance_pc > 1e5 else 0.   # peculiar-velocity dominated below ~100 kpc

@dataclass
class Observer:
    name: str = "obs0"; distance_pc: float = 10.; z: float = None; thresh: dict = field(default_factory=lambda: dict(o2=.02, ch4=1e-4, pigments=.05, tech=.01, h2o=.02))
    aperture: float = 1.      # sensitivity multiplier vs a 10 pc reference instrument
    def delay(self):
        return light_time_yr(self.distance_pc) if self.z is None else cosmo.lookback_gyr(self.z)*1e9
    def to_dict(self): return asdict(self)

class ObserverCatalog:
    """Queue of emitted BioSignFlux; poll(t_now) moves arrived packets into the detection log."""
    def __init__(self, observers=None): self.obs = observers or [Observer()]; self.pending = []; self.log = []; self.photons = []
    def emit(self, t_emit, planet, flux, every=1):
        f = flux.to_dict() if hasattr(flux, "to_dict") else dict(flux)
        lines = f.pop("lines", None); lines = biosign.build_lines(f) if lines is None else lines
        f = {k: v for k, v in f.items() if isinstance(v, (int, float))}
        for o in self.obs:
            zz = redshift(o.distance_pc, o.z); dim = (10./max(o.distance_pc, 1e-6))**2/(1+zz)**2
            self.pending.append(dict(observer=o.name, planet=planet, t_emit=float(t_emit), t_arr=float(t_emit + o.delay()), z=zz,
                                     flux={k: float(v*min(1., dim*o.aperture)) for k, v in f.items()},
                                     lines=biosign.observed(lines, zz, min(1., dim*o.aperture)), thr=o.thresh))
    def poll(self, t_now):
        keep = []
        for p in self.pending:
            if p["t_arr"] <= t_now:
                self.photons.append({k: p[k] for k in ("observer", "planet", "t_emit", "t_arr")})
                det = [k for k, v in p["flux"].items() if v >= p["thr"].get(k, np.inf)]
                ld = [dict(name=l["name"], rest_nm=l["rest_nm"], obs_nm=l["obs_nm"], strength=l["strength"]) for l in p.get("lines", [])
                      if l["strength"] >= p["thr"].get(l["species"], np.inf)*l["weight"]]
                det += [l["name"] for l in ld]
                if det: self.log.append(dict(observer=p["observer"], planet=p["planet"], t_emit=p["t_emit"], t_arr=p["t_arr"], z=p["z"], detected=det, lines=ld, flux=p["flux"]))
            else: keep.append(p)
        self.pending = keep; return self
    def first_detection(self, sig, planet=None):
        for d in self.log:
            if sig in d["detected"] and (planet is None or d["planet"] == planet): return d
        return None
    def summary(self):
        firsts = {}
        for d in self.log:
            obs = {l["name"]: l["obs_nm"] for l in d.get("lines", [])}
            for s in d["detected"]: firsts.setdefault(f'{d["planet"]}:{s}', dict(t_emit=d["t_emit"], t_arr=d["t_arr"], z=d["z"], **({"obs_nm": obs[s]} if s in obs else {})))
        return dict(observers=[o.to_dict() for o in self.obs], delay_yr={o.name: o.delay() for o in self.obs},
                    n_arrived=len(self.photons), n_pending=len(self.pending), n_detections=len(self.log), first=firsts, log_tail=self.log[-5:],
                    lines_detected=sorted({l["name"] for d in self.log for l in d.get("lines", [])}))
