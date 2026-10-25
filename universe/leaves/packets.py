"""Typed interface packets between leaves. All are dataclasses with .to_dict() for JSON."""
from dataclasses import dataclass, field, asdict
from .params import LIFE_SPACE, LIFE_DEFAULTS, clamp

class _P:
    def to_dict(self): return asdict(self)

@dataclass
class TideLink(_P):
    """Wraps sim2.tidal_tensors output: per-frame symmetric 3x3 galactic tidal tensor at the host star,
    in galaxy units; multiply by sim2.TIDE_PHYS (x scale) to get system units. Consumed by sim2.leapfrog_tidal."""
    Ts: list = field(default_factory=list); scale: float = 1.
    @classmethod
    def from_array(cls, Ts, scale=1.): return cls([t.tolist() for t in Ts], float(scale))

@dataclass
class HostWeather(_P):
    """Stellar forcing: M/L/Teff + Poisson flare model (rate per yr, duty cycle, UV/X boosts) + rolling irradiance
    history + toy wind mass loss dM_star (Msun, <=0).  Call .step(t, dt, rng, LP) each world step; *_now fields are
    the instantaneous (step-averaged) values consumed by World/Chemistry."""
    M_star: float = 1.; L_star: float = 1.; T_eff: float = 5800.; dM_star: float = 0.; flare_rate: float = 0.; uv_proxy: float = 1.
    flare_dur_yr: float = 1e-3; flare_duty: float = 0.; L_now: float = 1.; uv_now: float = 1.; xray_now: float = 0.
    n_flares: int = 0; irr_hist: list = field(default_factory=list); hist_len: int = 64; distance_pc: float = 10.
    @classmethod
    def from_star(cls, M, L, T, distance_pc=10.):
        fr = float(0.5/max(M, .08)**2 if M < .6 else .05); uv = float((T/5800.)**4)
        return cls(M, L, T, 0., fr, uv, L_now=float(L), uv_now=uv, distance_pc=float(distance_pc))
    def mdot(self): return 2e-14*self.L_star**.5*(1 + 10*self.flare_rate)      # Msun/yr, solar wind ~2e-14
    def step(self, t, dt, rng, LP=None):
        P = as_bus(LP); lam = self.flare_rate*P["flare_rate_mult"]*dt; n = int(rng.poisson(lam)) if lam > 0 else 0
        duty = min(1., n*self.flare_dur_yr/max(dt, 1e-9)); amp = P["flare_amp"]
        self.flare_duty = float(duty); self.n_flares += n
        self.uv_now = float(self.uv_proxy*(1 + amp*duty)); self.xray_now = float(amp*duty*(1 + 4*(self.M_star < .6)))
        self.L_now = float(self.L_star*(1 + .01*amp*duty)); self.dM_star -= self.mdot()*dt
        self.irr_hist.append([float(t), self.L_now, self.uv_now, self.xray_now, int(n)]); del self.irr_hist[:-self.hist_len]
        return self
    def summary(self):
        h = self.irr_hist; u = [x[2] for x in h]
        return dict(M_star=self.M_star, L_star=self.L_star, flare_rate=self.flare_rate, n_flares=self.n_flares, dM_star=self.dM_star,
                    uv_mean=float(sum(u)/len(u)) if u else self.uv_proxy, uv_max=float(max(u)) if u else self.uv_proxy,
                    flare_steps=int(sum(x[4] > 0 for x in h)), hist_tail=h[-8:], distance_pc=self.distance_pc)

@dataclass
class HabitatState(_P):
    planet_id: int; name: str; m: float; a: float; e: float; spin: float; tilt: float; L_star: float
    hz_in: float; hz_out: float; in_hz: bool; ptype: str; volatiles: float; impact_flux: float; irradiance: float

@dataclass
class ClimateStress(_P):
    T_K: float; liquid_water_frac: float; climate_class: str; stressors: dict = field(default_factory=dict)

@dataclass
class ChemInventory(_P):
    species: dict = field(default_factory=lambda: dict(C=1., H=4., O=2., N=.5, P=.05, S=.1, organics=0.))
    redox: float = 0.; polymer_ready: bool = False; vesicle_ready: bool = False

@dataclass
class AbiogenesisGate(_P):
    p: float = 0.; fired: bool = False; t_w: float = -1.; planted: bool = False

@dataclass
class NicheOffer(_P):
    capacity: float = 0.; guild_slots: int = 0; resources: float = 0.

@dataclass
class CreaturePacket(_P):
    births: float = 0.; deaths: float = 0.; n: float = 0.; trait_means: dict = field(default_factory=dict); energy_budget: float = 0.

@dataclass
class BioSignFlux(_P):
    """Observable gases/pigments/tech markers; consumed by leaves.radiation observers."""
    o2: float = 0.; ch4: float = 0.; pigments: float = 0.; tech: float = 0.

@dataclass
class LifeParamBus(_P):
    """Knobs a future NIM life mesh writes; values clamped to LIFE_SPACE."""
    knobs: dict = field(default_factory=lambda: dict(LIFE_DEFAULTS))
    def __post_init__(self):
        k = dict(LIFE_DEFAULTS); k.update(self.knobs or {}); self.knobs = {n: clamp(n, v) for n, v in k.items() if n in LIFE_SPACE}
    def __getitem__(self, k): return self.knobs[k]
    def set(self, k, v): self.knobs[k] = clamp(k, v)

def as_bus(LP): return LP if isinstance(LP, LifeParamBus) else LifeParamBus(dict(LP or {}))

class ScoreBus:
    """Residual helpers (target - observed, in [-1,1]) for a future life-mesh trainer."""
    @staticmethod
    def habitability(climates, target=.3):
        f = sum(c.liquid_water_frac > 0 for c in climates)/max(len(climates), 1); return float(target - f)
    @staticmethod
    def eco_stability(biomass_series, target=.1):
        import numpy as np
        b = np.asarray(biomass_series[len(biomass_series)//2:], float)
        cv = float(b.std()/b.mean()) if len(b) and b.mean() > 0 else 1.; return float(target - min(cv, 1.))
    @staticmethod
    def planted_life(survived, target=1.): return float(target - float(bool(survived)))
