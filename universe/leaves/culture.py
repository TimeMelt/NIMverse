"""Culture/tech leaf (v1): biological-first.  Activates only when organism cohorts have existed for
min_age world-years AND biomass >= tech_threshold AND diversity >= 2.  Tech level grows logistically at tech_rate
and is emitted as BioSignFlux.tech.  Without organisms it never starts (and decays if they vanish)."""
import numpy as np
from dataclasses import dataclass, asdict
from .packets import as_bus

@dataclass
class Culture:
    active: bool = False; level: float = 0.; t_start: float = -1.; org_age: float = 0.
    def to_dict(self): return asdict(self)

def step(cu, bio, pkt, LP, rng, t_w, dt_w_yr=1., min_age=1000.):
    P = as_bus(LP); c = Culture(**cu.to_dict())
    has_org = pkt is not None and pkt.n > 0
    c.org_age = c.org_age + dt_w_yr if has_org else 0.
    if not has_org:
        c.level = max(0., c.level*(1 - .01*dt_w_yr)); c.active = c.active and c.level > 1e-6; return c
    if not c.active and c.org_age >= min_age and bio.total() >= P["tech_threshold"] and bio.diversity >= 2:
        if rng.random() < 1 - np.exp(-P["tech_rate"]*dt_w_yr): c.active, c.t_start, c.level = True, float(t_w), 1e-3
    if c.active: c.level = float(min(1., c.level + P["tech_rate"]*50*c.level*(1-c.level)*dt_w_yr))
    return c
