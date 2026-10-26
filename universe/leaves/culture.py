"""Culture/tech leaf (v2): biological-first, coupled back into the biosphere.
Activates only when organisms have existed for min_age world-years AND biomass >= tech_threshold AND diversity >= 2.
Tech level grows logistically at tech_rate and is emitted as BioSignFlux.tech.  Without organisms it never starts
(and decays if they vanish).

Coupling (item 4; toy but consequential):
* resource draw: demand = tech_draw*level biomass/yr taken from producers (capped at DRAW_CAP of P per step).
  supply ratio s = taken/demand scales growth; s < .5 makes level decline (overshoot -> stall/crash).
* pollution: stock relaxes toward level*tech_pollute with e-fold POLL_TAU yr (persists after a collapse).
* climate: next-step stressor dT = +tech_warm_K*pollution (greenhouse), merged into world.step(stress).
* extinction risk: toxicity kill fraction tech_kill = TOX_K*max(0, pollution-TOX_ON)*dt/20 (per step, <= .5),
  applied like uv_kill to biomass and individuals.  footprint = taken/(P+taken) also trims next-step niche resources."""
import numpy as np
from dataclasses import dataclass, asdict
from .packets import as_bus

DRAW_CAP, POLL_TAU, TOX_ON, TOX_K = .5, 500., .3, .4

@dataclass
class Culture:
    active: bool = False; level: float = 0.; t_start: float = -1.; org_age: float = 0.
    pollution: float = 0.; draw: float = 0.; footprint: float = 0.; supply: float = 1.; warm_K: float = 0.; tech_kill: float = 0.
    def to_dict(self): return asdict(self)

def _pollute(c, P, dt):
    tgt = c.level*P["tech_pollute"] if c.active else 0.
    c.pollution = float(tgt + (c.pollution - tgt)*np.exp(-dt/POLL_TAU))
    c.warm_K = float(P["tech_warm_K"]*c.pollution)
    c.tech_kill = float(min(.5, TOX_K*max(0., c.pollution - TOX_ON)*dt/20.))

def step(cu, bio, pkt, LP, rng, t_w, dt_w_yr=1., min_age=1000.):
    P = as_bus(LP); c = Culture(**cu.to_dict()); c.draw = c.footprint = 0.; c.supply = 1.
    has_org = pkt is not None and pkt.n > 0
    c.org_age = c.org_age + dt_w_yr if has_org else 0.
    if not has_org:
        c.level = max(0., c.level*(1 - .01*dt_w_yr)); c.active = c.active and c.level > 1e-6; _pollute(c, P, dt_w_yr); return c
    if not c.active and c.org_age >= min_age and bio.total() >= P["tech_threshold"] and bio.diversity >= 2:
        if rng.random() < 1 - np.exp(-P["tech_rate"]*dt_w_yr): c.active, c.t_start, c.level = True, float(t_w), 1e-3
    if c.active:
        need = P["tech_draw"]*c.level*dt_w_yr
        if need > 0:
            c.draw = float(min(need, DRAW_CAP*bio.P)); c.supply = float(c.draw/need); c.footprint = float(c.draw/max(bio.P, 1e-12))
        g = P["tech_rate"]*50*c.level*(1-c.level)*dt_w_yr
        c.level = float(min(1., c.level + g*c.supply))
        if c.supply < .5: c.level = float(max(0., c.level*(1 - .02*(1 - c.supply)*dt_w_yr)))   # starved industry contracts
        c.active = c.level > 1e-6
    _pollute(c, P, dt_w_yr)
    return c

def apply(bio, c):
    """Same-step resource draw from producers. Returns new Biosphere."""
    from .biosphere import Biosphere
    b = Biosphere(**bio.to_dict()); b.P = max(0., b.P - c.draw); return b

def stress(c):
    """Next-step stressors to merge into world.step(stress): dT (added), tech_kill, tech_pollution."""
    return {"tech_dT": c.warm_K, "tech_kill": c.tech_kill, "tech_pollution": c.pollution}
