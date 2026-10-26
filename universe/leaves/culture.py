"""Culture/tech leaf (v3): biological-first, coupled back into biosphere, climate and chemistry.
Activates only when organisms have existed for min_age world-years AND biomass >= tech_threshold AND diversity >= 2.
Tech level grows logistically at tech_rate and is emitted as BioSignFlux.tech.  Without organisms it never starts
(and decays if they vanish).

v2 coupling: resource draw (tech_draw), pollution stock (POLL_TAU) -> greenhouse dT (tech_warm_K), toxicity tech_kill,
niche footprint.  v3 (culture leftovers closed; all toy but consequential):
* polities: 2-4 polities (n from diversity), deterministic shares from t_start (own RNG, main stream untouched).
  conflict = CONFLICT_K*inequality-free rivalry*stress (pollution/supply/warming) -> extra kill + wasted draw.
* energy_mix: fossil = tech_fossil_frac*(1-mitigation), rest renewable:nuclear 2:1.  Emission intensity
  fossil 1 / renewable .1 / nuclear .05 (normalised to the reference .7-fossil mix so defaults ~ v2); renewables raise draw
  (land/niche), nuclear adds a small chronic kill.
* aerosols: short-lived stock (AER_TAU 5 yr) of fossil burn; net warm_K = greenhouse_K - tech_aerosol_K*aerosols.
* acidification: slow stock (ACID_TAU 1500 yr) toward min(1, tech_acid*pollution*fossil); lowers producer capacity
  on wet worlds (biosphere.capacity reads stressor tech_acid) and is reported as chem pH proxy.
* apply_chem: industry outgasses organics (tech_chem) and fossil exhaust shifts redox reducing; pollution oxidises.
* mitigation: when pollution/warming/acidification/kill exceed thresholds, mitigation rises at tech_mitigate;
  shifts mix off fossil, cuts emission intensity, costs growth (opportunity cost); relaxes slowly w/o crisis."""
import numpy as np
from dataclasses import dataclass, asdict, field
from .packets import as_bus

DRAW_CAP, POLL_TAU, TOX_ON, TOX_K = .5, 500., .3, .4
AER_TAU, ACID_TAU, CONFLICT_K, NUC_KILL = 5., 1500., .5, 2e-4
INTENS = dict(fossil=1., renewable=.1, nuclear=.05)
MIT_ON = dict(pollution=.25, warm=3., acid=.3, kill=.02)

def mix_of(f0, m):
    f = float(np.clip(f0*(1 - m), 0, 1)); r = 1 - f
    return dict(fossil=f, renewable=2*r/3, nuclear=r/3)

def intensity(mix): return float(sum(INTENS[k]*v for k, v in mix.items()))
REF_FOSSIL = .7; REF_INT = intensity(mix_of(REF_FOSSIL, 0.))   # default mix -> v2-equivalent emissions

@dataclass
class Culture:
    active: bool = False; level: float = 0.; t_start: float = -1.; org_age: float = 0.
    pollution: float = 0.; draw: float = 0.; footprint: float = 0.; supply: float = 1.; warm_K: float = 0.; tech_kill: float = 0.
    greenhouse_K: float = 0.; aerosol_K: float = 0.; aerosols: float = 0.; acidification: float = 0.; mitigation: float = 0.
    conflict: float = 0.; energy_mix: dict = field(default_factory=lambda: dict(fossil=0., renewable=0., nuclear=0.))
    polities: list = field(default_factory=list)
    def to_dict(self): return asdict(self)

def _polities(c, diversity):
    if not c.active: return []
    n = int(np.clip(2 + int(diversity) % 3, 2, 4)); w = np.random.default_rng(int(abs(c.t_start)) + 7919*n).dirichlet(np.full(n, 3.))
    return [dict(id=i, share=float(s), level_contrib=float(s*c.level)) for i, s in enumerate(w)]

def _pollute(c, P, dt, mit_on=True):
    mix = mix_of(P["tech_fossil_frac"], c.mitigation) if c.active else dict(fossil=0., renewable=0., nuclear=0.)
    c.energy_mix = mix; ref = REF_INT
    rate = c.level*P["tech_pollute"]*intensity(mix)/max(ref, 1e-9)*(1 - .5*c.mitigation)*(1 + c.conflict) if c.active else 0.
    c.pollution = float(rate + (c.pollution - rate)*np.exp(-dt/POLL_TAU))
    aer = c.level*P["tech_pollute"]*mix["fossil"]/max(ref, 1e-9) if c.active else 0.
    c.aerosols = float(aer + (c.aerosols - aer)*np.exp(-dt/AER_TAU))
    acid = min(1., P["tech_acid"]*c.pollution*(mix["fossil"]/REF_FOSSIL if c.active else 1.))
    c.acidification = float(np.clip(acid + (c.acidification - acid)*np.exp(-dt/ACID_TAU), 0, 1))
    c.greenhouse_K = float(P["tech_warm_K"]*c.pollution); c.aerosol_K = float(P["tech_aerosol_K"]*c.aerosols)
    c.warm_K = float(c.greenhouse_K - c.aerosol_K)
    c.tech_kill = float(min(.5, TOX_K*max(0., c.pollution - TOX_ON)*dt/20. + c.conflict*.05*dt/20. + NUC_KILL*mix["nuclear"]*c.level*dt/20.))
    if mit_on and c.active:     # mitigation / learning response to crisis
        crisis = max(c.pollution/MIT_ON["pollution"], c.warm_K/MIT_ON["warm"], c.acidification/MIT_ON["acid"], c.tech_kill/MIT_ON["kill"])
        if crisis > 1: c.mitigation = float(min(1., c.mitigation + P["tech_mitigate"]*min(crisis - 1, 2.)*dt))
        else: c.mitigation = float(max(0., c.mitigation - 2e-4*dt))
    elif not c.active: c.mitigation = float(max(0., c.mitigation - 2e-4*dt))

def step(cu, bio, pkt, LP, rng, t_w, dt_w_yr=1., min_age=1000.):
    P = as_bus(LP); c = Culture(**cu.to_dict()); c.draw = c.footprint = 0.; c.supply = 1.
    has_org = pkt is not None and pkt.n > 0
    c.org_age = c.org_age + dt_w_yr if has_org else 0.
    if not has_org:
        c.level = max(0., c.level*(1 - .01*dt_w_yr)); c.active = c.active and c.level > 1e-6; c.conflict = 0.
        c.polities = _polities(c, bio.diversity); _pollute(c, P, dt_w_yr); return c
    if not c.active and c.org_age >= min_age and bio.total() >= P["tech_threshold"] and bio.diversity >= 2:
        if rng.random() < 1 - np.exp(-P["tech_rate"]*dt_w_yr): c.active, c.t_start, c.level = True, float(t_w), 1e-3
    if c.active:
        c.polities = _polities(c, bio.diversity); sh = np.array([p["share"] for p in c.polities])
        rivalry = 1 - float((sh**2).sum())        # 0 for a hegemon, -> 1-1/n when balanced
        stress_ = max(0., cu.pollution - .2) + max(0., .7 - cu.supply) + max(0., cu.warm_K - 2.)/10
        c.conflict = float(min(1., CONFLICT_K*P["tech_conflict"]*rivalry*stress_))
        mix = mix_of(P["tech_fossil_frac"], c.mitigation); mix0 = mix_of(REF_FOSSIL, 0.)
        land = (1 + .5*mix["renewable"])/(1 + .5*mix0["renewable"])
        need = P["tech_draw"]*c.level*dt_w_yr*land*(1 + c.conflict)*(1 - .3*c.mitigation)
        if need > 0:
            c.draw = float(min(need, DRAW_CAP*bio.P)); c.supply = float(c.draw/need); c.footprint = float(c.draw/max(bio.P, 1e-12))
        g = P["tech_rate"]*50*c.level*(1-c.level)*dt_w_yr*(1 - .3*c.mitigation)    # mitigation opportunity cost
        c.level = float(min(1., c.level + g*c.supply))
        if c.supply < .5: c.level = float(max(0., c.level*(1 - .02*(1 - c.supply)*dt_w_yr)))   # starved industry contracts
        c.active = c.level > 1e-6
        for p in c.polities: p["level_contrib"] = float(p["share"]*c.level)
    _pollute(c, P, dt_w_yr, P["tech_mitigate"] > 0)
    return c

def apply(bio, c):
    """Same-step resource draw from producers. Returns new Biosphere."""
    from .biosphere import Biosphere
    b = Biosphere(**bio.to_dict()); b.P = max(0., b.P - c.draw); return b

def apply_chem(chem, c, LP, dt_w_yr=1.):
    """Industrial outgassing: organics += tech_chem*1e-4*level*(fossil+.2) per yr; fossil exhaust makes redox more reducing
    (-), pollution stock oxidises (+). Returns new ChemInventory."""
    from .packets import ChemInventory
    P = as_bus(LP); s = dict(chem.species)
    if not c.active and c.pollution <= 0: return chem
    f = c.energy_mix.get("fossil", 0.)
    s["organics"] = float(s["organics"] + P["tech_chem"]*1e-4*c.level*(f + .2)*dt_w_yr)
    r = float(np.clip(chem.redox + P["tech_chem"]*1e-4*(c.pollution - 2*c.level*f)*dt_w_yr, -2., 2.))
    return ChemInventory(s, r, chem.polymer_ready, chem.vesicle_ready)

def stress(c):
    """Next-step stressors to merge into world.step(stress): dT (net, added), tech_kill, tech_pollution, aerosol/acid."""
    return {"tech_dT": c.warm_K, "tech_kill": c.tech_kill, "tech_pollution": c.pollution, "tech_greenhouse_K": c.greenhouse_K,
            "tech_aerosol_K": c.aerosol_K, "tech_acid": c.acidification, "tech_mitigation": c.mitigation, "tech_conflict": c.conflict}
