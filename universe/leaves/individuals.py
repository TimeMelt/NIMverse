"""Individuals leaf: literal organisms with ids, parent links, energy, heritable traits, and evolution.
Population is stored struct-of-arrays (numpy, vectorised) but every row is one organism with a unique id.
Per step: competition for niche food -> metabolism -> death (base + climate stress vs hardiness + starvation)
-> reproduction (Poisson, energy-paid, rate x fitness) with inherited + mutated traits (asexual) -> soft-cap cull of weakest.
Speciation-lite: child gets a new species_id when its traits drift > speciation_dist from its species' founder traits."""
import numpy as np
from dataclasses import dataclass, field
from .packets import CreaturePacket, as_bus
from .params import TRAITS, LIFE_SPACE

LO = np.array([LIFE_SPACE[t][0] for t in TRAITS]); HI = np.array([LIFE_SPACE[t][1] for t in TRAITS]); SPAN = HI - LO
H = TRAITS.index("hardiness"); MET = TRAITS.index("metabolism_eff"); MOB = TRAITS.index("mobility"); REP = TRAITS.index("reproduction_rate")
E_MAX = 12.         # energy store cap (~24 yr of basal burn: buffers famine steps)

@dataclass
class Population:
    id: np.ndarray = field(default_factory=lambda: np.zeros(0, int)); parent: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    birth: np.ndarray = field(default_factory=lambda: np.zeros(0)); energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    traits: np.ndarray = field(default_factory=lambda: np.zeros((0, len(TRAITS)))); gen: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    species: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    next_id: int = 0; next_species: int = 0; founders: dict = field(default_factory=dict)   # species_id -> founder traits (normalised)
    t: float = 0.; n_births: int = 0; n_deaths: int = 0; graveyard: list = field(default_factory=list); n_dead_seen: int = 0
    series: list = field(default_factory=list); last_series_t: float = -1e18
    def __len__(self): return len(self.id)
    def take(self, m):
        for k in ("id", "parent", "birth", "energy", "traits", "gen", "species"): setattr(self, k, getattr(self, k)[m])
    def append(self, **kw):
        for k, v in kw.items(): setattr(self, k, np.concatenate([getattr(self, k), v]))
    def record(self, i, t_death=None):
        d = dict(id=int(self.id[i]), parent_id=(int(self.parent[i]) if self.parent[i] >= 0 else None), birth_t_w=float(self.birth[i]),
                 age=float(self.t - self.birth[i]), energy=round(float(self.energy[i]), 4), generation=int(self.gen[i]), species_id=int(self.species[i]),
                 traits={t: round(float(v), 4) for t, v in zip(TRAITS, self.traits[i])}, alive=t_death is None)
        if t_death is not None: d["death_t_w"] = float(t_death)
        return d
    def individuals(self, k=None):
        idx = np.argsort(self.id); idx = idx if k is None or len(idx) <= k else idx[np.linspace(0, len(idx)-1, k).astype(int)]
        return [self.record(i) for i in idx]
    def stats(self):
        n = len(self); tr = self.traits
        return dict(t_w=float(self.t), n_alive=int(n), trait_means={t: float(tr[:, j].mean()) if n else None for j, t in enumerate(TRAITS)},
                    trait_stds={t: float(tr[:, j].std()) if n else None for j, t in enumerate(TRAITS)},
                    generations_max=int(self.gen.max()) if n else 0, generations_mean=float(self.gen.mean()) if n else 0.,
                    n_species=int(len(np.unique(self.species))) if n else 0, n_births=int(self.n_births), n_deaths=int(self.n_deaths))
    def summary(self, k=None):
        return dict(individuals=self.individuals(k), n_alive=len(self), deceased_sample=self.graveyard, evolution_series=self.series,
                    total_births=int(self.n_births), total_deaths=int(self.n_deaths), species_ever=int(self.next_species))

def _new_ids(pop, k): ids = np.arange(pop.next_id, pop.next_id + k); pop.next_id += k; return ids

def found(LP, rng, n=40, t_w=0., traits=None):
    """Founding population: n individuals around LP trait means (or explicit `traits` dict), spread mutation_sd."""
    P = as_bus(LP); pop = Population(); mu = np.array([(traits or {}).get(t, P[t]) for t in TRAITS])
    tr = np.clip(mu + P["mutation_sd"]*SPAN*rng.normal(size=(n, len(TRAITS))), LO, HI)
    pop.append(id=_new_ids(pop, n), parent=-np.ones(n, int), birth=np.full(n, float(t_w)), energy=np.ones(n), traits=tr,
               gen=np.zeros(n, int), species=np.zeros(n, int))
    pop.next_species = 1; pop.founders[0] = (mu - LO)/SPAN; pop.t = float(t_w); pop.series.append(pop.stats()); pop.last_series_t = pop.t
    return pop

def climate_stress(climate):
    """Chronic stress (water, temperature) felt every year; shocks (impacts, UV) are applied once per step in `shock`."""
    return float((1 - climate.liquid_water_frac) + abs(climate.T_K - 293.)/80.)

def shock_kill(climate):
    st = climate.stressors or {}; return float(1 - (1 - min(st.get("impact_kill", 0.), 1.))*(1 - min(st.get("uv_kill", 0.), 1.)))

def _bury(pop, dead, rng, grave_k, t):
    """Remove `dead` rows, keeping a reservoir sample of grave_k deceased records (compact lineage history)."""
    for i in np.flatnonzero(dead):
        pop.n_dead_seen += 1; j = pop.n_dead_seen - 1 if len(pop.graveyard) < grave_k else int(rng.integers(pop.n_dead_seen))
        if j < grave_k:
            rec = pop.record(i, t)
            if len(pop.graveyard) < grave_k: pop.graveyard.append(rec)
            else: pop.graveyard[j] = rec
    pop.take(~dead)

def step(pop, niche, climate, LP, rng, dt_w_yr=1., t_w=None, series_every=None, grave_k=200, h_max=1.):
    """Advance every individual by dt world-years (internal substeps <= h_max yr so demography is dt-stable).
    Returns (pop, CreaturePacket) summed over substeps. Mutates `pop` in place."""
    P = as_bus(LP); k = max(1, int(np.ceil(dt_w_yr/h_max - 1e-9))); h = float(dt_w_yr)/k; t1 = float(t_w if t_w is not None else pop.t + dt_w_yr)
    B = D = 0; f = shock_kill(climate)
    if f > 0 and len(pop):                                                 # one-off shock: hardiness halves the kill chance
        dead = rng.random(len(pop)) < f*(1 - .5*pop.traits[:, H]); D += int(dead.sum()); _bury(pop, dead, rng, grave_k, pop.t); pop.n_deaths += int(dead.sum())
    for i in range(k):
        b, d = _substep(pop, niche, climate, P, rng, h, t1 - (k-1-i)*h, grave_k); B += b; D += d
    if series_every and pop.t - pop.last_series_t >= series_every - 1e-9: pop.series.append(pop.stats()); pop.last_series_t = pop.t
    s = pop.stats(); means = {t: v for t, v in s["trait_means"].items() if v is not None}; means["cohorts"] = float(s["n_species"])
    cap = int(P["pop_cap"])
    return pop, CreaturePacket(float(B), float(D), float(len(pop)), means, float(pop.energy.sum()/max(cap, 1)) if len(pop) else 0.)

def _substep(pop, niche, climate, P, rng, dt, t, grave_k):
    n = len(pop); cap = int(P["pop_cap"]); sel = P["sel_strength"]; pop.t = float(t); births = deaths = 0
    if n:
        tr = pop.traits; hard, met, mob, rep = tr[:, H], tr[:, MET], tr[:, MOB], tr[:, REP]
        need = (1 - .5*met)*(1 + .6*hard)*(1 + .4*mob)                  # basal need/yr: hardiness & mobility are costly
        food = max(0., niche.resources)*cap*dt                             # niche food shared by up to pop_cap organisms
        w = np.sqrt(need)*(1 + .4*mob)*(1 + .3*rng.random(n))                       # competitive weight (mobility forages better)
        got = np.minimum(food*w/w.sum(), 1.5*need*dt) if food > 0 else np.zeros(n)
        burn = .5 if food > 0 else .5*(.4 - .3*hard)                       # famine -> dormancy (hardy forms encyst deeper)
        pop.energy = np.clip(pop.energy + got - burn*need*dt, 0., E_MAX)   # energy store never negative
        stress = climate_stress(climate); starving = pop.energy <= 0
        rate = P["eco_death"]*.5 + sel*.2*np.maximum(0., stress - hard) + (.25 if food > 0 else .03)*starving   # hazard per world-yr (dormant in famine)
        dead = rng.random(n) < 1 - np.exp(-rate*dt); deaths = int(dead.sum())
        if deaths: _bury(pop, dead, rng, grave_k, pop.t)
        n = len(pop)
    if n:
        tr = pop.traits; need = (1 - .5*tr[:, MET])*(1 + .6*tr[:, H])*(1 + .4*tr[:, MOB])
        fit = np.clip(pop.energy/need, 0., 2.)                             # energy surplus -> fecundity
        lam = tr[:, REP]*dt*fit*max(0., 1 - n/(1.5*cap)); kids = rng.poisson(lam)
        cost = .5*need; kids = np.minimum(kids, np.floor(np.maximum(pop.energy, 0)/cost).astype(int)); births = int(kids.sum())
        if births:
            par = np.repeat(np.arange(n), kids); pop.energy = pop.energy - kids*cost
            sd = P["mutation_sd"]*P["evolution_rate"]*SPAN*np.where(rng.random((births, 1)) < .02, 5., 1.)   # rare big jumps
            ct = np.clip(tr[par] + sd*rng.normal(size=(births, len(TRAITS))), LO, HI); sp = pop.species[par].copy()
            dist = np.linalg.norm((ct - LO)/SPAN - np.array([pop.founders[s] for s in sp]), axis=1)
            for i in np.flatnonzero(dist > P["speciation_dist"]):          # lineage split -> new species founded by this child
                sp[i] = pop.next_species; pop.founders[pop.next_species] = (ct[i] - LO)/SPAN; pop.next_species += 1
            pop.append(id=_new_ids(pop, births), parent=pop.id[par], birth=np.full(births, pop.t), energy=np.full(births, .5)*cost[par],
                       traits=ct, gen=pop.gen[par] + 1, species=sp)
        if len(pop) > cap:                                                 # soft cap: cull weakest (lowest energy/need)
            tr = pop.traits; need = (1 - .5*tr[:, MET])*(1 + .6*tr[:, H])*(1 + .4*tr[:, MOB])
            keep = np.argsort(-(pop.energy/need + .1*rng.random(len(pop))))[:cap]; m = np.zeros(len(pop), bool); m[keep] = True
            deaths += int((~m).sum()); pop.take(np.sort(np.flatnonzero(m)))
    pop.n_births += births; pop.n_deaths += deaths
    return births, deaths
