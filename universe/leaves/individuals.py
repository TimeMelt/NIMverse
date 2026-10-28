"""Individuals leaf: literal organisms with ids, parent links, positions, energy, heritable traits, and evolution.
Population is stored struct-of-arrays (numpy, vectorised) but every row is one organism with a unique id.
Per step: random walk on the unit sphere (step ~ mobility) -> local density (equal-area grid) -> competition for niche food
(diminishing-returns mobility bonus that only pays where crowded) -> metabolism (mobility cost convex) -> death (base + climate
stress vs hardiness, incl. slow internal climate drift + rare flips, + starvation) -> reproduction (Poisson, energy-paid; with
prob sex_rate a nearby same-species mate is chosen and traits are blended) -> soft-cap cull of weakest.
Speciation: children inherit the parent's species; every SPECIES_EVERY yr the living set is re-clustered (average linkage on
trait + geographic distance, threshold .75 x speciation_dist, tiny clusters merged into the nearest big one); cluster labels are
matched to previous species ids by majority so ids persist and species merge when they drift back together.
Impacts: per-individual kill ~ shock x (1-hardiness/2) x local crowding, refugia (shelter draw grows with ocean cover), and a
survivor floor max(3, impact_refuge x pre-impact n) unless chronic climate is already lethal."""
import numpy as np
from dataclasses import dataclass, field
from .packets import CreaturePacket, as_bus
from .params import TRAITS, LIFE_SPACE

LO = np.array([LIFE_SPACE[t][0] for t in TRAITS]); HI = np.array([LIFE_SPACE[t][1] for t in TRAITS]); SPAN = HI - LO
H = TRAITS.index("hardiness"); MET = TRAITS.index("metabolism_eff"); MOB = TRAITS.index("mobility"); REP = TRAITS.index("reproduction_rate")
E_MAX = 12.           # energy store cap (~24 yr of basal burn: buffers famine steps)
SPECIES_EVERY = 500.  # yr between re-clusterings of the living set
NEW_SPECIES_FRAC = .06 # a split founds a new species_id only if the daughter cluster holds >= 6% of the living
RECENT_YR = 10000.     # vanished species ids within this window can be re-used by a matching cluster
RECYCLE_R = 1.5       # a new cluster re-uses a recent vanished id if its centroid is within RECYCLE_R x cut (reach widens smoothly as mint tokens run out)
CL_CUT = .6          # dendrogram cut as a fraction of speciation_dist
MIN_SPECIES = 4       # clusters smaller than this are merged into the nearest larger one
SPLIT_SEP = .8         # a daughter founds a new id only if its centroid sits >= SPLIT_SEP x cut from its parent cluster (else merge back)
SOFT_MAX = 10         # soft cap: above SOFT_MAX living species ids, the daughter-size bar rises by SOFT_K per extra cluster (no hard ceiling)
SOFT_K = .5
MINT_TAU, MINT_CAP = 700., 3.  # soft mint-rate limit: new-id tokens refill 1 per MINT_TAU yr (bucket <= MINT_CAP); with no token a daughter takes the nearest recent vanished id (any distance) or merges back
SMOOTH_R = 1.
RECYCLE_FAR = 2.          # without a token, recycling reaches out to RECYCLE_FAR x RECYCLE_R x cut before merging back into the parent
GEO_W = .5            # weight of great-circle distance (rad/pi) in the speciation metric (allopatry)
WATER_W = .6         # weight of dry/frozen surface in chronic stress (ocean refugia buffer the water-fraction swings)
LAT_STRESS = .3       # poles harsher than equator (mean zero over the sphere): local adaptation -> geographic species
NZ, NL = 6, 12        # equal-area density grid (z bands x lon sectors)
KEYS = ("id", "parent", "parent2", "birth", "energy", "traits", "gen", "species", "pos", "trail")

UPT0, UPT1, UPT2 = 1.15, 0., .22  # rate-yield: uptake 1.15 - UPT1 x - UPT2 x^2 (x = normalised met); convex -> stable interior optimum (was linear .2 x: bang-bang)
UPT_REP = 1.          # brood growth ~ uptake (efficiency also slows reproduction, binding even when food is plentiful)
HARD_MET = .05       # efficient metabolisms have less stress reserve: hardiness_eff = hardiness x (1 - HARD_MET x met)
REP_UPKEEP = .6     # basal cost of reproductive machinery per unit reproduction_rate (rep in [.01,.5] -> up to +60% burn)
JUV_AGE, JUV_HAZ = 15., .1   # juvenile window (yr) and extra hazard scale: small offspring of fast breeders die young
NICHE_ALPHA = .8 
REFUGE_N = .5       # cold-spell refugia: below REFUGE_N x pop_cap survivors crowd into sheltered microhabitats -> stress hazard x n/(REFUGE_N cap) (>= .2)
ACCL_MAX, ACCL_TAU = .15, 600.  # non-heritable acclimation (hardiness plasticity): tracks mean stress excess with tau 600 yr, <= +.15 (keeps heritable hardiness under selection)
HAZ_SAT = 1.        # stress hazard ~ x/(1 + HAZ_SAT x): beyond tolerance organisms encyst rather than die linearly faster
DORM_STRESS = .5     # under chronic stress, basal burn falls (torpor) by up to 50%
FOOD_TAU = 5000.     # ecosystem storage (detritus/soil): food felt = lagged productivity cycle -> metabolism can't chase every phase
REST_K = .3           # mind rest action: burn x (1 - REST_K rest), forage weight x (1 - 2 REST_K rest)
TRAIL_K, TRAIL_EVERY = 24, 10.    # path history: last 24 positions, one per 10 yr (newborns inherit the parent's path = lineage track) for viewer replay    # frequency-dependent competition: food share / (phenotype-niche crowding)^alpha (rare niches pay)

def need_of(tr):
    m = tr[:, MOB]; r = (tr[:, REP] - LO[REP])/SPAN[REP]
    return (1 - .5*tr[:, MET])*(1 + .6*tr[:, H])*(1 + .2*m + .45*m*m)*(1 + REP_UPKEEP*.5*r)   # mobility cost convex; breeding costly

def uptake(tr):
    """Rate-yield trade-off: efficient metabolism (high yield) forages/assimilates slowly; convex: 1.15 at met=.1 -> .85 at met=.9."""
    x = (tr[:, MET] - LO[MET])/SPAN[MET]; return UPT0 - UPT1*x - UPT2*x*x   # convex cost: interior optimum, no bang-bang switching

NB_MOB, NB_HARD, NB_BELT = 4, 4, (.35, .75)   # niche grid: mobility bands x hardiness bands x climate belts (|sin lat| edges)
NICHE_KERNEL = 0.     # >0: continuous Gaussian kernel crowding (width in normalised trait units) instead of discrete bands

def niche_crowding(tr, pos):
    """Relative crowding (1 = even) of each organism's phenotype niche: mobility (NB_MOB bands: sessile/forager/roamer) x hardiness
    (NB_HARD bands) x climate belt (equatorial / temperate / polar). Competition is mostly within a niche, so rare niches get more food."""
    n = len(tr)
    if not n: return np.zeros(0)
    lb = np.digitize(np.abs(pos[:, 1]), NB_BELT)
    if NICHE_KERNEL > 0:
        z = np.stack([tr[:, MOB], tr[:, H]], 1); d2 = ((z[:, None] - z[None])**2).sum(-1)
        k = np.exp(-d2/(2*NICHE_KERNEL**2))*(lb[:, None] == lb[None]); c = k.sum(1); return c/c.mean()
    mb = np.clip((tr[:, MOB]*NB_MOB).astype(int), 0, NB_MOB-1); hb = np.clip((tr[:, H]*NB_HARD).astype(int), 0, NB_HARD-1)
    nb = len(NB_BELT) + 1; c = (mb*NB_HARD + hb)*nb + lb
    cnt = np.bincount(c, minlength=NB_MOB*NB_HARD*nb); occ = (cnt > 0).sum(); return cnt[c]/(n/occ)

def _rand_pos(rng, n):
    v = rng.normal(size=(n, 3)); return v/np.linalg.norm(v, axis=1, keepdims=True)

def latlon(pos): return np.arcsin(np.clip(pos[:, 1], -1, 1)), np.arctan2(pos[:, 2], pos[:, 0])   # matches life.js (y up)

def _jitter(pos, step, rng):
    p = pos + step[:, None]*rng.normal(size=pos.shape); return p/np.linalg.norm(p, axis=1, keepdims=True)

def local_density(pos):
    """Relative density (1 = uniform) of each individual's equal-area cell."""
    n = len(pos)
    if not n: return np.zeros(0)
    zb = np.clip(((pos[:, 1] + 1)/2*NZ).astype(int), 0, NZ-1); lb = ((np.arctan2(pos[:, 2], pos[:, 0]) + np.pi)/(2*np.pi)*NL).astype(int) % NL
    c = zb*NL + lb; cnt = np.bincount(c, minlength=NZ*NL); return cnt[c]/(n/(NZ*NL))

@dataclass
class Population:
    id: np.ndarray = field(default_factory=lambda: np.zeros(0, int)); parent: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    parent2: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    birth: np.ndarray = field(default_factory=lambda: np.zeros(0)); energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    traits: np.ndarray = field(default_factory=lambda: np.zeros((0, len(TRAITS)))); gen: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    species: np.ndarray = field(default_factory=lambda: np.zeros(0, int)); pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    recent: dict = field(default_factory=dict)
    next_id: int = 0; next_species: int = 0; last_cluster_t: float = -1e18
    T_K: float = None; drift_phase: float = 0.; flip: float = 0.; env_offset: float = 0.
    t: float = 0.; n_births: int = 0; n_deaths: int = 0; n_sexual: int = 0; graveyard: list = field(default_factory=list); n_dead_seen: int = 0
    series: list = field(default_factory=list); accl: float = 0.; food_buf: float = None; trail: np.ndarray = field(default_factory=lambda: np.zeros((0, TRAIL_K, 3))); last_trail_t: float = -1e18; last_series_t: float = -1e18; impact_log: list = field(default_factory=list)
    def __len__(self): return len(self.id)
    def take(self, m):
        for k in KEYS: setattr(self, k, getattr(self, k)[m])
    def append(self, **kw):
        for k, v in kw.items(): setattr(self, k, np.concatenate([getattr(self, k), v]))
    def record(self, i, t_death=None):
        la, lo = latlon(self.pos[i:i+1])
        d = dict(id=int(self.id[i]), parent_id=(int(self.parent[i]) if self.parent[i] >= 0 else None),
                 parent2_id=(int(self.parent2[i]) if self.parent2[i] >= 0 else None), birth_t_w=float(self.birth[i]),
                 age=float(self.t - self.birth[i]), energy=round(float(self.energy[i]), 4), generation=int(self.gen[i]), species_id=int(self.species[i]),
                 lat=round(float(la[0]), 5), lon=round(float(lo[0]), 5),
                 traits={t: round(float(v), 4) for t, v in zip(TRAITS, self.traits[i])}, alive=t_death is None)
        if t_death is None and len(self.trail) > i:
            tr = self.trail[i]; ok = np.isfinite(tr[:, 0])
            if ok.sum() >= 2:
                a, b = latlon(tr[ok]); d["trail"] = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(a, b)]   # oldest -> newest
        if t_death is not None: d["death_t_w"] = float(t_death)
        return d
    def individuals(self, k=None):
        idx = np.argsort(self.id); idx = idx if k is None or len(idx) <= k else idx[np.linspace(0, len(idx)-1, k).astype(int)]
        return [self.record(i) for i in idx]
    def stats(self):
        n = len(self); tr = self.traits; ns = int(len(np.unique(self.species))) if n else 0
        return dict(t_w=float(self.t), n_alive=int(n), trait_means={t: float(tr[:, j].mean()) if n else None for j, t in enumerate(TRAITS)},
                    trait_stds={t: float(tr[:, j].std()) if n else None for j, t in enumerate(TRAITS)},
                    generations_max=int(self.gen.max()) if n else 0, generations_mean=float(self.gen.mean()) if n else 0.,
                    n_species=ns, n_species_alive=ns, n_species_ever=int(self.next_species), env_offset=float(self.env_offset),
                    T_K=self.T_K, n_births=int(self.n_births), n_deaths=int(self.n_deaths), n_sexual=int(self.n_sexual))
    def summary(self, k=None):
        return dict(individuals=self.individuals(k), n_alive=len(self), deceased_sample=self.graveyard, evolution_series=self.series,
                    total_births=int(self.n_births), total_deaths=int(self.n_deaths), total_sexual_births=int(self.n_sexual),
                    species_ever=int(self.next_species), n_species_ever=int(self.next_species),
                    n_species_alive=int(len(np.unique(self.species))) if len(self) else 0, impact_log=self.impact_log)

def _new_trail(pos):
    t = np.full((len(pos), TRAIL_K, 3), np.nan); t[:, -1] = pos; return t

def _push_trail(pop):
    if len(pop): pop.trail = np.concatenate([pop.trail[:, 1:], pop.pos[:, None]], 1)
    pop.last_trail_t = pop.t

def _new_ids(pop, k): ids = np.arange(pop.next_id, pop.next_id + k); pop.next_id += k; return ids

def found(LP, rng, n=40, t_w=0., traits=None):
    """Founding population: n individuals around LP trait means (or explicit `traits` dict), spread mutation_sd, near one site."""
    P = as_bus(LP); pop = Population(); mu = np.array([(traits or {}).get(t, P[t]) for t in TRAITS])
    tr = np.clip(mu + P["mutation_sd"]*SPAN*rng.normal(size=(n, len(TRAITS))), LO, HI)
    pos = _jitter(np.repeat(_rand_pos(rng, 1), n, 0), np.full(n, .15), rng)
    pop.append(id=_new_ids(pop, n), parent=-np.ones(n, int), parent2=-np.ones(n, int), birth=np.full(n, float(t_w)), energy=np.ones(n), traits=tr,
               gen=np.zeros(n, int), species=np.zeros(n, int), pos=pos, trail=_new_trail(pos))
    pop.next_species = 1; pop.t = float(t_w); pop.last_cluster_t = pop.t; pop.drift_phase = float(rng.uniform(0, 2*np.pi))
    pop.series.append(pop.stats()); pop.last_series_t = pop.t
    return pop

def climate_stress(climate):
    """Chronic stress (water, temperature) felt every year; shocks (impacts, UV) are applied once per step in `shock`."""
    T = climate.T_K - (climate.stressors or {}).get("impact_dT", 0.)   # impact winter is the one-off shock, not chronic stress
    w = (climate.stressors or {}).get("water_chronic", climate.liquid_water_frac)
    return float(WATER_W*(1 - w) + abs(T - 293.)/80.)

def shock_kill(climate):
    st = climate.stressors or {}; return float(1 - (1 - min(st.get("impact_kill", 0.), 1.))*(1 - min(st.get("uv_kill", 0.), 1.))*(1 - min(st.get("tech_kill", 0.), 1.)))

def env_drift(pop, P, rng, dt, climate=None):
    """Multi-millennial nonstationarity felt by organisms: two incommensurate cycles (8.3 & 21 kyr) + rare telegraph flips
    (~1 per 12 kyr, +-1). Returns stress offset (scaled by climate_drift); also stored on pop.env_offset."""
    st = getattr(climate, "stressors", None) or {}
    if "drift_offset" in st:      # planetary drift already in climate.T_K/water (life_loop + world.ClimateDrift): no double count
        pop.env_offset = float(st["drift_offset"]); return 0.
    a = P["climate_drift"]
    if rng.random() < 1 - np.exp(-dt/12000.): pop.flip = -pop.flip if pop.flip else float(rng.choice([-1., 1.]))
    t = pop.t; ph = pop.drift_phase
    pop.env_offset = float(a*(.22*np.sin(2*np.pi*t/8300. + ph) + .12*np.sin(2*np.pi*t/21000. + 2*ph) + .12*pop.flip)); return pop.env_offset

def food_mult(pop, P):
    """Productivity cycle (~13 kyr) anti-correlated-ish with stress drift: shifts the metabolism/reproduction optimum."""
    return float(1 + .35*P["climate_drift"]*np.sin(2*np.pi*pop.t/13000. + 1.3*pop.drift_phase))

def food_felt(pop, P, dt, resources=1.):
    """Lagged productivity (ecosystem storage: detritus/soil/seed banks, tau FOOD_TAU): first-order low-pass of
    niche resources x food_mult, so short cold spells and productivity phases are buffered."""
    f = max(0., resources)*food_mult(pop, P)
    if pop.food_buf is None: pop.food_buf = f
    pop.food_buf += (f - pop.food_buf)*(1 - np.exp(-dt/FOOD_TAU)); return pop.food_buf

def _bury(pop, dead, rng, grave_k, t):
    """Remove `dead` rows, keeping a reservoir sample of grave_k deceased records (compact lineage history)."""
    for i in np.flatnonzero(dead):
        pop.n_dead_seen += 1; j = pop.n_dead_seen - 1 if len(pop.graveyard) < grave_k else int(rng.integers(pop.n_dead_seen))
        if j < grave_k:
            rec = pop.record(i, t)
            if len(pop.graveyard) < grave_k: pop.graveyard.append(rec)
            else: pop.graveyard[j] = rec
    pop.take(~dead)

def impact_deaths(pop, f, climate, P, rng):
    """Boolean death mask for a one-off shock of strength f (0..1). Crowded cells die more (density-dependent), hardiness halves
    the chance, a shelter draw (10% + 30% x ocean cover) survives outright, and a survivor floor max(3, impact_refuge x n) is
    kept (hardiest victims spared) unless chronic stress is already lethal for everyone."""
    n = len(pop)
    if not n or f <= 0: return np.zeros(n, bool)
    dens = local_density(pop.pos); crowd = .6 + .4*np.clip(dens, 0, 3)/max(1., float(np.clip(dens, 0, 3).max()))
    p = np.clip(f*(1 - .5*pop.traits[:, H])*crowd, 0, 1)
    shelter = rng.random(n) < .1 + .3*float(climate.liquid_water_frac)
    dead = (rng.random(n) < p) & ~shelter
    lethal = climate_stress(climate) - pop.traits[:, H].max() > 1.5
    floor = 0 if lethal else min(n, max(3, int(np.ceil(P["impact_refuge"]*n))))
    short = floor - int((~dead).sum())
    if short > 0:
        v = np.flatnonzero(dead); spare = v[np.argsort(-(pop.traits[v, H] + .2*rng.random(len(v))))[:short]]; dead[spare] = False
    return dead

def step(pop, niche, climate, LP, rng, dt_w_yr=1., t_w=None, series_every=None, grave_k=200, h_max=1., mind=None):
    """Advance every individual by dt world-years (internal substeps <= h_max yr so demography is dt-stable).
    Returns (pop, CreaturePacket) summed over substeps. Mutates `pop` in place. `mind` (leaves.mind, default None): soft clamped
    steering of movement / foraging / rest / mate choice; None or inactive -> identical code path and rng draws."""
    P = as_bus(LP); k = max(1, int(np.ceil(dt_w_yr/h_max - 1e-9))); h = float(dt_w_yr)/k; t1 = float(t_w if t_w is not None else pop.t + dt_w_yr)
    B = D = 0; f = shock_kill(climate); pop.T_K = float(climate.T_K)
    if f > 1e-3 and len(pop):
        n0 = len(pop); dead = impact_deaths(pop, f, climate, P, rng); D += int(dead.sum()); _bury(pop, dead, rng, grave_k, pop.t); pop.n_deaths += int(dead.sum())
        pop.impact_log.append(dict(t_w=float(pop.t), kill=round(f, 4), n_before=n0, n_after=len(pop)))
    for i in range(k):
        b, d = _substep(pop, niche, climate, P, rng, h, t1 - (k-1-i)*h, grave_k, mind); B += b; D += d
    if pop.t - pop.last_cluster_t >= SPECIES_EVERY - 1e-9: recluster(pop, P); pop.last_cluster_t = pop.t
    if series_every and pop.t - pop.last_series_t >= series_every - 1e-9: pop.series.append(pop.stats()); pop.last_series_t = pop.t
    s = pop.stats(); means = {t: v for t, v in s["trait_means"].items() if v is not None}; means["cohorts"] = float(s["n_species"])
    cap = int(P["pop_cap"])
    return pop, CreaturePacket(float(B), float(D), float(len(pop)), means, float(pop.energy.sum()/max(cap, 1)) if len(pop) else 0.)

def recluster(pop, P):
    """Assign species_id = average-linkage clusters of the living set (trait + GEO_W x geo distance < .75 x speciation_dist).
    Small clusters join the nearest big one. Each cluster keeps the majority previous id (largest cluster wins ties);
    clusters that lose the vote get a new id (speciation) if they hold >= NEW_SPECIES_FRAC of the living; ids absent afterwards are extinct or merged."""
    n = len(pop)
    if n < 2: return
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    x = (pop.traits - LO)/SPAN; Dt = np.sqrt(((x[:, None] - x[None])**2).sum(-1))
    Dg = np.arccos(np.clip(pop.pos @ pop.pos.T, -1, 1))/np.pi; Dm = Dt + GEO_W*Dg; np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), "average"), CL_CUT*P["speciation_dist"], "distance") - 1; sz = np.bincount(lab)
    big = np.flatnonzero(sz >= MIN_SPECIES)
    if not len(big): big = np.array([int(np.argmax(sz))])
    small = ~np.isin(lab, big)
    if small.any():
        cols = np.isin(lab, big); j = np.argmin(np.where(cols[None, :], Dm[small], np.inf), axis=1); lab[small] = lab[j]
    new = pop.species.copy(); used = set(); cut = CL_CUT*P["speciation_dist"]
    dt_ = pop.t - getattr(pop, "mint_t", pop.t); pop.mint_t = pop.t; pop.mint_tok = min(MINT_CAP, getattr(pop, "mint_tok", MINT_CAP) + dt_/MINT_TAU)
    near = []; frac = NEW_SPECIES_FRAC*(1 + SOFT_K*max(0, len(np.unique(pop.species)) - SOFT_MAX)); cen_of = {}
    for c in sorted(np.unique(lab), key=lambda c: -int((lab == c).sum())):
        m = lab == c; ids, cnt = np.unique(pop.species[m], return_counts=True); o = np.argsort(-cnt); pick = None; near = []
        for q in o:
            if int(ids[q]) not in used: pick = int(ids[q]); break
        if pick is None and m.sum() < frac*n: pick = int(ids[o[0]])   # small daughter cluster: no new id (hysteresis + soft cap)
        cen = np.concatenate([x[m].mean(0), GEO_W*pop.pos[m].mean(0)])
        if pick is None and int(ids[o[0]]) in cen_of and np.linalg.norm(cen - cen_of[int(ids[o[0]])]) < SPLIT_SEP*cut: pick = int(ids[o[0]])   # too close to parent: merge back
        if pick is None:                         # re-use a recently seen, now unused id whose centroid is close (no churn)
            near = [(np.linalg.norm(cen - c), i) for i, (c, tt) in pop.recent.items() if i not in used and pop.t - tt <= RECENT_YR]
            if near and min(near)[0] < RECYCLE_R*cut*(1 + (RECYCLE_FAR - 1)*SMOOTH_R*max(0., 1 - pop.mint_tok/MINT_CAP)): pick = int(min(near)[1])
        if pick is None and getattr(pop, "mint_tok", MINT_CAP) < 1:   # no token: wide recycle, else merge back under the parent id
            if near and min(near)[0] < RECYCLE_FAR*RECYCLE_R*cut: pick = int(min(near)[1])
            else: pick = int(ids[o[0]])
        if pick is None: pick = pop.next_species; pop.next_species += 1; pop.mint_tok = getattr(pop, "mint_tok", MINT_CAP) - 1
        used.add(pick); new[m] = pick; pop.recent[pick] = (cen, pop.t); cen_of.setdefault(pick, cen)
    pop.recent = {i: v for i, v in pop.recent.items() if pop.t - v[1] <= RECENT_YR}
    pop.species = new

def _substep(pop, niche, climate, P, rng, dt, t, grave_k, mind=None):
    n = len(pop); cap = int(P["pop_cap"]); sel = P["sel_strength"]; pop.t = float(t); births = deaths = 0
    off = env_drift(pop, P, rng, dt, climate)
    if n:
        tr = pop.traits; hard, met, mob, rep = tr[:, H], tr[:, MET], tr[:, MOB], tr[:, REP]
        acts = None
        if mind is not None and getattr(mind, "active", False):     # mind: own rng only; decided on pre-move state
            from . import mind as M
            st0 = climate_stress(climate) + off + LAT_STRESS*(np.abs(pop.pos[:, 1]) - .5); f0 = (pop.food_buf if pop.food_buf is not None else max(0., niche.resources))*cap/max(n, 1)
            acts = M.tick(mind, pop, st0, local_density(pop.pos), f0)
        stp = .04*(.05 + mob)*np.sqrt(dt)
        pop.pos = _jitter(pop.pos, stp, rng)    # brownian walk on the sphere, step ~ mobility
        if acts is not None and np.any(acts[0]):
            from .mind import tangent, MOVE_K
            p = pop.pos + MOVE_K*stp[:, None]*tangent(pop.pos, acts[0]); pop.pos = p/np.linalg.norm(p, axis=1, keepdims=True)
        dens = local_density(pop.pos)
        need = need_of(tr)
        food = cap*dt*food_felt(pop, P, dt, niche.resources)       # stored food outlasts a producer crash by ~FOOD_TAU          # niche food shared by up to pop_cap organisms
        bonus = 2.2*mob/(mob + .3)*np.clip(dens, 0, 2)/2                    # saturating, only pays where crowded
        upt = uptake(tr); crowd = niche_crowding(tr, pop.pos)
        w = np.sqrt(need/(1 + .2*mob + .45*mob*mob))*(1 + bonus)*upt*(1 + .3*rng.random(n))/crowd**NICHE_ALPHA
        if acts is not None: w = w*acts[1]*(1 - 2*REST_K*acts[3])     # forage effort (resting forages less)
        got = np.minimum(food*w/w.sum(), 1.5*upt*need*dt) if food > 0 else np.zeros(n)
        stress = climate_stress(climate) + off + LAT_STRESS*(np.abs(pop.pos[:, 1]) - .5)
        torpor = 1 - DORM_STRESS*np.clip(stress - .2, 0, 1)                 # cold/dry torpor lowers basal burn
        burn = .5*torpor if food > 1e-3*cap*dt else .5*(.4 - .3*hard)                # famine -> dormancy (hardy forms encyst deeper)
        if acts is not None: burn = burn*(1 - REST_K*acts[3])
        pop.energy = np.clip(pop.energy + got - burn*need*dt, 0., E_MAX)
        starving = pop.energy <= 0
        exc = float(np.mean(np.maximum(0., stress - hard))); pop.accl += (min(ACCL_MAX, exc) - pop.accl)*(1 - np.exp(-dt/ACCL_TAU))
        hard_eff = hard*(1 - HARD_MET*met) + pop.accl                            # lean metabolisms: less reserve; + acclimation (plastic)
        refuge = float(np.clip(n/(REFUGE_N*cap), .2, 1.))                   # few survivors -> sheltered microhabitats
        # stress hazard saturates in excess x (x/(1+2x)): beyond tolerance organisms encyst rather than die linearly faster
        juv = (pop.t - pop.birth < JUV_AGE)*JUV_HAZ*((rep - LO[REP])/SPAN[REP])**2*(1 + 2*max(0., stress.mean() - .3))
        rate = P["eco_death"]*.5 + sel*.2*refuge*(x := np.maximum(0., stress - hard_eff))/(1 + HAZ_SAT*x) + (.25 if food > 1e-3*cap*dt else .03)*starving + juv
        dead = rng.random(n) < 1 - np.exp(-rate*dt); deaths = int(dead.sum())
        if deaths: _bury(pop, dead, rng, grave_k, pop.t)
        if deaths and acts is not None: acts = tuple(x[~dead] for x in acts)
        n = len(pop)
    if n:
        tr = pop.traits; need = need_of(tr)
        fit = np.clip(pop.energy/need, 0., 2.)
        lam = tr[:, REP]*dt*fit*uptake(tr)**UPT_REP*max(0., 1 - n/(1.5*cap)); kids = rng.poisson(lam)   # slow assimilation -> slow brood growth
        cost = .5*need; kids = np.minimum(kids, np.floor(np.maximum(pop.energy, 0)/cost).astype(int)); births = int(kids.sum())
        if births:
            par = np.repeat(np.arange(n), kids); pop.energy = pop.energy - kids*cost
            base = tr[par].copy(); par2 = -np.ones(births, int); g = pop.gen[par].copy()
            sex = np.flatnonzero(rng.random(births) < P["sex_rate"]) if n > 1 else np.zeros(0, int)
            if len(sex):                                                   # mate: best of 8 random candidates (same species, nearby)
                cand = rng.integers(n, size=(len(sex), 8)); p1 = par[sex]
                score = np.arccos(np.clip((pop.pos[cand]*pop.pos[p1][:, None]).sum(-1), -1, 1)) + 10.*(pop.species[cand] != pop.species[p1][:, None]) + 100.*(cand == p1[:, None])
                if acts is not None and len(acts[2]) == n: score = score - .5*np.log(acts[2][cand])     # mind: attractive mates preferred
                m2 = cand[np.arange(len(sex)), np.argmin(score, 1)]; ok = m2 != p1; sex, m2 = sex[ok], m2[ok]
                wb = rng.random((len(sex), len(TRAITS))); base[sex] = wb*tr[par[sex]] + (1 - wb)*tr[m2]; par2[sex] = pop.id[m2]; g[sex] = np.maximum(g[sex], pop.gen[m2]); pop.n_sexual += len(sex)
            sd = P["mutation_sd"]*P["evolution_rate"]*SPAN*np.where(rng.random((births, 1)) < .02, 5., 1.)   # rare big jumps
            ct = np.clip(base + sd*rng.normal(size=(births, len(TRAITS))), LO, HI)
            pop.append(id=_new_ids(pop, births), parent=pop.id[par], parent2=par2, birth=np.full(births, pop.t), energy=np.full(births, .5)*cost[par],
                       traits=ct, gen=g + 1,
                       species=pop.species[par].copy(), pos=(cp := _jitter(pop.pos[par], np.full(births, .02), rng)), trail=np.concatenate([pop.trail[par][:, 1:], cp[:, None]], 1))
        if len(pop) > cap:                                                 # soft cap: cull weakest (lowest energy/need)
            need = need_of(pop.traits)
            keep = np.argsort(-(pop.energy/need + .1*rng.random(len(pop))))[:cap]; m = np.zeros(len(pop), bool); m[keep] = True
            deaths += int((~m).sum()); pop.take(np.sort(np.flatnonzero(m)))
    if pop.t - pop.last_trail_t >= TRAIL_EVERY - 1e-9: _push_trail(pop)
    pop.n_births += births; pop.n_deaths += deaths
    return births, deaths
