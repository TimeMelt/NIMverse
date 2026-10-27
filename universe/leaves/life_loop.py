"""Orchestrator: nested world-time t_w under a frozen system snapshot.
For each HZ/near-HZ rocky habitat: HostWeather(flares) -> impacts -> world (+history) -> chemistry -> abiogenesis gate
-> biosphere -> organisms -> culture (optional; draws producers same step, pollution warming/toxicity fed to the NEXT step's world/bio/organisms) -> BioSignFlux -> radiation observers (light-time delayed)."""
import copy, numpy as np
from ..bodies import lum, temp
from . import stellar, world, chemistry, biosphere, organisms, individuals, culture as culture_mod
from .packets import HostWeather, ScoreBus, as_bus
from .radiation import ObserverCatalog, Observer
from . import biosign
from .system_habitat import habitats_from_system, candidate

def host_weather(S, distance_pc=None, M_override=None, seed=None):
    """HostWeather for the system's primary.  With S["age_Gyr"] or a seed the star gets a system age prior
    (stellar.age_prior(M, seed): deterministic log-uniform, massive stars younger) used unless LP stellar_age_Gyr is non-default."""
    st = np.asarray(S["kind"]) == 0; M = float(M_override or S["m"][st].max())
    L = float(sum(lum(float(m)) for m in S["m"][st])) if not M_override else float(lum(M))
    d = distance_pc if distance_pc is not None else float(S.get("distance_pc", 10.))
    hw = HostWeather.from_star(M, L, float(temp(M)), d)
    if S.get("age_Gyr") is not None: hw.age_prior_Gyr = float(S["age_Gyr"])
    elif seed is not None: hw.age_prior_Gyr = stellar.age_prior(M, seed)
    return hw

def sync_habitat(h, hw):
    """Live habitat snapshot from the (evolving) host: hz_in/out, L_star, irradiance (baseline L_star/a^2, flares excluded).
    Orbit a and the initial in_hz (reference for hz_drift) stay fixed."""
    h.L_star = float(hw.L_star); h.hz_in, h.hz_out = float(hw.hz_in_now), float(hw.hz_out_now); h.irradiance = float(hw.L_star/max(h.a, 1e-9)**2)

class HabitatRun:
    """Stepwise habitat simulation (the body of run_habitat).  step() advances one dt; report() builds the run_habitat dict.
    years=None -> open-ended (continuous mode): ser_every falls back to `ser_every` (default 50*dt)."""
    def __init__(self, h, hw, LP, rng, years=2000, dt=10., plant=False, organisms_after=0., disturb_at=None,
                 culture=True, catalog=None, emit_every=1, culture_min_age=1000., record=None, model="individuals", n_founders=40, ser_every=None):
        self.P = P = as_bus(LP); self.hw = copy.deepcopy(hw); self.h = h = copy.deepcopy(h); self.chem = chemistry.init_chem(h); self.gate = chemistry.planted_gate(0.) if plant else None
        self.rng, self.dt, self.years, self.organisms_after, self.disturb_at = rng, dt, years, organisms_after, disturb_at
        self.culture, self.catalog, self.emit_every, self.culture_min_age, self.record, self.model, self.n_founders = culture, catalog, emit_every, culture_min_age, record, model, n_founders
        self.bio = biosphere.Biosphere(); self.coh = []; self.pop = None; self.series = []; self.clim = None; self.pkt = None; self.flux = None
        self.ser_every = max(dt, years/200.) if years is not None else (ser_every or 50*dt)
        self.drng = np.random.default_rng([int(abs(h.a)*1e6) % 2**31, 11])        # own stream: drift does not perturb the main rng draws
        self.orng = np.random.default_rng([int(abs(h.a)*1e6) % 2**31, 23])        # own stream for organisms: life draws never shift impacts/abiogenesis timing
        self.drift = world.ClimateDrift(P["climate_drift"], self.drng) if P["climate_drift"] > 0 else None
        self.hist = world.ClimateHistory(n=4000); self.cu = culture_mod.Culture(); self.tstress = {}; self.peak = {}; self.n_imp = 0; self.impact_loss = 0.; self.k = 0

    @property
    def t(self): return self.k*self.dt          # world time of the NEXT step

    def step(self):
        """One dt step (identical math/rng order to the original run_habitat loop body). Returns t_w of the step."""
        P, h, hw, rng, dt, k = self.P, self.h, self.hw, self.rng, self.dt, self.k
        t = k*dt; dist = 0.; tstress = self.tstress
        if self.disturb_at is not None and abs(t - self.disturb_at) < dt/2: dist = .8
        hw.step(t, dt, rng, P); sync_habitat(h, hw)
        stress = {"dT": 0.}; imp = world.impacts(h, P, rng, dt)
        if self.drift is not None: stress.update(self.drift.step(t, dt, self.drng))          # planetary climate drift (T, water, class, stress)
        stress.update({kk: v for kk, v in imp.items() if kk != "dT"}); stress["dT"] += imp.get("dT", 0.); stress["impact_dT"] = imp.get("dT", 0.)
        if tstress: stress.update(tstress); stress["dT"] += tstress["tech_dT"]       # previous step's tech feedback
        clim = world.step(h, hw, P, dt, stress); self.hist.push(t, clim)
        if stress["impact_dT"]:      # impact winter freezes water briefly: the shock is the kill, chronic stress uses pre-impact water
            clim.stressors["water_chronic"] = world.step(h, hw, P, dt, dict(stress, dT=stress["dT"] - stress["impact_dT"])).liquid_water_frac
        chem = chemistry.step(self.chem, clim, hw, P, dt); cu = self.cu
        if self.culture and (cu.active or cu.pollution > 0): chem = culture_mod.apply_chem(chem, cu, P, dt)   # last step's industry -> organics/redox
        gate = self.gate = chemistry.abiogenesis_roll(chem, clim, P, rng, t, dt, self.gate)
        bio = biosphere.seed(self.bio, gate); m0 = bio.total()
        bio, niche, flux = biosphere.step(bio, clim, chem, P, dt, dist)
        if tstress.get("tech_footprint"): niche.resources = float(niche.resources*(1 - min(tstress["tech_footprint"], 1.)))   # last step's draw
        if clim.stressors.get("impacts"): self.n_imp += clim.stressors["impacts"]; self.impact_loss += m0*clim.stressors["impact_kill"]
        coh, pop, pkt = self.coh, self.pop, self.pkt
        if self.model == "individuals":
            if bio.alive and t >= self.organisms_after and pop is None and bio.P > .05: pop = individuals.found(P, self.orng, self.n_founders, t)
            if pop is not None and len(pop):
                pop, pkt = individuals.step(pop, niche, clim, P, self.orng, dt, t, self.ser_every); bio = biosphere.apply_creatures(bio, pkt)
            coh = pop is not None and len(pop) > 0
        elif bio.alive and t >= self.organisms_after and not coh and bio.P > .05: coh = organisms.new_cohorts(P, rng)
        if coh and self.model != "individuals":
            coh, pkt = organisms.step(coh, niche, clim, P, rng, dt); bio = biosphere.apply_creatures(bio, pkt)
        elif not coh and pkt is not None and pkt.n > 0: pkt = None
        if self.culture:
            cu = culture_mod.step(cu, bio, pkt if coh else None, P, rng, t, dt, self.culture_min_age); flux.tech = float(cu.level); biosign.attach(flux)
            bio = culture_mod.apply(bio, cu); tstress = culture_mod.stress(cu)
            tstress["tech_footprint"] = cu.footprint; self.peak = {kk: max(self.peak.get(kk, 0.), v) for kk, v in tstress.items()}
        if self.catalog is not None:
            if k % self.emit_every == 0: self.catalog.emit(t, h.name, flux)
            self.catalog.poll(t)
        self.series.append(bio.total())
        self.chem, self.bio, self.flux, self.clim, self.cu, self.tstress, self.coh, self.pop, self.pkt = chem, bio, flux, clim, cu, tstress, coh, pop, pkt
        if self.record is not None:
            self.record(dict(planet=h.name, t_w=float(t), T=float(clim.T_K), drift=float(clim.stressors.get("drift_offset", 0.)), water=float(clim.liquid_water_frac), biomass=float(bio.total()),
                        alive=bool(bio.alive), o2=float(flux.o2), ch4=float(flux.ch4), pigments=float(flux.pigments), tech=float(flux.tech), tech_dT=float(tstress.get("tech_dT", 0.)), pollution=float(tstress.get("tech_pollution", 0.)),
                        aerosol_K=float(tstress.get("tech_aerosol_K", 0.)), acid=float(tstress.get("tech_acid", 0.)), mitigation=float(tstress.get("tech_mitigation", 0.)),
                        n_org=float(pkt.n) if (coh and pkt is not None) else 0., impacts=int(self.n_imp),
                        trait_means=(dict(pkt.trait_means) if (coh and pkt is not None) else {})))
        if len(self.series) > 200000: self.series = self.series[-100000:]   # continuous mode: bound memory (finite runs never reach this at sane years/dt)
        self.k += 1
        return float(t)

    def frames(self, max_years=None, emit_every=1, stop=None):
        """Generator: step forever (or until t >= max_years / stop() true), yielding self every `emit_every` steps."""
        while (max_years is None or max_years <= 0 or self.t < max_years) and not (stop and stop()):
            self.step()
            if self.k % max(1, int(emit_every)) == 0: yield self

    def tick(self, seed=0, max_show=48, trail_max=24):
        """Small JSON-able live frame (see run_life_live). Creatures carry the server-side movement trail
        (oldest -> newest [lat, lon], last `trail_max` stored points + current position; 0 = no trails)."""
        from . import creature_look as CL
        clim, flux, cu, pop = self.clim, self.flux, self.cu, self.pop
        n_alive = len(pop) if pop is not None else 0
        crs = []
        if n_alive:
            idx = np.argsort((pop.id.astype(np.uint64)*np.uint64(2654435761)) % np.uint64(2**32))[:max_show]   # stable hashed subset: same ids frame to frame
            inds = [pop.record(int(i)) for i in idx]
            for d in inds:
                if trail_max and d.get("trail"): d["trail"] = d["trail"][-int(trail_max):]
                else: d.pop("trail", None)
            crs = [{kk: c[kk] for kk in ("id", "lat", "lon", "species_id", "plan", "size", "limbs", "segments", "shell", "shell_k", "color", "accent", "glow", "speed", "wobble",
                                          "phase", "heading", "traits", "generation", "age", "energy", "parent_id", "parent2_id", "trail") if kk in c}
                   for c in CL.creatures_from_individuals(inds, seed, flux.to_dict(), self.h.ptype, max_show)]
        cd = cu.to_dict() if self.culture else {}
        cult = dict(level=float(cu.level), active=bool(cu.active), pollution=float(cu.pollution), warm_K=float(self.tstress.get("tech_dT", 0.)),
                    polities_n=len(cd.get("polities") or []), footprint=float(cu.footprint)) if self.culture else None
        return dict(t_w=float((self.k-1)*self.dt), step=int(self.k), planet=self.h.name, ptype=self.h.ptype,
                    climate=dict(T_K=float(clim.T_K), water=float(clim.liquid_water_frac), cls=clim.climate_class),
                    biomass=float(self.bio.total()), alive=bool(self.bio.alive),
                    biosign=dict(o2=float(flux.o2), ch4=float(flux.ch4), pigments=float(flux.pigments), tech=float(flux.tech), h2o=float(getattr(flux, "h2o", 0.) or 0.)),
                    tech=float(flux.tech), culture=cult, n_alive=int(n_alive), n_species=int(len(np.unique(pop.species))) if n_alive else 0,
                    births=int(pop.n_births) if pop is not None else 0, impacts=int(self.n_imp), creatures=crs)

    def report(self):
        P, h, hw, pop, clim, flux, cu, pkt, series, model = self.P, self.h, self.hw, self.pop, self.clim, self.flux, self.cu, self.pkt, self.series, self.model
        if pop is not None and (not pop.series or pop.series[-1]["t_w"] < pop.t): pop.series.append(pop.stats())
        evo = None
        if pop is not None:
            sm = pop.summary(); ser = sm["evolution_series"]
            evo = dict(model="individuals", founding_traits=ser[0]["trait_means"] if ser else None, final_traits=ser[-1]["trait_means"] if ser else None,
                       **{k: v for k, v in sm.items()})
        return dict(habitat=h.to_dict(), climate=clim.to_dict(), climate_history=self.hist.summary(), chem=self.chem.to_dict(), gate=self.gate.to_dict(),
                    biosphere=self.bio.to_dict(), biomass=float(self.bio.total()), biosign=flux.to_dict(), host=hw.summary(),
                    impacts=dict(events=int(self.n_imp), biomass_lost=float(self.impact_loss), rate_per_kyr=float(h.impact_flux*P["impact_scale"])),
                    culture=dict(cu.to_dict(), peak=self.peak) if self.culture else None, tech=float(flux.tech),
                    creatures=pkt.to_dict() if pkt else None, organism_model=model,
                    individuals=(evo or {}).get("individuals"), evolution=({k: v for k, v in evo.items() if k != "individuals"} if evo else None), biomass_series=[float(x) for x in series[::max(1, len(series)//50)]],
                    score=dict(eco_stability=ScoreBus.eco_stability(series), planted_life=ScoreBus.planted_life(self.bio.alive)))

def run_habitat(h, hw, LP, rng, years=2000, dt=10., plant=False, organisms_after=0., disturb_at=None,
                culture=True, catalog=None, emit_every=1, culture_min_age=1000., record=None, model="individuals", n_founders=40):
    """model="individuals" (default): literal organisms with ids/parents + evolution (leaves.individuals); "cohorts": legacy cohort model.
    Finite wrapper over HabitatRun (same rng order -> identical reports)."""
    run = HabitatRun(h, hw, LP, rng, years, dt, plant, organisms_after, disturb_at, culture, catalog, emit_every, culture_min_age, record, model, n_founders)
    for _ in range(int(years/dt)): run.step()
    return run.report()

def run_habitat_continuous(h, hw, LP, rng, dt=10., plant=False, culture=True, max_years=None, emit_every=1, stop=None, **kw):
    """Open-ended generator of HabitatRun states (yield every emit_every steps) until max_years / stop() / KeyboardInterrupt."""
    run = HabitatRun(h, hw, LP, rng, None, dt, plant, culture=culture, **kw)
    yield from run.frames(max_years, emit_every, stop)

def life_loop(S, seed, LP=None, years=2000, dt=10., plant=False, rng=None, culture=True, distance_pc=None, observers=None,
              emit_every=None, M_override=None, record=None, model="individuals"):
    P = as_bus(LP); rng = rng or np.random.default_rng(seed); hw = host_weather(S, distance_pc, M_override, seed=seed)
    cat = ObserverCatalog(observers or [Observer("obs0", hw.distance_pc)])
    ee = emit_every or max(1, int(years/dt)//400)
    habs = habitats_from_system(S, seed, P); rep = []
    for h in habs:
        if candidate(h): rep.append(run_habitat(h, hw, P, rng, years, dt, plant, culture=culture, catalog=cat, emit_every=ee, record=record, model=model))
        else: rep.append(dict(habitat=h.to_dict(), skipped="not rocky near-HZ"))
    cat.poll(years)
    clims = [world.step(h, hw, P) for h in habs]
    live = [r for r in rep if "skipped" not in r]
    h0 = live[0]["host"] if live else {}
    return dict(seed=int(seed), host=hw.to_dict(), stellar_age=dict(prior_Gyr=hw.age_prior_Gyr, source=h0.get("age_source"), age0_Gyr=h0.get("age0_Gyr")), host_flares=(live[0]["host"] if live else hw.summary()), LP=P.knobs, years=years, dt=dt,
                planted=plant, culture_enabled=culture, habitability_residual=ScoreBus.habitability(clims), planets=rep,
                with_life=[r["habitat"]["name"] for r in rep if r.get("biosphere", {}).get("alive")],
                with_tech=[r["habitat"]["name"] for r in rep if r.get("tech", 0) > 0],
                radiation=cat.summary())
