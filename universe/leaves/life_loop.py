"""Orchestrator: nested world-time t_w under a frozen system snapshot.
For each HZ/near-HZ rocky habitat: HostWeather(flares) -> impacts -> world (+history) -> chemistry -> abiogenesis gate
-> biosphere -> organisms -> culture (optional) -> BioSignFlux -> radiation observers (light-time delayed)."""
import copy, numpy as np
from ..bodies import lum, temp
from . import world, chemistry, biosphere, organisms, individuals, culture as culture_mod
from .packets import HostWeather, ScoreBus, as_bus
from .radiation import ObserverCatalog, Observer
from .system_habitat import habitats_from_system, candidate

def host_weather(S, distance_pc=None, M_override=None):
    st = np.asarray(S["kind"]) == 0; M = float(M_override or S["m"][st].max())
    L = float(sum(lum(float(m)) for m in S["m"][st])) if not M_override else float(lum(M))
    d = distance_pc if distance_pc is not None else float(S.get("distance_pc", 10.))
    return HostWeather.from_star(M, L, float(temp(M)), d)

def run_habitat(h, hw, LP, rng, years=2000, dt=10., plant=False, organisms_after=0., disturb_at=None,
                culture=True, catalog=None, emit_every=1, culture_min_age=1000., record=None, model="individuals", n_founders=40):
    """model="individuals" (default): literal organisms with ids/parents + evolution (leaves.individuals); "cohorts": legacy cohort model."""
    P = as_bus(LP); hw = copy.deepcopy(hw); chem = chemistry.init_chem(h); gate = chemistry.planted_gate(0.) if plant else None
    bio = biosphere.Biosphere(); coh = []; pop = None; ser_every = max(dt, years/200.); series = []; clim = None; pkt = None; flux = None
    drng = np.random.default_rng([int(abs(h.a)*1e6) % 2**31, 11])        # own stream: drift does not perturb the main rng draws
    drift = world.ClimateDrift(P["climate_drift"], drng) if P["climate_drift"] > 0 else None
    hist = world.ClimateHistory(n=4000); cu = culture_mod.Culture(); kill_tot = 0.; n_imp = 0; impact_loss = 0.
    for k in range(int(years/dt)):
        t = k*dt; dist = 0.
        if disturb_at is not None and abs(t - disturb_at) < dt/2: dist = .8
        hw.step(t, dt, rng, P)
        stress = {"dT": 0.}; imp = world.impacts(h, P, rng, dt)
        if drift is not None: stress.update(drift.step(t, dt, drng))          # planetary climate drift (T, water, class, stress)
        stress.update({k: v for k, v in imp.items() if k != "dT"}); stress["dT"] += imp.get("dT", 0.); stress["impact_dT"] = imp.get("dT", 0.)
        clim = world.step(h, hw, P, dt, stress); hist.push(t, clim)
        chem = chemistry.step(chem, clim, hw, P, dt)
        gate = chemistry.abiogenesis_roll(chem, clim, P, rng, t, dt, gate)
        bio = biosphere.seed(bio, gate); m0 = bio.total()
        bio, niche, flux = biosphere.step(bio, clim, chem, P, dt, dist)
        if clim.stressors.get("impacts"): n_imp += clim.stressors["impacts"]; impact_loss += m0*clim.stressors["impact_kill"]
        if model == "individuals":
            if bio.alive and t >= organisms_after and pop is None and bio.P > .05: pop = individuals.found(P, rng, n_founders, t)
            if pop is not None and len(pop):
                pop, pkt = individuals.step(pop, niche, clim, P, rng, dt, t, ser_every); bio = biosphere.apply_creatures(bio, pkt)
            coh = pop is not None and len(pop) > 0
        elif bio.alive and t >= organisms_after and not coh and bio.P > .05: coh = organisms.new_cohorts(P, rng)
        if coh and model != "individuals":
            coh, pkt = organisms.step(coh, niche, clim, P, rng, dt); bio = biosphere.apply_creatures(bio, pkt)
        elif not coh and pkt is not None and pkt.n > 0: pkt = None
        if culture: cu = culture_mod.step(cu, bio, pkt if coh else None, P, rng, t, dt, culture_min_age); flux.tech = float(cu.level)
        if catalog is not None:
            if k % emit_every == 0: catalog.emit(t, h.name, flux)
            catalog.poll(t)
        series.append(bio.total())
        if record is not None:
            record(dict(planet=h.name, t_w=float(t), T=float(clim.T_K), drift=float(clim.stressors.get("drift_offset", 0.)), water=float(clim.liquid_water_frac), biomass=float(bio.total()),
                        alive=bool(bio.alive), o2=float(flux.o2), ch4=float(flux.ch4), pigments=float(flux.pigments), tech=float(flux.tech),
                        n_org=float(pkt.n) if (coh and pkt is not None) else 0., impacts=int(n_imp),
                        trait_means=(dict(pkt.trait_means) if (coh and pkt is not None) else {})))
    if pop is not None and (not pop.series or pop.series[-1]["t_w"] < pop.t): pop.series.append(pop.stats())
    evo = None
    if pop is not None:
        sm = pop.summary(); ser = sm["evolution_series"]
        evo = dict(model="individuals", founding_traits=ser[0]["trait_means"] if ser else None, final_traits=ser[-1]["trait_means"] if ser else None,
                   **{k: v for k, v in sm.items()})
    return dict(habitat=h.to_dict(), climate=clim.to_dict(), climate_history=hist.summary(), chem=chem.to_dict(), gate=gate.to_dict(),
                biosphere=bio.to_dict(), biomass=float(bio.total()), biosign=flux.to_dict(), host=hw.summary(),
                impacts=dict(events=int(n_imp), biomass_lost=float(impact_loss), rate_per_kyr=float(h.impact_flux*P["impact_scale"])),
                culture=cu.to_dict() if culture else None, tech=float(flux.tech),
                creatures=pkt.to_dict() if pkt else None, organism_model=model,
                individuals=(evo or {}).get("individuals"), evolution=({k: v for k, v in evo.items() if k != "individuals"} if evo else None), biomass_series=[float(x) for x in series[::max(1, len(series)//50)]],
                score=dict(eco_stability=ScoreBus.eco_stability(series), planted_life=ScoreBus.planted_life(bio.alive)))

def life_loop(S, seed, LP=None, years=2000, dt=10., plant=False, rng=None, culture=True, distance_pc=None, observers=None,
              emit_every=None, M_override=None, record=None, model="individuals"):
    P = as_bus(LP); rng = rng or np.random.default_rng(seed); hw = host_weather(S, distance_pc, M_override)
    cat = ObserverCatalog(observers or [Observer("obs0", hw.distance_pc)])
    ee = emit_every or max(1, int(years/dt)//400)
    habs = habitats_from_system(S, seed, P); rep = []
    for h in habs:
        if candidate(h): rep.append(run_habitat(h, hw, P, rng, years, dt, plant, culture=culture, catalog=cat, emit_every=ee, record=record, model=model))
        else: rep.append(dict(habitat=h.to_dict(), skipped="not rocky near-HZ"))
    cat.poll(years)
    clims = [world.step(h, hw, P) for h in habs]
    live = [r for r in rep if "skipped" not in r]
    return dict(seed=int(seed), host=hw.to_dict(), host_flares=(live[0]["host"] if live else hw.summary()), LP=P.knobs, years=years, dt=dt,
                planted=plant, culture_enabled=culture, habitability_residual=ScoreBus.habitability(clims), planets=rep,
                with_life=[r["habitat"]["name"] for r in rep if r.get("biosphere", {}).get("alive")],
                with_tech=[r["habitat"]["name"] for r in rep if r.get("tech", 0) > 0],
                radiation=cat.summary())
