"""Orchestrator: nested world-time t_w under a frozen system snapshot.
For each HZ/near-HZ rocky habitat: world -> chemistry -> abiogenesis gate -> biosphere -> organisms."""
import numpy as np
from ..bodies import lum, temp
from . import world, chemistry, biosphere, organisms
from .packets import HostWeather, ScoreBus, as_bus
from .system_habitat import habitats_from_system, candidate

def host_weather(S):
    st = np.asarray(S["kind"]) == 0; M = float(S["m"][st].max()); L = float(sum(lum(float(m)) for m in S["m"][st]))
    return HostWeather.from_star(M, L, float(temp(M)))

def run_habitat(h, hw, LP, rng, years=2000, dt=10., plant=False, organisms_after=0., disturb_at=None):
    P = as_bus(LP); chem = chemistry.init_chem(h); gate = chemistry.planted_gate(0.) if plant else None
    bio = biosphere.Biosphere(); coh = []; series = []; clim = None; pkt = None; flux = None
    for k in range(int(years/dt)):
        t = k*dt; stress = {"dT": 0.}; dist = 0.
        if disturb_at is not None and abs(t - disturb_at) < dt/2: dist = .8
        clim = world.step(h, hw, P, dt, stress)
        chem = chemistry.step(chem, clim, hw, P, dt)
        gate = chemistry.abiogenesis_roll(chem, clim, P, rng, t, dt, gate)
        bio = biosphere.seed(bio, gate)
        bio, niche, flux = biosphere.step(bio, clim, chem, P, dt, dist)
        if bio.alive and t >= organisms_after and not coh and bio.P > .05: coh = organisms.new_cohorts(P, rng)
        if coh:
            coh, pkt = organisms.step(coh, niche, clim, P, rng, dt); bio = biosphere.apply_creatures(bio, pkt)
        series.append(bio.total())
    return dict(habitat=h.to_dict(), climate=clim.to_dict(), chem=chem.to_dict(), gate=gate.to_dict(),
                biosphere=bio.to_dict(), biomass=float(bio.total()), biosign=flux.to_dict(),
                creatures=pkt.to_dict() if pkt else None, biomass_series=[float(x) for x in series[::max(1, len(series)//50)]],
                score=dict(eco_stability=ScoreBus.eco_stability(series), planted_life=ScoreBus.planted_life(bio.alive)))

def life_loop(S, seed, LP=None, years=2000, dt=10., plant=False, rng=None):
    P = as_bus(LP); rng = rng or np.random.default_rng(seed); hw = host_weather(S)
    habs = habitats_from_system(S, seed, P); rep = []
    for h in habs:
        if candidate(h): rep.append(run_habitat(h, hw, P, rng, years, dt, plant))
        else: rep.append(dict(habitat=h.to_dict(), skipped="not rocky near-HZ"))
    clims = [world.step(h, hw, P) for h in habs]
    return dict(seed=int(seed), host=hw.to_dict(), LP=P.knobs, years=years, dt=dt, planted=plant,
                habitability_residual=ScoreBus.habitability(clims), planets=rep,
                with_life=[r["habitat"]["name"] for r in rep if r.get("biosphere", {}).get("alive")])
