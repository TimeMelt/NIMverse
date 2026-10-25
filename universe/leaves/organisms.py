"""Organisms leaf: cohort model. Each cohort = (n, energy, traits). No reproduction without resources."""
import numpy as np
from .packets import CreaturePacket, as_bus
from .params import TRAITS, clamp

def new_cohorts(LP, rng, k=3, n=10.):
    P = as_bus(LP); return [dict(n=float(n), energy=0., traits={t: clamp(t, P[t] + P["mutation_sd"]*rng.normal()) for t in TRAITS}) for _ in range(k)]

def step(cohorts, niche, climate, LP, rng, dt_w_yr=1.):
    """Consume niche resources, reproduce if energy budget closes, die under stress. Returns (cohorts, CreaturePacket)."""
    P = as_bus(LP); res = max(0., niche.resources)*dt_w_yr; births = deaths = 0.; out = []
    demand = sum(c["n"]*(1 - .5*c["traits"]["metabolism_eff"]) for c in cohorts) + 1e-12
    stress = (1 - climate.liquid_water_frac) + (abs(climate.T_K-293.)/80.)
    for c in cohorts:
        tr = c["traits"]; need = c["n"]*(1 - .5*tr["metabolism_eff"])
        got = res*need/demand*(1 + .2*tr["mobility"]) if res > 0 else 0.; got = min(got, res*need/demand*1.2)
        e = max(0., c["energy"] + got - need*dt_w_yr*.1)               # energy budget never negative
        b = tr["reproduction_rate"]*c["n"]*dt_w_yr if (got > 0 and e > need*.1) else 0.
        cost = b*.1; b = b if e >= cost else 0.; e -= cost if b else 0.
        d = c["n"]*min(1., (.05 + max(0., stress - tr["hardiness"]) + (1. if got <= 0 else 0.)*.5)*dt_w_yr)
        n = max(0., c["n"] + b - d); births += b; deaths += d
        if n >= .5:
            if b and rng.random() < .05:                               # mutant split
                t2 = {t: clamp(t, v + P["mutation_sd"]*rng.normal()) for t, v in tr.items()}
                out.append(dict(n=n*.1, energy=e*.1, traits=t2)); n, e = n*.9, e*.9
            out.append(dict(n=n, energy=max(0., e), traits=tr))
    out = out[:50]; N = sum(c["n"] for c in out)
    means = {t: float(sum(c["n"]*c["traits"][t] for c in out)/N) for t in TRAITS} if N else {}
    means["cohorts"] = float(len(out))
    return out, CreaturePacket(float(births), float(deaths), float(N), means, float(sum(c["energy"] for c in out)))
