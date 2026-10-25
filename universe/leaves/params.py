"""Learnable life-parameter space (parallel to sim2.SPACE / events.ESPACE). Every knob is (lo, hi)."""
LIFE_SPACE = {
    # climate
    "greenhouse_K": (0., 60.), "albedo_mult": (.6, 1.4), "snowball_K": (230., 260.), "runaway_K": (330., 400.),
    # chemistry
    "chem_rate": (.01, .3), "photo_boost": (0., 2.), "organic_thresh": (.05, .5),
    # abiogenesis prior (per world-year probability when ready)
    "abio_p": (0., 1e-3),
    # ecology
    "eco_growth": (.05, .8), "eco_stiffness": (.5, 2.), "eco_death": (.01, .2),
    # trait means + mutation
    "metabolism_eff": (.1, .9), "mobility": (0., 1.), "reproduction_rate": (.01, .5), "hardiness": (0., 1.),
    "mutation_sd": (0., .1),
    # individuals + evolution
    "evolution_rate": (0., 5.), "pop_cap": (50., 500.), "sel_strength": (0., 3.), "speciation_dist": (.1, .6),
    # host weather (flares)
    "flare_rate_mult": (0., 5.), "flare_amp": (0., 10.),
    # impacts (scale on HabitatState.impact_flux; kill = biomass fraction lost per unit shock)
    "impact_scale": (0., 20.), "impact_kill": (0., 1.),
    # culture / tech
    "tech_rate": (0., 1e-2), "tech_threshold": (.1, 2.),
}
LIFE_DEFAULTS = {k: (lo+hi)/2 for k, (lo, hi) in LIFE_SPACE.items()}
LIFE_DEFAULTS.update(greenhouse_K=33., albedo_mult=1., abio_p=2e-4, flare_rate_mult=1., flare_amp=2., impact_scale=1., impact_kill=.5, tech_rate=1e-3, tech_threshold=.5,
                     evolution_rate=1., pop_cap=300., sel_strength=1., speciation_dist=.4)
TRAITS = ("metabolism_eff", "mobility", "reproduction_rate", "hardiness")

def clamp(k, v):
    lo, hi = LIFE_SPACE[k]; return min(max(float(v), lo), hi)
