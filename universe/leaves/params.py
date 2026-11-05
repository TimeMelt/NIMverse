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
    "sex_rate": (0., 1.), "climate_drift": (0., 1.), "impact_refuge": (0., .5),
    # host weather (flares)
    "flare_rate_mult": (0., 5.), "flare_amp": (0., 10.),
    # stellar evolution: initial MS age (Gyr), dL/dt scale (1 = real rate), time compression (stellar Gyr per world Gyr)
    "stellar_age_Gyr": (.05, 12.), "stellar_evolve": (0., 1.), "stellar_age_rate": (1., 1e6),
    # impacts (scale on HabitatState.impact_flux; kill = biomass fraction lost per unit shock)
    "impact_scale": (.25, 20.), "impact_kill": (0., 1.),
    # culture / tech
    "tech_rate": (0., 1e-2), "tech_threshold": (.1, 2.),
    # culture <-> biosphere coupling: biomass/yr drawn at level 1, pollution stock at level 1, K warming per unit pollution
    "tech_draw": (0., .5), "tech_pollute": (0., 2.), "tech_warm_K": (0., 60.),
    # culture v3: fossil share of energy mix, K cooling per unit aerosol, acidification gain, chem outgassing, mitigation rate, conflict gain
    "tech_fossil_frac": (0., 1.), "tech_aerosol_K": (0., 30.), "tech_acid": (0., 5.), "tech_chem": (0., 10.),
    "tech_mitigate": (0., .01), "tech_conflict": (0., 2.),
    # physics15: latitudinal climate diffusion (W m^-2 K^-1), K per CO2 doubling, carbon-cycle time compression (1 = real
    # ~0.5 Myr inorganic-carbon timescale), soft individual predation rate (1/yr), heritable log-mass mutation sd
    "lat_diffusion": (.1, 3.), "co2_sens_K": (1., 8.), "carbon_rate": (0., 200.), "pred_rate": (0., .3), "mass_sd": (0., .5),
}
LIFE_DEFAULTS = {k: (lo+hi)/2 for k, (lo, hi) in LIFE_SPACE.items()}
LIFE_DEFAULTS.update(greenhouse_K=33., albedo_mult=1., abio_p=4e-4, snowball_K=238., organic_thresh=.18, flare_rate_mult=1., flare_amp=2., stellar_age_Gyr=4.6, stellar_evolve=0., stellar_age_rate=1., impact_scale=1., impact_kill=.5, tech_rate=1e-3, tech_threshold=.5, tech_draw=.001, tech_pollute=.1, tech_warm_K=5., tech_fossil_frac=.7, tech_aerosol_K=2., tech_acid=1., tech_chem=.1, tech_mitigate=2e-4, tech_conflict=1.,
                     evolution_rate=1., pop_cap=300., sel_strength=1., speciation_dist=.4,
                     sex_rate=.3, climate_drift=.5, impact_refuge=.05,
                     lat_diffusion=.3, co2_sens_K=4., carbon_rate=20., pred_rate=.03, mass_sd=.2)
TRAITS = ("metabolism_eff", "mobility", "reproduction_rate", "hardiness")

def clamp(k, v):
    lo, hi = LIFE_SPACE[k]; return min(max(float(v), lo), hi)
