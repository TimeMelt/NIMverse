# nim-universe

Procedural 3D universe generator that uses NIM (https://github.com/TimeMelt/NIM) inceptron meshes and the QrnLOpt optimizer to learn its generation parameters. It covers galaxies, stars, planets, supernovae, black holes, TDEs and AGN, and includes a three.js viewer.

- `universe/`: the code. It's run as the `universe` package next to a NIM checkout, e.g. `python -m universe.learn_alt2`.
- `results/`: the learned parameters, training logs and convergence plots.

Large rendered outputs (videos, HTML viewers, frames, .npz) are not included.

## Life leaves (v1)

`universe/leaves/` adds a physics+life stack on top of a planetary system snapshot. These are v1 box models: they get the gates right, not the detail.

- `packets.py`: typed dataclass contracts (TideLink, HostWeather, HabitatState, ClimateStress, ChemInventory, AbiogenesisGate, NicheOffer, CreaturePacket, BioSignFlux, LifeParamBus, ScoreBus)
- `params.py`: LIFE_SPACE / LIFE_DEFAULTS learnable ranges (like sim2.SPACE / events.ESPACE)
- `system_habitat.py`: `habitats_from_system(S, seed)`. The HZ becomes sim state, using the same math as `bodies.classify`
- `world.py` (energy balance) → `chemistry.py` (prebiotic pools + abiogenesis gate) → `biosphere.py` (3-guild ODE) → `organisms.py` (trait cohorts)
- `life_loop.py`: nested world-time loop over rocky habitats in or near the HZ

Run (NIM checkout on the path):

    PYTHONPATH=/workspace/NIM python -m universe.run_life --seed 4 --years 20000 --dt 20   # -> /workspace/nim_universe/out/life_demo_seed4.json
    PYTHONPATH=/workspace/NIM python -m pytest -q universe/leaves/test_life_leaves.py

## Deferred leaves (v2)

- **HostWeather** (`packets.py`): Poisson flares (`flare_rate` per yr, M dwarfs flare more), duty cycle, UV/X-ray boosts, `L_now`, a rolling irradiance history (`irr_hist`), and toy wind mass loss `dM_star`. World uses `L_now` and turns flare UV into a `uv_kill` stressor. Chemistry gets a photochemistry boost from `uv_now`/`xray_now`. Knobs: `flare_rate_mult`, `flare_amp`.
- **Climate history + impacts** (`world.py`): `ClimateHistory` keeps the rolling (t, T, water) record. `impacts()` draws Poisson impacts from `HabitatState.impact_flux*impact_scale`. Each impact gives an impact-winter dT and an `impact_kill` biomass fraction, and both biosphere and organism hardiness consume them.
- **Radiation / observers** (`radiation.py`): an `ObserverCatalog` takes BioSignFlux emitted at t_emit and delivers it at t_emit + light time (distance in pc, default 10). It applies 1/d² dimming (with an `aperture` multiplier) and a toy Hubble-flow redshift, or a cosmological z through lookback time.
- **Culture** (`culture.py`): biological-first. It switches on only after organisms have lived ≥1000 world-yr, biomass ≥ `tech_threshold`, and diversity ≥2. Tech then grows logistically at `tech_rate` and shows up in `BioSignFlux.tech`. It is on by default; pass `--no-culture` to disable it.
- **Cosmology stub** (`cosmology.py`): flat ΛCDM a(t), H(a), Hubble flow, luminosity distance, lookback time, and `CosmoDrive`. It is not wired into sim2.
- **Strong-field stub** (`strong_field.py`): r_s, Kerr ISCO, TDE radius, γ(r) static and orbital, 1PN precession, and `newtonian_limit_ok()`. It is not a GR solver.
- **Life training** (`learn_life.py`): numpy SPSA over normalised LIFE_SPACE. The loss is |habitability| + |eco stability| + 2·planted-seed recovery after an 80% disturbance. QrnLOpt was skipped because it is too heavy for this inner loop.
- **System hook** (`run_life_from_system.py`): make_system → short `leapfrog_tidal` orbit integration (optional toy tide) → life_loop. Optional `--host-mass` / `--dm-frac` hints. The event training scripts are untouched.

Run:

    PYTHONPATH=/workspace/NIM python -m universe.run_life --seed 4 --years 20000 --dt 20 [--distance-pc 10 --no-culture --params results/learned_life.json]
    PYTHONPATH=/workspace/NIM python -m universe.run_life_from_system --seed 22 --years 20000 --dt 20 --orbit-yr 5 [--tide 1e-6 --host-mass .8]
    PYTHONPATH=/workspace/NIM python -m universe.learn_life --iters 6 --years 600 --seeds 3   # -> results/learned_life.json + .log
    PYTHONPATH=/workspace/NIM python -m pytest -q universe/leaves/

Still stubs: no spectra or radiative transfer, no stellar evolution track (dM is wind-only), no GR dynamics, cosmology isn't coupled to the sim, and culture is a single logistic marker.
Backup from before the leaves: `/workspace/backups/universe_pre_leaves_20261025_163027{.tar.gz,/}`.

## Life surface viewer (see the organisms, not just charts)
```
PYTHONPATH=/workspace/NIM python -m universe.viewer_life --seed 4        # -> /workspace/nim_universe/out/viewer_life_seed4.html
PYTHONPATH=/workspace/NIM python -m universe.viewer_life_shots --seed 4 [--tour 84]   # PNGs life_surface_seed4_*.png (+ life_creatures_seed4.mp4)
```
Living planet uses the same FS_ROCKY/FS_CLOUD/FS_ATM shaders as the gallery closeups; creatures are low-poly meshes from `universe/leaves/creature_look.py`:
mobility -> body plan (swimmer/crawler/flyer/floater) + speed + limb count; metabolism_eff -> size & glow strength; reproduction_rate -> segments, smaller bodies;
hardiness -> shell + spines; biosign pigments -> hue/saturation + bioluminescence; biosign tech -> lit structures. Final cohorts are not stored in the life report,
so 8 cohorts are synthesised around the recorded final trait means (spread = mutation_sd), deterministic per seed. Click a creature for its traits.

## Individuals + evolution (leaves/individuals.py)
Organisms are now **literal individuals** (default `model="individuals"` in `life_loop.run_habitat`; legacy cohorts via `model="cohorts"`).
Each has `id`, `parent_id`, `birth_t_w`, `age`, `energy`, `traits`, `generation`, `species_id`; storage is numpy struct-of-arrays, stepped
in ≤1-yr substeps. Per year: food (niche resources × `pop_cap`) is split by competitive weight (√need × mobility), basal burn
depends on metabolism/hardiness/mobility (hardiness and mobility are costly), hazard = base + `sel_strength`·max(0, climate stress − hardiness)
+ starvation; impacts/UV are one-off shocks that hardiness halves; famine (no food) → dormancy (lower burn and starvation hazard).
Reproduction is Poisson with rate = reproduction_rate × energy-surplus fitness, energy-paid; children inherit the parent's traits +
gaussian mutation (`mutation_sd × evolution_rate`, 2% of births ×5 jumps), clamped to LIFE_SPACE. Over `pop_cap` the weakest
(energy/need) are culled. Speciation-lite: a child farther than `speciation_dist` (normalised trait space) from its species founder starts a new species_id.
New LIFE_SPACE knobs: `evolution_rate`, `pop_cap`, `sel_strength`, `speciation_dist`.
Report per living planet: `individuals` (all living), `evolution.{evolution_series, deceased_sample (reservoir 200), founding_traits, final_traits, total_births, species_ever}`.
`viewer_life` builds one mesh per saved individual (≤64 shown, real ids, click shows id/parent/species/generation); the synthesised-herd path is only used for old cohort reports.

**Rough-edge fixes (sexual repro, positions, speciation, drift, impacts).**
- *Positions*: every individual has a unit vector on the sphere (`lat`/`lon` in the report), brownian-walked each year with step ∝ mobility; births land near the parent. `viewer_life`/`life.js` place meshes at the simulated lat/lon (small local wander only); herd placement is kept only for old reports.
- *Sexual reproduction*: with probability `sex_rate` (default .3) a mate is picked (best of 8 random candidates: same species, nearest); child traits = per-trait random blend + mutation; `parent2_id` recorded.
- *Speciation*: children inherit the parent's species; every 200 yr the living set is re-clustered (average linkage on trait distance + 0.5 × great-circle distance/π, cut at 0.75 × `speciation_dist`). Clusters keep the plurality previous id; a new id is minted only for a daughter cluster with ≥10% of the living, and ids that drift together merge. Poles are harsher than the equator (`LAT_STRESS`), so local adaptation can split species. Reported as `n_species_alive` / `n_species_ever`.
- *Mobility*: foraging bonus saturates (`mob/(mob+0.3)`) and only pays in crowded cells; the metabolic cost is convex in mobility, so mobility no longer runs away to ~0.95.
- *Nonstationarity*: `climate_drift` adds organism-felt stress cycles (8.3 and 21 kyr) plus rare ±flips, and a ~13 kyr productivity cycle, so selection on hardiness keeps moving.
- *Impacts*: kill chance scales with local crowding and is halved by hardiness; shelter draws (10% + 30% × ocean cover) survive outright, and a floor of max(3, `impact_refuge` × n) survivors holds unless the climate is already lethal. Shocks are logged in `evolution.impact_log`.
New knobs: `sex_rate`, `climate_drift`, `impact_refuge`.

    PYTHONPATH=/workspace/NIM python -m universe.run_life --seed 4 --years 40000 --dt 20 --out results/life_demo_seed4.json
    PYTHONPATH=/workspace/NIM python -m universe.plot_evolution        # -> results/life_evolution_traits_seed4.png
    PYTHONPATH=/workspace/NIM python -m universe.viewer_life --seed 4 --life results/life_demo_seed4.json && python -m universe.viewer_life_shots --seed 4 --tour 96
    PYTHONPATH=/workspace/NIM python -m pytest universe/leaves/          # incl. test_evolution.py
