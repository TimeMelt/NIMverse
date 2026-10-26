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

**Imperfection fixes (trade-offs, niches, planetary drift).**
- *Metabolism trade-off (rate vs yield)*: `uptake = 1.15 − 0.2·met_norm` scales both the competitive food share / intake cap and the brood rate, and efficient metabolisms lose a little stress reserve (`hardiness_eff = h·(1 − .05·met)`). Efficiency no longer pins at the 0.9 ceiling.
- *Reproduction trade-off*: breeding machinery raises basal need (`× (1 + .3·rep_norm)`), and juveniles (<15 yr) of fast breeders face extra hazard `0.1·rep_norm²` that grows under climate stress.
- *Niches*: competition is frequency-dependent: food share ÷ (phenotype-niche crowding)^0.8 over mobility band (sessile/forager/roamer) × hardiness band × climate belt (equatorial/temperate/polar), so rare niches pay and several species coexist. Clustering cut lowered to 0.6 × `speciation_dist`, min cluster 4, a daughter needs ≥4% of the living, and recently vanished ids are re-used by a matching cluster (less id churn).
- *Planetary climate drift*: `world.ClimateDrift` (same 8.3/21 kyr cycles + flips, own rng stream) feeds `dT = −35 K × offset` into `world.step`, so T_K, water fraction, climate class and chronic stress all move; organisms read the offset from `climate.stressors` (no double count). Impact-winter dT counts as the one-off shock, not chronic stress; water weight in chronic stress is 0.6 (ocean refugia). `climate_history.series` and per-series `T_K` are reported; `plot_evolution` adds a planet-T vs hardiness panel.
- seed 4, 40 kyr: end means metabolism .55 (was .87, cap .9), reproduction .21 (was .41, cap .5), mobility .49, hardiness .62; 15 species alive at the end (median 11, 99% of samples in 3–15), 285 ever; planet T 242–272 K (σ 3.7 K; σ ≈ 0 with drift=0).

    PYTHONPATH=/workspace/NIM python -m universe.run_life --seed 4 --years 40000 --dt 20 --out results/life_demo_seed4.json
    PYTHONPATH=/workspace/NIM python -m universe.plot_evolution        # -> results/life_evolution_traits_seed4.png
    PYTHONPATH=/workspace/NIM python -m universe.viewer_life --seed 4 --life results/life_demo_seed4.json && python -m universe.viewer_life_shots --seed 4 --tour 96
    PYTHONPATH=/workspace/NIM python -m pytest universe/leaves/          # incl. test_evolution.py

## Imperfection fixes 2 (seed 4, 40k yr)

| metric | before (15d5812) | after |
|---|---|---|
| min living pop (after founding) | 37 | 110 (no extinction; drift .8/1.0 min ≈106/95) |
| species ever / alive (median, end) | 285 / 11, 15 | 84 / 5, 6 |
| metabolism mean ± std | .68 ± .118 (2nd half .50–.84, boom-bust) | .39 ± .055 (2nd half .23–.52) |

- Cold spells: refugia (hazard × clip(n/.5cap,.2,1)), non-heritable acclimation (≤.15, τ 600 yr), saturating stress hazard, torpor; impact-winter freeze no longer counts as chronic water stress.
- Species churn: reclustering every 500 yr, new-species threshold .06, hysteresis, min 4 clusters.
- Metabolism: food storage low-pass (τ 5000 yr) on resources × food_mult; convex rate–yield trade-off (interior optimum, no bang-bang).
- Viewer: individuals record a 24-point movement trail (every 10 yr, inherited from parent); `life.js` replays it over ~10 s with a fading line, then wanders; jitter fallback when no trail.
- Tests: `test_cold_spell_floor_species_churn_metabolism_seed4`, `test_strong_drift_no_extinction`, `test_food_lag_bounds_metabolism_oscillation`, `test_cold_refuge_and_acclimation_unit`.

## Imperfection fixes 3 (seed 4, 40k yr)

| metric | before (34708f7) | after |
|---|---|---|
| metabolism mean ± std (range .1–.9) | .39 ± .055 (std 1st/2nd half .049/.060) | .53 ± .068 (1st/2nd half std .068/.068; 2nd half .34–.66) |
| reproduction (2nd-half mean, cap .5) | .18 | .18 |
| min living pop / species ever / alive (median, end) | 110 / 84 / 5, 6 | 126 / 102 / 5, 7 |

- Metabolism: rate–yield curvature `UPT2` .3 → .22 moves the interior optimum to mid-range (still convex, off ceiling).
- RNG: organisms draw from their own stream (`life_loop`: `default_rng([a, 23])`), so founder count / organism draws never shift impact or abiogenesis timing (test `test_organism_rng_stream_isolated`). Seed-robustness test: 4 rng seeds × drift 1.0, 18 kyr, no pop < 80, 2nd-half metabolism means within .15.
  Sweep of 8 rng seeds × drift .5/1.0 (24 kyr, cold niche): metabolism .45–.53, min pop 139–186, no extinction.
- Cold floor: polymer chemistry works down to 240 K (eutectic brines; `chemistry.T_BRINE`, was 250 K) → cold ocean worlds like seed 7 become abiogenesis-ready.
- Seed scan (40 kyr, dt 20, defaults): life on 4, 6, 8 of seeds 0–9 and 12 of 10–19 (4/20). The rest have no candidate wet world: frozen (no liquid water, 190–245 K: seeds 1, 5, 10, 11, 15), dry deserts (water < .1 or organics below threshold) or gas giants only; seed 7's 247 K ocean is now ready but p≈5e-4/step did not fire in 40 kyr.
- `viewer_life_shots` copies all stills (01–06) into `results/`.

## Planet gen + abiogenesis (imperfection fixes 4)

Seed scan 0–19 (40 kyr, dt 20, defaults): **4/20 → 12/20** with life.

| | seeds with life |
|---|---|
| before (baafe50) | 4, 6, 8, 12 |
| after | 3, 4, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19 (new: 3, 7, 13, 14, 15, 16, 17, 19; 6, 7, 8, 12 now have two living worlds) |

- Deserts: `system_habitat.VOLATILES["desert"]` .1 → .35 (groundwater / playa lakes; draw .25–.45) so inner-edge deserts reach w ≈ .3–.65 and enough C/N for organics. `bodies.classify` and gallery looks unchanged.
- Candidate margin .25 → .35 (rocky worlds at ≥ .65·hz_in, typical hot-desert orbits at .66–.71·hz_in, 320–345 K, below runaway).
- `organic_thresh` default .275 → .18 (sparse-volatile deserts top out at organics ≈ .8·volatiles).
- Cold: `snowball_K` default 245 → 238; polymer chemistry tests T_BRINE (240 K, unchanged) against the warm-season temperature `T + season_K` (equatorial/summer brines), mean T still bounds the hot side.
- Abiogenesis: `abio_p` 2e-4 → 4e-4/yr, scaled by `min(1, 2·w)` (was `w`): a w = .1 ready world fires within 40 kyr with p ≈ .96. Not-ready → never (unchanged).
- Leftovers: 1, 5, 16(outer), 17(outer) ice worlds are 187–199 K (true snowballs, kept sterile); 10 (238 K ocean) and 11 (232 K terran) stay below brine water thresholds; 0, 2, 9, 18 have no rocky world within the widened HZ margin (gas giants / far ice).
- seed 4 @ 40 kyr: life (abiogenesis t = 1480 yr, was 4780), min pop 117 (was 126), metabolism 2nd half .53 (was .53), no extinction; 7 species alive at end.
- Tests: `test_desert_near_hz_reaches_polymer_ready`, `test_abiogenesis_fires_when_ready_40kyr` (200 rng seeds, ≥90%), `test_multiseed_ready_proxy` (≥10/20 seeds have a polymer+vesicle-ready candidate; ice worlds stay sterile).

## Stellar evolution + HostWeather depth

`universe/leaves/stellar.py` adds a toy main-sequence track (no post-MS, no spectra). `HostWeather.step` now calls `evolve()` to age the star first, then adds Poisson flares on top of the evolved baseline (`L_now = L_star(t)·(1+flare)`, `uv_now` from the evolved `uv_proxy`).

- **Track:** `t_MS = 10 Gyr·M^-2.5`. With fractional age `x = age/t_MS`, `L ∝ 1/(1+0.4(1−x/0.457))`, which is the Gough faint-young-Sun law. Under it the Sun starts at ZAMS L≈0.7 and brightens ~8–9 %/Gyr today, while M dwarfs stay almost flat (0.3 Msun gains <2 % in 3.4 Gyr) and heavier stars evolve faster. `Teff ∝ L^0.1`, `R` comes from `L = R²Teff⁴`, and `uv_proxy = (Teff/5800)⁴`. The `(L, Teff, flare_rate)` passed to `from_star` is the star's state at the initial age, and the track only rescales it.
- **Activity:** for M ≥ 0.6, `flare_rate ∝ age^-1/2` (Skumanich). Fully convective M < 0.6 stars stay saturated (flat rate) for `8 Gyr·(0.6/M)`.
- **Wind:** `dM_star` adds up over stellar time (`mdot·dt·max(1, evolve·age_rate)`), and the track reads `M_star+dM_star`.
- **Knobs (LIFE_SPACE):**
  - `stellar_age_Gyr` (0.05–12, default 4.6): initial age, capped at 0.9·t_MS so the star is still on the MS.
  - `stellar_evolve` (0–1, default **0**): scale on dL/dt.
  - `stellar_age_rate` (1–1e6, default 1): stellar years per world year. For example, 1e6 makes a 4 kyr run cover 4 Gyr.
- **HZ / climate coupling:** HostWeather keeps `hz_in_now/hz_out_now = 0.95/1.67·√L_star`. `world.step` already uses `L_now` for teq, and its stressors now also report `L_host`, `irradiance_now`, `hz_in_now`, `hz_out_now`, `in_hz_now` and `hz_drift`. `hz_drift` is True once the planet (orbit `a` fixed) leaves or enters the moving HZ relative to the snapshot `in_hz`.
- **Host summary fields:**
  - Before: M, L, flare_rate, n_flares, dM, uv_mean/max, flare_steps, hist_tail, distance.
  - Now also: `T_eff, R_star, uv_proxy, age_Gyr, age0_Gyr, t_ms_Gyr, L0, dL_frac, hz_in_now, hz_out_now, L_track` ([t_w, age, L, Teff], ≤~65 points).
- **Effect (1 Msun, a=1, no drift, 4 kyr, `stellar_evolve=1, stellar_age_rate=1e6`):** age goes 4.6→8.6 Gyr, L +54 %, Teff 5800→6056 K, uv_proxy 1.19, flare_rate .05→.037, HZ 0.95–1.67 → 1.18–2.07 AU, so `hz_drift` becomes True. Planet T climbs from a first-quarter mean of 290.7 K to a last-quarter mean of 312.4 K; the control stays flat at 287.8 K.
- **Defaults:** `stellar_evolve=0` keeps the star fixed. At the real rate (`evolve=1, rate=1`), 40 kyr changes L by only ~1e-5, but even that perturbation chaotically shifts the seed-4 trajectory past a borderline metabolism-std assertion. With the default of 0, seed-4 is bit-identical to before: life on planet2, min pop 117, 2nd-half metabolism mean 0.53, no extinction. The seed-4 host (1.45 Msun, t_MS 3.9 Gyr) is anchored at age 3.53 Gyr.
- Tests: `test_stellar_ms_track_unit`, `test_hostweather_stellar_evolve_and_fixed`, `test_stellar_brightening_warms_climate`.
- Leftovers: no post-MS (subgiant/RGB), no spectra/blackbody UV bands, the habitat snapshot `hz_in/hz_out/irradiance` stays fixed (the live values are in the stressors), and there is no MS-age prior from the system generator.
