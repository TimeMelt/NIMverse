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
- **Cosmology** (`cosmology.py`): flat ΛCDM a(t), H(a), Hubble flow, comoving/luminosity distance, lookback time, z(d_C) inversion, and `CosmoDrive`. Wired into radiation observers (see item 6 below); it does not drive sim2 time.
- **Strong-field** (`strong_field.py`): r_s, Kerr ISCO, TDE radius, γ(r) static and orbital, 1PN precession, and `newtonian_limit_ok()`. Wired into `events.py` TDE/AGN log fidelity (item 6). It is not a GR solver.
- **Life training** (`learn_life.py`): numpy SPSA over normalised LIFE_SPACE. The loss is |habitability| + |eco stability| + 2·planted-seed recovery after an 80% disturbance. QrnLOpt was skipped because it is too heavy for this inner loop.
- **System hook** (`run_life_from_system.py`): make_system → short `leapfrog_tidal` orbit integration (optional toy tide) → life_loop. Optional `--host-mass` / `--dm-frac` hints. The event training scripts are untouched.

Run:

    PYTHONPATH=/workspace/NIM python -m universe.run_life --seed 4 --years 20000 --dt 20 [--distance-pc 10 --no-culture --params results/learned_life.json]
    PYTHONPATH=/workspace/NIM python -m universe.run_life_from_system --seed 22 --years 20000 --dt 20 --orbit-yr 5 [--tide 1e-6 --host-mass .8]
    PYTHONPATH=/workspace/NIM python -m universe.learn_life --iters 6 --years 600 --seeds 3   # -> results/learned_life.json + .log
    PYTHONPATH=/workspace/NIM python -m pytest -q universe/leaves/

Still stubs: no radiative transfer (host spectra are toy blackbody bands only), stellar evolution is a toy MS + subgiant/RGB track (not MESA; no He flash, HB or white dwarf), no GR dynamics (strong-field enters only as event-log fidelity), and cosmology couples to observers/life reports but not to N-body time. Culture is the v3 model (resources, pollution, polities, energy mix; see below), not a logistic marker.
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

`universe/leaves/stellar.py` adds a toy main-sequence track, extended below with a toy post-MS and blackbody bands. `HostWeather.step` now calls `evolve()` to age the star first, then adds Poisson flares on top of the evolved baseline (`L_now = L_star(t)·(1+flare)`, `uv_now` from the evolved `uv_proxy`).

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
- **Effect (1 Msun, a=1, no drift, 4 kyr, `stellar_evolve=1, stellar_age_rate=1e6`):** age goes 4.6→8.6 Gyr, L +54 %, Teff 5800→6056 K, uv_proxy 1.44 (blackbody NUV band; was 1.19 with the old (T/5800)⁴), flare_rate .05→.037, HZ 0.95–1.67 → 1.18–2.07 AU, so `hz_drift` becomes True. Planet T climbs from a first-quarter mean of 290.7 K to a last-quarter mean of 312.4 K; the control stays flat at 287.8 K.
- **Defaults:** `stellar_evolve=0` keeps the star fixed. At the real rate (`evolve=1, rate=1`), 40 kyr changes L by only ~1e-5, but even that perturbation chaotically shifts the seed-4 trajectory past a borderline metabolism-std assertion. With the default of 0, seed-4 is bit-identical to before: life on planet2, min pop 117, 2nd-half metabolism mean 0.53, no extinction. The seed-4 host (1.45 Msun, t_MS 3.9 Gyr) now takes its age from the system prior (0.17 Gyr, `age_source="system"`; it was the 3.53 Gyr knob cap). With evolve=0 that changes only report fields.
- Tests: `test_stellar_ms_track_unit`, `test_hostweather_stellar_evolve_and_fixed`, `test_stellar_brightening_warms_climate`.

### Stellar stubs closed

The four leftovers from the pass above are now covered (all toy-level):
- **Post-MS toy track (`stellar.py`):** for x = age/t_MS > 1, `L = L_MS_end·(1 + 8(x−1)^1.5)` and `Teff ∝ (L/L_MS_end)^-0.15`, so R swells. `stellar.phase()` returns `ms` / `subgiant` (x < 1.15) / `rgb`. Activity drops a further `1/(1+10(x−1))`. During `evolve()`, age is clamped at `1.4·t_MS` (L ≈ 3× the MS-end value), so the track stays finite. The start age is still capped at 0.9·t_MS, because `from_star` L/Teff are MS values. The host summary gains `phase` and `post_ms`. For example, 1 Msun at 12 Gyr gives L 3.3, Teff 5700 K, R 1.85.
- **Blackbody bands:** `stellar.band_fractions(T)` uses Planck integrals over uvc 100–280, uvb 280–315, uva 315–400, nuv 200–400, optical 400–700 and ir >700 nm. `uv_surface_rel(T)` is the surface NUV flux relative to a 5772 K blackbody (exactly 1 at the Sun), and `uv_bands(T, L)` returns a snapshot. `HostWeather.uv_bands` is set at `from_star` and refreshed while evolving. When evolving, `uv_proxy = uv0·uv_surface_rel(T)/uv_surface_rel(T0)`, which is continuous with the from_star value, so evolve=0 stays bit-identical. Chemistry and flares consume it unchanged. Radiation observers stay scalar (no RT).
- **Live habitat snapshot:** `run_habitat` deep-copies the habitat and calls `life_loop.sync_habitat(h, hw)` every step, which sets `L_star`, `hz_in`, `hz_out` and `irradiance = L_star/a²` (baseline, flares excluded) from the host. `a` and the initial `in_hz`, the reference for `hz_drift`, stay fixed. The caller's habitat object is never mutated.
- **System age prior:** `stellar.age_prior(M, seed)` is a deterministic log-uniform draw in [0.1, min(0.9·t_MS, 12)] Gyr, so massive stars come out younger. `life_loop.host_weather(S, seed=...)` stores it in `hw.age_prior_Gyr`, or uses `S["age_Gyr"]` if the system carries one. Precedence at the first `evolve()`: a non-default LP `stellar_age_Gyr` wins (`age_source="knob"`). Otherwise the system prior is used (`age_source="system"`). Bare `HostWeather.from_star` has no prior and uses the knob (4.6). The `life_loop` report has `stellar_age={prior_Gyr, source, age0_Gyr}`.
- Tests: `test_stellar_post_ms_track`, `test_blackbody_uv_bands`, `test_live_habitat_hz_sync`, `test_system_age_prior`. The seed-4 40 kyr demo was rerun: every life/biosphere/radiation field is identical; only host/habitat report fields changed.
- Still out of scope: full radiative transfer, real MESA tracks (He flash, HB, AGB, WD), cosmology coupling, GR (culture depth: see culture v3).

## Biosignatures with spectra (improvement item 3)

`universe/leaves/biosign.py` turns `BioSignFlux` scalars into spectral features. This is a toy, physically directed model with no radiative transfer. `BioSignFlux` keeps `o2, ch4, pigments, tech` and adds `h2o`, set from climate as `.5·liquid_water_frac·sigmoid((T−260)/12)`, plus `lines`.

| line | species | rest λ (nm) | kind | weight |
|---|---|---|---|---|
| o2_760 | O₂ A-band | 760 | absorption | 1.0 |
| o2_690 | O₂ B-band | 690 | absorption | 0.4 |
| ch4_1650 | CH₄ | 1650 | absorption | 1.0 |
| ch4_2300 | CH₄ | 2300 | absorption | 0.7 |
| h2o_1400 | H₂O | 1400 | absorption | 1.0 |
| h2o_1900 | H₂O | 1900 | absorption | 0.8 |
| redge_720 | pigments (red edge) | 720 | continuum step | 1.0 |
| tech_589 | tech (Na-lamp city lights) | 589 | emission | 1.0 |
| tech_10um | tech (waste-heat IR excess) | 10000 | IR excess | 0.6 |

- **Producers:** `biosphere.step` sets `h2o` and `lines` (`strength = scalar·weight`). `life_loop` rebuilds `lines` after culture sets `tech`.
- **Observers:** `ObserverCatalog.emit` dims the lines exactly as it dims the scalars (`1/d²·(1+z)⁻²·aperture`, capped at 1) and records `obs_nm = rest_nm·(1+z)`. `poll` detects a line when `strength ≥ thresh[species]·weight`, which gives parity with the scalar test. Line names are appended to `detected`, and each log entry carries `lines=[{name, rest_nm, obs_nm, strength}]`. `summary()` adds `lines_detected`, and `first[planet:line]` includes `z` and `obs_nm`. Legacy scalar keys are unchanged, and `h2o` (thresh .02) is a new scalar key.
- **Seed 4 @ 40 kyr:** every life/biosphere field is identical (biosign only gains `h2o` and `lines`). Detections went from 385 to 400, because H₂O is visible from t=0, before life. Lines detected: o2_760, o2_690, h2o_1400, h2o_1900, redge_720, tech_589, tech_10um. CH₄ stays below threshold (9.4e-5 < 1e-4), as before.
- **Test:** `test_biosign_spectra` checks O₂ vs producers, H₂O vs wetness, tech lines, empty lines on a dead world, and obs_nm = 3·rest at z=2. It also runs an integration case with a z=2 observer, where o2_760 is detected at 2280 nm.
- **Leftovers:** no full RT, no line profiles or continuum shape, no instrument noise or S/N, no retrieval, no stellar-type contrast scaling (host `uv_bands` not folded in), and no false-positive (abiotic O₂) modelling.

## Culture ↔ biosphere coupling (improvement item 4)

`leaves/culture.py` v2: tech is no longer a free logistic marker — it eats, pollutes, warms and can kill.

- **Resource draw**: active tech demands `tech_draw × level` biomass/yr from producers (capped at 50% of P per step). `supply = taken/demand` scales logistic growth; `supply < .5` makes `level` contract (overshoot → stall/crash). `footprint = taken/P` trims the *next* step's `NicheOffer.resources` for organisms.
- **Pollution / climate**: a pollution stock relaxes toward `level × tech_pollute` (e-fold 500 yr, persists after collapse). Next step it is fed into `world.step` stress as `dT += tech_warm_K × pollution` (plus `tech_dT`, `tech_pollution` stressors), so `ClimateHistory` T/water carry the tech trend.
- **Extinction risk**: toxicity `tech_kill = .4 × max(0, pollution − .3) × dt/20` (≤ .5/step) is applied like `uv_kill` to biomass guilds and as an individual shock kill. Organisms crashing → culture decays (existing rule) → pollution relaxes.
- **Order**: biosphere → organisms → culture.step → draw applied to producers same step; warming/toxicity/niche footprint applied on the next step (one-step lag, documented in `life_loop`). `--no-culture` disables everything.
- **Knobs** (LIFE_SPACE): `tech_draw` (0–.5, default .001; .003 + warming tipped the seed-4 metabolism-stability test, chaotic sensitivity), `tech_pollute` (0–2, default .1), `tech_warm_K` (0–60, default 5). `Culture.to_dict` adds `pollution, draw, footprint, supply, warm_K, tech_kill`; run report `culture.peak`; record rows add `tech_dT, pollution`.
- **Seed 4, 40 kyr** (planet2, before c048813 → after): tech 1.0 → 1.0 (t_start 3640 both); tech warming 0 → +0.5 K (pollution .10, no toxicity); T_mean 263.62 → 264.07 K, T_end 267.01 → 267.51 K; min pop 117 → 119, end pop 202 → 206; metabolism mean (t>3 kyr) .549 → .562 (end .536 → .635); biomass 1.139 → 1.124; peak footprint .063; no extinction.
- **Test** `test_culture_biosphere_coupling`: draw lowers P, starved tech contracts; `tech_pollute=1, tech_warm_K=30` → T_max > control +5 K; `tech_pollute=2, tech_warm_K=60, tech_draw=.5` → toxicity, biomass crash, lower final tech.
- **Leftovers**: closed in v3 below.

## Culture leftovers closed (culture v3)

`leaves/culture.py` v3 — every former leftover now moves climate, chemistry, biosphere or culture state (toy, tested):

- **Polities / conflict**: active culture splits into 2–4 polities (`n = 2 + diversity mod 3`), Dirichlet shares from a private RNG seeded by `t_start` (main RNG stream untouched). `rivalry = 1 − Σshare²`; `conflict = .5 × tech_conflict × rivalry × stress` (stress = pollution > .2, supply < .7, net warming > 2 K) inflates draw and emissions by `(1+conflict)` and adds `.05 × conflict × dt/20` to `tech_kill` without adding tech. Planet-level `BioSignFlux.tech` stays the aggregate level (`Σ level_contrib = level`).
- **Energy mix**: `energy_mix = {fossil: tech_fossil_frac × (1 − mitigation), renewable: ⅔ rest, nuclear: ⅓ rest}`. Emission intensity fossil 1 / renewable .1 / nuclear .05, normalised to the reference .7-fossil mix (so defaults ≈ v2 pollution). Renewables raise draw (land: `×(1+.5·renewable)`), nuclear adds a small chronic kill (`2e-4 × nuclear × level`).
- **Aerosol cooling**: short-lived stock (e-fold 5 yr) of fossil burn; `aerosol_K = tech_aerosol_K × aerosols`; net `warm_K = greenhouse_K − aerosol_K` (can go negative) is what enters next-step `world.step` dT. Both terms in `stress()` / record rows.
- **Ocean acidification**: slow stock (e-fold 1500 yr) toward `min(1, tech_acid × pollution × fossil/.7)`; passed as stressor `tech_acid`; `biosphere.capacity` × `(1 − .6 × acid × water_frac)` (wet worlds hit hardest).
- **Tech → chemistry**: `culture.apply_chem` (called in `life_loop` right after `chemistry.step`, using last step's culture): organics `+= tech_chem·1e-4·level·(fossil+.2)·dt`; redox `+= tech_chem·1e-4·(pollution − 2·level·fossil)·dt` (fossil exhaust reducing, pollution oxidising; clipped ±2).
- **Mitigation / learning**: `crisis = max(pollution/.25, warm_K/3, acid/.3, tech_kill/.02)`; if > 1, `mitigation += tech_mitigate × min(crisis−1, 2) × dt` (≤1), else relaxes 2e-4/yr. Mitigation shifts the mix off fossil, multiplies emissions by `(1 − .5m)` and draw/growth by `(1 − .3m)` (opportunity cost).
- **Knobs** (LIFE_SPACE, defaults): `tech_fossil_frac` (0–1, .7), `tech_aerosol_K` (0–30, 2), `tech_acid` (0–5, 1), `tech_chem` (0–10, .1; 1 pinned seed-4 redox at −2), `tech_mitigate` (0–.01, 2e-4; 1e-3 defused the v2 extreme-toxicity test), `tech_conflict` (0–2, 1). `tech_draw` stays .001. `Culture.to_dict` adds `greenhouse_K, aerosol_K, aerosols, acidification, mitigation, conflict, energy_mix, polities[id, share, level_contrib]`; record rows add `aerosol_K, acid, mitigation`; planet card shows a "Tech climate" row. `--no-culture` disables all of it.
- **Seed 4, 40 kyr** (planet2, a740eab → v3): tech 1.0 → 1.0 (t_start 3640 both); pollution .10 → .10; net tech dT +0.50 → +0.31 K (GHG +.50, aerosol −.19); acidification 0 → .10; mitigation 0 → 0 (no crisis); mix fossil/renewable/nuclear .7/.2/.1; 2 polities (.25/.75), conflict 0; T_end 267.51 → 267.31 K; min pop 119 → 119, end pop 206 → 221; metabolism 2nd half mean .585 → .551 (σ .059 → .058); biomass 1.124 → 1.248; organics .82 → 1.10; redox −.53 → −1.00; no extinction.
- **Test** `test_culture_leftovers_closed`: 2–4 polities with shares/level summing correctly, stress → conflict → extra kill; fossil-only vs renewable-only → >3× pollution and acidification, more greenhouse, renewables draw more; aerosols lower net dT (negative under extreme aerosol); acidification lowers capacity and final producers in a full run; `apply_chem` changes organics + redox and full-run organics differ vs `tech_chem=0`; mitigation rises under crisis, cuts fossil share, pollution and acidification vs `tech_mitigate=0`, stays 0 without crisis. Existing coupling and seed-4 metabolism tests unchanged and passing.
- **Still out of scope**: full geopolitics/trade/diplomacy, detailed GCM aerosol/carbon-cycle chemistry (no real pH/carbonate system), real industrial LCA or sector economics.

## Species churn / niches (improvement item 5)
Seed 4 @ 40 kyr (`run_life --seed 4 --years 40000 --dt 20`, default knobs):

| metric | before (f5130de) | after |
|---|---|---|
| species ever | 96 | **51** |
| species alive (median / end / last-10 min / max) | 5 / 6 / 4 / 8 | 5 / 5 / 4 / 9 |
| min pop (after warm-up) | 119 | 116 |
| metabolism 2nd half (mean / std) | .551 / .058 | .514 / .071 |

What drives churn: alive stays ~5, so the ~100 ids came from turnover (clusters vanish and come back under a fresh id), not from too many species at once.
- `RECYCLE_R` 1.5 → **2.0**: a new cluster takes a recently vanished id (inside `RECENT_YR`) when its centroid is within 2 × cut. This is the main lever.
- `SPLIT_SEP = .8` (new): a daughter cluster only gets a new id if its centroid is ≥ .8 × cut from its parent cluster's centroid. Otherwise it merges back under the parent id (merge pressure, so near-twins don't count as a split).
- `SOFT_MAX = 10`, `SOFT_K = .5` (new): a soft cap. Above 10 living ids, the daughter-size bar `NEW_SPECIES_FRAC` goes up by 50% for each extra id. There's no hard ceiling, so allopatry can still found species. It doesn't trigger on seed 4; it's a guard only.
- Unchanged: `SPECIES_EVERY`, `CL_CUT`, `NEW_SPECIES_FRAC`, `NICHE_ALPHA`, niche grid, `speciation_dist`.

Sweep notes: raising `NICHE_ALPHA` to 1.1–1.2 *raises* species ever (114–128) because rare niches survive and split more. Raising `NEW_SPECIES_FRAC` to ≥ .08 or `SPECIES_EVERY` to 1000 drops alive to 1–3, and a soft cap at ≤ 6 collapses to a single species. So I kept niche crowding as it was and went through id hysteresis and recycling.
- Test: `test_cold_spell_floor_species_churn_metabolism_seed4` now asserts species ever ≤ 70 (was ≤ 120), median alive 3–12, and ≥ 3 alive over the last 10 samples. Full suite: 48 passed.
- Leftovers: closed below.

## Species leftovers closed
All 40 kyr runs use `run_life --seed S --years 40000 --dt 20` with default knobs.
- **Mint-rate limiter (new):** `MINT_TAU = 700`, `MINT_CAP = 3`. New-id tokens refill at 1 per 700 yr, and the bucket holds at most 3. When no token is left, a daughter cluster takes the nearest recently vanished id within `RECYCLE_FAR` (2) × `RECYCLE_R` × cut. If there is none, it merges back under its parent id. The recycle reach also widens smoothly as tokens run out (`SMOOTH_R`). Over 40 kyr this puts a soft ceiling of about 60 on species ever, independent of the recycle radius. `RECYCLE_R` goes 2.0 → **1.5**, because the limiter now does most of the work.
- **Finer niche grid:** `niche_crowding` now has 4 mobility bands × 4 hardiness bands × 3 climate belts (`NB_MOB`, `NB_HARD`, `NB_BELT`; was 3×3×3). An optional continuous Gaussian-kernel mode (`NICHE_KERNEL > 0`) is available but off by default.

Sensitivity on seed 4 (species ever):

| RECYCLE_R | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|
| before (d317c45) | – | 78 | **51** | 38 | 14 | 14 |
| after | 58 | **45** | 39 | 25 | – | 18 |

Around the default (±0.5), the spread drops from 78→38 (40 ids) to 58→39 (19 ids). Across all knob values, alive median stays at 5 and min pop at 110–117. Large radii still lower species ever, which is expected: more recycling means fewer ids. The ceiling stops the low-radius side from blowing up.

Multi-seed, 40 kyr, defaults:

| seed | ever before | ever after | alive median | last-10 min | max alive | min pop |
|---|---|---|---|---|---|---|
| 4 | 51 | **45** | 5 | 4 | 9 | 114 |
| 3 | 90 | 56 | 5 | 4 | 10 | 59 |
| 6 | 94 | 53 | 5 | 2 | 9 | 46 |
| 7 | 87 | 58 | 5 | 3 | 11 | 71 |
| 8 | 47 | 46 | 5 | 3 | 10 | 129 |
| 12 | 21 | 57 | 5 | 2 | 11 | 96 |
| 13 | 36 | 56 | 5 | 4 | 11 | 79 |

On every seed, species ever falls between 45 and 58, where it used to range from 21 to 94. No seed collapses to a single living species, and none goes extinct. Seed 12 rises from 21, because the old setting suppressed splits there. The desert seeds (3, 6, 7) have a pre-existing low point of about 46–60 organisms, unchanged by this pass.

Niche-grid outcome on seed 4 (with the limiter on): 3×3 gives 52, **4×4 gives 45**, 5×5 gives 52 (min pop 108), 4×4 with `NICHE_ALPHA` .6 gives 47, 4×4 with 4 belts gives 48, and the kernel at σ .12 or .2 gives 51. Across the six other seeds, 4×4 is neutral to slightly better than 3×3 (last-10 min alive is higher on 3 and 13). The finer grid no longer raises churn, because the limiter absorbs the extra rare-niche splits. Shipped 4×4.

Tests: the seed-4 test bounds are unchanged (ever ≤ 70, median alive 3–12, metabolism band). The new `test_species_churn_multiseed_light` runs seeds 8 and 12 at 15 kyr as a cheaper proxy (16 and 23 species ever) and asserts ever ≤ 45, median alive ≥ 3, last-10 alive ≥ 2 and min pop ≥ 50. Full suite: 49 passed.

### Cosmology + strong-field wiring (improvement item 6)

**Cosmology → observers** (`leaves/radiation.py`)
- `Observer.from_z(z)` builds an observer from redshift alone: `distance_pc` = comoving distance, so dimming is `(10 pc/d_L)²` with `d_L = (1+z) d_C` (`radiation.dimming`). That is the same formula as before, now read consistently as comoving distance.
- Delay is the lookback time whenever `z` is set **or** `distance_pc > 10 Mpc` (z then comes from `cosmology.z_of_comoving`). Below 10 Mpc the delay is still light time, and between 100 kpc and 10 Mpc the toy `z_hubble` is unchanged.
- `ObserverCatalog.summary()["cosmology"]` gives `{observer: {age_gyr, a, H, z, d_C_mpc, d_L_mpc, d_L_obs_mpc, lookback_gyr}}` (from `CosmoDrive.at_z(z).snapshot()`) for observers with z set or beyond 100 kpc. The default 10 pc observer gives `{}`, so the seed-4 life report is bit-identical; `results/life_demo_seed4.json` was refreshed and only gained `radiation.cosmology = {}`.

**Strong-field → TDE/AGN** (`events.py`: `bh_fidelity`, `agn_fid`)
- Unit mapping: sim mass unit = 1e8 M☉, so SMBH M☉ = `m × MASS_UNIT_MSUN`. Stellar BHs use their physical remnant mass `0.3 × smass`. The disrupted star has `m_* = smass` M☉ and `R_* = R☉ m^0.8`. All radii are in AU. The kpc `tde_r` capture radius only selects candidates.
- Each `tidal disruption` entry gets `M_bh_msun, r_s, r_tde, r_isco, visible (r_tde > r_s), swallowed, gamma (circular-orbit γ at max(r_tde, r_isco)), eta`. Each `quasar ignition` entry gets `M_bh_msun, r_s, r_isco, eta (thin-disk 1−E_isco, 0.057 for χ=0), L_edd, gamma_isco`.
- The gate is set by `EP["sf_gate"]` or `C["sf_gate"]`:
  - `"tag"` (default): fields are logged, but types, AGN fuel and scores don't change.
  - `"strict"`/`True`: R_tde < r_s is logged as `swallowed whole`, which is not luminous and does not fuel an AGN.
  - `"off"`/`False`: legacy log entries with no fields.
- Dynamics are identical in every mode. Strict is opt-in because default SMBHs (~3e8–1e9 M☉) sit above the Hills mass for the sub-solar star particles, so nearly every TDE gets swallowed. Smoke run (seed 7, 400 stars/galaxy, 300 steps):

| EP | gate | TDE / swallowed | events score | agn |
|---|---|---|---|---|
| defaults | off = tag | 34 / 0 | 0.727 | 0.000 |
| defaults | strict | 0 / 34 | 0.708 | 0.000 |
| learned_alt2 | off = tag | 4 / 0 | 0.796 | 0.779 |
| learned_alt2 | strict | 0 / 4 | 0.738 | 0.368 |

Tests: `test_cosmo_strong_field_wiring` in `leaves/test_deferred_leaves.py`.

Remaining limits: no GR integrator (geodesics, Kerr orbits, GW inspiral), no perturbation cosmology or lensing, `CosmoDrive` does not drive sim2 leapfrog time, and no spin evolution (χ=0 by default).
