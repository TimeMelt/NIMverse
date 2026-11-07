# Live browser universe — final verification

Verified locally on November 7, 2026 (America/New_York / EST). This is an
experimental, completely offline static browser build, not a validated physical
cosmology or a quantum-hardware experiment. Nothing was pushed or published.

## Reproduce

From `/workspace/nim-universe`:

```sh
PYTHONPATH=.:/workspace/NIM /workspace/nimenv/bin/python -m universe.export_itch_live
# Serve /workspace/itch statically on port 8874 in another terminal.
PYTHONPATH=/workspace/itch/python-tools /workspace/nimenv/bin/python web/live/verify_gate.py
PYTHONPATH=/workspace/itch/python-tools /workspace/nimenv/bin/python web/live/verify_browser.py
PYTHONPATH=/workspace/itch/python-tools /workspace/nimenv/bin/python web/live/verify_perf.py
node --test web/live/test_*.js
```

The browser scripts use locally installed Playwright/Chromium. Performance values
below are measurements, not universal promises. They use headless Chromium with
software SwiftShader WebGL; no hardware-GPU result is claimed.

## Stages

1. **Feasibility:** official Pyodide 314.0.7 module worker, NumPy 2.4.6,
   SciPy 1.18.0, and REBOUND 4.4.7 all run offline inside a statically served
   sandboxed iframe. Real seed-4 planet2 leaf simulation advances, including
   24-band climate, carbon/CNP and foodweb. REBOUND smoke integration succeeds.
   No native-JS life fallback was needed.
2. **Live galaxy:** independent ID-keyed seeds; bounded Barnes–Hut/leapfrog
   history and event recipes; explicitly labeled offline-learned NIM preset.
   Three finite galaxies / 768 star tracers plus halo/SMBH bodies.
3. **Systems:** real raycast star click opens its mass/age/metallicity-coupled
   procedural system. Breadcrumbs, local pause/speed/seed/reset and units verified.
4. **Life and circuits:** active real Python world, bounded four-world cache,
   coarse inactive evolution, JS QWRK creature actions, procedural photoreal and
   creature views. Live bounded circuit-drive physical forcing verified.
5. **Acceptance:** focused native and JS tests, browser parity, deterministic
   generation, conservation sanity, rendering/throughput probes, iframe/input/
   fullscreen, no-network audit, screenshots and offline zip completed.

## Numerical and behavioral evidence

- 21 focused native pytest cases pass: physics15, physics_real, conserving/diverse
  foodweb, REBOUND Earth/Sun orbit, circuit structure and browser bridge/cache/
  forcing tests. The full historical native suite was not completed (a slow
  exploratory run was interrupted after 15 passing tests).
- 14 Node tests pass, covering circuit fixtures, generation, host coupling,
  deterministic exploration, bounded state, mass and gravity sanity.
- QWRK vs PennyLane fixture maximum error: `4.440892098500626e-16`.
- NIM16 vs PennyLane fixture maximum error: `2.426947531830592e-13`.
- Closed 100-step galaxy energy drift: `0.001219204587620774` (~0.122%);
  two-body energy drift below 1e-5; bounded event mass ledger tests pass.
  These are numerical sanity checks, not proof of global cosmological accuracy.
- Native Python vs WASM five-tick leaf run: temperature max difference
  `1.1368683772161603e-13 K`; biomass <=`1.776e-15`; organism counts, births
  and species match. Both use the explicit browser dependency adapter.
- Real live population: 114 initially in the reference habitat; 30 in a second
  generated world. QWRK steering reports zero evaluation failures.
- Actual clicked star -> generated system -> world -> live surface -> creature;
  photoreal mode; second generated world; pause; quantum toggle; fullscreen;
  seed/reset regression all pass.
- Full browser run: zero page errors, zero console errors, zero HTTP failures,
  zero requests to external origins. Worker core, packages, shaders, source
  archive, fonts/materials and Three.js are bundled, with no CDN/backend.
- Tests run in nested `allow-scripts allow-same-origin` sandboxed iframes using
  an ordinary static HTTP server (no special COOP/COEP headers). The actual
  itch.io upload/player has not been tested or published.

## Performance and memory

Final full browser acceptance boot: **3716 ms**, initial WASM allocation
**78,512,128 bytes (~75 MiB)**. Detailed parity tick median **58.1 ms**
(range 54.7–62 ms).

Dedicated worker-clock probe:

| View | Detailed ticks/s | Typical Python service | Render FPS (software) |
| --- | ---: | ---: | ---: |
| Planet orbit | 7.466 | 52.7 ms | 12.7 |
| Photoreal surface | 7.454 | 56.5 ms | 28.3 |

Creature-closeup acceptance reported ~11 FPS. The 5–20 detailed tick/s target
is met in this probe; the 30–60 render FPS target is **not consistently met**.
The worker advances independently of rendering with bounded snapshot backpressure.

WASM allocation reached **94,240,768 bytes (~90 MiB)** in the throughput probe;
main-thread JS heap was ~68 MB. These are partial memory metrics, not whole-tab
or GPU memory. The 256–512 MB total-memory target remains unverified. Hardware
GPU rendering, Safari, mobile and long-duration memory behavior remain untested.

## Exact quantum-drive contract

These are deterministic seeded CPU statevector evaluations, not a quantum
computer and not established quantum gravity. Fixed policy weights are used;
there is no online browser training.

- NIM 16-wire structure circuit: initial galaxy radius and disk fraction ±4%.
  These initial conditions change on regeneration, not retroactively on toggle.
- Per-world QWRK readouts feed actual Python insolation/climate/chemistry:
  bolometric flux ±1%, UV ±5%, flare multiplier ±20%, and clamped secular
  eccentricity offset ±0.002.
- Event QWRK readouts: supernova kick amplitude ±15%; GRB/AGN probabilities ±20%.
- Individual QWRK actions steer movement, forage, mating and rest.
- Quantum-drive toggle disables future physics perturbations; independent
  creature steering remains active. Outputs and drive status are visible.

This is genuine circuit-driven input to classical solvers, with a deliberately
phenomenological, bounded coupling; not a claim of scientifically derived
quantum control of real astrophysical laws.

## Exact versus simplified / known issues

Real leaf modules run in Python/WASM: 24-band climate, chemistry/carbon/CNP,
foodweb, individuals, culture and SciPy clustering. Circuit gate/statevector math
matches independent PennyLane fixtures. Numerical gravity and event evolution
are live, never recorded frame playback.

Finite tracer galaxies, fitted formation/event/strong-field recipes and analytic
Kepler display orbits are simplified. REBOUND is bundled and smoke-tested but
is not the visible moon integrator. Near-HZ organisms are seeded, not spontaneous
abiogenesis. There is no browser MESA, ExoPlaSim, full radiative transfer,
LALSuite, numerical relativity, or online policy training. “Photoreal” denotes
procedural detailed materials, not physical image synthesis.

Worlds are generated lazily; FIFO cache holds at most four. Evicted worlds
regenerate and there is no disk save. Active/inactive/galaxy clocks are separately
accelerated, not one synchronized cosmic chronology. Galaxy supernovae do not
sterilize cached habitats. Render meshes are adaptively capped at 4–8 while the
full population remains simulated. Existing seed-4 reference mode is retained.

Runtime licensing is mixed: project/Three.js MIT, Pyodide MPL2, REBOUND GPL3+
(with corresponding source tar included), NumPy/SciPy BSD and CPython license.
See bundled `THIRD_PARTY.txt` and `third_party/`; do not label the entire runtime MIT.

## Deliverables

- Static build: `/workspace/itch/live_universe/` (`index.html` at root).
- Archive: `/workspace/itch/nimverse_live_universe.zip` (about 24.1 MB, 37 files;
  final archive integrity and precise byte size recorded in handoff).
- Screenshots: `/workspace/itch/live_shots/01_galaxy.png`, `02_supernova.png`,
  `03_clicked_star_system.png`, `04_generated_planet.png`, `05_live_surface.png`,
  `06_photoreal.png`, `07_live_creature.png`, `08_second_generated_world.png`.
- Reports: `/workspace/itch/live_browser_report.json`,
  `/workspace/itch/live_performance_report.json`,
  `/workspace/itch/live_gate_final.json`, `/workspace/itch/native_live_reference.json`.
- Pre-change backup:
  `/workspace/backups/nim-universe-pre-live-20261107-165811.tar.gz`.

The generated runtime/cache/archive remain gitignored. Only source, tests,
documentation and lightweight circuit fixtures are committed locally on main.
