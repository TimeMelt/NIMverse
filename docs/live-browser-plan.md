# Approved live-browser universe plan (2026-11-07)

Local-only implementation; main; one commit per stage; no push or itch publication.
Backup: /workspace/backups/nim-universe-pre-live-20261107-165811.tar.gz.

1. Offline module worker: bundled Pyodide 314.0.7, NumPy/SciPy/REBOUND; exact leaf HabitatRun, seed4/planet2 gate in a static sandboxed iframe.
2. Deterministic stream-per-ID JS procedural galaxies, bounded leapfrog/Barnes–Hut and events. NIM learned preset labelled as offline-trained. No frame accumulation.
3. Each star's mass, age and Z feed a repeatable generated system; navigation and local time controls. Physical units kept separate from render radii.
4. Worker active-world leaf simulation and bounded visited-world LOD; QWRK statevector creature policy and bounded quantum-driven classical forcing, user toggle. Existing planet/creature shaders reused.
5. Automated circuit/native-worker parity, determinism and physics sanity; iframe, request audit, screenshots; offline archive under 100 MB / 1000 files.

Simplifications are explicit: finite sample of galaxies/stars, independent display/stellar/planet clocks, local detail only, approximate galaxy/event recipes (not numerical relativity). Browser never trains NIM and never represents a quantum computer. Exact numerical leaf code is distinguished from parameter-driven classical physics. No product/public wording claims quantum hardware.

Stage 1 passed: Pyodide uses module workers (classic workers are unsupported in 314.0.7). REBOUND 4.4.7 integrates a two-body orbit in WASM. A WASM32 NumPy repeat index-width portability fix was required. Adapter clears the existing equilibrium cache because its warm path integrates an extra year and otherwise makes revisits order-dependent. Runtime is ~23.5 MB zipped. No external browser requests.

Stage 2: seeded per-body streams, three finite galaxies, 768 star tracers + 3 halos + 3 SMBHs. JS procedural port follows sim2 morphology/IMF recipes, but is not bit-identical to NumPy generation. Streaming Barnes-Hut with leapfrog; independent accelerated event clock. Mass loss recorded as unresolved ejecta/radiation, event ring limited to 64. WD/SN/NS/BH/GRB and radius-gated capture/TDE, phenomenological AGN, SMBH mergers included. Events are simplified classical prescriptions, not external solver parity.

Stage 3: deterministic planet + moon generation, tied to clicked host mass/age/Z; Keplerian orbit positions and full-system secular forcing inputs. Galaxy/system/world breadcrumbs, any seed, pause/speed/reset/fullscreen. Inner worlds deliberately sampled rocky for exploration; not an unbiased exoplanet survey. System rendering normalizes AU by sqrt(L/Lsun), radii/moon separations exaggerated; numerical habitats retain AU and solar masses. Browser transition smoke and 8 JS tests pass.

Stage 4: actual 24-band climate, carbon/CNP, chemistry, evolving individual populations and SciPy clustering execute in the worker. JS QWRK creature queries succeed with zero failures (40 world-year cadence; 16 sampled individuals; fixed seeded circuit weights, no online learning). Generated worlds additionally use the species-div foodweb. Four worlds are kept in memory; active world full cadence, inactive cached worlds round-robin one exact tick per 5 seconds. This is wall-clock LOD, not equal elapsed physical time everywhere; evicted worlds regenerate. Existing surface/creature meshes and photoreal option consume live ticks via postMessage, not playback.

Quantum drive is exact CPU statevector evaluation, not quantum hardware. NIM 16-wire free-think structure ansatz modulates initial galaxy size/disk fraction ±4%. Live 5-wire QWRK readouts modulate planet stellar flux ±1%, UV ±5%, flare-rate multiplier ±20%, secular eccentricity offset ±0.002 (clamped), supernova kick amplitude ±15%, GRB/AGN probability ±20%. Classical solvers consume those values. Toggle changes future forcing; regenerate for structural toggle. Chemical/ecological effects propagate from climate/UV and creature decisions. This is a phenomenological coupling, not a validated quantum theory of physics.


## Stage 5 — acceptance and packaging (complete)

Final architecture supersedes the initial main-thread request cadence: the worker
owns the detailed-life clock (~8 ticks/s at normal speed), with at most one
unacknowledged visual snapshot. Slow rendering drops/coalesces snapshots, never
queues unbounded RPCs or stalls physics. Inactive cached worlds receive one exact
coarse tick round-robin every five seconds. Creature visibility is adaptively
limited to 4–8 meshes; all organisms still participate in Python simulation.

Acceptance: static nested sandboxed iframe, actual clicked-star navigation,
generated systems/rocky worlds, live surface and creature views, photoreal mode,
second inhabited world, pause/seed/reset/fullscreen and quantum toggle passed.
Browser audit found no external requests, HTTP failures, console errors or page
errors. REBOUND imports and integrates successfully inside Pyodide.

Native/Pyodide five-tick parity: temperature difference <=1.14e-13 K,
biomass difference <=1.78e-15; organism counts/births/species agree. Circuit
parity: QWRK <=4.45e-16 and NIM16 <=2.43e-13 versus PennyLane fixtures.
Focused native regression: 21 passed; JavaScript regression: 14 passed.

Final throughput: 7.47 detailed ticks/s in orbit and 7.45 ticks/s on the
photoreal surface; service time ~53–57 ms. Pyodide cold boot ~3.7 s;
WASM memory ~75 MiB at boot and ~90 MiB during the throughput probe.
Main JS heap ~68 MB in that probe; total process/GPU/tab memory was not measured.
Headless Chromium SwiftShader rendering ~12.7 FPS in orbit, ~28.3 FPS on
the photoreal surface, ~11 FPS creature closeup: the 30–60 FPS render target
is NOT consistently met on software WebGL. Hardware GPU/Safari/mobile untested.

Output: /workspace/itch/live_universe/ and
/workspace/itch/nimverse_live_universe.zip (about 24.1 MB, 37 files).
Screenshots and machine-readable reports: /workspace/itch/live_shots/ and
/workspace/itch/live_*report.json. See docs/live-browser-verification.md.
Local commits only; no push, publication, deployment or external announcement.
