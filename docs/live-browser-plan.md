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
