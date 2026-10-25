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

Not built yet: culture/tech, radiation observers for BioSignFlux, the NIM life-mesh training loop, and PN/cosmology. The galaxy and event pipeline is unchanged.
Backup from before the leaves: `/workspace/backups/universe_pre_leaves_20261025_163027{.tar.gz,/}`.
