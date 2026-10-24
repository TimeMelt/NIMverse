# nim-universe

Procedural 3D universe generator that uses NIM (https://github.com/TimeMelt/NIM) inceptron meshes and the QrnLOpt optimizer to learn its generation parameters. It covers galaxies, stars, planets, supernovae, black holes, TDEs and AGN, and includes a three.js viewer.

- `universe/`: the code. It's run as the `universe` package next to a NIM checkout, e.g. `python -m universe.learn_alt2`.
- `results/`: the learned parameters, training logs and convergence plots.

Large rendered outputs (videos, HTML viewers, frames, .npz) are not included.
