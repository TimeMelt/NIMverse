# MESA inlists

Minimal templates used by `universe.solvers.ext.mesa_runner` when `MESA_DIR` points at a working MESA install.
This repository does **not** ship MESA. Build MESA elsewhere (needs gfortran + MESA SDK), set `MESA_DIR`, then:

```
--stellar-solver mesa
# or config: {"mesa_dir": "/path/to/MESA"}
```

Without `MESA_DIR`, the mesa path raises `UnavailableError` and tests skip.
