"""External science-stack adapters (opt-in). See README "External stacks".

Live on this box (nimenv): rebound, lalsimulation, mist proxy (+ mistery/isochrones when network works),
npz/csv table loaders for climate / RT / NR products.

Needs offline installs: MESA (MESA_DIR), ExoPlaSim Fortran binary (gfortran), ROCKE-3D / SpEC / Einstein Toolkit
product tables (loaded via climate_table / nr_waveform / rt_table).

Import submodules directly, e.g. `from universe.solvers.ext import rebound_orbit, lal_wave`.
"""

class UnavailableError(RuntimeError):
    """Raised when an external stack is selected but not installed / compiled / pointed at."""


__all__ = [
    "UnavailableError",
    "mist_tracks", "mesa_runner", "rebound_orbit", "lal_wave",
    "exoplasim_climate", "rt_tables", "et_loader",
]

# Eager-import submodules so `from universe.solvers.ext import rebound_orbit` yields the module
# (not a shadowing accessor). Soft-fail individual imports so a broken optional dep does not
# block the package.
def _load():
    import importlib
    mods = {}
    for name in ("mist_tracks", "mesa_runner", "rebound_orbit", "lal_wave",
                 "exoplasim_climate", "rt_tables", "et_loader"):
        try:
            mods[name] = importlib.import_module(f".{name}", __name__)
        except Exception as e:  # pragma: no cover
            mods[name] = e
    return mods

_mods = _load()
for _k, _v in _mods.items():
    if not isinstance(_v, Exception):
        globals()[_k] = _v
