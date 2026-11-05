"""Column RT table loader (--rt-table PATH).

Drop-in for solvers.rt climate_tables cache: load precomputed OLR/albedo vs T, pCO2 (, buffer)
from npz/csv produced offline by HELIOS / SOCRATES / petitRADTRANS (or our own solvers.rt).
Ships a tiny example built from solvers.rt so the path is tested without external codes.
"""
from __future__ import annotations
import os
import numpy as np

from . import UnavailableError

EXAMPLE_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "rt_tables", "example_rt.npz")
_ACTIVE = None  # path currently patched into rt.climate_tables


def _grids_from_rt():
    from .. import rt
    return dict(TS=np.asarray(rt.TS_GRID, float), LCO2=np.asarray(rt.LCO2_GRID, float),
                RH=np.asarray(rt.RH_GRID, float), MU=np.asarray(rt.MU_GRID, float), g=float(rt.G_EARTH))


def build_example_table(path=EXAMPLE_TABLE, **kw):
    """Build a compact example table from the live solvers.rt climate_tables (no external RT codes)."""
    from .. import rt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tab = rt.climate_tables(**{k: kw[k] for k in ("g", "T_eff", "x_ch4", "o3_du", "ps", "x_h2") if k in kw})
    g = _grids_from_rt()
    np.savez_compressed(path, olr=tab["olr"], sw=tab["sw"], g=np.array(tab["g"]),
                        runaway=np.array(bool(tab["runaway"])),
                        TS=g["TS"], LCO2=g["LCO2"], RH=g["RH"], MU=g["MU"],
                        source="solvers.rt", note="offline product stand-in for HELIOS/SOCRATES/petitRADTRANS")
    return path


def load_rt_table(path):
    """Load npz/csv RT table -> dict compatible with solvers.rt.climate_tables return."""
    path = os.path.abspath(path)
    if not os.path.isfile(path): raise FileNotFoundError(path)
    if path.lower().endswith(".csv"):
        raise UnavailableError("csv RT tables need a documented columnar layout; use npz (see build_example_table)")
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    if "olr" not in keys or "sw" not in keys:
        raise ValueError(f"rt table {path} missing olr/sw arrays (has {sorted(keys)})")
    out = dict(olr=np.asarray(d["olr"], float), sw=np.asarray(d["sw"], float),
               g=float(np.asarray(d["g"]).reshape(-1)[0]) if "g" in keys else 9.81,
               runaway=bool(np.asarray(d["runaway"]).reshape(-1)[0]) if "runaway" in keys else False,
               source=str(d["source"]) if "source" in keys else os.path.basename(path),
               path=path)
    for k in ("TS", "LCO2", "RH", "MU"):
        if k in keys: out[k] = np.asarray(d[k], float)
    return out


def install_rt_table(path):
    """Monkey-patch solvers.rt.climate_tables to return the loaded table (opt-in). Returns previous callable."""
    from .. import rt
    global _ACTIVE
    tab = load_rt_table(path)
    # Optionally align module grids if the table carries them (lookups use module-level grids)
    for name, attr in (("TS", "TS_GRID"), ("LCO2", "LCO2_GRID"), ("RH", "RH_GRID"), ("MU", "MU_GRID")):
        if name in tab:
            setattr(rt, attr, np.asarray(tab[name], float))
    prev = rt.climate_tables

    def _tables(*a, **k):
        # Ignore args: table is precomputed for its own (g,Teff,...) — caller must match.
        return dict(olr=tab["olr"], sw=tab["sw"], g=tab["g"], runaway=tab["runaway"],
                    rt_table=tab.get("source", path))

    rt.climate_tables = _tables
    # Clear RT caches that keyed on phys
    for cname in ("_CACHE", "_climate_cache", "CACHE"):
        c = getattr(rt, cname, None)
        if isinstance(c, dict): c.clear()
    _ACTIVE = path
    return prev


def uninstall_rt_table(prev=None):
    """Restore solvers.rt.climate_tables if we patched it."""
    from .. import rt
    global _ACTIVE
    if prev is not None:
        rt.climate_tables = prev
    else:
        # re-import original binding via wrapping: set back to _climate_tables path
        def _tables(g=None, T_eff=5772., x_ch4=7e-7, o3_du=0., ps=1., x_h2=0.):
            if g is None: g = rt.G_EARTH
            return rt._climate_tables(float(g), float(T_eff), float(x_ch4), float(o3_du), float(ps), float(x_h2), rt.phys_key())
        rt.climate_tables = _tables
    _ACTIVE = None


def ensure_example():
    if not os.path.isfile(EXAMPLE_TABLE): build_example_table(EXAMPLE_TABLE)
    return EXAMPLE_TABLE
