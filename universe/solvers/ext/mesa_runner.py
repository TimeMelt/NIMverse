"""Optional real MESA runner (--stellar-solver mesa via MESA_DIR / config mesa_dir).

This box has no gfortran/cmake, so MESA cannot be compiled here. When MESA_DIR points at a working
install with `./star`, we write a minimal inlist, run it, and parse `LOGS/history.data` into the
same track table as solvers.stellar_track.solve_track. Otherwise raise UnavailableError (tests skip).
"""
from __future__ import annotations
import math, os, shutil, subprocess, tempfile
import numpy as np

from . import UnavailableError
from ..stellar_track import ZSUN, T_SUN

INLIST_DIR = os.path.join(os.path.dirname(__file__), "mesa_inlists")


def mesa_dir(cfg=None):
    """Resolve MESA_DIR from env or config dict/path string."""
    if isinstance(cfg, dict): d = cfg.get("mesa_dir") or cfg.get("MESA_DIR")
    else: d = cfg
    d = d or os.environ.get("MESA_DIR")
    return os.path.abspath(d) if d else None


def available(cfg=None):
    d = mesa_dir(cfg)
    if not d or not os.path.isdir(d): return False
    star = os.path.join(d, "star")
    if os.path.isfile(star) and os.access(star, os.X_OK): return True
    # work directory layout: star binary next to inlist
    return any(os.path.isfile(os.path.join(d, p)) and os.access(os.path.join(d, p), os.X_OK)
               for p in ("star", "star.exe"))


def _require(cfg=None):
    if not available(cfg):
        raise UnavailableError(
            "MESA not available: set MESA_DIR (or config mesa_dir) to a working MESA install with an executable `star`. "
            "This box cannot compile MESA (no gfortran/cmake)."
        )
    return mesa_dir(cfg)


def _write_inlist(work, M, Z, max_age_Gyr=15.):
    os.makedirs(work, exist_ok=True)
    tmpl = os.path.join(INLIST_DIR, "inlist_project")
    if os.path.isfile(tmpl):
        text = open(tmpl).read()
        text = text.replace("@MASS@", f"{float(M):.6f}").replace("@Z@", f"{float(Z):.6e}")
        text = text.replace("@MAX_AGE@", f"{float(max_age_Gyr)*1e9:.6e}")
    else:
        text = f"""&star_job
  create_pre_main_sequence_model = .true.
/ ! end of star_job

&controls
  initial_mass = {float(M):.6f}
  initial_z = {float(Z):.6e}
  max_age = {float(max_age_Gyr)*1e9:.6e}
  history_interval = 1
/ ! end of controls

&pgstar
/ ! end of pgstar
"""
    open(os.path.join(work, "inlist"), "w").write(text)
    open(os.path.join(work, "inlist_project"), "w").write(text)
    return os.path.join(work, "inlist")


def _parse_history(path):
    """Parse MESA history.data (header + rows) -> track dict."""
    lines = open(path).read().splitlines()
    # find column header (line after one starting with 'columns')
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("star_age") and "star_age" not in lines[i]:
        i += 1
    if i >= len(lines):
        # alternate: skip comment lines starting with #
        cols = None; rows = []
        for ln in lines:
            if not ln.strip() or ln.lstrip().startswith("#"): continue
            if cols is None:
                cols = ln.split(); continue
            rows.append([float(x) for x in ln.split()])
        if not cols or not rows: raise ValueError(f"unreadable MESA history: {path}")
        arr = {c: np.array([r[k] for r in rows], float) for k, c in enumerate(cols)}
    else:
        cols = lines[i].split(); rows = []
        for ln in lines[i + 1:]:
            if not ln.strip() or ln.lstrip().startswith("#"): continue
            parts = ln.split()
            if len(parts) < len(cols): continue
            try: rows.append([float(x) for x in parts[:len(cols)]])
            except ValueError: continue
        arr = {c: np.array([r[k] for r in rows], float) for k, c in enumerate(cols)}
    t = arr.get("star_age", arr.get("age")) / 1e9
    if "log_L" in arr: L = 10 ** arr["log_L"]
    elif "luminosity" in arr: L = arr["luminosity"]
    else: raise ValueError("history missing luminosity")
    if "log_R" in arr: R = 10 ** arr["log_R"]
    elif "radius" in arr: R = arr["radius"]
    else:
        Teff = 10 ** arr["log_Teff"] if "log_Teff" in arr else T_SUN
        R = np.sqrt(L) * (T_SUN / Teff) ** 2
    Mm = arr.get("star_mass", np.full_like(t, np.nan))
    Teff = T_SUN * (L / np.maximum(R, 1e-6) ** 2) ** .25
    # crude phase from center H / He if present
    ph = np.array(["ms"] * len(t))
    if "center_h1" in arr:
        ph = np.where(arr["center_h1"] > .1, "ms", ph)
        ph = np.where((arr["center_h1"] <= .1) & (arr.get("center_he4", np.ones_like(t)) > .1), "rgb", ph)
    t_pms = float(t[0]); ms = ph == "ms"
    t_ms = float(t[ms][-1]) if ms.any() else float(t[len(t) // 2])
    return dict(t=t, L=L, R=R, Teff=Teff, M=Mm, phase=ph, t_pms=t_pms, t_ms=t_ms, t_end=float(t[-1]),
                n_pulses=0, M_wd=None, Z=None, mist_source="mesa", mode="mesa")


def run_star(M, Z=ZSUN, cfg=None, work=None, timeout=3600):
    """Run MESA `./star` for mass M, metallicity Z. Returns track dict."""
    root = _require(cfg)
    star = os.path.join(root, "star")
    if not os.path.isfile(star): star = os.path.join(root, "star.exe")
    own = work is None
    work = work or tempfile.mkdtemp(prefix="mesa_nim_")
    try:
        _write_inlist(work, M, Z)
        # copy or link star binary into work if MESA expects local cwd
        env = dict(os.environ); env["MESA_DIR"] = root
        cmd = [star] if os.path.dirname(star) == work else [star]
        # Prefer running from work with MESA_DIR set
        try:
            subprocess.run(cmd, cwd=work, env=env, check=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            # try `$MESA_DIR/star` via shell
            subprocess.run(["bash", "-lc", f'cd "{work}" && "{star}"'], env=env, check=True, timeout=timeout)
        hist = os.path.join(work, "LOGS", "history.data")
        if not os.path.isfile(hist):
            # sometimes history is in work/
            alt = os.path.join(work, "history.data")
            if os.path.isfile(alt): hist = alt
            else: raise UnavailableError(f"MESA finished but no history.data under {work}")
        d = _parse_history(hist); d["Z"] = float(Z); return d
    finally:
        if own: shutil.rmtree(work, ignore_errors=True)


def solve_track(M, Z=ZSUN, cfg=None):
    return run_star(M, Z, cfg=cfg)


X_START_MAX = .9

def _interp(d, key, age):
    a = float(np.clip(age, 0, d["t_end"])); v = d[key]
    return float(np.exp(np.interp(a, d["t"], np.log(np.maximum(v, 1e-30))))) if key != "M" else float(np.interp(a, d["t"], v))


class _Api:
    X_START_MAX = X_START_MAX
    def __init__(self, Z, cfg=None): self.Z = float(Z); self.cfg = cfg; self.mode = "mesa"
    def t_ms_Gyr(self, M): return solve_track(M, self.Z, self.cfg)["t_ms"]
    def max_age_Gyr(self, M): return solve_track(M, self.Z, self.cfg)["t_end"]
    def phase(self, M, age_Gyr):
        d = solve_track(M, self.Z, self.cfg); i = int(np.clip(np.searchsorted(d["t"], age_Gyr), 0, len(d["t"]) - 1)); return str(d["phase"][i])
    def track(self, M, L0, T0, age0_Gyr, age_Gyr):
        d = solve_track(M, self.Z, self.cfg)
        L = L0 * _interp(d, "L", age_Gyr) / _interp(d, "L", age0_Gyr)
        T = T0 * _interp(d, "Teff", age_Gyr) / _interp(d, "Teff", age0_Gyr)
        return float(L), float(T), float(math.sqrt(max(L, 1e-12)) * (T_SUN / T) ** 2)
    def mass(self, M, age_Gyr): return _interp(solve_track(M, self.Z, self.cfg), "M", age_Gyr)


def api(Z=None, cfg=None):
    _require(cfg)
    return _Api(ZSUN if Z is None or Z < 0 else float(Z), cfg)
