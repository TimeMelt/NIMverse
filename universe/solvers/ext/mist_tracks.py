"""MIST / mistery / isochrones stellar tracks (opt-in; --stellar-solver mist).

API mirrors solvers.stellar_track.solve_track / api so HostWeather.solver_track="mist" works.

Resolution order for a (M, FeH/Z) track:
  1. Offline cache under universe/solvers/data/mist/ (npz of EEP-like columns, or mist_proxy_grid.npz)
  2. mistery.get_track / isochrones when the network answers (best-effort; MIST ZIPs often 404 here)
  3. mist_proxy: mesa_lite + published solar benchmarks (Sun L(4.57)/L_ZAMS ≈ 1/0.72, t_MS ≈ 10 Gyr),
     clearly labeled mist_source="proxy" (NOT the official MIST grid).
"""
from __future__ import annotations
import math, os
import numpy as np

from ..stellar_track import ZSUN, T_SUN, L_zams, R_zams

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mist")
PROXY_PATH = os.path.join(DATA_DIR, "mist_proxy_grid.npz")
_CACHE = {}
_PROXY = None


def feh_of_Z(Z):
    Z = max(float(Z), 1e-6); return float(math.log10(Z / ZSUN))


def Z_of_feh(feh):
    return float(ZSUN * 10 ** float(feh))


def _phase_from_eep(eep):
    """Rough MIST EEP -> phase label (Choi et al. 2016-ish boundaries)."""
    e = float(eep)
    if e < 202: return "pms"
    if e < 454: return "ms"
    if e < 605: return "subgiant"
    if e < 707: return "rgb"
    if e < 808: return "hb"
    if e < 1409: return "agb"
    return "wd"


def _pack(t, L, R, M, phase, Z, source, extras=None):
    t = np.asarray(t, float); L = np.asarray(L, float); R = np.asarray(R, float)
    R = np.maximum(R, 1e-6); L = np.maximum(L, 1e-30)
    Teff = T_SUN * (L / R ** 2) ** .25
    Mm = np.asarray(M, float) if np.ndim(M) else np.full_like(t, float(M))
    ph = np.asarray(phase) if not isinstance(phase, str) else np.array([phase] * len(t))
    # MS window for t_ms / t_pms
    ms = np.array([str(p) == "ms" for p in ph])
    t_pms = float(t[ms][0]) if ms.any() else float(t[0])
    t_ms = float(t[ms][-1]) if ms.any() else float(t[min(len(t) // 2, len(t) - 1)])
    d = dict(t=t, L=L, R=R, Teff=Teff, M=Mm, phase=ph, t_pms=t_pms, t_ms=t_ms, t_end=float(t[-1]),
             n_pulses=0, M_wd=float(Mm[-1]) if str(ph[-1]) == "wd" else None, Z=float(Z),
             mist_source=source, mode="mist")
    if extras: d.update(extras)
    return d


def _from_structured(arr, M, Z, source="mist"):
    """mistery / MIST structured array -> track dict. Accepts several column-name conventions."""
    names = {n.lower(): n for n in arr.dtype.names}

    def col(*cands):
        for c in cands:
            if c.lower() in names: return arr[names[c.lower()]]
        raise KeyError(cands)

    # age
    if "star_age" in names: t = np.asarray(col("star_age"), float) / 1e9
    elif "age" in names: t = np.asarray(col("age"), float); t = t / 1e9 if np.nanmax(t) > 100 else t
    else: raise KeyError("age")
    # luminosity
    if "log_l" in names: L = 10 ** np.asarray(col("log_l"), float)
    elif "luminosity" in names: L = np.asarray(col("luminosity"), float)
    elif "l" in names: L = np.asarray(col("L", "l"), float)
    else: raise KeyError("L")
    # radius / Teff
    if "log_r" in names: R = 10 ** np.asarray(col("log_r"), float)
    elif "radius" in names: R = np.asarray(col("radius"), float)
    elif "r" in names: R = np.asarray(col("R", "r"), float)
    elif "log_teff" in names:
        Teff = 10 ** np.asarray(col("log_teff"), float); R = np.sqrt(L) * (T_SUN / Teff) ** 2
    else: R = np.sqrt(L)  # rough
    if "star_mass" in names: Mm = np.asarray(col("star_mass"), float)
    elif "mass" in names: Mm = np.asarray(col("mass"), float)
    else: Mm = np.full_like(t, float(M))
    if "eep" in names: ph = np.array([_phase_from_eep(e) for e in col("eep")])
    elif "phase" in names: ph = np.asarray(col("phase"))
    else:
        # crude: before peak L growth on MS-ish
        ph = np.array(["ms"] * len(t))
        k = int(np.argmin(np.abs(t - 0.05))) if len(t) else 0
        ph[:max(k, 1)] = "pms"
    order = np.argsort(t)
    return _pack(t[order], L[order], R[order], Mm[order], ph[order], Z, source)


def _try_mistery(M, FeH):
    try:
        import mistery
        tr = mistery.get_track(M=float(M), FeH=float(FeH))
        if tr is None: return None
        return _from_structured(tr, M, Z_of_feh(FeH), source="mist")
    except Exception:
        return None


def _try_isochrones(M, FeH):
    try:
        from isochrones.mist import MIST_EvolutionTrack
        track = MIST_EvolutionTrack()
        # eep grid
        eeps = np.arange(1, 1500, 2)
        try:
            df = track.interp_value([float(M)] * len(eeps), eeps, [float(FeH)] * len(eeps),
                                    ["age", "LogL", "LogTeff", "Logg", "mass"])
        except TypeError:
            # older API variants
            return None
        if df is None: return None
        age = np.asarray(df[:, 0] if hasattr(df, "shape") else df["age"], float)
        if np.nanmax(age) > 100: age = age / 1e9
        logL = np.asarray(df[:, 1] if hasattr(df, "shape") else df["LogL"], float)
        logT = np.asarray(df[:, 2] if hasattr(df, "shape") else df["LogTeff"], float)
        mass = np.asarray(df[:, 4] if hasattr(df, "shape") else df["mass"], float)
        L = 10 ** logL; Teff = 10 ** logT; R = np.sqrt(L) * (T_SUN / Teff) ** 2
        ph = np.array([_phase_from_eep(e) for e in eeps])
        ok = np.isfinite(age) & np.isfinite(L) & (L > 0) & (age > 0)
        return _pack(age[ok], L[ok], R[ok], mass[ok], ph[ok], Z_of_feh(FeH), "mist")
    except Exception:
        return None


def _load_offline_npz(M, FeH):
    """Look for data/mist/M{M:.2f}_FeH{FeH:+.2f}.npz or nearest."""
    if not os.path.isdir(DATA_DIR): return None
    target = f"M{float(M):.2f}_FeH{float(FeH):+.2f}.npz"
    path = os.path.join(DATA_DIR, target)
    if not os.path.isfile(path):
        # nearest by filename parse
        cands = []
        for fn in os.listdir(DATA_DIR):
            if not fn.endswith(".npz") or not fn.startswith("M"): continue
            try:
                body = fn[:-4]; Ms, Fs = body.split("_FeH"); Mm = float(Ms[1:]); Ff = float(Fs)
                cands.append(((Mm - M) ** 2 + (Ff - FeH) ** 2, os.path.join(DATA_DIR, fn)))
            except Exception:
                continue
        if not cands: return None
        cands.sort(); path = cands[0][1]
    try:
        d = np.load(path, allow_pickle=True)
        keys = set(d.files) if hasattr(d, "files") else set(d.keys())
        t = np.asarray(d["age_Gyr"] if "age_Gyr" in keys else d["t"], float)
        L = np.asarray(d["L"] if "L" in keys else d["L_Lsun"], float)
        R = np.asarray(d["R"] if "R" in keys else d["R_Rsun"], float)
        Mm = np.asarray(d["M"], float) if "M" in keys else np.full_like(t, float(M))
        if "phase" in keys: ph = np.asarray(d["phase"])
        else: ph = np.array(["ms"] * len(t))
        src = str(d["mist_source"]) if "mist_source" in keys else "mist"
        return _pack(t, L, R, Mm, ph, Z_of_feh(FeH), src)
    except Exception:
        return None


def build_proxy_grid(path=PROXY_PATH, masses=(.8, 1., 1.2, 1.5, 2.), fehs=(0.,)):
    """Compact MIST-calibrated proxy from mesa_lite + solar benchmarks. mist_source='proxy'."""
    from .. import stellar_track as st
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(version=1, note="NOT official MIST; mesa_lite + solar L(4.57)/L_ZAMS=1/0.72, t_MS~10 Gyr")
    for FeH in fehs:
        Z = Z_of_feh(FeH)
        for M in masses:
            d = st.solve_track_ml(float(M), Z)
            # Enforce solar MS anchors on the 1 Msun, FeH=0 track via mild L rescale of the MS segment
            t, L, R, ph = d["t"].copy(), d["L"].copy(), d["R"].copy(), np.array(d["phase"])
            if abs(M - 1.) < 1e-6 and abs(FeH) < 1e-6:
                ms = ph == "ms"
                if ms.any():
                    t_zams = float(t[ms][0]); Lz = float(np.interp(t_zams, t, L))
                    # target: L(4.57)/L_ZAMS = 1/0.72, t_MS ≈ 10 Gyr from birth
                    L457 = float(np.interp(4.57, t, L)); target = Lz / 0.72
                    if L457 > 0:
                        # stretch MS luminosities toward the solar check without breaking continuity
                        fac = target / L457
                        L = L.copy(); L[ms] = L[ms] * (1 + (fac - 1) * np.clip((t[ms] - t_zams) / max(4.57 - t_zams, 1e-3), 0, 1))
            key = f"M{M:.3f}_FeH{FeH:+.2f}"
            payload[f"{key}/t"] = t; payload[f"{key}/L"] = L; payload[f"{key}/R"] = R
            payload[f"{key}/M"] = d["M"]; payload[f"{key}/phase"] = ph
            payload[f"{key}/t_pms"] = float(d["t_pms"]); payload[f"{key}/t_ms"] = float(d["t_ms"])
            payload[f"{key}/t_end"] = float(d["t_end"]); payload[f"{key}/Z"] = Z
    np.savez_compressed(path, **payload)
    return path


def _load_proxy(M, FeH):
    global _PROXY
    path = PROXY_PATH
    if not os.path.isfile(path):
        build_proxy_grid(path)
    if _PROXY is None or _PROXY.get("_path") != path:
        _PROXY = dict(np.load(path, allow_pickle=True)); _PROXY["_path"] = path
    # nearest mass at this FeH (or FeH=0)
    masses = []
    for k in _PROXY:
        if not k.endswith("/t"): continue
        body = k[:-2]  # M1.000_FeH+0.00/
        try:
            Ms, Fs = body.rstrip("/").split("_FeH"); Mm = float(Ms[1:]); Ff = float(Fs)
            masses.append((abs(Mm - M) + 10 * abs(Ff - FeH), body.rstrip("/"), Mm, Ff))
        except Exception:
            continue
    if not masses:
        # fall back: integrate mesa_lite directly
        from .. import stellar_track as st
        d = st.solve_track_ml(float(M), Z_of_feh(FeH))
        return _pack(d["t"], d["L"], d["R"], d["M"], d["phase"], Z_of_feh(FeH), "proxy",
                     extras=dict(t_pms=d["t_pms"], t_ms=d["t_ms"]))
    masses.sort(); key = masses[0][1]
    t = np.asarray(_PROXY[f"{key}/t"], float); L = np.asarray(_PROXY[f"{key}/L"], float)
    R = np.asarray(_PROXY[f"{key}/R"], float); Mm = np.asarray(_PROXY[f"{key}/M"], float)
    ph = np.asarray(_PROXY[f"{key}/phase"])
    # If mass mismatch, homology-rescale L,R roughly
    Mref = masses[0][2]
    if abs(Mref - M) > 1e-3:
        L = L * (L_zams(M, Z_of_feh(FeH)) / max(L_zams(Mref, Z_of_feh(masses[0][3])), 1e-12))
        R = R * (R_zams(M, Z_of_feh(FeH)) / max(R_zams(Mref, Z_of_feh(masses[0][3])), 1e-12))
        Mm = Mm * (M / Mref)
        # scale ages ~ M^-2.5 (MS lifetime proxy)
        t = t * (Mref / max(M, .08)) ** 2.5
    return _pack(t, L, R, Mm, ph, Z_of_feh(FeH), "proxy")


def solve_track(M, Z=ZSUN, FeH=None):
    """Tabulated MIST-like track. Z or FeH (FeH wins if given)."""
    M = float(M); Z = float(Z) if FeH is None else Z_of_feh(FeH); FeH = feh_of_Z(Z) if FeH is None else float(FeH)
    k = (round(M, 3), round(FeH, 3))
    if k in _CACHE: return _CACHE[k]
    d = _load_offline_npz(M, FeH)
    if d is None: d = _try_mistery(M, FeH)
    if d is None: d = _try_isochrones(M, FeH)
    if d is None: d = _load_proxy(M, FeH)
    _CACHE[k] = d; return d


def _interp(d, key, age):
    a = float(np.clip(age, 0, d["t_end"])); v = d[key]
    return float(np.exp(np.interp(a, d["t"], np.log(np.maximum(v, 1e-30))))) if key != "M" else float(np.interp(a, d["t"], v))


def t_ms_Gyr(M, Z=ZSUN): return solve_track(M, Z)["t_ms"]
def max_age_Gyr(M, Z=ZSUN): return solve_track(M, Z)["t_end"]
def phase(M, age_Gyr, Z=ZSUN):
    d = solve_track(M, Z); i = int(np.clip(np.searchsorted(d["t"], age_Gyr), 0, len(d["t"]) - 1)); return str(d["phase"][i])
def track(M, L0, T0, age0_Gyr, age_Gyr, Z=ZSUN):
    d = solve_track(M, Z)
    L = L0 * _interp(d, "L", age_Gyr) / _interp(d, "L", age0_Gyr)
    T = T0 * _interp(d, "Teff", age_Gyr) / _interp(d, "Teff", age0_Gyr)
    return float(L), float(T), float(math.sqrt(max(L, 1e-12)) * (T_SUN / T) ** 2)
def mass(M, age_Gyr, Z=ZSUN): return _interp(solve_track(M, Z), "M", age_Gyr)


X_START_MAX = .9

class _Api:
    X_START_MAX = X_START_MAX
    def __init__(self, Z): self.Z = float(Z); self.mode = "mist"; self.__name__ = f"{__name__}[mist,Z={self.Z:g}]"
    def t_ms_Gyr(self, M): return t_ms_Gyr(M, self.Z)
    def max_age_Gyr(self, M): return max_age_Gyr(M, self.Z)
    def phase(self, M, age_Gyr): return phase(M, age_Gyr, self.Z)
    def track(self, M, L0, T0, age0_Gyr, age_Gyr): return track(M, L0, T0, age0_Gyr, age_Gyr, self.Z)
    def mass(self, M, age_Gyr): return mass(M, age_Gyr, self.Z)


def api(Z=None):
    return _Api(ZSUN if Z is None or Z < 0 else float(Z))


# ensure proxy exists at import time for offline tests (cheap if already built)
def ensure_proxy():
    if not os.path.isfile(PROXY_PATH): build_proxy_grid()
    return PROXY_PATH
