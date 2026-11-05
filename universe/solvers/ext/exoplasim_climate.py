"""ExoPlaSim climate hook + external climate-table loader (--climate-model exoplasim / --climate-table).

ExoPlaSim's Python shell is installed, but the Fortran binary cannot be built here (no gfortran).
Calling run_exoplasim raises UnavailableError with install instructions.

ROCKE-3D / ExoPlaSim offline products enter via load_climate_table(path): NetCDF (if netCDF4 present)
or npz/csv of T, albedo, OLR, precip on lat-lon, which drive HabitatRun stressors through TableClimate.
"""
from __future__ import annotations
import os
import numpy as np

from . import UnavailableError

EXAMPLE_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "climate_tables", "example_latlon.npz")


def exoplasim_binary_ok():
    """True if a compiled PlaSim/ExoPlaSim Fortran binary looks runnable.

    The Python package ships `most_plasim_run` (a shell driver) and sources; those alone are not enough.
    We require an actual compiled `plasim*` binary (typically produced by gfortran via exoplasim.compile()).
    """
    try:
        import exoplasim
        root = os.path.dirname(exoplasim.__file__)
    except Exception:
        return False
    for dirpath, _, files in os.walk(os.path.join(root, "plasim")):
        if "/src" in dirpath.replace("\\", "/"):  # skip Fortran sources
            continue
        for f in files:
            fl = f.lower()
            if fl.startswith("plasim") and (fl.endswith(".x") or fl.endswith(".so") or fl == "plasim" or fl.startswith("plasim_t")):
                pth = os.path.join(dirpath, f)
                # must be a real binary, not a tiny wrapper script
                if os.path.isfile(pth) and os.access(pth, os.X_OK) and os.path.getsize(pth) > 50_000:
                    return True
    return False


def run_exoplasim(*args, **kwargs):
    """Attempt an ExoPlaSim run; raise UnavailableError if the Fortran core is missing."""
    if not exoplasim_binary_ok():
        raise UnavailableError(
            "ExoPlaSim Fortran binary is not compiled. Install gfortran (and a working apt/build chain), "
            "then rebuild exoplasim (`import exoplasim; exoplasim.compile()` / its most_compiler). "
            "Alternatively produce lat-lon NetCDF/npz tables with ROCKE-3D or ExoPlaSim elsewhere and pass "
            "--climate-table PATH (see load_climate_table)."
        )
    import exoplasim
    # Minimal Model construction — real runs need many kwargs; this is the live hook.
    Model = getattr(exoplasim, "Model", None) or getattr(exoplasim, "ExoPlaSim", None)
    if Model is None:
        raise UnavailableError("exoplasim.Model not found in this install")
    m = Model(*args, **kwargs)
    return m


def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    def get(*names, default=None):
        for n in names:
            if n in keys: return np.asarray(d[n])
        if default is not None: return default
        raise KeyError(names)

    lat = get("lat", "latitude", "lat_deg")
    lon = get("lon", "longitude", "lon_deg", default=np.array([0.]))
    T = get("T", "T_K", "tsurf", "tas")
    alb = get("albedo", "alb", "planetary_albedo", default=np.full_like(T, .3, dtype=float))
    olr = get("OLR", "olr", default=np.full_like(T, 240., dtype=float))
    precip = get("precip", "precipitation", "pr", default=np.zeros_like(T, dtype=float))
    return dict(lat=np.asarray(lat, float), lon=np.asarray(lon, float), T=np.asarray(T, float),
                albedo=np.asarray(alb, float), OLR=np.asarray(olr, float), precip=np.asarray(precip, float),
                source=str(d["source"]) if "source" in keys else os.path.basename(path))


def _load_csv(path):
    """CSV with columns lat,lon,T,albedo,OLR,precip (header required)."""
    import csv
    rows = list(csv.DictReader(open(path)))
    def col(name, *alts, default=None):
        for n in (name,) + alts:
            if rows and n in rows[0]: return np.array([float(r[n]) for r in rows], float)
        if default is not None: return np.full(len(rows), default)
        raise KeyError(name)
    lat = col("lat", "latitude"); lon = col("lon", "longitude", default=0.)
    T = col("T", "T_K", "tsurf"); alb = col("albedo", "alb", default=.3)
    olr = col("OLR", "olr", default=240.); pr = col("precip", "pr", default=0.)
    # pivot to unique lat/lon grid if possible
    ulat = np.unique(lat); ulon = np.unique(lon)
    if len(ulat) * len(ulon) == len(lat):
        shape = (len(ulat), len(ulon))
        def grid(v):
            g = np.zeros(shape); 
            for la, lo, vv in zip(lat, lon, v):
                g[np.searchsorted(ulat, la), np.searchsorted(ulon, lo) % len(ulon)] = vv
            return g
        return dict(lat=ulat, lon=ulon, T=grid(T), albedo=grid(alb), OLR=grid(olr), precip=grid(pr), source=os.path.basename(path))
    return dict(lat=lat, lon=lon, T=T, albedo=alb, OLR=olr, precip=pr, source=os.path.basename(path))


def _load_netcdf(path):
    try:
        from netCDF4 import Dataset
    except ImportError as e:
        raise UnavailableError(f"netCDF4 not installed; convert the climate table to npz/csv or pip install netCDF4 ({e})") from e
    ds = Dataset(path)
    def find(*cands):
        for c in cands:
            if c in ds.variables: return np.asarray(ds.variables[c][:])
        # case-insensitive
        lower = {k.lower(): k for k in ds.variables}
        for c in cands:
            if c.lower() in lower: return np.asarray(ds.variables[lower[c.lower()]][:])
        raise KeyError(cands)
    lat = find("lat", "latitude", "lat_deg")
    try: lon = find("lon", "longitude", "lon_deg")
    except KeyError: lon = np.array([0.])
    T = find("T", "T_K", "tsurf", "ts", "tas")
    # squeeze time if present
    while T.ndim > 2: T = T[-1] if T.shape[0] < T.shape[-1] else T[..., -1]  # prefer last time
    try: alb = find("albedo", "alb", "as", "rsut")
    except KeyError: alb = np.full(T.shape, .3)
    try: olr = find("OLR", "olr", "rlut", "toa_lw")
    except KeyError: olr = np.full(T.shape, 240.)
    try: precip = find("precip", "pr", "precipitation")
    except KeyError: precip = np.zeros(T.shape)
    while alb.ndim > T.ndim: alb = alb[-1]
    while olr.ndim > T.ndim: olr = olr[-1]
    while precip.ndim > T.ndim: precip = precip[-1]
    ds.close()
    return dict(lat=np.asarray(lat, float), lon=np.asarray(lon, float), T=np.asarray(T, float),
                albedo=np.asarray(alb, float), OLR=np.asarray(olr, float), precip=np.asarray(precip, float),
                source=os.path.basename(path))


def load_climate_table(path):
    """Load lat-lon climate fields from npz / csv / NetCDF. ROCKE-3D & ExoPlaSim are external producers of these tables."""
    path = os.path.abspath(path)
    if not os.path.isfile(path): raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith(".npz"): return _load_npz(path)
    if low.endswith(".csv"): return _load_csv(path)
    if low.endswith(".nc") or low.endswith(".nc4") or low.endswith(".netcdf"): return _load_netcdf(path)
    # try npz then netcdf
    try: return _load_npz(path)
    except Exception:
        return _load_netcdf(path)


def build_example_table(path=EXAMPLE_TABLE, nlat=12, nlon=24, T_mean=288., amp=30.):
    """Tiny synthetic lat-lon grid for tests (no external GCM)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    T = T_mean - amp * np.sin(np.radians(LAT)) ** 2 + 2 * np.cos(np.radians(LON))
    alb = .3 + .4 * np.clip((250 - T) / 40., 0, 1)  # brighter when cold
    olr = 5.67e-8 * T ** 4 * .9
    precip = np.maximum(0., 2e-5 * (T - 260.)) * (1 - alb)
    np.savez_compressed(path, lat=lat, lon=lon, T=T, albedo=alb, OLR=olr, precip=precip,
                        source="synthetic_example", note="stand-in for ROCKE-3D/ExoPlaSim offline products")
    return path


class TableClimate:
    """Drop-in climate stepper driven by an external lat-lon table (ROCKE-3D / ExoPlaSim products).

    Matches LatClimate call shape: step(h, hw, P, dt, stress, t) -> ClimateStress.
    """
    def __init__(self, table, h=None, orbit=None):
        if isinstance(table, (str, bytes, os.PathLike)): table = load_climate_table(table)
        self.tab = table; self.orbit = orbit
        self.lat = np.asarray(table["lat"], float); self.lon = np.asarray(table["lon"], float)
        self.Tg = np.asarray(table["T"], float)
        if self.Tg.ndim == 1: self.Tg = self.Tg[:, None]
        self.alb = np.asarray(table["albedo"], float)
        self.olr = np.asarray(table["OLR"], float)
        self.pr = np.asarray(table["precip"], float)
        for name in ("alb", "olr", "pr"):
            v = getattr(self, name)
            if v.ndim == 1:
                setattr(self, name, v.reshape(self.Tg.shape) if v.size == self.Tg.size else np.broadcast_to(v, self.Tg.shape).copy())
            elif v.shape != self.Tg.shape:
                setattr(self, name, np.broadcast_to(v, self.Tg.shape).copy())
        self.n = int(self.Tg.shape[0]); self.nlon = int(self.Tg.shape[1])
        self.T = self.Tg.mean(1).astype(float); self.Td = self.T.copy()
        self.pCO2 = 3e-4; self.t = 0.; self.rt = False
        self.x_o2 = 0.; self.x_ch4 = 1e-7; self.tech_co2_mult = 1.; self.limit_cycle = None
        ice = (self.T < 271.).astype(float)
        self.last = dict(toa_imbalance=0., ice=ice, climate_table=self.tab.get("source", "table"))

    def set_trace(self, x_o2=None, x_ch4=None):
        if x_o2 is not None: self.x_o2 = float(x_o2)
        if x_ch4 is not None: self.x_ch4 = float(x_ch4)

    def step(self, h, hw, LP=None, dt_w_yr=1., stress=None, t=None):
        from ...leaves.packets import ClimateStress
        stress = dict(stress or {}); t = 0. if t is None else float(t); self.t = t
        dT = float(stress.get("dT", 0.) or 0.) + float(stress.get("impact_dT", 0.) or 0.)
        Tg = self.Tg + dT
        self.T = Tg.mean(1).astype(float)
        Tmean = float(np.mean(self.T))
        if not np.isfinite(Tmean): Tmean = 288.
        water = float(np.mean(np.clip((self.T - 273.15) / 20. + .5, 0, 1)))
        band = "temperate"
        if Tmean < 250: band = "snowball"
        elif Tmean > 320: band = "hot"
        orb = self.orbit.at(t) if self.orbit is not None else dict(
            e=float(getattr(h, "e", 0.)), varpi=0., obliquity=float(abs(getattr(h, "tilt", 0.))))
        ice = (self.T < 271.).astype(float)
        asr = float(np.mean((1 - self.alb) * 340.))
        self.last = dict(toa_imbalance=float(np.mean(self.olr) - asr), ice=ice,
                         climate_table=self.tab.get("source", "table"))
        lat_T = self.T.tolist()
        stressors = dict(climate_model="climate_table", lat_deg=self.lat.tolist(), lat_T=lat_T,
                         lat_water=[float(np.clip((tt - 273.15) / 20. + .5, 0, 1)) for tt in self.T],
                         lat_ice=ice.tolist(),
                         grid_T=Tg.tolist(), grid_albedo=np.asarray(self.alb, float).tolist(),
                         grid_OLR=np.asarray(self.olr, float).tolist(),
                         grid_precip=np.asarray(self.pr, float).tolist(),
                         pCO2_bar=float(self.pCO2),
                         ecc=float(orb["e"]), obliquity_deg=float(np.degrees(orb["obliquity"])),
                         toa_imbalance=self.last["toa_imbalance"], table_source=self.tab.get("source", "table"))
        return ClimateStress(Tmean, water, band, stressors)


class TableCarbon:
    """Minimal carbon-cycle stub compatible with Biogeochem / HabitatRun (weathering held fixed)."""
    def __init__(self, pCO2=3e-4):
        from ...leaves.world import P_REF
        self.C_atm = float(pCO2) / float(P_REF); self.C_carb = 0.; self.fossil = 0.
        self.cum_V = 0.; self.cum_W = 0.; self.cum_tech = 0.; self.V0 = 1e-6; self.volc = 1.
        self.ocean = .7; self.land = .3; self.last = {}
        self._pref = float(P_REF)

    @property
    def C_inorg(self):
        return self.C_atm * (1 + 10. * self.ocean)

    @C_inorg.setter
    def C_inorg(self, v):
        self.C_atm = max(1e-6, v / (1 + 10. * self.ocean))

    def pCO2(self, P_REF=None):
        return float((P_REF if P_REF is not None else self._pref) * self.C_atm)

    def step(self, lat, P, dt, tech_level=0., fossil_int=0.):
        lat.pCO2 = self.pCO2()
        self.last = dict(outgas=0., weathering=0., tech_co2=0., frost_sink=0.)
        return dict(pCO2_bar=float(lat.pCO2), C_atm=float(self.C_atm), C_inorg=float(self.C_inorg),
                    C_carb=float(self.C_carb), outgas_flux=0., weathering_flux=0., tech_co2_flux=0.,
                    frost_flux=0., fossil_left=float(self.fossil))

    def total_C(self):
        return float(self.C_inorg + self.C_carb + self.fossil)


def init_table_climate(h, hw, LP=None, orbit=None, path=None):
    path = path or EXAMPLE_TABLE
    if not os.path.isfile(path): build_example_table(path)
    clim = TableClimate(path, h=h, orbit=orbit)
    # seed pCO2 from table mean T via a mild greenhouse guess
    cc = TableCarbon(pCO2=float(getattr(clim, "pCO2", 3e-4)))
    clim.pCO2 = cc.pCO2()
    return clim, cc
