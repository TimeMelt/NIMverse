"""Out-of-leaf numerical solvers (opt-in). See README "Solvers (out-of-leaf)".

rt             two-stream k-distribution radiative transfer (OLR/albedo tables, transmission line depths)
climate2d      lat x lon energy-balance GCM (+ rt, seasons, land-cell weathering)      climate_model="gcm2d"
foodweb        mass-conserving trophic guild model or N-species size spectrum         solvers["foodweb"] (True/'guild'|'species')
secular_nbody  Wisdom-Holman N-body / Laplace-Lagrange orbit + spin drivers (+ tides)  solvers["orbit_solver"] (+ "orbit_tides": on|moon)
stellar_track  pre-MS + MS + subgiant/RGB/HB/AGB/TP-AGB + WD, Z-scaled   solvers["stellar_solver"] (True/'semi'|'mesa_lite'|'mist'|'mesa', + "stellar_Z")
gr             Kerr geodesics, ISCO, Peters inspiral, remnant + ringdown, PW pull   config "gr_solver" (events; True=tags, "dynamics"; LAL Phenom when available)
ext/           MIST/MESA/REBOUND/LALSuite/ExoPlaSim + climate/RT/NR table loaders   see README "External stacks"

Nothing here is imported by the leaves unless one of the switches above is set."""

def add_args(ap):
    """CLI flags shared by run_life / run_life_live (defaults = leaf behaviour)."""
    ap.add_argument("--climate-model", default="lat", choices=["lat", "box", "gcm2d", "climate_table", "exoplasim"], help="lat (default EBM), box (v1 0-D), gcm2d (solvers.climate2d), climate_table (external NetCDF/npz via --climate-table), exoplasim (needs compiled binary)")
    ap.add_argument("--rt", action="store_true", help="two-stream radiative transfer in the gcm2d climate (needs --climate-model gcm2d)")
    ap.add_argument("--gcm-physics", default=None, help="gcm2d extras, comma list of clouds,diurnal,winds,dynamics,frost or 'all' (= clouds,diurnal,winds; needs --climate-model gcm2d)")
    ap.add_argument("--rt-buffer", default=None, metavar="{n2[:bar],h2[:bar[:x_h2]],auto}",
                    help="gcm2d RT background gas: n2 (3 bar N2), h2 (3 bar, 10%% H2, N2-H2 CIA), auto (ladder; first non-snowball else warmest); needs --rt")
    ap.add_argument("--stellar-solver", nargs="?", const="semi", default=None, choices=["semi", "mesa_lite", "mist", "mesa"],
                    help="host star track: semi (default when given), mesa_lite, mist (MIST/proxy), mesa (needs MESA_DIR); "
                         "needs stellar_evolve > 0 in the params to age")
    ap.add_argument("--stellar-z", type=float, default=None, help="host metallicity Z for --stellar-solver (default: system Z/feh, else solar 0.0142)")
    ap.add_argument("--foodweb", nargs="?", const="guild", default=None, choices=["guild", "species", "species-div"],
                    help="food web (solvers.foodweb) instead of the leaf biosphere: guild (default when given) or species (N-species size spectrum); species-div adds niche packing, immigration, self-limitation and T_opt evolution")
    ap.add_argument("--orbit-solver", default=None, choices=["nbody", "secular", "rebound", "off"], help="orbit/spin drivers from solvers.secular_nbody (or ext.rebound_orbit)")
    ap.add_argument("--orbit-tides", default=None, choices=["on", "moon", "moon-ring"], help="constant-lag tides on the orbit/spin drivers (needs --orbit-solver nbody|secular|rebound); moon = + satellite (N-body: massive moon particle), moon-ring = orbit-averaged satellite")
    ap.add_argument("--nbody-kyr", type=float, default=None, help="N-body window before the Laplace-Lagrange hand-over (kyr, default 20)")
    ap.add_argument("--megno-kyr", type=float, default=None, help="report the N-body system's MEGNO over this many kyr (needs --orbit-solver nbody)")
    ap.add_argument("--climate-table", default=None, metavar="PATH",
                    help="lat-lon NetCDF/npz/csv of T,albedo,OLR,precip from ROCKE-3D/ExoPlaSim offline runs (sets climate_model=climate_table if still lat)")
    ap.add_argument("--rt-table", default=None, metavar="PATH",
                    help="precomputed OLR/albedo npz (HELIOS/SOCRATES/petitRADTRANS offline or solvers.rt example) used as climate_tables drop-in")
    ap.add_argument("--nr-waveform", default=None, metavar="PATH",
                    help="SpEC/SXS/Einstein Toolkit strain ASCII/HDF5/npz; else LALSuite Phenom when gr_solver is on")

def from_args(a):
    """-> (climate_model, solvers dict or None)."""
    sv = {k: v for k, v in dict(rt=a.rt, gcm_physics=getattr(a, "gcm_physics", None), rt_buffer=getattr(a, "rt_buffer", None), stellar_solver=a.stellar_solver, stellar_Z=getattr(a, "stellar_z", None), foodweb=a.foodweb, orbit_solver=a.orbit_solver,
                                       orbit_tides=getattr(a, "orbit_tides", None), nbody_kyr=getattr(a, "nbody_kyr", None),
                                       megno_kyr=getattr(a, "megno_kyr", None),
                                       climate_table=getattr(a, "climate_table", None),
                                       rt_table=getattr(a, "rt_table", None),
                                       nr_waveform=getattr(a, "nr_waveform", None)).items() if v}
    cm = a.climate_model
    if sv.get("climate_table") and cm == "lat": cm = "climate_table"
    if sv.get("orbit_tides") and sv.get("orbit_solver") not in ("nbody", "secular", "rebound"): raise SystemExit("--orbit-tides needs --orbit-solver nbody|secular|rebound")
    if sv.get("rt") and cm != "gcm2d": raise SystemExit("--rt needs --climate-model gcm2d")
    if sv.get("gcm_physics") and cm != "gcm2d": raise SystemExit("--gcm-physics needs --climate-model gcm2d")
    if sv.get("rt_buffer") and not sv.get("rt"): raise SystemExit("--rt-buffer needs --rt (and --climate-model gcm2d)")
    if sv.get("rt_buffer") and sv["rt_buffer"] != "auto":
        from .climate2d import parse_buffer
        try: parse_buffer(sv["rt_buffer"])
        except ValueError as e: raise SystemExit(str(e))
    if sv.get("rt_table"):
        from .ext import rt_tables
        rt_tables.install_rt_table(sv["rt_table"])
    return cm, (sv or None)
