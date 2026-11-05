"""Out-of-leaf numerical solvers (opt-in). See README "Solvers (out-of-leaf)".

rt             two-stream k-distribution radiative transfer (OLR/albedo tables, transmission line depths)
climate2d      lat x lon energy-balance GCM (+ rt, seasons, land-cell weathering)      climate_model="gcm2d"
foodweb        mass-conserving trophic guild model or N-species size spectrum         solvers["foodweb"] (True/'guild'|'species')
secular_nbody  Wisdom-Holman N-body / Laplace-Lagrange orbit + spin drivers (+ tides)  solvers["orbit_solver"] (+ "orbit_tides": on|moon)
stellar_track  pre-MS + one-zone MS + subgiant/RGB/HB/AGB/TP-AGB + WD cooling, Z-scaled solvers["stellar_solver"]
gr             Kerr geodesics, ISCO, Peters inspiral, remnant + ringdown, PW pull   config "gr_solver" (events; True=tags, "dynamics")

Nothing here is imported by the leaves unless one of the switches above is set."""

def add_args(ap):
    """CLI flags shared by run_life / run_life_live (defaults = leaf behaviour)."""
    ap.add_argument("--climate-model", default="lat", choices=["lat", "box", "gcm2d"], help="lat (default EBM), box (v1 0-D), gcm2d (solvers.climate2d)")
    ap.add_argument("--rt", action="store_true", help="two-stream radiative transfer in the gcm2d climate (needs --climate-model gcm2d)")
    ap.add_argument("--gcm-physics", default=None, help="gcm2d extras, comma list of clouds,diurnal,winds or 'all' (needs --climate-model gcm2d)")
    ap.add_argument("--stellar-solver", action="store_true", help="host star follows solvers.stellar_track (needs stellar_evolve > 0 in the params to age)")
    ap.add_argument("--foodweb", nargs="?", const="guild", default=None, choices=["guild", "species"],
                    help="food web (solvers.foodweb) instead of the leaf biosphere: guild (default when given) or species (N-species size spectrum)")
    ap.add_argument("--orbit-solver", default=None, choices=["nbody", "secular", "off"], help="orbit/spin drivers from solvers.secular_nbody")
    ap.add_argument("--orbit-tides", default=None, choices=["on", "moon"], help="constant-lag tides on the orbit/spin drivers (needs --orbit-solver nbody|secular); moon = + satellite torque")

def from_args(a):
    """-> (climate_model, solvers dict or None)."""
    sv = {k: v for k, v in dict(rt=a.rt, gcm_physics=getattr(a, "gcm_physics", None), stellar_solver=a.stellar_solver, foodweb=a.foodweb, orbit_solver=a.orbit_solver,
                                       orbit_tides=getattr(a, "orbit_tides", None)).items() if v}
    if sv.get("orbit_tides") and sv.get("orbit_solver") not in ("nbody", "secular"): raise SystemExit("--orbit-tides needs --orbit-solver nbody|secular")
    if sv.get("rt") and a.climate_model != "gcm2d": raise SystemExit("--rt needs --climate-model gcm2d")
    if sv.get("gcm_physics") and a.climate_model != "gcm2d": raise SystemExit("--gcm-physics needs --climate-model gcm2d")
    return a.climate_model, (sv or None)
