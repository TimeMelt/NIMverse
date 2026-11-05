"""Out-of-leaf numerical solvers (opt-in). See README "Solvers (out-of-leaf)".

rt             two-stream k-distribution radiative transfer (OLR/albedo tables, transmission line depths)
climate2d      lat x lon energy-balance GCM (+ rt, seasons, land-cell weathering)      climate_model="gcm2d"
foodweb        mass-conserving trophic guild model (drop-in for biosphere.step)       solvers["foodweb"]
secular_nbody  Wisdom-Holman N-body / Laplace-Lagrange orbit + spin drivers             solvers["orbit_solver"]
stellar_track  one-zone MS burning + post-MS (subgiant/RGB/HB) stellar track            solvers["stellar_solver"]
gr             Kerr geodesics, ISCO, periastron advance, Peters inspiral              config "gr_solver" (events)

Nothing here is imported by the leaves unless one of the switches above is set."""

def add_args(ap):
    """CLI flags shared by run_life / run_life_live (defaults = leaf behaviour)."""
    ap.add_argument("--climate-model", default="lat", choices=["lat", "box", "gcm2d"], help="lat (default EBM), box (v1 0-D), gcm2d (solvers.climate2d)")
    ap.add_argument("--rt", action="store_true", help="two-stream radiative transfer in the gcm2d climate (needs --climate-model gcm2d)")
    ap.add_argument("--stellar-solver", action="store_true", help="host star follows solvers.stellar_track (needs stellar_evolve > 0 in the params to age)")
    ap.add_argument("--foodweb", action="store_true", help="trophic guild food web (solvers.foodweb) instead of the leaf biosphere")
    ap.add_argument("--orbit-solver", default=None, choices=["nbody", "secular", "off"], help="orbit/spin drivers from solvers.secular_nbody")

def from_args(a):
    """-> (climate_model, solvers dict or None)."""
    sv = {k: v for k, v in dict(rt=a.rt, stellar_solver=a.stellar_solver, foodweb=a.foodweb, orbit_solver=a.orbit_solver).items() if v}
    if sv.get("rt") and a.climate_model != "gcm2d": raise SystemExit("--rt needs --climate-model gcm2d")
    return a.climate_model, (sv or None)
