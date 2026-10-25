"""Thin hook galaxy/event pipeline -> life leaves.  make_system -> (optional) short leapfrog_tidal orbit integration
with a galactic tidal tensor (zero by default, or a constant toy tensor --tide) -> rebuild snapshot from the last frame
-> life_loop.  --host-mass overrides the host star mass (e.g. a star mass taken from events.simulate_ev output) and
--dm-frac applies a toy evolutionary mass-loss hint (HostWeather.dM_star).  Does not touch the event training scripts.
python -m universe.run_life_from_system --seed 4 --years 5000 --orbit-yr 20 [--tide 1e-6 --host-mass .8]"""
import argparse, json, os, numpy as np
from .sim2 import make_system, leapfrog_tidal, DEFAULTS
from .leaves import life_loop
from .leaves.radiation import Observer
from .run_life import common_args, load_lp, summarize

def evolve_system(S, orbit_yr=0., dt_yr=.002, tide=0.):
    """Integrate system orbits for orbit_yr years (system units: t = 2*pi*yr).  Returns new S + drift stats."""
    if orbit_yr <= 0: return S, {}
    x0, v0, m = np.asarray(S["x"], float), np.asarray(S["v"], float), np.asarray(S["m"], float)
    dt = 2*np.pi*dt_yr; steps = int(orbit_yr/dt_yr); T = np.diag([1., 1., -2.])*tide        # traceless toy tidal tensor
    fx, fv = leapfrog_tidal(x0, v0, m, dt, steps, 4, [T], 1.)
    S2 = dict(S); S2["x"], S2["v"] = fx[-1], fv[-1]
    pl = np.asarray(S["kind"]) == 1
    return S2, dict(orbit_yr=orbit_yr, steps=steps, tide=tide, max_planet_r_drift=float(np.abs(np.linalg.norm(fx[-1][pl], axis=1)-np.linalg.norm(x0[pl], axis=1)).max()))

def main(argv=None):
    ap = argparse.ArgumentParser(); common_args(ap)
    ap.add_argument("--orbit-yr", type=float, default=5.); ap.add_argument("--tide", type=float, default=0.)
    ap.add_argument("--host-mass", type=float, default=None); ap.add_argument("--dm-frac", type=float, default=0.)
    a = ap.parse_args(argv); C = json.load(open(a.config)); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    r = np.random.default_rng(a.seed); S = make_system(r, C, DEFAULTS); S, orb = evolve_system(S, a.orbit_yr, tide=a.tide)
    rep = life_loop(S, a.seed, load_lp(a.params), a.years, a.dt, a.plant, culture=a.culture, distance_pc=a.distance_pc, M_override=a.host_mass)
    if a.dm_frac: rep["host"]["dM_star"] += -a.dm_frac*rep["host"]["M_star"]
    rep["system_hook"] = dict(orbit=orb, host_mass_override=a.host_mass, dm_frac=a.dm_frac)
    out = a.out or os.path.join(C.get("out_dir", "/workspace/nim_universe/out"), f"life_from_system_seed{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1); summarize(rep, out); print("  orbit:", orb); return rep

if __name__ == "__main__": main()
