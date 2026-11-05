"""Seed-4 short-horizon health smoke: python -m universe.smoke_physics15 [--seed 4 --years 6000 --dt 20]"""
import argparse, json, os, time, numpy as np
from .sim2 import make_system, DEFAULTS
from .leaves import life_loop

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4); ap.add_argument("--years", type=float, default=6000.)
    ap.add_argument("--dt", type=float, default=20.); ap.add_argument("--out", default=None); a = ap.parse_args(argv)
    C = json.load(open(os.path.join(os.path.dirname(__file__), "config3.json"))); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    S = make_system(np.random.default_rng(a.seed), C, DEFAULTS); t0 = time.time(); rows = []
    rep = life_loop(S, a.seed, None, a.years, a.dt, record=rows.append)
    for p in rep["planets"]:
        if "skipped" in p: continue
        nm = p["habitat"]["name"]; r = [x for x in rows if x["planet"] == nm]; ev = p.get("evolution") or {}
        n = [x["n_org"] for x in r]; T = [x["T"] for x in r]
        print(f"{nm}: alive={p['biosphere']['alive']} biomass={p['biomass']:.3f} T={np.mean(T):.1f}[{min(T):.1f},{max(T):.1f}] water_end={r[-1]['water']:.2f} "
              f"n_org end={n[-1]:.0f} min(after found)={min([v for v in n if v>0] or [0]):.0f} species={ev.get('n_species_alive', ev.get('species_alive'))} "
              f"o2={p['biosign']['o2']:.3f} ch4={p['biosign']['ch4']:.2e} clim={p['climate']['climate_class']} extra={ {k: p['climate']['stressors'].get(k) for k in ('pCO2_bar','T_eq_lat','ice_lat_deg','obliquity_deg','ecc','O2_pool','CH4_pool','growth_mult','C_drift','C_total')} }")
    print(f"with_life={rep['with_life']} elapsed={time.time()-t0:.1f}s")
    if a.out: json.dump(rep, open(a.out, "w"), indent=1, default=float)
    return rep

if __name__ == "__main__": main()
