"""CLI: make a small planetary system (sim2.make_system + config defaults) and run the life leaves on it.
python -m universe.run_life --seed 7 [--years 2000 --dt 10 --plant --rng numpy|nim --no-culture --distance-pc 10
                                      --params results/learned_life.json]"""
import argparse, json, os, numpy as np
from .sim2 import make_system, DEFAULTS
from .leaves import life_loop

def load_lp(path):
    if not path: return None
    d = json.load(open(path)); return d.get("params", d)

def common_args(ap):
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--years", type=float, default=2000.)
    ap.add_argument("--dt", type=float, default=10.); ap.add_argument("--plant", action="store_true"); ap.add_argument("--rng", default="numpy")
    ap.add_argument("--culture", dest="culture", action="store_true", default=True); ap.add_argument("--no-culture", dest="culture", action="store_false")
    ap.add_argument("--distance-pc", type=float, default=10.); ap.add_argument("--params", default=None)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config3.json")); ap.add_argument("--out", default=None)

def summarize(rep, out):
    hf = rep["host_flares"]; rad = rep["radiation"]
    print(f"wrote {out}; planets={len(rep['planets'])} with_life={rep['with_life']} with_tech={rep['with_tech']}")
    print(f"  host flares={hf['n_flares']} uv_max={hf['uv_max']:.2f} dM={hf['dM_star']:.2e}; observer delay={rad['delay_yr']} detections={rad['n_detections']} first={list(rad['first'])[:6]}")

def main(argv=None):
    ap = argparse.ArgumentParser(); common_args(ap)
    a = ap.parse_args(argv); C = json.load(open(a.config)); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    if a.rng == "nim": from .nim_rng import NIMRandom
    r = NIMRandom(a.seed) if a.rng == "nim" else np.random.default_rng(a.seed)
    S = make_system(r, C, DEFAULTS)
    rep = life_loop(S, a.seed, load_lp(a.params), a.years, a.dt, a.plant, culture=a.culture, distance_pc=a.distance_pc)
    out = a.out or os.path.join(C.get("out_dir", "/workspace/nim_universe/out"), f"life_demo_seed{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1); summarize(rep, out); return rep

if __name__ == "__main__": main()
