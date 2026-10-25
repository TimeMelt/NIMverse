"""CLI: make a small planetary system (sim2.make_system + config defaults) and run the life leaves on it.
python -m universe.run_life --seed 7 [--years 2000 --dt 10 --plant --rng numpy|nim]"""
import argparse, json, os, numpy as np
from .sim2 import make_system, DEFAULTS
from .leaves import life_loop

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--years", type=float, default=2000.)
    ap.add_argument("--dt", type=float, default=10.); ap.add_argument("--plant", action="store_true"); ap.add_argument("--rng", default="numpy")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config3.json")); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv); C = json.load(open(a.config)); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    if a.rng == "nim": from .nim_rng import NIMRandom
    r = NIMRandom(a.seed) if a.rng == "nim" else np.random.default_rng(a.seed)
    S = make_system(r, C, DEFAULTS); rep = life_loop(S, a.seed, None, a.years, a.dt, a.plant)
    out = a.out or os.path.join(C.get("out_dir", "/workspace/nim_universe/out"), f"life_demo_seed{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1)
    print(f"wrote {out}; planets={len(rep['planets'])} with_life={rep['with_life']}"); return rep

if __name__ == "__main__": main()
