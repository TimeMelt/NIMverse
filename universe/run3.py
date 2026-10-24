"""Generate, simulate and render a v2 universe.  python -m universe.run3 [--seed 42] [--params learned_params_v3.json] [--force-encounter] [--focus encounter]"""
import sys, os, json, argparse, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from universe.sim2 import DEFAULTS, simulate
from universe.score2 import score
from universe.nim_rng import NIMRandom
from universe.render3 import render_movie

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config3.json"))
ap.add_argument("--seed", type=int); ap.add_argument("--params"); ap.add_argument("--tag", default="")
ap.add_argument("--galaxies", type=int); ap.add_argument("--stars", type=int); ap.add_argument("--frames", type=int)
ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--rng", choices=["nim", "numpy"])
ap.add_argument("--force-encounter", action="store_true"); ap.add_argument("--focus", choices=["all", "encounter"], default="all")
ap.add_argument("--tide-boost", type=float); ap.add_argument("--gravity", choices=["direct", "bh"]); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--no-render", action="store_true")
a = ap.parse_args(); C = json.load(open(a.config))
for k, v in dict(seed=a.seed, n_galaxies=a.galaxies, stars_per_galaxy=a.stars, frames=a.frames, width=a.width, height=a.height, rng=a.rng, tide_boost=a.tide_boost, gravity=a.gravity, force_encounter=a.force_encounter or None).items():
    if v is not None: C[k] = v
P = dict(DEFAULTS); label = "defaults"
if a.params: P.update(json.load(open(a.params))["params"]); label = "NIM-learned"
t0 = time.time(); r = NIMRandom(C["seed"]) if C["rng"] == "nim" else np.random.default_rng(C["seed"])
Gx, S, gx, gv, sx, sv, host = simulate(r, C, P, C["frames"]); print(f"sim {time.time()-t0:.1f}s: {len(Gx['m'])} galactic bodies, {len(S['m'])} system bodies, binary={S['binary']}, belt={S['belt'] is not None}, encounter={Gx['encounter']}", flush=True)
print("tides:", S.get("tide"))
sc = score(Gx, S, gx, gv, sx, sv); print("score:", {k: round(v, 3) for k, v in sc.items()})
o = C["out_dir"]; os.makedirs(o, exist_ok=True); pre = f"{o}/universe_v3_seed{C['seed']}{a.tag}"
json.dump(dict(config=C, params=P, score=sc, tide=S.get("tide"), system_bodies=list(S["name"]), encounter=Gx["encounter"]), open(pre+".json", "w"), indent=1)
np.savez_compressed(pre+".npz", galaxy_frames=gx.astype(np.float32), system_frames=sx.astype(np.float32), kind=Gx["kind"], gid=Gx["gid"], smass=Gx["smass"])
if a.no_render: sys.exit()
t1 = time.time(); render_movie(Gx, S, gx, sx, host, pre, C["width"], C["height"], focus=a.focus, workers=a.workers, title=f"NIM universe v3 | seed {C['seed']} | {label} | {C['n_galaxies']} galaxies, {len(Gx['m'])} bodies | score {sc['total']:.3f}")
print(f"render {time.time()-t1:.0f}s"); print(f"done in {time.time()-t0:.0f}s -> {pre}.png/.mp4/.gif")
