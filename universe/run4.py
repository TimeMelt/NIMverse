"""Full universe with inceptron-driven high-energy events.  python -m universe.run4 --seed 7 --events learned_events_v4.json [--tag _events]"""
import sys, os, json, argparse, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from universe.events import EDEFAULTS, simulate_ev
from universe.score_ev import score_ev
from universe.nim_rng import NIMRandom
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--events"); ap.add_argument("--tag", default="_events")
ap.add_argument("--stars", type=int); ap.add_argument("--force-encounter", action="store_true")
a = ap.parse_args(); here = os.path.dirname(os.path.abspath(__file__)); C = json.load(open(f"{here}/config3.json")); C["seed"] = a.seed
if a.stars: C["stars_per_galaxy"] = a.stars
if a.force_encounter: C["force_encounter"] = True
o = C["out_dir"]; P = json.load(open(f"{o}/learned_params_v3.json"))["params"]; EP = dict(EDEFAULTS); lab = "defaults"
if a.events: EP.update(json.load(open(a.events if os.path.isabs(a.events) else f"{o}/{a.events}"))["event_params"]); lab = "learned"
if a.events and "params" in json.load(open(a.events if os.path.isabs(a.events) else f"{o}/{a.events}")): P = json.load(open(a.events if os.path.isabs(a.events) else f"{o}/{a.events}"))["params"]; lab = "alternating"
t0 = time.time(); Gx, S, gx, gv, sx, sv, host, gk, info = simulate_ev(NIMRandom(a.seed), C, P, EP, C["frames"], a.seed)
sc = score_ev(Gx, S, gx, gv, sx, sv, gk, info, C); print(f"sim {time.time()-t0:.0f}s", {k: round(float(v), 3) for k, v in sc.items()}, flush=True)
pre = f"{o}/universe_v4_seed{a.seed}{a.tag}"; cv = lambda x: x.tolist() if hasattr(x, "tolist") else x
json.dump(dict(config=C, params=P, event_params=EP, event_label=lab, score={k: float(v) for k, v in sc.items()}, encounter=Gx["encounter"], active={str(k): v for k, v in info["active"].items()}, smbh=[int(b) for b in info["smbh"]],
               events=[{k: cv(v) for k, v in e.items()} for e in info["log"]], system_bodies=list(S["name"])), open(pre+".json", "w"), indent=1)
np.savez_compressed(pre+".npz", galaxy_frames=gx.astype(np.float32), system_frames=sx.astype(np.float32), kind=Gx["kind"], gid=Gx["gid"], smass=Gx["smass"],
                    kind_frames=gk.astype(np.int8), m_final=info["m_final"], host=host)
from collections import Counter; print("events:", Counter(e["type"] for e in info["log"]), f"-> {pre}.json/.npz")
