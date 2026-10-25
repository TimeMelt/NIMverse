"""Life-mesh trainer (lightweight numpy SPSA over normalized LIFE_SPACE; the QrnLOpt/pennylane meshes of
learn_alt2 are too heavy for this inner loop).  Loss = |habitability residual| + |eco_stability residual|
+ planted-seed recovery (ScoreBus.planted_life after a mid-run 80% disturbance) averaged over fixed train seeds.
python -m universe.learn_life --iters 6 --years 600 --seeds 3"""
import argparse, json, os, time, numpy as np
from .sim2 import make_system, DEFAULTS
from .leaves import LIFE_SPACE, LIFE_DEFAULTS, LifeParamBus, ScoreBus
from .leaves.life_loop import host_weather, run_habitat
from .leaves.system_habitat import habitats_from_system, candidate
from .leaves import world

KEYS = list(LIFE_SPACE); LO = np.array([LIFE_SPACE[k][0] for k in KEYS]); HI = np.array([LIFE_SPACE[k][1] for k in KEYS])
def decode(u): return {k: float(v) for k, v in zip(KEYS, LO + np.clip(u, 0, 1)*(HI-LO))}
def encode(P): return np.array([(P[k]-LIFE_SPACE[k][0])/(LIFE_SPACE[k][1]-LIFE_SPACE[k][0]) for k in KEYS])

def loss(P, seeds, C, years, dt):
    LP = LifeParamBus(P); L = []; parts = []
    for s in seeds:
        S = make_system(np.random.default_rng(s), C, DEFAULTS); hw = host_weather(S); habs = habitats_from_system(S, s, LP)
        hab = abs(ScoreBus.habitability([world.step(h, hw, LP) for h in habs]))
        cands = [h for h in habs if candidate(h)] or sorted(habs, key=lambda h: abs(np.log(h.a/np.sqrt(h.hz_in*h.hz_out))))[:1]
        rng = np.random.default_rng(1000+s); eco = rec = 0.
        for h in cands:
            r = run_habitat(h, hw, LP, rng, years, dt, plant=True, disturb_at=years/2, culture=False)
            eco += abs(r["score"]["eco_stability"]); rec += r["score"]["planted_life"]
        eco /= len(cands); rec /= len(cands); parts.append((hab, eco, rec)); L.append(hab + eco + 2*rec)
    return float(np.mean(L)), np.mean(parts, 0).tolist()

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=6); ap.add_argument("--years", type=float, default=600.)
    ap.add_argument("--dt", type=float, default=10.); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=.05); ap.add_argument("--ck", type=float, default=.08)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "learned_life.json"))
    a = ap.parse_args(argv); C = json.load(open(os.path.join(os.path.dirname(__file__), "config3.json"))); C = {**C, "asteroids": 20}
    seeds = list(range(100, 100+a.seeds)); rng = np.random.default_rng(a.seed); u = encode(LIFE_DEFAULTS)
    f0, p0 = loss(decode(u), seeds, C, a.years, a.dt); best = (f0, u.copy(), p0); hist = [dict(it=0, loss=f0, parts=p0)]
    log = open(a.out.replace(".json", ".log"), "w"); w = lambda s: (print(s), log.write(s+"\n"), log.flush())
    w(f"[learn_life] it0 loss={f0:.4f} parts(hab,eco,rec)={np.round(p0, 4).tolist()}"); t0 = time.time()
    for it in range(1, a.iters+1):
        ck = a.ck/it**.101; lr = a.lr/(it+2)**.602; d = rng.choice([-1., 1.], len(u))
        fp, _ = loss(decode(u+ck*d), seeds, C, a.years, a.dt); fm, _ = loss(decode(u-ck*d), seeds, C, a.years, a.dt)
        g = (fp-fm)/(2*ck)*d; u = np.clip(u - lr*g/max(1., np.abs(g).max()), 0, 1)
        f, p = loss(decode(u), seeds, C, a.years, a.dt); hist.append(dict(it=it, loss=f, parts=p))
        if f < best[0]: best = (f, u.copy(), p)
        w(f"[learn_life] it{it} loss={f:.4f} parts={np.round(p, 4).tolist()} best={best[0]:.4f} ({time.time()-t0:.0f}s)")
    res = dict(params=decode(best[1]), loss0=f0, best_loss=best[0], best_parts=dict(zip(("habitability", "eco", "planted_recovery"), best[2])),
               history=hist, seeds=seeds, years=a.years, method="numpy SPSA over normalized LIFE_SPACE")
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); w(f"wrote {a.out}"); return res

if __name__ == "__main__": main()
