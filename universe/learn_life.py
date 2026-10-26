"""Life-mesh trainer (lightweight numpy SPSA over normalized LIFE_SPACE; the QrnLOpt/pennylane meshes of
learn_alt2 are too heavy for this inner loop).

Loss per seed = |habitability residual| + |eco_stability residual| + 2*planted_recovery_fail + W_EXT*extinction
  extinction = 1 if founded population/biosphere is dead at end of run, else max(0, 1 - min_pop_after_founding/POP_THRESH)*0.5
Planted runs: plant=True, disturb_at=years/2, culture=False, model=individuals (or --model cohorts).
python -m universe.learn_life --iters 24 --years 3000 --seed-list 3,4,6,7,8,12,13"""
import argparse, fnmatch, json, os, time, numpy as np
from .sim2 import make_system, DEFAULTS
from .leaves import LIFE_SPACE, LIFE_DEFAULTS, LifeParamBus, ScoreBus
from .leaves.life_loop import host_weather, run_habitat
from .leaves.system_habitat import habitats_from_system, candidate
from .leaves import world

KEYS = list(LIFE_SPACE); LO = np.array([LIFE_SPACE[k][0] for k in KEYS]); HI = np.array([LIFE_SPACE[k][1] for k in KEYS])
W_EXT = 2.0; POP_THRESH = 10
DEFAULT_FREEZE = "stellar_evolve,stellar_age_Gyr,stellar_age_rate,tech_*,impact_scale,impact_kill"
# impacts frozen at LIFE_DEFAULTS (impact_scale=1, impact_kill=.5): the 2026-10 run drove impact_scale->~0 to dodge
# extinctions (unphysical).  LIFE_SPACE also floors impact_scale at .25 so even --freeze '' cannot switch impacts off.
def decode(u): return {k: float(v) for k, v in zip(KEYS, LO + np.clip(u, 0, 1)*(HI-LO))}
def encode(P): return np.array([(P[k]-LIFE_SPACE[k][0])/(LIFE_SPACE[k][1]-LIFE_SPACE[k][0]) for k in KEYS])
def freeze_mask(spec):
    pats = [p.strip() for p in (spec or "").split(",") if p.strip()]
    return np.array([not any(fnmatch.fnmatch(k, p) for p in pats) for k in KEYS], float)

def extinction(r):
    """0 = healthy, 1 = founded life went extinct; partial penalty for near-collapse (min pop < POP_THRESH)."""
    ev = r.get("evolution")
    if ev:
        ns = [x["n_alive"] for x in ev.get("evolution_series") or []]
        if ev.get("n_alive", 0) == 0 or not r["biosphere"].get("alive", True): return 1.
        return 0.5*max(0., 1 - min(ns or [POP_THRESH])/POP_THRESH)
    return 0. if r["biosphere"].get("alive", True) else 1.

_REF = LifeParamBus(LIFE_DEFAULTS)
def plantable(cands, hw):
    """Drop candidates that are outside the HZ AND have no liquid water under the fixed LIFE_DEFAULTS climate
    (e.g. seed 4 planet3: 187 K ice, a=4.4 > hz_out 3.53 but inside the 35% candidate margin).  Planting there is a
    guaranteed death independent of the trained params, so it only adds a constant (or rewards greenhouse cheats).
    The reference is LIFE_DEFAULTS, not P, so the filter cannot be optimized."""
    return [h for h in cands if h.in_hz or world.step(h, hw, _REF).liquid_water_frac > 0]

def loss(P, seeds, C, years, dt, model="individuals", culture=False, detail=False):
    LP = LifeParamBus(P); L = []; parts = []; per = {}
    for s in seeds:
        S = make_system(np.random.default_rng(s), C, DEFAULTS); hw = host_weather(S); habs = habitats_from_system(S, s, LP)
        hab = abs(ScoreBus.habitability([world.step(h, hw, LP) for h in habs]))
        cands = plantable([h for h in habs if candidate(h)], hw) or sorted(habs, key=lambda h: abs(np.log(h.a/np.sqrt(h.hz_in*h.hz_out))))[:1]
        rng = np.random.default_rng(1000+s); eco = rec = ext = 0.; mins = []; sp = 0
        for h in cands:
            r = run_habitat(h, hw, LP, rng, years, dt, plant=True, disturb_at=years/2, culture=culture, model=model)
            eco += abs(r["score"]["eco_stability"]); rec += r["score"]["planted_life"]; ext += extinction(r)
            ev = r.get("evolution") or {}; mins.append(min([x["n_alive"] for x in ev.get("evolution_series") or []] or [0])); sp = max(sp, ev.get("n_species_ever", 0))
        n = len(cands); eco /= n; rec /= n; ext /= n
        parts.append((hab, eco, rec, ext)); L.append(hab + eco + 2*rec + W_EXT*ext)
        per[s] = dict(hab=hab, eco=eco, rec=rec, ext=ext, min_pop=min(mins), species_ever=sp)
    out = (float(np.mean(L)), np.mean(parts, 0).tolist())
    return out + (per,) if detail else out

def parse_seeds(a):
    return [int(x) for x in a.seed_list.split(",")] if a.seed_list else list(range(100, 100+a.seeds))

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=6); ap.add_argument("--years", type=float, default=600.)
    ap.add_argument("--dt", type=float, default=10.); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-list", default=None, help="comma list, overrides --seeds")
    ap.add_argument("--model", default="individuals", choices=["individuals", "cohorts"])
    ap.add_argument("--freeze", default=DEFAULT_FREEZE, help="comma globs of LIFE_SPACE keys to hold at LIFE_DEFAULTS ('' = none)")
    ap.add_argument("--lr", type=float, default=.05); ap.add_argument("--ck", type=float, default=.08)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "learned_life.json"))
    a = ap.parse_args(argv); C = json.load(open(os.path.join(os.path.dirname(__file__), "config3.json"))); C = {**C, "asteroids": 20}
    seeds = parse_seeds(a); rng = np.random.default_rng(a.seed); u = encode(LIFE_DEFAULTS); m = freeze_mask(a.freeze)
    L = lambda uu: loss(decode(uu), seeds, C, a.years, a.dt, a.model)
    log = open(a.out.replace(".json", ".log"), "w"); w = lambda s: (print(s, flush=True), log.write(s+"\n"), log.flush())
    t0 = time.time(); f0, p0 = L(u); best = (f0, u.copy(), p0); hist = [dict(it=0, loss=f0, parts=p0)]
    w(f"[learn_life] seeds={seeds} years={a.years} model={a.model} free={int(m.sum())}/{len(KEYS)}")
    w(f"[learn_life] it0 loss={f0:.4f} parts(hab,eco,rec,ext)={np.round(p0, 4).tolist()} ({time.time()-t0:.0f}s)")
    def ckpt(it, final=False):
        res = dict(params=decode(best[1]), loss0=f0, best_loss=best[0],
                   best_parts=dict(zip(("habitability", "eco", "planted_recovery", "extinction"), best[2])),
                   history=hist, seeds=seeds, years=a.years, iters=it, model=a.model, freeze=a.freeze, w_ext=W_EXT,
                   wall_s=time.time()-t0, complete=final, method="numpy SPSA over normalized LIFE_SPACE (frozen keys masked)")
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); return res
    for it in range(1, a.iters+1):
        ck = a.ck/it**.101; lr = a.lr/(it+2)**.602; d = rng.choice([-1., 1.], len(u))*m
        fp, _ = L(u+ck*d); fm, _ = L(u-ck*d)
        g = (fp-fm)/(2*ck)*d; u = np.clip(u - lr*g/max(1., np.abs(g).max()), 0, 1)
        f, p = L(u); hist.append(dict(it=it, loss=f, parts=p))
        if f < best[0]: best = (f, u.copy(), p)
        ckpt(it)
        w(f"[learn_life] it{it} loss={f:.4f} parts={np.round(p, 4).tolist()} best={best[0]:.4f} ({time.time()-t0:.0f}s)")
    res = ckpt(a.iters, True); w(f"wrote {a.out}"); return res

if __name__ == "__main__": main()
