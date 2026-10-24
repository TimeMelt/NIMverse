"""Alternating ("taking turns") training of the two NIM inceptron meshes.
Round r:  (a) structure mesh (3 nodes x 16 qubits, SPACE) trained with the event settings frozen, on the COMBINED
              objective score_ev.total = .4*structure + .6*events;
          (b) event mesh (2 nodes x 11 qubits, ESPACE + tde_eff + agn_duty_myr) trained with the new structure frozen.
Both meshes warm-start from their previous weights every round (round 0 starts from weights fitted to reproduce the
one-way v3 structure settings and v4 event settings, new keys at mid-range).  Recipe per node-epoch = learn3/learn_ev:
SPSA gradient on fixed train universes (NIM RNG), QrnlStepOptimizer step with peer weights from the mesh, decayed lr,
best-on-validation.  Weight exchange runs over ONE NIM ServerClass server (TCP BaseManager) with 5 queues: 0-2 structure,
3-4 events; every put/get is issued from the main thread in fixed order -> deterministic.  Universe sims are farmed out to
a fork pool (pure function of (seed, P, EP)), so results are identical to serial.
Stop early when max |param delta| (fraction of range, both meshes) < --tol-dp AND round val gain < --tol-val.
   python -m universe.learn_alt [--rounds 4 --epochs 3]"""
import sys, os, json, argparse, time, socket, secrets, tempfile, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pennylane as qml
from pennylane import numpy as pnp
from QrnlOptimizerClass import QrnlStepOptimizer
from universe.sim2 import SPACE
from universe.events import ESPACE, ESPACE5_EXTRA, simulate_ev
from universe.score_ev import score_ev
from universe.nim_rng import NIMRandom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ES5 = dict(ESPACE, **ESPACE5_EXTRA); NL = 2

class Mesh:
    def __init__(s, space):
        s.space = space; s.keys = list(space); s.nw = len(s.keys); dev = qml.device("default.qubit", wires=s.nw)
        @qml.qnode(dev, interface="autograd")
        def circ(W):
            for i in range(s.nw): qml.Hadamard(wires=i)
            for l in range(NL):
                for i in range(s.nw): qml.RX(W[l, i, 0], wires=i); qml.RY(W[l, i, 1], wires=i); qml.RZ(W[l, i, 2], wires=i)
                if l < NL-1:
                    for i in range(s.nw): qml.CNOT(wires=[i, (i+1) % s.nw])
            return [qml.expval(qml.PauliZ(i)) for i in range(s.nw)]
        s.circ = circ
    def decode(s, W):
        e = np.array(s.circ(W), float)
        return {k: float((s.space[k][0]+s.space[k][1])/2 + (s.space[k][1]-s.space[k][0])/2*x) for k, x in zip(s.keys, e)}
    def norm(s, P): return np.array([(P[k]-s.space[k][0])/(s.space[k][1]-s.space[k][0]) for k in s.keys])
    def fit(s, P, seed, iters=150):
        """weights whose decode reproduces settings P (warm start from the one-way result)."""
        tgt = pnp.array(np.clip(2*s.norm(P)-1, -.98, .98), requires_grad=False)
        W = pnp.array(np.random.default_rng(seed).normal(0, .05, (NL, s.nw, 3)), requires_grad=True)
        cost = lambda W: pnp.sum((pnp.stack(s.circ(W)) - tgt)**2)
        opt = qml.AdamOptimizer(.1)
        for _ in range(iters): W = opt.step(cost, W)
        return np.array(W)

E_ = None
def _sim(job):
    s, P, EP = job
    o = simulate_ev(NIMRandom(s), E_, P, EP, 40, s); return score_ev(*o[:6], o[7], o[8], E_)

def main():
    global E_
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--snodes", type=int, default=3); ap.add_argument("--enodes", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slr", type=float, default=.08); ap.add_argument("--elr", type=float, default=.1)
    ap.add_argument("--sck", type=float, default=.1); ap.add_argument("--eck", type=float, default=.15)
    ap.add_argument("--train", type=int, default=4); ap.add_argument("--val", type=int, default=3); ap.add_argument("--heldout", type=int, default=10)
    ap.add_argument("--stars", type=int, default=1500); ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--tol-dp", type=float, default=.02); ap.add_argument("--tol-val", type=float, default=.002)
    ap.add_argument("--mesh", choices=["server", "local"], default="server"); ap.add_argument("--tag", default="_alt2"); ap.add_argument("--oneway", action="store_true")
    a = ap.parse_args(); a.rounds = 1 if a.oneway else a.rounds; here = os.path.dirname(os.path.abspath(__file__)); C = json.load(open(f"{here}/config3.json"))
    E_ = dict(C, **C["eval"]); E_.update(stars_per_galaxy=a.stars, gravity="bh"); out = C["out_dir"]
    pool = mp.get_context("fork").Pool(a.procs)      # forked before the mesh server thread exists
    TR, VA, HO = [100+i for i in range(a.train)], [200+i for i in range(a.val)], [9000+i for i in range(a.heldout)]
    P0 = json.load(open(f"{out}/learned_params_v3.json"))["params"]; EP4 = json.load(open(f"{out}/learned_events_v4.json"))["event_params"]
    EP0 = dict(EP4, **{k: (v[0]+v[1])/2 for k, v in ESPACE5_EXTRA.items()})
    def evalm(P, EP, seeds, full=False):
        out_ = pool.map(_sim, [(s, P, EP) for s in seeds]); t = float(np.mean([o["total"] for o in out_]))
        return (t, {k: float(np.mean([o[k] for o in out_])) for k in out_[0]}) if full else t
    def evalb(jobs):   # batch of (P, EP, seeds) -> list of mean totals, one pool.map
        flat = [(s, P, EP) for P, EP, S in jobs for s in S]; r = pool.map(_sim, flat); k = 0; res = []
        for P, EP, S in jobs: res.append(float(np.mean([x["total"] for x in r[k:k+len(S)]]))); k += len(S)
        return res
    t0 = time.time(); SM, EM = Mesh(SPACE), Mesh(ES5); N = a.snodes + a.enodes
    if a.mesh == "server":
        import Config
        from ServerClass import Server, QueueManager
        port = (lambda s: (s.bind(("127.0.0.1", 0)), s.getsockname()[1], s.close())[1])(socket.socket())
        Config.settings.update(queue_port=port, queue_authkey=secrets.token_bytes(16), log_dir=tempfile.mkdtemp(prefix="nim_mesh_alt_"))
        srv = Server(N); srv.daemon = True; srv.start(); time.sleep(.5)
        cli = QueueManager(address=("127.0.0.1", port), authkey=Config.settings["queue_authkey"]); cli.connect()
        Q = [getattr(cli, f"get_queue_{i}")() for i in range(N)]; print(f"NIM ServerClass mesh on 127.0.0.1:{port} ({N} queues)", flush=True)
    else:
        import queue; Q = [queue.Queue() for _ in range(N)]
    groups = {"structure": list(range(a.snodes)), "events": list(range(a.snodes, N))}; pend = [0]*N
    Ws = SM.fit(P0, a.seed); We = EM.fit(EP0, a.seed+1)
    print(f"warm-start fit: max|dP| struct {np.abs(SM.norm(SM.decode(Ws))-SM.norm(P0)).max():.4f} events {np.abs(EM.norm(EM.decode(We))-EM.norm(EP0)).max():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    W = [Ws.copy() for _ in range(a.snodes)] + [We.copy() for _ in range(a.enodes)]
    opt = [QrnlStepOptimizer(W[i].size, lr=1., decode="adaptive", random_state=a.seed*100+i) for i in range(N)]
    def push(i):
        for j in groups["structure" if i < a.snodes else "events"]:
            if j != i: Q[j].put(W[i].copy()); pend[j] += 1
    for i in range(N): push(i)
    P, EP = dict(P0), dict(EP0)
    v_ow, d_ow = evalm(P0, EP0, VA, True); v_ow4, _ = evalm(P0, EP4, VA, True)
    print(f"one-way (v3+v4, new keys mid) val {v_ow:.4f}   [old v4 model: {v_ow4:.4f}]", flush=True)
    best = dict(val=v_ow, P=dict(P), EP=dict(EP), where="one-way"); rounds = []; curve = [("start", v_ow)]
    step = 0
    for r in range(a.rounds):
        rec = dict(round=r); Pprev, EPprev, vprev = dict(P), dict(EP), best["val"]
        for phase, mesh, nodes, lr0, ck in [("structure", SM, groups["structure"], a.slr, a.sck), ("events", EM, groups["events"], a.elr, a.eck)]:
            # each phase restarts from the current best weights of its mesh (warm start) and the other mesh's frozen settings
            pbest = dict(val=evalm(P, EP, VA), W=None)
            for ep in range(a.epochs):
                lr = lr0/(1+(r*a.epochs+ep)/4); row = []
                for i in nodes:
                    peers = [Q[i].get(timeout=60).ravel() for _ in range(pend[i])]; pend[i] = 0
                    D = np.random.default_rng([a.seed, r, ep, i]).choice([-1., 1.], W[i].shape)
                    cand = lambda Wx: (mesh.decode(Wx), EP) if phase == "structure" else (P, mesh.decode(Wx))
                    (Pp, EPp), (Pm, EPm) = cand(W[i]+ck*D), cand(W[i]-ck*D)
                    fp, fm = evalb([(Pp, EPp, TR), (Pm, EPm, TR)])
                    g = -(fp-fm)/(2*ck)*D; opt[i].lr = lr
                    W[i] = W[i] + opt[i].step(g, W[i], np.array(peers) if peers else None).reshape(W[i].shape)
                    Pc, EPc = cand(W[i]); tr, va = evalb([(Pc, EPc, TR), (Pc, EPc, VA)]); row.append((tr, va)); step += 1
                    curve.append((f"r{r}{phase[0]}{ep}n{i}", va))
                    if va > pbest["val"]: pbest = dict(val=va, W=W[i].copy(), Pc=Pc, EPc=EPc)
                    push(i)
                print(f"round {r} {phase:9s} epoch {ep} lr {lr:.3f} (train,val) {[tuple(round(x, 3) for x in t) for t in row]} phase-best {pbest['val']:.4f} [{time.time()-t0:.0f}s]", flush=True)
            if pbest["W"] is not None:      # adopt the phase winner; all nodes of this mesh warm-start from it next round
                P, EP = pbest["Pc"], pbest["EPc"]
                for i in nodes: W[i] = pbest["W"].copy()
            rec[f"val_after_{phase}"] = pbest["val"]
            if pbest["val"] > best["val"]: best = dict(val=pbest["val"], P=dict(P), EP=dict(EP), where=f"round {r} {phase}")
        rec["dP_struct"] = float(np.abs(SM.norm(P)-SM.norm(Pprev)).max()); rec["dP_events"] = float(np.abs(EM.norm(EP)-EM.norm(EPprev)).max())
        rec["val_gain"] = best["val"]-vprev; rec["P"] = P; rec["EP"] = EP; rounds.append(rec)
        print(f"== round {r}: val {rec['val_after_events']:.4f} gain {rec['val_gain']:+.4f} max|dP| struct {rec['dP_struct']:.3f} events {rec['dP_events']:.3f}", flush=True)
        if (not a.oneway) and r > 0 and max(rec["dP_struct"], rec["dP_events"]) < a.tol_dp and rec["val_gain"] < a.tol_val:
            print("converged -> stop", flush=True); break
    # held-out: same seeds for every arm
    ho = {"one-way v3+v4 (old event model)": evalm(P0, EP4, HO, True)[1], "one-way v3+v4 (+fixes at mid-range)": evalm(P0, EP0, HO, True)[1],
          "trained (best-on-val)": evalm(best["P"], best["EP"], HO, True)[1]}
    ho_seed = [float(x["total"]) for x in pool.map(_sim, [(s, best["P"], best["EP"]) for s in HO])]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); vals = [c[1] for c in curve]
    ax[0].plot(vals, ".-", alpha=.6, label="node val after step"); ax[0].plot(np.maximum.accumulate(vals), "k", lw=2, label="best")
    k = 1
    for rec in rounds:
        n = a.epochs*a.snodes; ax[0].axvspan(k, k+n, color="C0", alpha=.08); ax[0].axvspan(k+n, k+n+a.epochs*a.enodes, color="C1", alpha=.08); k += n+a.epochs*a.enodes
    ax[0].set_title("validation (combined) — blue=structure turn, orange=event turn"); ax[0].set_xlabel("node step"); ax[0].legend(fontsize=8)
    R = range(len(rounds)); ax[1].plot(R, [x["dP_struct"] for x in rounds], "o-", label="structure"); ax[1].plot(R, [x["dP_events"] for x in rounds], "s-", label="events")
    ax[1].axhline(a.tol_dp, ls="--", c="r", label="tol"); ax[1].set_title("max |param delta| per round (fraction of range)"); ax[1].set_xlabel("round"); ax[1].legend()
    ax[2].bar(R, [x["val_gain"] for x in rounds]); ax[2].axhline(a.tol_val, ls="--", c="r"); ax[2].set_title("best-val gain per round"); ax[2].set_xlabel("round")
    fig.tight_layout(); fig.savefig((f"{out}/oneway{a.tag}_convergence.png" if a.oneway else f"{out}/alt2_convergence.png"), dpi=110)
    res = dict(params=best["P"], event_params=best["EP"], best_where=best["where"], val_best=best["val"], val_oneway=v_ow, val_oneway_oldmodel=v_ow4,
               rounds=rounds, curve=curve, heldout_seeds=HO, heldout=ho, heldout_per_seed=ho_seed, train_seeds=TR, val_seeds=VA, args=vars(a), mesh=a.mesh, seconds=time.time()-t0)
    json.dump(res, open((f"{out}/learned_oneway{a.tag}.json" if a.oneway else f"{out}/learned_alt2.json"), "w"), indent=1)
    keys = ["total", "base", "events", "sn_rate", "agn", "visible", "merger", "n_tde", "n_sn"]
    print("HELD-OUT", HO); print(f"{'arm':40s}" + "".join(f"{k:>9s}" for k in keys if k in ho['trained (best-on-val)']))
    for n, d in ho.items(): print(f"{n:40s}" + "".join(f"{d[k]:9.3f}" for k in keys if k in d))
    print(f"done {time.time()-t0:.0f}s"); pool.close()

if __name__ == "__main__": main()
