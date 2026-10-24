"""v2 training: NIM inceptron mesh + QrnLOpt learns 16 universe-generation settings.
Each node = 16-qubit NIM-style circuit (H, RX/RY/RZ per wire, CNOT ring, 2 layers); <Z_i> -> setting i
(<Z>=0 == hand-picked default, so training starts at the defaults and must improve on them).
Gradient: SPSA averaged over K directions on 6 fixed training universes (common random numbers, NIM quantum RNG).
Update: NIM QrnlStepOptimizer (KDE->RBM) with peer weights from the node's mesh queue; decayed lr;
best weights selected on 4 validation universes; final report on 12 held-out universes."""
import sys, os, json, queue, argparse, time, numpy as np, pennylane as qml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QrnlOptimizerClass import QrnlStepOptimizer
from universe.sim2 import SPACE, DEFAULTS, simulate
from universe.score2 import score, W as WEIGHTS
from universe.nim_rng import NIMRandom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

KEYS = list(SPACE); NW, NL = len(KEYS), 2
dev = qml.device("default.qubit", wires=NW)
@qml.qnode(dev)
def circuit(W):
    for i in range(NW): qml.Hadamard(wires=i)
    for l in range(NL):
        for i in range(NW): qml.RX(W[l, i, 0], wires=i); qml.RY(W[l, i, 1], wires=i); qml.RZ(W[l, i, 2], wires=i)
        if l < NL-1:   # entangle between layers; last layer stays local so each <Z_i> is first-order in its weights
            for i in range(NW): qml.CNOT(wires=[i, (i+1) % NW])
    return [qml.expval(qml.PauliZ(i)) for i in range(NW)]
def decode(W):
    e = np.array(circuit(W), float)
    return {k: float((SPACE[k][0]+SPACE[k][1])/2 + (SPACE[k][1]-SPACE[k][0])/2*x) for k, x in zip(KEYS, e)}

def evaluate(P, E, seeds, rng="nim", full=False):
    out = []
    for s in seeds:
        r = NIMRandom(s) if rng == "nim" else np.random.default_rng(s)
        out.append(score(*simulate(r, E, P, 40)[:6]))
    tot = float(np.mean([o["total"] for o in out]))
    return (tot, {k: float(np.mean([o[k] for o in out])) for k in out[0]}) if full else tot

def make_mesh(a):
    """Use NIM's own ServerClass (multiprocessing BaseManager over TCP) as the inceptron weight mesh.
    Deterministic because every put/get happens from this one thread in a fixed round-robin order."""
    import socket, secrets, tempfile, threading, Config
    from ServerClass import Server, QueueManager
    port = a.port or (lambda s: (s.bind(("127.0.0.1", 0)), s.getsockname()[1], s.close())[1])(socket.socket())
    Config.settings.update(queue_port=port, queue_authkey=secrets.token_bytes(16),
                           log_dir=tempfile.mkdtemp(prefix="nim_mesh_"))    # fresh dir -> Server never deletes logs
    srv = Server(a.nodes); srv.daemon = True; srv.start(); time.sleep(.5)
    cli = QueueManager(address=("127.0.0.1", port), authkey=Config.settings["queue_authkey"]); cli.connect()
    print(f"NIM ServerClass mesh on 127.0.0.1:{port}", flush=True)
    return [getattr(cli, f"get_queue_{i}")() for i in range(a.nodes)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--spsa-k", type=int, default=2); ap.add_argument("--ck", type=float, default=0.1)
    ap.add_argument("--rng", choices=["nim", "numpy"], default="nim")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config3.json"))
    ap.add_argument("--train", type=int, default=10); ap.add_argument("--val", type=int, default=6); ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--mesh", choices=["local", "server"], default="local", help="server = NIM ServerClass BaseManager queues over TCP")
    ap.add_argument("--port", type=int, default=0); ap.add_argument("--tag", default="")
    a = ap.parse_args(); C = json.load(open(a.config)); E = dict(C, **C["eval"]); out = C["out_dir"]; os.makedirs(out, exist_ok=True)
    TR, VA, HO = [100+i for i in range(a.train)], [200+i for i in range(a.val)], [9000+i for i in range(a.heldout)]
    ev = lambda P, S: evaluate(P, E, S, a.rng)
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    Wn = [rng.normal(0, .05, (NL, NW, 3)) for _ in range(a.nodes)]
    opt = [QrnlStepOptimizer(Wn[0].size, lr=a.lr, decode="adaptive", random_state=a.seed*100+i) for i in range(a.nodes)]
    Q = make_mesh(a) if a.mesh == "server" else [queue.Queue() for _ in range(a.nodes)]
    pending = [0]*a.nodes   # explicit message counts: mp.Queue.empty() is racy (feeder thread), counts are not
    def push(i):
        for j in range(a.nodes):
            if j != i: Q[j].put(Wn[i].copy()); pending[j] += 1
    for i in range(a.nodes): push(i)
    d_tr, d_va = ev(DEFAULTS, TR), ev(DEFAULTS, VA); print(f"defaults: train {d_tr:.4f} val {d_va:.4f}", flush=True)
    d_dist = evaluate(DEFAULTS, E, VA, a.rng, True)[1]["distinct"]   # constraint: never accept worse distinctness
    best = dict(val=d_va, W=None, P=dict(DEFAULTS), epoch=-1); hist = []
    for ep in range(a.epochs):
        lr = a.lr/(1+ep/4); row = []
        for i in range(a.nodes):
            peers = []
            for _ in range(pending[i]): peers.append(Q[i].get(timeout=30).ravel())
            pending[i] = 0
            g = np.zeros_like(Wn[i])
            for k in range(a.spsa_k):
                D = np.random.default_rng([a.seed, ep, i, k]).choice([-1., 1.], Wn[i].shape)
                g += -(ev(decode(Wn[i]+a.ck*D), TR) - ev(decode(Wn[i]-a.ck*D), TR))/(2*a.ck)*D
            opt[i].lr = lr
            Wn[i] = Wn[i] + opt[i].step(g/a.spsa_k, Wn[i], np.array(peers) if peers else None).reshape(Wn[i].shape)
            P = decode(Wn[i]); tr, va = ev(P, TR), ev(P, VA); row.append((tr, va))
            if va > best["val"] and evaluate(P, E, VA, a.rng, True)[1]["distinct"] >= d_dist - .01: best = dict(val=va, W=Wn[i].copy(), P=P, epoch=ep)
            push(i)
        hist.append(row)
        print(f"epoch {ep} lr {lr:.3f}: (train,val) {[tuple(round(x, 3) for x in t) for t in row]} best val {best['val']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    H = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i in range(a.nodes):
        ax[0].plot(H[:, i, 0], label=f"inceptron {i}"); ax[1].plot(H[:, i, 1], label=f"inceptron {i}")
    for k, (axx, ref) in enumerate(zip(ax, [d_tr, d_va])):
        axx.axhline(ref, ls="--", c="r", label="hand-picked defaults"); axx.set_xlabel("epoch"); axx.legend(fontsize=8)
        axx.set_title([f"train score ({a.train} universes)", f"validation score ({a.val} universes)"][k])
    ax[1].plot(np.maximum.accumulate(np.r_[d_va, H[:, :, 1].max(1)])[1:], "k", lw=2, label="best")
    fig.suptitle("NIM mesh + QrnLOpt training (NIM quantum RNG)"); fig.tight_layout(); fig.savefig(f"{out}/training_curve_v3{a.tag}.png", dpi=110)
    hd, hdb = evaluate(DEFAULTS, E, HO, a.rng, True); hl, hlb = evaluate(best["P"], E, HO, a.rng, True)
    res = dict(params=best["P"], best_epoch=best["epoch"], val_best=best["val"], val_default=d_va, train_default=d_tr,
               heldout_seeds=HO, heldout_default=hdb, heldout_learned=hlb, history=H.tolist(), args=vars(a), seconds=time.time()-t0)
    json.dump(res, open(f"{out}/learned_params_v3{a.tag}.json", "w"), indent=1)
    if best["W"] is not None: np.save(f"{out}/learned_weights_v3{a.tag}.npy", best["W"])
    print("HELD-OUT  default vs learned"); [print(f"  {k:11s} {hdb[k]:.3f}  {hlb[k]:.3f}") for k in hdb]
    print(json.dumps(best["P"], indent=1))

if __name__ == "__main__": main()
