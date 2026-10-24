"""Train the inceptron "event node": a dedicated 9-qubit NIM-style circuit (H, RX/RY/RZ, CNOT ring) whose <Z_i>
set the 9 high-energy event settings (ESPACE).  Same recipe as learn3: SPSA gradients on fixed training universes
(NIM RNG), NIM QrnlStepOptimizer update with peer weights over the mesh queues, best-on-validation, held-out report.
Structure settings stay fixed at the v3 learned values.   python -m universe.learn_ev [--epochs 8 --nodes 2]"""
import sys, os, json, queue, argparse, time, numpy as np, pennylane as qml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QrnlOptimizerClass import QrnlStepOptimizer
from universe.events import ESPACE, EDEFAULTS, simulate_ev
from universe.score_ev import score_ev
from universe.nim_rng import NIMRandom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
KEYS = list(ESPACE); NW, NL = len(KEYS), 2
dev = qml.device("default.qubit", wires=NW)
@qml.qnode(dev)
def circuit(W):
    for i in range(NW): qml.Hadamard(wires=i)
    for l in range(NL):
        for i in range(NW): qml.RX(W[l, i, 0], wires=i); qml.RY(W[l, i, 1], wires=i); qml.RZ(W[l, i, 2], wires=i)
        if l < NL-1:
            for i in range(NW): qml.CNOT(wires=[i, (i+1) % NW])
    return [qml.expval(qml.PauliZ(i)) for i in range(NW)]
def decode(W):
    e = np.array(circuit(W), float)
    return {k: float((ESPACE[k][0]+ESPACE[k][1])/2 + (ESPACE[k][1]-ESPACE[k][0])/2*x) for k, x in zip(KEYS, e)}
def evaluate(EP, P, E, seeds, full=False):
    out = []
    for s in seeds:
        o = simulate_ev(NIMRandom(s), E, P, EP, 40, s); out.append(score_ev(*o[:6], o[7], o[8], E))
    tot = float(np.mean([o["total"] for o in out]))
    return (tot, {k: float(np.mean([o[k] for o in out])) for k in out[0]}) if full else tot
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=.1); ap.add_argument("--ck", type=float, default=.15)
    ap.add_argument("--train", type=int, default=5); ap.add_argument("--val", type=int, default=3); ap.add_argument("--heldout", type=int, default=10)
    ap.add_argument("--stars", type=int, default=1500)
    a = ap.parse_args(); here = os.path.dirname(os.path.abspath(__file__)); C = json.load(open(f"{here}/config3.json"))
    E = dict(C, **C["eval"]); E.update(stars_per_galaxy=a.stars, gravity="bh"); out = C["out_dir"]
    P = json.load(open(f"{out}/learned_params_v3.json"))["params"]
    TR, VA, HO = [100+i for i in range(a.train)], [200+i for i in range(a.val)], [9000+i for i in range(a.heldout)]
    rng = np.random.default_rng(a.seed); Wn = [rng.normal(0, .05, (NL, NW, 3)) for _ in range(a.nodes)]
    opt = [QrnlStepOptimizer(Wn[0].size, lr=a.lr, decode="adaptive", random_state=a.seed*100+i) for i in range(a.nodes)]
    Q = [queue.Queue() for _ in range(a.nodes)]; pend = [0]*a.nodes
    def push(i):
        for j in range(a.nodes):
            if j != i: Q[j].put(Wn[i].copy()); pend[j] += 1
    for i in range(a.nodes): push(i)
    t0 = time.time(); d_va = evaluate(EDEFAULTS, P, E, VA); print(f"defaults val {d_va:.4f}", flush=True)
    best = dict(val=d_va, P=dict(EDEFAULTS), W=None, epoch=-1); hist = []
    for ep in range(a.epochs):
        lr = a.lr/(1+ep/4); row = []
        for i in range(a.nodes):
            peers = [Q[i].get().ravel() for _ in range(pend[i])]; pend[i] = 0
            D = np.random.default_rng([a.seed, ep, i]).choice([-1., 1.], Wn[i].shape)
            g = -(evaluate(decode(Wn[i]+a.ck*D), P, E, TR) - evaluate(decode(Wn[i]-a.ck*D), P, E, TR))/(2*a.ck)*D
            opt[i].lr = lr; Wn[i] = Wn[i] + opt[i].step(g, Wn[i], np.array(peers) if peers else None).reshape(Wn[i].shape)
            EP = decode(Wn[i]); tr, va = evaluate(EP, P, E, TR), evaluate(EP, P, E, VA); row.append((tr, va))
            if va > best["val"]: best = dict(val=va, P=EP, W=Wn[i].copy(), epoch=ep)
            push(i)
        hist.append(row); print(f"epoch {ep}: {[tuple(round(x, 3) for x in t) for t in row]} best {best['val']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    H = np.array(hist); fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(a.nodes): ax.plot(H[:, i, 1], label=f"event node {i} (val)")
    ax.axhline(d_va, ls="--", c="r", label="defaults"); ax.legend(); ax.set_xlabel("epoch"); ax.set_title("Event-node training (QrnLOpt, NIM RNG)")
    fig.tight_layout(); fig.savefig(f"{out}/training_curve_events_v4.png", dpi=110)
    _, hd = evaluate(EDEFAULTS, P, E, HO, True); _, hl = evaluate(best["P"], P, E, HO, True)
    json.dump(dict(event_params=best["P"], params=P, best_epoch=best["epoch"], val_default=d_va, val_best=best["val"], heldout_seeds=HO,
                   heldout_default=hd, heldout_learned=hl, history=H.tolist(), args=vars(a), seconds=time.time()-t0), open(f"{out}/learned_events_v4.json", "w"), indent=1)
    print("HELD-OUT default vs learned"); [print(f"  {k:10s} {hd[k]:.3f} {hl[k]:.3f}") for k in hd]; print(json.dumps(best["P"], indent=1))
if __name__ == "__main__": main()
