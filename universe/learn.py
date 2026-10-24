"""Train NIM to generate better universes.
Mesh: N inceptron nodes, each a parameterised quantum circuit (NIM ansatz: Hadamard + RX/RY/RZ per wire,
plus a CNOT ring) whose PauliZ expvals decode to generation parameters. Nodes push their weights into each
other's queues (NIM mesh pattern); each node trains with NIM's QrnLOpt (QrnlStepOptimizer = KDE -> RBM via
QRNL.fit_rows) fed an SPSA gradient of a universe-quality objective, plus peer weights from its queue."""
import sys, os, json, queue, argparse, time, numpy as np, pennylane as qml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from QrnlOptimizerClass import QrnlStepOptimizer
from universe.generate import make_galaxies, make_system, leapfrog, accel
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PARAMS = [("spread_mult", .5, 2.5), ("gal_mass_mult", .3, 2.), ("gal_radius_mult", .5, 1.5),
          ("vel_disp", 0., 1.), ("spacing_mult", .8, 1.5), ("moon_dist_mult", .5, 3.)]
NW, NL = len(PARAMS), 2
dev = qml.device("default.qubit", wires=NW)
@qml.qnode(dev)
def circuit(W):
    for i in range(NW): qml.Hadamard(wires=i)
    for l in range(NL):
        for i in range(NW): qml.RX(W[l, i, 0], wires=i); qml.RY(W[l, i, 1], wires=i); qml.RZ(W[l, i, 2], wires=i)
        for i in range(NW): qml.CNOT(wires=[i, (i+1) % NW])
    return [qml.expval(qml.PauliZ(i)) for i in range(NW)]

def decode(W):
    e = np.array(circuit(W), float)
    return {k: float(lo + (hi-lo)*(x+1)/2) for (k, lo, hi), x in zip(PARAMS, e)}

DEFAULTS = {"spread_mult": 1., "gal_mass_mult": 1., "gal_radius_mult": 1., "vel_disp": .3, "spacing_mult": 1., "moon_dist_mult": 1.}

def energy(x, v, m, eps):
    d = x[None]-x[:, None]; r = np.sqrt((d**2).sum(-1)+eps**2); iu = np.triu_indices(len(m), 1)
    return .5*(m*(v**2).sum(1)).sum() - (m[:, None]*m[None]/r)[iu].sum()

def score_universe(C, seed):
    """Short sim -> dict of sub-scores in [0,1] and weighted total."""
    r = np.random.default_rng(seed)
    x, v, m, kind, gid = make_galaxies(r, C); cores = np.where(kind == 0)[0]
    fg = leapfrog(x, v, m, C["galaxy_dt"], C["galaxy_steps"], 40, C["softening"], src=cores)
    R = np.array([np.percentile(np.linalg.norm(x[(gid == g) & (kind == 1)] - x[cores[g]], axis=1), 90) for g in range(len(cores))])
    cp = fg[:, cores]; sep = np.linalg.norm(cp[:, :, None]-cp[:, None], axis=-1)
    iu = np.triu_indices(len(cores), 1); minsep = sep.min(0)[iu]; rr = (R[:, None]+R[None])[iu]
    distinct = np.mean(np.clip(minsep/(2*rr), 0, 1))
    nn0 = np.where(np.eye(len(cores), dtype=bool), np.inf, sep[0]).min(1)
    compact = float(np.mean(np.exp(-np.maximum(0, nn0/R - 8)/4)))   # galaxies shouldn't be tiny specks far apart
    st = np.where(kind == 1)[0]; dist = np.linalg.norm(fg[-1, st]-fg[-1, cores[gid[st]]], axis=1)
    gal_bound = float(np.mean(dist < 3*R[gid[st]]))
    names, sx, sv, sm, sk = make_system(r, C); eps = 1e-4
    E0 = energy(sx, sv, sm, eps)
    xs, vs = sx.copy(), sv.copy(); a = accel(xs, sm, eps); dt = C["system_dt"]
    for _ in range(C["system_steps"]):
        vs += .5*dt*a; xs += dt*vs; a = accel(xs, sm, eps); vs += .5*dt*a
    drift = abs((energy(xs, vs, sm, eps)-E0)/E0)
    ok = []
    for i, n in enumerate(names[1:], 1):
        if n.startswith("planet"):
            rel = xs[i]-xs[0]; vr = vs[i]-vs[0]; ok.append(.5*vr@vr - sm[0]/np.linalg.norm(rel) < 0 and np.linalg.norm(rel) < 3*np.linalg.norm(sx[i]))
        else:
            p = names.index("planet"+n[4:].split(".")[0]); hill = np.linalg.norm(xs[p]-xs[0])*(sm[p]/(3*sm[0]))**(1/3)
            ok.append(np.linalg.norm(xs[i]-xs[p]) < .7*hill)
    sys_stable = float(np.mean(ok)) if ok else 1.
    en = float(np.exp(-drift/1e-3))
    s = dict(distinct=float(distinct), compact=compact, gal_bound=gal_bound, sys_stable=sys_stable, energy=en)
    s["total"] = .25*distinct + .15*compact + .2*gal_bound + .25*sys_stable + .15*en
    return s

def objective(P, base, seeds):
    C = dict(base, **P); return float(np.mean([score_universe(C, s)["total"] for s in seeds]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_config.json"))
    a = ap.parse_args(); base = json.load(open(a.config)); out = base["out_dir"]; os.makedirs(out, exist_ok=True)
    # cheap training universe (same physics, fewer bodies/steps)
    base.update(n_galaxies=4, stars_per_galaxy=40, galaxy_steps=1500, galaxy_dt=0.01, system_steps=2000, system_dt=0.005)
    seeds = [1000+a.seed, 2000+a.seed]            # fixed evaluation universes -> reproducible objective
    rng = np.random.default_rng(a.seed)
    W = [rng.normal(0, .3, (NL, NW, 3)) for _ in range(a.nodes)]
    opt = [QrnlStepOptimizer(W[0].size, lr=a.lr, decode="adaptive", random_state=a.seed*100+i) for i in range(a.nodes)]
    Q = [queue.Queue() for _ in range(a.nodes)]   # one inbound queue per inceptron (NIM mesh)
    for i in range(a.nodes):
        for j in range(a.nodes):
            if j != i: Q[j].put(W[i].copy())
    init_params = decode(W[0]); init_score = objective(init_params, base, seeds); print(f"untrained mesh (node0) score: {init_score:.3f}")
    hist = []; best = (-1, None, None); ck = 0.15
    base_score = objective(DEFAULTS, base, seeds); print(f"untrained default params score: {base_score:.3f}")
    for ep in range(a.epochs):
        row = []
        for i in range(a.nodes):
            peers = []
            while not Q[i].empty(): peers.append(Q[i].get().ravel())
            prng = np.random.default_rng([a.seed, ep, i]); D = prng.choice([-1., 1.], W[i].shape)
            fp, fm = objective(decode(W[i]+ck*D), base, seeds), objective(decode(W[i]-ck*D), base, seeds)
            g = -(fp-fm)/(2*ck)*D                     # gradient of loss = -score (SPSA)
            W[i] = W[i] + opt[i].step(g, W[i], np.array(peers) if peers else None).reshape(W[i].shape)
            sc = objective(decode(W[i]), base, seeds); row.append(sc)
            if sc > best[0]: best = (sc, W[i].copy(), decode(W[i]))
            for j in range(a.nodes):                  # push to every other inceptron's queue
                if j != i: Q[j].put(W[i].copy())
        hist.append(row); print(f"epoch {ep}: node scores {np.round(row, 3)}  best {best[0]:.3f}", flush=True)
    hist = np.array(hist)
    plt.figure(figsize=(7, 4))
    for i in range(a.nodes): plt.plot(hist[:, i], label=f"inceptron {i}", alpha=.7)
    plt.plot(np.maximum.accumulate(hist.max(1)), "k-", lw=2, label="best so far")
    plt.axhline(base_score, ls="--", c="r", label="untrained defaults")
    plt.xlabel("epoch"); plt.ylabel("universe quality score"); plt.title("NIM mesh + QrnLOpt training"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/training_curve.png", dpi=110)
    full = lambda P: {k: score_universe(dict(base, **P), s) for k, s in zip(["eval_a", "eval_b"], seeds)}
    res = {"params": best[2], "best_score": best[0], "untrained_default_score": base_score,
           "untrained_mesh_score": init_score, "untrained_mesh_params": init_params,
           "breakdown_learned": full(best[2]), "breakdown_default": full(DEFAULTS), "history": hist.tolist(), "args": vars(a)}
    json.dump(res, open(f"{out}/learned_params.json", "w"), indent=1); np.save(f"{out}/learned_weights.npy", best[1])
    print(json.dumps({k: res[k] for k in ["params", "best_score", "untrained_default_score", "untrained_mesh_score"]}, indent=1))

if __name__ == "__main__": main()
