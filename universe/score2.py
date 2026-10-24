"""Universe-quality objective (each part in [0,1])."""
import numpy as np
W = {"distinct": .20, "interest": .05, "coherence": .15, "gal_bound": .10, "compact": .10,
     "sys_stable": .15, "belt": .05, "energy": .10, "richness": .10}

def sys_energy(x, v, m):
    k = m > 0; x, v, m = x[k], v[k], m[k]; d = x[None]-x[:, None]; r = np.sqrt((d**2).sum(-1)+1e-8)
    iu = np.triu_indices(len(m), 1); return .5*(m*(v**2).sum(1)).sum() - (m[:, None]*m[None]/r)[iu].sum()

def score(Gx, S, gx, gv, sx, sv):
    kind, gid, R = Gx["kind"], Gx["gid"], Gx["R"]; cores = np.where(kind == 0)[0]; st = np.where(kind == 1)[0]; ng = len(cores)
    cp = gx[:, cores]; sep = np.linalg.norm(cp[:, :, None]-cp[:, None], axis=-1); iu = np.triu_indices(ng, 1)
    rr = (R[:, None]+R[None])[iu]; ms = sep.min(0)[iu]
    distinct = float(np.mean(np.clip(ms/(.8*rr), 0, 1))) if ng > 1 else 1.
    interest = 1. if np.any((ms < 2*rr) & (ms > .5*rr)) else .3
    nn0 = np.where(np.eye(ng, dtype=bool), np.inf, sep[0]).min(1); compact = float(np.mean(np.exp(-np.maximum(0, nn0/R-8)/4)))
    rel = gx[-1, st]-gx[-1, cores[gid[st]]]; vrel = gv[-1, st]-gv[-1, cores[gid[st]]]; dist = np.linalg.norm(rel, axis=1)
    inside = dist < 3*R[gid[st]]; gal_bound = float(inside.mean())
    rel0 = gx[0, st]-gx[0, cores[gid[st]]]; vrel0 = gv[0, st]-gv[0, cores[gid[st]]]
    coh = []
    for g in range(ng):                                   # disk keeps rotating coherently about its initial spin axis
        s = (gid[st] == g); L0 = np.cross(rel0[s], vrel0[s]).sum(0); n = L0/np.linalg.norm(L0); s &= inside
        if s.sum() < 5: coh.append(0.); continue
        Lz = np.cross(rel[s], vrel[s])@n; coh.append(float(np.clip(np.mean(Lz/(np.linalg.norm(rel[s], axis=1)*np.linalg.norm(vrel[s], axis=1)+1e-9)), 0, 1)))
    coherence = float(np.mean(coh))
    names, kd, m = list(S["name"]), S["kind"], S["m"]; x0, xs, vs = sx[0], sx[-1], sv[-1]
    stars = np.where(kd == 0)[0]; com = (m[stars, None]*xs[stars]).sum(0)/m[stars].sum(); vcom = (m[stars, None]*vs[stars]).sum(0)/m[stars].sum()
    ok = []; alive = 0.
    if S["binary"]:
        d = xs[stars[0]]-xs[stars[1]]; dv = vs[stars[0]]-vs[stars[1]]; ok.append(.5*dv@dv - m[stars].sum()/np.linalg.norm(d) < 0)
    for i, n in enumerate(names):
        if kd[i] == 1:
            rl = xs[i]-com; vr = vs[i]-vcom; good = .5*vr@vr - S["Mt"]/np.linalg.norm(rl) < 0 and np.linalg.norm(rl) < 3*np.linalg.norm(x0[i]); ok.append(good); alive += good
        elif kd[i] == 2:
            p = names.index("planet"+n[4:].split(".")[0]); hill = np.linalg.norm(xs[p]-com)*(m[p]/(3*S["Mt"]))**(1/3)
            good = np.linalg.norm(xs[i]-xs[p]) < .7*hill; ok.append(good); alive += .5*good
    sys_stable = float(np.mean(ok)) if ok else 1.
    if S["belt"] is not None:
        lo, hi = S["belt"]; ra = np.linalg.norm(xs[kd == 3]-com, axis=1); belt = float(np.mean((ra > .7*lo) & (ra < 1.3*hi)))
    else: belt = .5
    E0, E1 = sys_energy(sx[0], sv[0], m), sys_energy(xs, vs, m); energy = float(np.exp(-abs((E1-E0)/E0)/1e-3))
    richness = float(min(1., (alive + (S["belt"] is not None) + S["binary"])/10))
    s = dict(distinct=distinct, interest=interest, coherence=coherence, gal_bound=gal_bound, compact=compact,
             sys_stable=sys_stable, belt=belt, energy=energy, richness=richness)
    # hinge constraint: galaxies that overlap (distinct < 0.85) cost extra, so learning can't trade them away
    s["total"] = sum(W[k]*s[k] for k in W) - .5*max(0., .85-distinct); return s
