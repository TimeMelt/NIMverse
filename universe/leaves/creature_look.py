"""Phenotype leaf: cohort traits (+ biosign, planet type) -> deterministic stylized creature look for the life viewer.
Mapping: mobility -> body plan & speed; metabolism_eff -> size & glow; reproduction_rate -> count & smallness;
hardiness -> shell/armour; biosign pigments -> hue/saturation + bioluminescence; ocean world -> swimmers/floaters favoured."""
import hashlib, colorsys, numpy as np
from .params import TRAITS, clamp

def _rng(seed, i): return np.random.default_rng(int(hashlib.sha256(f"creature:{seed}:{i}".encode()).hexdigest()[:12], 16))

def cohorts_from_report(pl, LP, seed, k=8):
    """Final cohorts are not stored in the life report; synthesise k cohorts around the recorded final trait means
    (spread = LP mutation_sd, as organisms.step mutant splits do). Population share follows the recorded n."""
    cr = pl.get("creatures") or {}; tm = cr.get("trait_means") or {}; sd = LP.get("mutation_sd", .05); out = []
    for i in range(k):
        r = _rng(seed, 1000+i); tr = {t: clamp(t, tm.get(t, LP.get(t, .5)) + (sd*2*r.normal() if i else 0)) for t in TRAITS}
        out.append(dict(n=float(cr.get("n", 1.))*float(r.dirichlet(np.ones(k))[0] if i else .4), traits=tr, synthetic=bool(i)))
    return out

def phenotype(tr, seed, i, pig=0., tech=0., ptype="ocean"):
    r = _rng(seed, i); mob, met, rep, hard = (tr[t] for t in TRAITS)
    water = ptype in ("ocean", "terran")
    w = np.array([(.8 if water else .1)*(.5+mob), (1.2-mob)*(.6 if water else 1.2), max(0, mob-.45)*2.0, (.6 if water else .1)*(1.1-mob)])
    plan = ["swimmer", "crawler", "flyer", "floater"][int(r.choice(4, p=w/w.sum()))]
    size = float((.6+1.1*met)*(1.3-.6*rep)*(.8+.4*r.random()))
    hue = (.33 + .45*pig + .7*r.random()) % 1.; sat = .35+.6*min(1, pig*1.4)
    col = colorsys.hsv_to_rgb(hue, sat, .55+.4*r.random()); col2 = colorsys.hsv_to_rgb((hue+.5+.1*r.random()) % 1, sat, .9)
    return dict(plan=plan, size=size, limbs=int(0 if plan == "floater" else 2+2*round(2*mob+2*r.random())),
                segments=int(2+round(3*rep+r.random())), shell=bool(hard > .45 + .1*r.random()), shell_k=float(hard),
                color=[float(c) for c in col], accent=[float(c) for c in col2], glow=float(min(1, pig*1.6*(.4+met)) if r.random() < .3+pig else 0.),
                speed=float(.2+1.6*mob*(.8+.4*r.random())), wobble=float(r.random()*6.283), tech=float(tech))

def creatures(cohorts, seed, biosign, ptype, max_total=60):
    """List of individual creatures (JSON-serialisable): each cohort contributes individuals proportional to n (>=2)."""
    N = sum(c["n"] for c in cohorts) or 1.; pig, tech = biosign.get("pigments", 0.), biosign.get("tech", 0.); out = []
    for ci, c in enumerate(cohorts):
        base = phenotype(c["traits"], seed, ci, pig, tech, ptype); cnt = max(3, int(round(max_total*c["n"]/N)))
        rc = _rng(seed, 500+ci); clat, clon = float(np.arcsin(rc.uniform(-.6, .6))), float(rc.uniform(0, 6.283))  # cohort home range (herd)
        for j in range(cnt):
            r = _rng(seed, 10000+100*ci+j); u, th = np.sin(np.clip(clat+.22*r.normal(), -1.3, 1.3)), clon+.3*r.normal()
            out.append(dict(base, cohort=ci, traits={k: round(float(v), 3) for k, v in c["traits"].items()},
                            size=base["size"]*float(.8+.4*r.random()), lat=float(np.arcsin(u)), lon=float(th), heading=float(r.uniform(0, 6.283)), phase=float(r.random()*6.283)))
    return out

def creatures_from_individuals(inds, seed, biosign, ptype, max_show=64):
    """1:1 meshes from saved literal individuals (life report `individuals`). If there are more than max_show, an evenly
    spaced subsample over species (then id) is shown; every creature keeps its real id/parent/species/generation.
    Look is seeded by the individual id (stable across runs); placement = the individual's simulated lat/lon
    (legacy reports without positions fall back to species home-range herds)."""
    pig, tech = biosign.get("pigments", 0.), biosign.get("tech", 0.)
    inds = sorted(inds, key=lambda d: (d.get("species_id", 0), d["id"]))
    if len(inds) > max_show: inds = [inds[i] for i in np.linspace(0, len(inds)-1, max_show).astype(int)]
    out = []
    for d in inds:
        tr = {t: float(d["traits"][t]) for t in TRAITS}; sp = int(d.get("species_id", 0))
        base = phenotype(tr, seed, 10**6 + int(d["id"]), pig, tech, ptype)
        r = _rng(seed, 7*10**6 + int(d["id"]))
        if d.get("lat") is not None and d.get("lon") is not None: lat, lon, real = float(d["lat"]), float(d["lon"]), True   # simulated position
        else:                                   # legacy reports without positions: species home-range herd placement
            rs = _rng(seed, 5*10**6 + sp); clat, clon = float(np.arcsin(rs.uniform(-.6, .6))), float(rs.uniform(0, 6.283))
            lat, lon, real = float(np.clip(clat+.22*r.normal(), -1.3, 1.3)), float(clon+.3*r.normal()), False
        out.append(dict(base, cohort=sp, id=int(d["id"]), parent_id=d.get("parent_id"), parent2_id=d.get("parent2_id"), species_id=sp, real_pos=real, generation=int(d.get("generation", 0)),
                        age=float(d.get("age", 0.)), energy=float(d.get("energy", 0.)), traits={k: round(v, 3) for k, v in tr.items()},
                        lat=lat, lon=lon, heading=float(r.uniform(0, 6.283)), phase=float(r.random()*6.283)))
    return out
