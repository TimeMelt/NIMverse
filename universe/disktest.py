"""Isolated-galaxy disk heating test: rms thickness and coherence vs. fixes."""
import numpy as np, json, sys, time
from universe import sim2
def run(C, P, label):
    C = dict(C, n_galaxies=1); r = np.random.default_rng(7)
    G = sim2.make_galaxies(r, C, P); t = time.time()
    fx, fv = sim2.leapfrog(G["x"], G["v"], G["m"], G["e2"], C["galaxy_dt"], C["galaxy_steps"], 24)
    Rm = G["Rot"] if "Rot" in G else None
    st = G["kind"] == 1; out = []
    # spin axis from initial L
    x0 = G["x"][st]-G["x"][0]; v0 = G["v"][st]-G["v"][0]; Lz = np.cross(x0, v0).sum(0); z = Lz/np.linalg.norm(Lz)
    for k in [0, len(fx)//2, len(fx)-1]:
        x = fx[k][st]-fx[k][0]; v = fv[k][st]-fv[k][0]; zz = x@z; rr = np.linalg.norm(x, axis=1)
        inn = rr < 1.6*G["R"][0]
        coh = np.mean((np.cross(x, v)@z)/(rr*np.linalg.norm(v, axis=1)+1e-9))
        out.append((round(float(np.sqrt(np.mean(zz[inn]**2))/G["R"][0]), 4), round(float(coh), 3)))
    print(f"{label:34s} (z_rms/R, coherence) t=0,mid,end: {out}  {time.time()-t:.1f}s", flush=True)
base = json.load(open("universe/config2.json")); P = sim2.DEFAULTS
for label, mod in [("v2 baseline", {}), ("dt 0.01 (2x steps)", dict(galaxy_dt=.01, galaxy_steps=2880)),
                   ("softening 0.7", dict(star_soft=.7)), ("equal-mass tracers", dict(equal_mass=True)),
                   
                   ("equal-mass + equilibrium", dict(equal_mass=True, equilibrium=True)),
                   ("equal-mass + soft .5", dict(equal_mass=True, star_soft=.5)),
                   ("equal-mass + equilibrium + soft .5", dict(equal_mass=True, equilibrium=True, star_soft=.5))]:
    run(dict(base, **mod), P, label)
