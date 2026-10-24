import json, numpy as np, sys; sys.path.insert(0, '.')
from universe import sim2
from universe.nim_rng import NIMRandom
C = json.load(open('universe/config3.json')); E = dict(C, **C['eval'], force_encounter=True)
r = NIMRandom(7); G = sim2.make_galaxies(r, E, sim2.DEFAULTS); S = sim2.make_system(r, E, sim2.DEFAULTS)
gx, gv = sim2.leapfrog(G['x'], G['v'], G['m'], G['e2'], E['galaxy_dt'], E['galaxy_steps'], 120); host = sim2.host_star(G, E)
Ts = sim2.tidal_tensors(gx, G['m'], G['e2'], host); pl = S['kind'] == 1
base, _ = sim2.leapfrog(S['x'], S['v'], S['m'], np.full(len(S['m']), 1e-8), E['system_dt'], E['system_steps'], 120)
aout = np.linalg.norm(S['x'][pl], axis=1).max(); lam = np.abs(np.linalg.eigvalsh(Ts)).max()
for b in [1, 1e8, 1e10, 1e11, 3e11]:
    sc = sim2.TIDE_PHYS*b; fx, _ = sim2.leapfrog_tidal(S['x'], S['v'], S['m'], E['system_dt'], E['system_steps'], 120, Ts, sc)
    dev = np.abs(np.linalg.norm(fx[:, pl], axis=2) - np.linalg.norm(base[:, pl], axis=2)).max()/aout
    print(f"boost {b:8.0e}: tidal/Kepler accel at outer planet {sc*lam*aout/(S['Mt']/aout**2):.1e}, max orbit-radius change {dev:.1e} of a_out")
