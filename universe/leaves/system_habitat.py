"""HabitatState from a sim2.make_system dict: HZ is sim state, using the same math as bodies.classify/lum."""
import numpy as np
from ..bodies import lum, classify, body_rng, EARTH
from .packets import HabitatState

VOLATILES = {"terran": .5, "ocean": .9, "desert": .1, "ice": .8, "lava": .02, "gas giant": 1., "ice giant": 1.}

def system_luminosity(S): return float(sum(lum(float(m)) for m, k in zip(S["m"], S["kind"]) if k == 0))

def _orbit(S, i, Mt):
    x, v = np.asarray(S["x"][i], float), np.asarray(S["v"][i], float)
    st = np.asarray(S["kind"]) == 0; com = (S["x"][st]*S["m"][st, None]).sum(0)/S["m"][st].sum()
    r = x - com; vv = v - (S["v"][st]*S["m"][st, None]).sum(0)/S["m"][st].sum()
    mu = Mt; rn = np.linalg.norm(r); E = .5*vv@vv - mu/rn; a = -mu/(2*E) if E < 0 else rn
    e = np.linalg.norm(np.cross(vv, np.cross(r, vv))/mu - r/rn); return float(a), float(e)

def habitats_from_system(S, seed, LP=None, rng="numpy"):
    L = system_luminosity(S); Mt = float(S.get("Mt", S["m"][np.asarray(S["kind"]) == 0].sum()))
    hz_in, hz_out = .95*np.sqrt(L), 1.67*np.sqrt(L)
    ast = np.asarray(S["kind"]) == 3; belt = S.get("belt"); out = []
    for i in np.where(np.asarray(S["kind"]) == 1)[0]:
        a, e = _orbit(S, i, Mt); r = body_rng(seed, i, rng); u = r.random(6)
        m = float(S["m"][i]); t = classify(m, a, L, u[0])
        imp = 0.
        if belt is not None and ast.any(): imp = float(ast.sum()/200.*np.exp(-min(abs(a-belt[0]), abs(a-belt[1]))/max(a, 1e-3)))
        out.append(HabitatState(int(i), str(S["name"][i]), m, a, e, float(.3+u[5]*1.5), float((u[4]-.5)*.9), L,
                                float(hz_in), float(hz_out), bool(hz_in <= a <= hz_out), t,
                                float(VOLATILES[t]*(.7+.6*u[2])), imp, float(L/a**2)))
    return out

def is_rocky(h): return h.m/EARTH <= 12

def candidate(h, margin=.25):
    return is_rocky(h) and (1-margin)*h.hz_in <= h.a <= (1+margin)*h.hz_out
