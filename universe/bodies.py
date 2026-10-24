"""Procedural per-body detail (deterministic from universe seed + body id, randomness from the NIM quantum RNG).
Planet types from mass and distance relative to the habitable zone; stars from temperature; moons get craters."""
import numpy as np
from .nim_rng import NIMRandom

EARTH = 3.0e-6   # Msun
TYPES = ["terran", "desert", "ocean", "ice", "lava", "gas giant", "ice giant"]

def body_rng(seed, bid, rng="nim"):
    s = (int(seed)*1000003 + int(bid)*7919 + 12345) % (2**31)
    return NIMRandom(s, n_wires=6, shots=512) if rng == "nim" else np.random.default_rng(s)

def lum(M): return M**4 if M < 2 else 1.4*M**3.5
def temp(M): return 5800*np.clip(M, .08, 60)**.55

def classify(m, a, L, u):
    """m [Msun], a [AU-like system units], L [Lsun], u: one uniform draw for terran/ocean split."""
    hz_in, hz_out = .95*np.sqrt(L), 1.67*np.sqrt(L); me = m/EARTH
    if me > 12: return "gas giant" if a < 2.5*hz_out else "ice giant"
    if a < .55*hz_in: return "lava"
    if a < hz_in: return "desert"
    if a <= hz_out: return "ocean" if u < .3 else "terran"
    return "ice"

PAL = {  # deep water, shallow, beach/low, mid, high, peak/snow  (rgb 0-1)
    "terran": [[.02, .08, .25], [.05, .3, .5], [.76, .7, .5], [.16, .45, .14], [.4, .33, .25], [.95, .95, .97]],
    "ocean":  [[.01, .05, .22], [.04, .25, .5], [.7, .68, .5], [.2, .5, .2], [.35, .35, .3], [.95, .95, .97]],
    "desert": [[.25, .15, .08], [.5, .3, .15], [.85, .65, .4], [.75, .5, .28], [.55, .35, .22], [.9, .8, .65]],
    "ice":    [[.15, .25, .4], [.4, .6, .75], [.8, .88, .95], [.85, .9, .97], [.7, .75, .85], [1, 1, 1]],
    "lava":   [[1., .35, .05], [.9, .15, .02], [.12, .08, .07], [.2, .12, .1], [.3, .2, .18], [.45, .4, .38]],
}
ATM = {"terran": [.35, .6, 1.], "ocean": [.3, .55, 1.], "desert": [.95, .7, .45], "ice": [.7, .85, 1.], "lava": [1., .4, .15]}

def planet_detail(seed, bid, m, a, L, rng="nim"):
    r = body_rng(seed, bid, rng); u = r.random(24)
    t = classify(m, a, L, u[0]); me = m/EARTH
    d = dict(id=int(bid), type=t, mass_earth=float(me), a=float(a), noise_off=(u[1:4]*200-100).tolist(),
             tilt=float((u[4]-.5)*.9), spin=float(.3+u[5]*1.5))
    if "giant" in t:
        base = np.array([[.85, .7, .5], [.7, .5, .35], [.95, .9, .8]]) if t == "gas giant" else np.array([[.45, .7, .85], [.3, .5, .75], [.7, .85, .95]])
        d.update(bands=float(6+u[6]*14), turb=float(.3+u[7]*.9), colors=(np.clip(base*(.85+.3*u[8:17].reshape(3, 3)), 0, 1)).tolist(),
                 storm=[float((u[17]-.5)*1.2), float(u[18]*6.283), float(.08+u[19]*.12)], radius_rel=float(np.clip(me**.15, 2, 6)))
    else:
        sea = {"terran": .45+.2*u[6], "ocean": .7+.15*u[6], "desert": .1*u[6], "ice": .3+.2*u[6], "lava": .35+.2*u[6]}[t]
        d.update(sea=float(sea), rough=float(.45+.2*u[7]), cont_scale=float(1.2+u[8]*1.6), mountains=float(.4+u[9]*.8),
                 moisture=float(u[10]-.5), cap=float({"ice": .2, "lava": 1.5, "desert": 1.2}.get(t, .72+.15*u[11])),
                 clouds=float({"terran": .45+.3*u[12], "ocean": .55+.3*u[12], "desert": .1*u[12], "ice": .2*u[12], "lava": .3*u[12]}[t]),
                 colors=PAL[t], atm=ATM[t], radius_rel=float(np.clip(me**.27, .6, 2.2)))
    return d

def moon_detail(seed, bid, m, rng="nim"):
    r = body_rng(seed, bid, rng); u = r.random(8)
    g = .35+.4*u[0]; tint = np.clip(np.array([g, g*(.92+.1*u[1]), g*(.85+.15*u[2])]), 0, 1)
    return dict(id=int(bid), type="moon", craters=float(4+u[3]*8), crater_depth=float(.4+.6*u[4]), color=tint.tolist(),
                noise_off=(u[5:8]*200-100).tolist())

def star_detail(seed, bid, M, rng="nim"):
    r = body_rng(seed, bid, rng); u = r.random(4); T = float(temp(M))
    return dict(id=int(bid), type="star", mass=float(M), T=T, L=float(lum(M)), radius_rel=float(M**.8),
                gran=float(18+u[0]*20), spots=float(u[1]*.6), noise_off=(u[1:4]*100).tolist())

def system_details(S, seed, rng="nim"):
    k, m, x = S["kind"], S["m"], S["x"]; stars = np.where(k == 0)[0]; L = sum(lum(m[i]) for i in stars)
    out = {}
    for i in np.where(k <= 2)[0]:
        if k[i] == 0: out[int(i)] = star_detail(seed, i, m[i], rng)
        elif k[i] == 1: out[int(i)] = planet_detail(seed, i, m[i], float(np.linalg.norm(x[i])), L, rng)
        else: out[int(i)] = moon_detail(seed, i, m[i], rng)
    return out
