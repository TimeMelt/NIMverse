"""Example --mind script policy: forage harder when hungry, rest under stress, drift toward the equator, prefer well-fed mates.
python -m universe.run_life --seed 4 --plant --mind script --mind-script universe/mind_example_policy.py --mind-learn"""
import math
stats = dict(n=0, d_energy=0.)

def decide(obs):
    out = []
    for o in obs:
        la, lo = o["lat"], o["lon"]; north = [-math.sin(la)*math.cos(lo), math.cos(la), -math.sin(la)*math.sin(lo)]
        k = -.5*math.copysign(1, la) if abs(la) > .4 else 0.                       # head toward the milder equator
        out.append(dict(move=[k*x for x in north], forage=1.3 if o["energy"] < .3 else 1., mate=.5 + 1.5*o["energy"], rest=min(1., max(0., o["stress"]))))
    return out

def learn(experiences):
    for e in experiences: stats["n"] += 1; stats["d_energy"] += e["energy1"] - e["energy0"]

def culture_advise(state): return dict(mitigation_boost=1. if state["pollution"] > .3 else 0.)
