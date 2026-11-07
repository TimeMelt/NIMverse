"""Explicit browser dependency boundary; the numerical leaf modules are unmodified.
No numba, NIM training, torch, JAX or PennyLane is imported by this entry point.
"""
import json
import numpy as np
from universe.leaves.life_loop import HabitatRun
from universe.leaves.packets import HabitatState, HostWeather

def make_run(spec, mind=None):
    # Existing warm-cache path integrates one extra year; isolate worlds from it.
    from universe.leaves import biogeochem
    biogeochem._EQ_CACHE.clear()
    h=HabitatState(**spec['habitat'])
    st=spec['star']
    hw=HostWeather.from_star(st['mass'],st['luminosity'],st['temperature'])
    hw.age_prior_Gyr=st.get('age',4.6); hw.Z_star=st.get('Z',.0142)
    from universe.leaves.orbit_climate import SecularOrbit
    orb=spec.get('orbits',dict(a=[h.a],m=[h.m],idx=0))
    orbit=SecularOrbit(orb['a'],orb['m'],st['mass'],orb['idx'],h.tilt,h.spin,seed=int(spec['seed']))
    return HabitatRun(h,hw,spec.get('params'),np.random.default_rng(spec['seed']),None,20.,
                      plant=spec.get('plant',True),culture=True,mind=mind,orbit=orbit,
                      solvers=spec.get('solvers',{}))

runs={}
def create(spec):
    key=spec['id']; runs.pop(key,None)
    # An LRU of four live worlds. Evicted worlds are regenerated on re-entry.
    while len(runs)>=4: runs.pop(next(iter(runs)))
    mind=BrowserMind(key,spec['seed']) if spec.get('circuit') and circuit_eval else None
    runs[key]=make_run(spec,mind=mind)
    return key

def advance(key,n=1,z=None):
    r=runs[key]
    if z is not None: apply_drive(r,z)
    for _ in range(n): r.step()
    # Explicit continuous-mode bounds, independent of internal defaults.
    del r.series[:-256]
    if r.pop is not None: del r.pop.series[:-128]
    tick=r.tick(seed=0,max_show=32,trail_max=8)
    tick['mind']=r.mind.summary() if r.mind else None
    tick['quantum_drive']=list(z or [0.]*5)
    return json.dumps(tick)

# Quantum drive is a phenomenological forcing layer on classical solvers, NOT a
# quantum-gravity solver. Each coefficient is bounded, exposed, and switchable.
class DrivenWeather(HostWeather):
    def step(self,t,dt,rng,LP=None):
        super().step(t,dt,rng,LP)
        self.L_now*=1+.01*self.qdrive[0]
        self.uv_now*=1+.05*self.qdrive[1]
        return self

class DrivenOrbit:
    def __init__(self,base): self.base=base; self.delta=0.
    def at(self,t):
        o=dict(self.base.at(t));o['e']=float(np.clip(o['e']+self.delta,0,.9));return o
    def key(self): return self.base.key()

def apply_drive(r,z):
    if not isinstance(r.hw,DrivenWeather):
        r.hw.__class__=DrivenWeather
        r._flare_base=r.P['flare_rate_mult']
        r.lat.orbit=DrivenOrbit(r.lat.orbit)
    r.hw.qdrive=np.clip(np.asarray(z,float),-1,1)
    r.P.set('flare_rate_mult',r._flare_base*(1+.2*r.hw.qdrive[1]))
    r.lat.orbit.delta=.002*float(r.hw.qdrive[4])

from universe.leaves.mind import Mind,featurize,_frame
class BrowserMind(Mind):
    kind='qwrk-js'; learn_enabled=False
    def __init__(self,key,seed):
        super().__init__(seed=seed,every=40.,batch=16)
        self.key=key
    def decide(self,obs):
        zz=json.loads(str(circuit_eval(json.dumps(dict(key=self.key,features=[featurize(o).tolist() for o in obs])))))
        out=[]
        for o,z in zip(obs,zz):
            n,e=_frame(o['lat'],o['lon'])
            out.append(dict(move=(z[0]*n+z[1]*e).tolist(),forage=1+.5*z[2],mate=1+.75*z[3],rest=.5+.5*z[4]))
        return out

# Installed only by the worker. Native tests and parity runs have no JS bridge.
circuit_eval=None
