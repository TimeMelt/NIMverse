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
    runs[key]=make_run(spec)
    return key

def advance(key,n=1):
    r=runs[key]
    for _ in range(n): r.step()
    # Explicit continuous-mode bounds, independent of internal defaults.
    del r.series[:-256]
    if r.pop is not None: del r.pop.series[:-128]
    return json.dumps(r.tick(seed=0,max_show=32,trail_max=8))
