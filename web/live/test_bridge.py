"""Native half of the worker parity contract (browser runs identical bridge)."""
import json
from pathlib import Path
import numpy as np

def test_real_leaf_tick_is_deterministic():
    import importlib.util
    spec=importlib.util.spec_from_file_location('live_bridge', Path(__file__).with_name('bridge.py'))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    s=json.loads(Path('.build-live/seed4.json').read_text())
    a=m.make_run(s); b=m.make_run(s)
    for _ in range(5): a.step(); b.step()
    assert a.tick(4)==b.tick(4)
    assert len(a.lat.T)==24
    assert np.isfinite(a.clim.T_K)

def test_drive_changes_classical_forcing_and_turns_off():
    import importlib.util
    spec=importlib.util.spec_from_file_location('live_bridge', Path(__file__).with_name('bridge.py'))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    s=json.loads(Path('.build-live/seed4.json').read_text())
    a=m.make_run(s); b=m.make_run(s)
    m.apply_drive(a,[1.,1.,0,0,1.]);m.apply_drive(b,[0.]*5)
    a.step();b.step()
    assert a.hw.L_now>b.hw.L_now
    assert a.P['flare_rate_mult']>b.P['flare_rate_mult']
    assert a.lat.orbit.at(0)['e']>b.lat.orbit.at(0)['e']
    m.apply_drive(a,[0.]*5)
    assert a.P['flare_rate_mult']==b.P['flare_rate_mult']

def test_world_cache_is_bounded_with_real_runs():
    import importlib.util
    spec=importlib.util.spec_from_file_location('live_bridge',Path(__file__).with_name('bridge.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s=json.loads(Path('.build-live/seed4.json').read_text())
    for i in range(6):m.create(dict(s,id=f'world-{i}',seed=i))
    assert list(m.runs)==['world-2','world-3','world-4','world-5']
    assert json.loads(m.advance('world-5'))['step']==1
