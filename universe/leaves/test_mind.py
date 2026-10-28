"""Mind plug-in: default identity, heuristic learning + determinism, script / http adapters, clamping."""
import json, numpy as np, pytest
from .life_loop import HabitatRun, run_habitat
from . import individuals as I, mind as M
from .test_deferred_leaves import hab, SUN

def _run(mind=None, years=1500, seed=1):
    r = HabitatRun(hab(1.), SUN, None, np.random.default_rng(seed), years, 10., plant=True, culture=False, mind=mind)
    for _ in r.frames(years): pass
    return r
def _js(x): return json.dumps(x, sort_keys=True, default=str)

def test_no_mind_bit_identical():
    a = run_habitat(hab(1.), SUN, None, np.random.default_rng(1), 1500, 10., plant=True, culture=False)
    b = _run(None).report(); c = _run(M.make_mind("none")).report(); d = _run(M.NullMind()).report()
    assert a["biosphere"]["alive"] and _js(a) == _js(b) == _js(c) == _js(d)

def test_heuristic_learns_and_is_deterministic(tmp_path):
    mk = lambda tp=None: M.make_mind("heuristic", every=20, batch=16, learn=True, seed=3, trace_path=tp)
    m1, m2 = mk(str(tmp_path/"t.jsonl")), mk()
    r1, r2 = _run(m1), _run(m2)
    assert m1.n_learn > 5 and np.abs(m1.W).sum() > 0 and np.array_equal(m1.W, m2.W)
    assert _js(r1.report()) == _js(r2.report())
    assert _js(r1.report()["individuals"]) != _js(_run(None).report()["individuals"])      # behaviour actually differs
    rows = [json.loads(l) for l in open(tmp_path/"t.jsonl")]; assert rows and {"energy0", "energy1", "alive", "obs", "action"} <= set(rows[0])
    m3 = M.make_mind("heuristic", every=20, batch=16, learn=False, seed=3); _run(m3); assert not np.abs(m3.W).sum()

def test_script_mind_forage(tmp_path):
    p = tmp_path/"pol.py"; p.write_text("def decide(obs):\n    return [dict(forage=1.2) for o in obs]\nseen=[]\ndef learn(e): seen.append(len(e))\n")
    m = M.make_mind("script", path=str(p), every=10, batch=10**6, learn=True)
    pop = I.found(None, np.random.default_rng(0), 40)
    class N: resources = .3
    import copy
    clim = HabitatRun(hab(1.), SUN, None, np.random.default_rng(1), 100, 10., plant=True, culture=False); clim.step(); C = clim.clim
    a = copy.deepcopy(pop); b = copy.deepcopy(pop)
    I._substep(a, N, C, I.as_bus(None), np.random.default_rng(5), 1., 1., 50)
    I._substep(b, N, C, I.as_bus(None), np.random.default_rng(5), 1., 1., 50, m)
    assert m.n_queries == 1 and np.allclose(m.act[1], 1.2)
    assert not np.array_equal(a.energy, b.energy) or len(a) != len(b)
    with pytest.raises(AttributeError):
        q = tmp_path/"bad.py"; q.write_text("x = 1\n"); M.make_mind("script", path=str(q))

def test_script_forage_wins_food():
    """Organisms with forage 1.5 out-eat the forage .5 half under shared food."""
    class Half(M.Mind):
        def decide(self, obs): return [dict(forage=1.5 if o["id"] % 2 else .5) for o in obs]
    m = Half(every=1, batch=10**6); pop = I.found(None, np.random.default_rng(0), 60)
    class N: resources = .2
    r = HabitatRun(hab(1.), SUN, None, np.random.default_rng(1), 100, 10., plant=True, culture=False); r.step()
    e0 = dict(zip(pop.id, pop.energy)); I._substep(pop, N, r.clim, I.as_bus(None), np.random.default_rng(2), 1., 1., 50, m)
    d = np.array([pop.energy[i] - e0[k] for i, k in enumerate(pop.id) if k in e0]); ids = np.array([k for k in pop.id if k in e0])
    assert d[ids % 2 == 1].mean() > d[ids % 2 == 0].mean()

def test_http_mind_error_and_ok():
    def boom(*a, **k): raise TimeoutError("no server")
    m = M.HttpMind(url="http://127.0.0.1:9/v1", model="x", post=boom)
    obs = [dict(id=1), dict(id=2)]; out = m.decide(obs); assert out == [M.NEUTRAL]*2 and m.n_fail == 1
    seen = {}
    def ok(url, body, h, t):
        seen.update(url=url, body=body); return {"choices": [{"message": {"content": 'sure: {"actions":[{"id":2,"forage":9},{"id":1,"mate":1.7}]}'}}]}
    m2 = M.HttpMind(url="http://h/v1/", model="m", post=ok); out = m2.decide(obs)
    assert seen["url"] == "http://h/v1/chat/completions" and seen["body"]["model"] == "m"
    assert M.clamp_action(out[0])["mate"] == 1.7 and M.clamp_action(out[1])["forage"] == 1.5
    m3 = M.HttpMind(post=lambda *a: {"choices": [{"message": {"content": "garbage"}}]}); assert m3.decide(obs) == [M.NEUTRAL]*2
    r = _run(M.HttpMind(post=boom, every=50, batch=8)); assert r.report()["mind"]["failures"] > 0 and r.bio.alive   # sim survives

def test_clamp():
    c = M.clamp_action(dict(move=[30, 0, 40], forage=99, mate=-1, rest=float("nan")))
    assert np.isclose(np.linalg.norm(c["move"]), 1) and c["forage"] == 1.5 and c["mate"] == .5 and c["rest"] == 0.
    assert M.clamp_action("junk") == M.NEUTRAL and M.clamp_action(dict(move=[1, "a"]))["move"] is None
    assert M.clamp_action(dict(move=[.1, 0, 0], forage=.1, mate=5, rest=3)) == dict(move=[.1, 0., 0.], forage=.5, mate=2., rest=1.)

def test_make_mind_and_cli():
    import argparse
    assert M.make_mind("none") is None and isinstance(M.make_mind("heuristic"), M.HeuristicMind)
    with pytest.raises(ValueError): M.make_mind("gpt")
    ap = argparse.ArgumentParser(); M.add_args(ap)
    a = ap.parse_args(["--mind", "http", "--mind-url", "http://127.0.0.1:11434/v1", "--mind-model", "qwen", "--mind-every", "5", "--mind-batch", "4"])
    m = M.from_args(a, 1); assert m.url.endswith("11434/v1") and m.model == "qwen" and m.every == 5 and m.batch == 4

def test_deepcopy_checkpoint():
    import copy
    m = M.make_mind("heuristic", learn=True, seed=2); r = _run(m, 800); c = copy.deepcopy(r)
    for _ in range(20): r.step(); c.step()
    assert np.array_equal(r.mind.W, c.mind.W) and _js(r.report()) == _js(c.report())
