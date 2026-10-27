"""Continuous mode: HabitatRun generator + run_life_live SSE server."""
import json, threading, urllib.request, numpy as np
from .life_loop import HabitatRun, run_habitat, run_habitat_continuous
from .test_deferred_leaves import hab, SUN

def test_generator_t_increases_and_unbounded():
    g = run_habitat_continuous(hab(1.), SUN, None, np.random.default_rng(1), dt=10., plant=True, culture=False, emit_every=5)
    ts = [next(g).tick(1)["t_w"] for _ in range(30)]          # no years given: keeps producing frames
    assert all(b > a for a, b in zip(ts, ts[1:])) and ts[-1] == 1490.

def test_finite_wrapper_matches_stepper():
    r = run_habitat(hab(1.), SUN, None, np.random.default_rng(1), 2000, 10., plant=True, culture=False)
    run = HabitatRun(hab(1.), SUN, None, np.random.default_rng(1), 2000, 10., plant=True, culture=False)
    for _ in run.frames(2000): pass
    r2 = run.report()
    assert r["biosphere"]["alive"] and r2["biosphere"]["alive"]
    assert json.dumps(r, sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)

def test_live_server_sse():
    from ..run_life_live import LiveSim, serve, shutdown
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=16)
    srv, th = serve(sim, port=0, max_years=3000, emit_every=5, realtime_s=0.)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"EventSource" in urllib.request.urlopen(base + "/", timeout=10).read()
        assert json.load(urllib.request.urlopen(base + "/info", timeout=10))["planet"] == "planet2"
        ts = []
        with urllib.request.urlopen(base + "/stream", timeout=30) as s:
            for line in s:
                if line.startswith(b"data: ") and line.strip() != b"data: {}":
                    tk = json.loads(line[6:]); ts.append(tk["t_w"]); assert {"climate", "biosign", "creatures", "n_alive", "culture"} <= set(tk)
                    if len(ts) >= 4: break
        assert len(ts) >= 4 and all(b > a for a, b in zip(ts, ts[1:]))
        th.join(60); assert not th.is_alive()
        snap = json.load(urllib.request.urlopen(base + "/snapshot", timeout=10))
        assert snap["t_w"] >= 2900 and snap["alive"] and len(snap["creatures"]) <= 16
    finally: shutdown(srv, sim)
