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

def _post(base, body):
    req = urllib.request.Request(base + "/control", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=10))

def _wait(pred, timeout=60):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred(): return True
        time.sleep(.05)
    return False

def test_live_multi_pause_history():
    import time
    from ..run_life_live import LiveSim, serve, shutdown
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=8, multi=True, history=50)
    assert sim.multi and len(sim.runs) >= 2
    srv, th = serve(sim, port=0, emit_every=5, realtime_s=0.01)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        info = json.load(urllib.request.urlopen(base + "/info", timeout=10))
        assert info["multi"] and len(info["planets"]) >= 2 and all("look" in q for q in info["planets"]) and info["focus"] == "planet2"
        assert _wait(lambda: sim.seq >= 12)
        snap = json.load(urllib.request.urlopen(base + "/snapshot", timeout=10))
        names = [q["planet"] for q in snap["planets"]]
        assert len(names) >= 2 and len(set(names)) == len(names) and snap["focus"] == "planet2"
        assert len({q["t_w"] for q in snap["planets"]}) == 1 and snap["planets"][0]["t_w"] == snap["t_w"]     # lockstep
        # pause: seq frozen; resume: advances
        assert _post(base, {"action": "pause"})["paused"]
        time.sleep(.2); s0 = sim.seq; time.sleep(.4); assert sim.seq == s0
        st = _post(base, {"action": "set_speed", "speed": 0}); assert st["realtime_s"] == 0.
        assert _post(base, {"action": "resume"})["paused"] is False and _wait(lambda: sim.seq > s0 + 3)
        # history + replay
        h = json.load(urllib.request.urlopen(base + "/history?creatures=0", timeout=10))
        assert len(h["ticks"]) >= 10 and "creatures" not in h["ticks"][0]["planets"][0]
        st = _post(base, {"action": "rewind", "steps": 5}); assert st["mode"] == "replay" and st["paused"]
        rep = json.load(urllib.request.urlopen(base + "/snapshot", timeout=10)); live_t = sim.latest["t_w"]
        assert rep["mode"] == "replay" and rep["t_w"] < live_t and rep["seq"] == st["replay_seq"]
        old = h["ticks"][0]["seq"]; rep2 = _post(base, {"action": "seek", "seq": old}); assert rep2["replay_seq"] == old
        time.sleep(.3); s1 = sim.seq; time.sleep(.3); assert sim.seq == s1        # replay pauses the sim (an in-flight step may finish first)
        assert _post(base, {"action": "live"})["mode"] == "live" and _wait(lambda: sim.seq > s1)
        q = urllib.request.Request(base + "/control?pause=1&realtime_s=0.2", data=b"", method="POST")
        st = json.load(urllib.request.urlopen(q, timeout=10)); assert st["paused"] and st["realtime_s"] == .2
    finally: shutdown(srv, sim)
