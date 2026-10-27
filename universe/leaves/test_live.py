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

def test_live_auth_token():
    from ..run_life_live import LiveSim, serve, shutdown
    import urllib.error
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=4)
    srv, th = serve(sim, port=0, max_years=400, emit_every=5, realtime_s=0., token="s3cret")
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"EventSource" in urllib.request.urlopen(base + "/", timeout=10).read()     # viewer HTML always served
        assert _wait(lambda: sim.seq >= 2)
        for path, hdr in [("/snapshot", {}), ("/info", {}), ("/stream", {}), ("/snapshot?token=nope", {}), ("/snapshot", {"Authorization": "Bearer nope"})]:
            try: urllib.request.urlopen(urllib.request.Request(base + path, headers=hdr), timeout=10); assert False, path
            except urllib.error.HTTPError as e: assert e.code == 401
        try: _post(base, {"action": "pause"}); assert False
        except urllib.error.HTTPError as e: assert e.code == 401
        r = urllib.request.urlopen(urllib.request.Request(base + "/snapshot", headers={"Authorization": "Bearer s3cret"}), timeout=10)
        assert r.status == 200 and json.load(r)["seq"] >= 2
        assert urllib.request.urlopen(base + "/state?token=s3cret", timeout=10).status == 200
    finally: shutdown(srv, sim)

def test_live_restore_physics_branch():
    """Restore = checkpoint + deterministic fast-forward: re-derived state equals the original tick at that seq, history after it is dropped."""
    from ..run_life_live import LiveSim
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=8, ckpt_every=3, history=100); sim.emit_every = 5
    for _ in range(14):
        sim._advance(5); tk = sim._emit(); sim.seq += 1; tk["seq"] = sim.seq; sim.latest = tk; sim.hist.append(tk)
        if sim.seq % sim.ckpt_every == 0: sim._checkpoint(sim.seq)
    orig = {x["seq"]: x for x in sim.hist}; k_end, t_end = sim.run.k, sim.latest["t_w"]
    st = sim.control("seek", seq=7, physics=True)
    assert st["restored"] == dict(seq=7, from_ckpt=6, fast_forward=1, t_w=orig[7]["t_w"]) and st["paused"] and st["mode"] == "live"
    assert sim.seq == 7 and sim.latest["t_w"] == orig[7]["t_w"] and sim.latest["n_alive"] == orig[7]["n_alive"] and sim.run.k < k_end
    assert json.dumps({k: v for k, v in sim.latest.items() if k != "seq"}, sort_keys=True) == json.dumps({k: v for k, v in orig[7].items() if k != "seq"}, sort_keys=True)
    assert max(x["seq"] for x in sim.hist) == 7 and max(sim.ckpts) <= 7
    sim._advance(5); tk = sim._emit()                                   # continuing from the restored past reproduces seq 8 exactly
    assert tk["t_w"] == orig[8]["t_w"] and tk["n_alive"] == orig[8]["n_alive"] and tk["biomass"] == orig[8]["biomass"]
    sim.control("restore", seq=0); assert sim.seq == 1 and sim.latest["t_w"] == orig[1]["t_w"] and len(sim.hist) == 1    # clamps to seq 1, via checkpoint 0 (initial state)

def test_live_restore_over_http_and_resume():
    from ..run_life_live import LiveSim, serve, shutdown
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=4, ckpt_every=4)
    srv, th = serve(sim, port=0, emit_every=5, realtime_s=0.)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert _wait(lambda: sim.seq >= 15)
        _post(base, {"action": "pause"}); t9 = next(x["t_w"] for x in sim.hist if x["seq"] == 9)
        st = _post(base, {"action": "restore", "seq": 9}); assert st["restored"]["seq"] == 9 and st["live_seq"] == 9 and st["hist_to"] == 9
        snap = json.load(urllib.request.urlopen(base + "/snapshot", timeout=10)); assert snap["seq"] == 9 and snap["t_w"] == t9
        _post(base, {"action": "resume"}); assert _wait(lambda: sim.seq >= 12)
        seqs = [x["seq"] for x in sim.hist]; assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and sim.latest["t_w"] > t9
    finally: shutdown(srv, sim)

def test_live_server_side_trails():
    from ..run_life_live import LiveSim
    sim = LiveSim(seed=4, dt=20., plant=True, max_show=16, ckpt_every=0)
    for _ in range(60): sim._advance(5)                                  # 6000 yr: trails (every 10 yr, 24 pts) are full
    tk = sim._emit(); trs = [c["trail"] for c in tk["creatures"] if c.get("trail")]
    assert tk["creatures"] and trs and all(len(t) >= 2 and len(t[0]) == 2 for t in trs) and max(map(len, trs)) <= 25
    tk0 = sim.run.tick(4, 16, 0); assert tk0["creatures"] and all(c.get("trail") is None for c in tk0["creatures"])   # trail_max=0 -> none
