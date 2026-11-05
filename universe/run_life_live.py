"""Continuous (open-ended) life simulation streamed over localhost Server-Sent Events.
python -m universe.run_life_live --seed 4 --dt 20 [--port 8765] [--emit-every 1] [--realtime-s 0.05] [--max-years 0] [--plant] [--planet planet2]
  GET /         live Three.js viewer (viewer_src/life_live_template.html + life_live.js)
  GET /stream   text/event-stream; each event `data: {tick JSON}` (see HabitatRun.tick)
  GET /snapshot latest tick JSON
  GET /info     run info (seed, planets + looks, dt, paused, realtime_s, ...)
  GET /state    paused / realtime_s / mode (live|replay) / history range
  GET /history?from=SEQ&to=SEQ[&creatures=0]  ring buffer of the last --history ticks
  POST /control {action: pause|resume|toggle|set_speed(speed|realtime_s)|seek(seq[,physics])|restore(seq)|rewind(steps)|forward(steps)|live}
  --token SECRET / --token-env VAR / --auth (auto-generate): every endpoint but GET / needs `Authorization: Bearer T` or `?token=T` (else 401)
  --ckpt-every N / --ckpt-depth D: physics checkpoints every N emits; restore(seq) = true branch point (restore + deterministic fast-forward)
  --multi streams every plantable candidate in lockstep: {t_w, step, focus, planets:[tick...]}
Ctrl+C stops.  Unlike run_life (fixed --years -> JSON) this never needs an end time; --max-years caps it (0 = unbounded)."""
import argparse, collections, copy, hmac, json, os, secrets, threading, time, zlib, numpy as np
from urllib.parse import parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from .sim2 import make_system, DEFAULTS
from .run_life import load_lp
from . import bodies
from .leaves import world
from .leaves.packets import as_bus
from .leaves.life_loop import host_weather, HabitatRun
from .leaves import mind as mind_mod
from . import solvers as solvers_mod
from .leaves.system_habitat import habitats_from_system, candidate
HERE = os.path.dirname(__file__)

def pick_habitat(habs, hw, name=None):
    """Primary plantable habitat: --planet name, else in-HZ candidates first, then most liquid water (seed 4 -> planet2 ocean)."""
    from .learn_life import plantable
    if name: return next(h for h in habs if h.name == name)
    c = plantable([h for h in habs if candidate(h)], hw) or [h for h in habs if candidate(h)] or list(habs)
    return max(c, key=lambda h: (bool(h.in_hz), world.step(h, hw, as_bus(None)).liquid_water_frac))

def planet_look(seed, hd):
    bid = hd["planet_id"]
    for _ in range(60):          # same as viewer_life: bodies.py look forced to the life-stack type
        d = bodies.planet_detail(seed, bid, hd["m"], hd["a"], hd["L_star"])
        if d["type"] == hd["ptype"]: break
        bid += 100
    return d

def multi_habitats(habs, hw):
    """All system_habitat candidates (plantable ones + marginal ones like seed-4 planet3 ice, which then just stays lifeless),
    ordered by planet name so lockstep order and per-planet RNG streams are stable."""
    return sorted([h for h in habs if candidate(h)], key=lambda h: h.name)

def planet_seed(seed, name):
    """Independent, order-stable RNG stream per planet (base seed + crc32(name))."""
    return [int(seed), zlib.crc32(name.encode()) & 0x7fffffff]

class LiveSim:
    """Steps one (default) or all plantable habitats (multi=True) in lockstep, with server-side pause/speed and a ring-buffer history.
    Single mode: payload is the plain HabitatRun.tick (+ seq/paused/... meta), unchanged default_rng(seed) trajectory.
    Multi mode: payload = {t_w, step, focus, planets:[tick per planet], + meta}."""
    def __init__(self, seed=4, dt=20., plant=False, culture=True, params=None, config=None, distance_pc=10., planet=None, max_show=48, rng="numpy", multi=False, history=200, realtime_s=.05, ckpt_every=10, ckpt_depth=None, trail_max=24, mind=None, climate_model="lat", solvers=None):
        """mind: None (default) or a factory seed -> leaves.mind object (one independent mind per planet run).
        climate_model / solvers: opt-in universe.solvers (see life_loop); defaults keep the leaf behaviour."""
        C = json.load(open(config or os.path.join(HERE, "config3.json"))); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
        if rng == "nim": from .nim_rng import NIMRandom
        r = NIMRandom(seed) if rng == "nim" else np.random.default_rng(seed)
        S = make_system(r, C, DEFAULTS); LP = load_lp(params); P = as_bus(LP)
        hw = host_weather(S, distance_pc, None, seed=seed); habs = habitats_from_system(S, seed, P)
        h0 = pick_habitat(habs, hw, planet)
        hs = multi_habitats(habs, hw) if (multi and not planet) else [h0]
        if h0 not in hs: hs = [h0] + hs
        self.multi = len(hs) > 1 or (multi and not planet)
        self.seed, self.max_show = seed, max_show
        sv = dict(solvers or {}); orbs = {}
        from .leaves.life_loop import apply_stellar_solver; apply_stellar_solver(hw, sv)
        if sv.get("orbit_solver") and climate_model in ("lat", "gcm2d", "climate_table", "exoplasim"):
            from .solvers import secular_nbody; orbs = secular_nbody.drivers_from_system(S, habs, seed, sv["orbit_solver"], **secular_nbody.tide_kw(sv.get("orbit_tides")))
        self.runs = [HabitatRun(h, hw, LP, np.random.default_rng(planet_seed(seed, h.name) if self.multi else seed), None, dt, plant, culture=culture,
                                mind=mind(planet_seed(seed, h.name) if self.multi else seed) if mind else None,
                                climate_model=climate_model, orbit=orbs.get(h.planet_id), solvers=sv or None) for h in hs]
        self.run = self.runs[[h.name for h in hs].index(h0.name)]
        hd = h0.to_dict(); self.focus = hd["name"]
        pls = [dict(planet=x.to_dict()["name"], ptype=x.to_dict()["ptype"], in_hz=bool(x.in_hz), look=planet_look(seed, x.to_dict())) for x in hs]
        self.info = dict(seed=seed, dt=dt, planet=hd["name"], ptype=hd["ptype"], plant=plant, culture=culture, T_eff=float(hw.T_eff) if hasattr(hw, "T_eff") else 5800.,
                         look=next(q["look"] for q in pls if q["planet"] == self.focus), multi=self.multi, focus=self.focus, planets=pls, history=history,
                         title=f"NIM life LIVE · seed {seed} · " + (f"{len(hs)} planets" if self.multi else hd["name"]))
        self.latest = None; self.seq = 0; self.cv = threading.Condition(); self.stop_ev = threading.Event(); self.done = False
        self.paused = False; self.realtime_s = float(realtime_s); self.base_realtime_s = float(realtime_s); self.wake = threading.Event()
        self.hist = collections.deque(maxlen=max(1, int(history))); self.replay = None; self.version = 0
        self.trail_max = int(trail_max); self.emit_every = 1; self.restored = None
        # physics checkpoints: {seq: deepcopy(runs)} every ckpt_every emits (seq 0 = initial state). step_lock serialises stepping vs restore.
        self.ckpt_every = max(0, int(ckpt_every)); self.ckpts = collections.OrderedDict(); self.step_lock = threading.Lock()
        self.ckpt_depth = max(1, int(ckpt_depth if ckpt_depth else max(2, int(history)//max(1, self.ckpt_every) + 2)))
        if self.ckpt_every: self._checkpoint(0)

    # ---- state / control ----
    def _meta(self):
        return dict(paused=self.paused, realtime_s=self.realtime_s, mode="replay" if self.replay is not None else "live", live_seq=self.seq,
                    hist_from=self.hist[0]["seq"] if self.hist else 0, hist_to=self.hist[-1]["seq"] if self.hist else 0)
    def current(self):
        """Tick the viewer should show now: the live frontier, or the buffered tick under replay. Meta fields are refreshed."""
        with self.cv:
            tk = self.latest if self.replay is None else next((x for x in self.hist if x["seq"] == self.replay), self.latest)
            return None if tk is None else {**tk, **self._meta()}
    def state(self):
        with self.cv: return dict(self._meta(), replay_seq=self.replay, done=self.done, multi=self.multi, n_planets=len(self.runs), ckpt_every=self.ckpt_every, ckpts=list(self.ckpts))
    def _checkpoint(self, seq):
        """Deep copy of every HabitatRun (biosphere, chem, culture, climate history, Population arrays incl. trails, rng/drng/orng
        bit-generator state, HostWeather copy, k / series ...). No creature/mesh data: ticks are rebuilt from state."""
        self.ckpts[seq] = copy.deepcopy(self.runs)
        while len(self.ckpts) > self.ckpt_depth + (0 in self.ckpts):     # seq-0 (initial state) is always kept on top of the ring
            del self.ckpts[next(k for k in self.ckpts if k != 0)]
    def _advance(self, ee):
        r0 = self.runs[0]
        for _ in range(ee):
            for run in self.runs: run.step()            # lockstep: same step index / world time on every planet
            if r0.k % ee == 0: break
    def _emit(self):
        return self._payload([run.tick(self.seed, self.max_show, self.trail_max) for run in self.runs])
    def restore(self, seq):
        """TRUE reverse (relative to view-only buffer replay): restore the newest physics checkpoint <= seq, deterministically
        fast-forward emits to reach seq exactly, make that the live frontier, discard history/checkpoints after seq, pause.
        This is checkpoint restore + replay of the same RNG stream, NOT backward ODE integration. Resuming branches from there."""
        if not self.ckpt_every: raise ValueError("checkpoints disabled (--ckpt-every 0)")
        seq = int(seq)
        with self.step_lock:
            with self.cv:
                seq = max(1, min(seq, self.seq)); cs = [c for c in self.ckpts if c <= seq]
                if not cs: raise ValueError(f"no checkpoint <= seq {seq} (oldest {next(iter(self.ckpts), None)})")
                c = max(cs); self.paused = True
            runs = copy.deepcopy(self.ckpts[c])            # keep the checkpoint itself pristine for later restores
            name = self.run.h.name; self.runs = runs; self.run = next(r for r in runs if r.h.name == name)
            for _ in range(seq - c): self._advance(self.emit_every)
            tk = self._emit(); tk["seq"] = seq
            with self.cv:
                self.seq = seq; self.latest = tk; self.replay = None
                kept = [x for x in self.hist if x["seq"] < seq]; self.hist.clear(); self.hist.extend(kept); self.hist.append(tk)
                for k in [k for k in self.ckpts if k > seq]: del self.ckpts[k]
                self.restored = dict(seq=seq, from_ckpt=c, fast_forward=seq - c, t_w=tk["t_w"]); self._bump()
        return dict(self.state(), restored=self.restored)
    def _bump(self): self.version += 1; self.cv.notify_all(); self.wake.set()
    def control(self, action, **kw):
        if action == "restore" or (action == "seek" and str(kw.get("physics", "")).lower() in ("1", "true")):
            return self.restore(kw["seq"])
        return self._control(action, **kw)
    def _control(self, action, **kw):
        """pause | resume | toggle | set_speed(speed=x relative to --realtime-s, or realtime_s=abs; 0 = max) | seek(seq) | rewind(steps) | forward(steps) | live."""
        with self.cv:
            if action == "pause": self.paused = True
            elif action == "resume": self.paused = False; self.replay = None
            elif action == "toggle": self.paused = not self.paused; self.replay = None if not self.paused else self.replay
            elif action == "set_speed":
                if kw.get("realtime_s") is not None: self.realtime_s = max(0., float(kw["realtime_s"]))
                else:
                    sp = float(kw.get("speed", 1.)); self.realtime_s = 0. if sp <= 0 else self.base_realtime_s/sp
            elif action in ("seek", "rewind", "forward"):
                if not self.hist: raise ValueError("history empty")
                lo, hi = self.hist[0]["seq"], self.hist[-1]["seq"]; cur = self.replay if self.replay is not None else hi
                tgt = int(kw["seq"]) if action == "seek" else cur + (-1 if action == "rewind" else 1)*int(kw.get("steps", 1))
                self.replay = min(hi, max(lo, tgt)); self.paused = True          # rewind is view-only over the buffer: sim pauses
            elif action == "live": self.replay = None; self.paused = bool(kw.get("paused", False))
            else: raise ValueError(f"unknown action {action!r}")
            self._bump()
        return self.state()
    def history(self, frm=None, to=None, creatures=True):
        with self.cv: xs = [x for x in self.hist if (frm is None or x["seq"] >= frm) and (to is None or x["seq"] <= to)]
        if creatures: return xs
        strip = lambda t: {k: v for k, v in t.items() if k != "creatures"}
        return [dict(strip(x), planets=[strip(q) for q in x["planets"]]) if "planets" in x else strip(x) for x in xs]

    def _payload(self, ticks):
        if not self.multi: return ticks[0]
        return dict(t_w=ticks[0]["t_w"], step=ticks[0]["step"], focus=self.focus, planets=ticks)

    def loop(self, max_years=None, emit_every=1, realtime_s=None):
        if realtime_s is not None: self.realtime_s = self.base_realtime_s = float(realtime_s)
        ee = self.emit_every = max(1, int(emit_every))
        try:
            while not self.stop_ev.is_set() and (max_years is None or max_years <= 0 or self.runs[0].t < max_years):
                with self.cv: self.cv.wait_for(lambda: not self.paused or self.stop_ev.is_set())
                if self.stop_ev.is_set(): break
                with self.step_lock:
                    with self.cv:
                        if self.paused: continue                # a restore/pause slipped in between
                    self._advance(ee); tk = self._emit()
                    with self.cv:
                        self.seq += 1; tk["seq"] = self.seq; self.latest = tk; self.hist.append(tk)
                        if self.ckpt_every and self.seq % self.ckpt_every == 0: self._checkpoint(self.seq)
                        self._bump()
                if self.realtime_s > 0: self.wake.clear(); self.wake.wait(self.realtime_s)
        finally:
            with self.cv: self.done = True; self._bump()

def make_handler(sim, token=None):
    class H(BaseHTTPRequestHandler):
        def _authed(self, qs):
            """No token -> open (unchanged). Else Bearer header or ?token= must match (constant-time); GET / is always served."""
            if not token: return True
            h = self.headers.get("Authorization") or ""; got = h[7:].strip() if h.lower().startswith("bearer ") else qs.get("token", "")
            if hmac.compare_digest(str(got).encode(), token.encode()): return True
            self.send_response(401); b = b'{"error":"unauthorized"}'
            self.send_header("WWW-Authenticate", 'Bearer realm="nim-live"'); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            return False
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        def _json(self, code, obj): return self._send(code, json.dumps(obj, separators=(",", ":")), "application/json")
        def do_POST(self):
            p, _, q = self.path.partition("?")
            if p != "/control": return self._send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
            qs = {k: v[-1] for k, v in parse_qs(q).items()}
            if not self._authed(qs): return
            try:
                d = json.loads(body) if body.strip() else {}
                if not d.get("action"):          # query form: ?pause=1 / ?pause=0 / ?realtime_s=0.2 / ?speed=4 / ?seq=N / ?rewind=10 / ?live=1
                    if "pause" in qs: sim.control("pause" if qs["pause"] not in ("0", "false") else "resume")
                    if "realtime_s" in qs or "speed" in qs: sim.control("set_speed", realtime_s=qs.get("realtime_s"), speed=qs.get("speed", 1))
                    if "seq" in qs: sim.control("restore" if qs.get("physics") in ("1", "true") else "seek", seq=qs["seq"])
                    if "restore" in qs: sim.control("restore", seq=qs["restore"])
                    if "rewind" in qs: sim.control("rewind", steps=qs["rewind"])
                    if "live" in qs: sim.control("live")
                    return self._json(200, sim.state())
                return self._json(200, sim.control(d.pop("action"), **d))
            except Exception as e: return self._json(400, dict(error=str(e)))
        def do_GET(self):
            p, _, q = self.path.partition("?"); qs = {k: v[-1] for k, v in parse_qs(q).items()}
            if p == "/": return self._send(200, viewer_html(), "text/html; charset=utf-8")
            if p in ("/info", "/state", "/snapshot", "/history", "/stream") and not self._authed(qs): return
            if p == "/info": return self._json(200, dict(sim.info, **sim.state()))
            if p == "/state": return self._json(200, sim.state())
            if p == "/snapshot": tk = sim.current(); return self._json(200 if tk else 503, tk or {})
            if p == "/history":
                xs = sim.history(int(qs["from"]) if "from" in qs else None, int(qs["to"]) if "to" in qs else None, qs.get("creatures", "1") not in ("0", "false"))
                return self._json(200, dict(sim.state(), ticks=xs))
            if p != "/stream": return self._send(404, "not found", "text/plain")
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            self.close_connection = True; seen = -1; ended = False
            try:
                self.wfile.write(b"retry: 2000\n\n"); self.wfile.flush()
                while not sim.stop_ev.is_set():
                    with sim.cv:
                        sim.cv.wait_for(lambda: sim.version != seen or sim.stop_ev.is_set(), timeout=15)
                        ver, done = sim.version, sim.done
                    tk = sim.current()
                    if ver != seen:    # slow clients skip frames: always send the current view (live frontier or replay tick)
                        seen = ver
                        if tk is not None: self.wfile.write(b"data: " + json.dumps(tk, separators=(",", ":")).encode() + b"\n\n")
                    elif ver == seen: self.wfile.write(b": keepalive\n\n")
                    if done and not ended: ended = True; self.wfile.write(b"event: end\ndata: {}\n\n")   # stream stays open so replay seeks still reach the viewer
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
    return H

def viewer_html():
    R = lambda p: open(os.path.join(HERE, p)).read(); html = R("viewer_src/life_live_template.html")
    for key, val in [("/*THREE*/", R("vendor/three.min.js")), ("/*ORBIT*/", R("vendor/OrbitControls.js")), ("/*SHADERS*/", R("viewer_src/shaders.js")), ("/*APP*/", R("viewer_src/creature_mesh.js")+"\n"+R("viewer_src/life_live.js"))]:
        html = html.replace(key, val.replace("</script", "<\\/script"), 1)
    return html

def serve(sim, port=8765, host="127.0.0.1", max_years=None, emit_every=1, realtime_s=None, token=None):
    """Start sim + HTTP threads; returns (server, sim_thread). server.server_address[1] is the bound port (port=0 -> ephemeral)."""
    srv = ThreadingHTTPServer((host, port), make_handler(sim, token)); srv.daemon_threads = True
    th = threading.Thread(target=sim.loop, args=(max_years, emit_every, realtime_s), daemon=True); th.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, th

def shutdown(srv, sim):
    sim.stop_ev.set()
    with sim.cv: sim._bump()
    srv.shutdown(); srv.server_close()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4); ap.add_argument("--dt", type=float, default=20.); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--emit-every", type=int, default=1); ap.add_argument("--realtime-s", type=float, default=.05); ap.add_argument("--max-years", type=float, default=0.)
    ap.add_argument("--plant", action="store_true"); ap.add_argument("--no-culture", dest="culture", action="store_false", default=True)
    ap.add_argument("--params", default=None); ap.add_argument("--config", default=None); ap.add_argument("--planet", default=None)
    ap.add_argument("--max-show", type=int, default=48); ap.add_argument("--rng", default="numpy"); ap.add_argument("--distance-pc", type=float, default=10.)
    ap.add_argument("--multi", action="store_true", help="stream all plantable candidates in lockstep (--planet forces single)"); ap.add_argument("--history", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=10, help="physics checkpoint every N emits (0 = off; restore needs it)")
    ap.add_argument("--ckpt-depth", type=int, default=0, help="max checkpoints kept (0 = cover --history)")
    ap.add_argument("--trail-max", type=int, default=24, help="server-side trail points per creature (0 = none)")
    ap.add_argument("--token", default=None, help="require Bearer token / ?token= on every endpoint but /")
    ap.add_argument("--token-env", default=None, help="read the token from this environment variable")
    ap.add_argument("--auth", action="store_true", help="require a token; auto-generated and printed if --token/--token-env not given")
    mind_mod.add_args(ap); solvers_mod.add_args(ap); a = ap.parse_args(argv); cm, sv = solvers_mod.from_args(a)
    token = a.token or (os.environ.get(a.token_env) if a.token_env else None)
    if a.token_env and not token: ap.error(f"--token-env {a.token_env}: variable empty or unset")
    if a.auth and not token: token = secrets.token_urlsafe(16)
    sim = LiveSim(a.seed, a.dt, a.plant, a.culture, a.params, a.config, a.distance_pc, a.planet, a.max_show, a.rng, a.multi, a.history, a.realtime_s,
                  a.ckpt_every, a.ckpt_depth or None, a.trail_max, mind=(lambda s: mind_mod.from_args(a, s)) if a.mind != "none" else None,
                  climate_model=cm, solvers=sv)
    srv, th = serve(sim, a.port, a.host, a.max_years or None, a.emit_every, a.realtime_s, token)
    q = f"?token={token}" if token else ""
    print(f"live: {sim.info['title']} ({sim.info['ptype']}) -> http://{a.host}:{srv.server_address[1]}/{q}  (Ctrl+C to stop)" + ("  [auth on]" if token else ""), flush=True)
    try:
        while th.is_alive():
            th.join(5); tk = sim.latest
            if tk and "planets" in tk: print(f"  t_w={tk['t_w']:.0f} " + " | ".join(f"{q['planet']}: T={q['climate']['T_K']:.0f}K n={q['n_alive']} tech={q['tech']:.2f}" for q in tk["planets"]), flush=True)
            elif tk: print(f"  t_w={tk['t_w']:.0f} T={tk['climate']['T_K']:.0f}K biomass={tk['biomass']:.3f} n_alive={tk['n_alive']} species={tk['n_species']} tech={tk['tech']:.3f}", flush=True)
        print("max-years reached; still serving last tick (Ctrl+C to stop)", flush=True)
        while True: time.sleep(3600)
    except KeyboardInterrupt: print("stopping", flush=True)
    finally: shutdown(srv, sim)

if __name__ == "__main__": main()
