"""Continuous (open-ended) life simulation streamed over localhost Server-Sent Events.
python -m universe.run_life_live --seed 4 --dt 20 [--port 8765] [--emit-every 1] [--realtime-s 0.05] [--max-years 0] [--plant] [--planet planet2]
  GET /         live Three.js viewer (viewer_src/life_live_template.html + life_live.js)
  GET /stream   text/event-stream; each event `data: {tick JSON}` (see HabitatRun.tick)
  GET /snapshot latest tick JSON
  GET /info     static run info (seed, planet look, dt, ...)
Ctrl+C stops.  Unlike run_life (fixed --years -> JSON) this never needs an end time; --max-years caps it (0 = unbounded)."""
import argparse, json, os, threading, time, numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from .sim2 import make_system, DEFAULTS
from .run_life import load_lp
from . import bodies
from .leaves import world
from .leaves.packets import as_bus
from .leaves.life_loop import host_weather, HabitatRun
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

class LiveSim:
    def __init__(self, seed=4, dt=20., plant=False, culture=True, params=None, config=None, distance_pc=10., planet=None, max_show=48, rng="numpy"):
        C = json.load(open(config or os.path.join(HERE, "config3.json"))); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
        if rng == "nim": from .nim_rng import NIMRandom
        r = NIMRandom(seed) if rng == "nim" else np.random.default_rng(seed)
        S = make_system(r, C, DEFAULTS); LP = load_lp(params); P = as_bus(LP)
        hw = host_weather(S, distance_pc, None, seed=seed); h = pick_habitat(habitats_from_system(S, seed, P), hw, planet)
        self.seed, self.max_show = seed, max_show
        self.run = HabitatRun(h, hw, LP, np.random.default_rng(seed), None, dt, plant, culture=culture)
        hd = h.to_dict()
        self.info = dict(seed=seed, dt=dt, planet=hd["name"], ptype=hd["ptype"], plant=plant, culture=culture, T_eff=float(hw.T_eff) if hasattr(hw, "T_eff") else 5800.,
                         look=planet_look(seed, hd), title=f"NIM life LIVE · seed {seed} · {hd['name']}")
        self.latest = None; self.seq = 0; self.cv = threading.Condition(); self.stop_ev = threading.Event(); self.done = False

    def loop(self, max_years=None, emit_every=1, realtime_s=0.):
        try:
            for run in self.run.frames(max_years, emit_every, self.stop_ev.is_set):
                tk = run.tick(self.seed, self.max_show)
                with self.cv: self.latest = tk; self.seq += 1; self.cv.notify_all()
                if realtime_s > 0: self.stop_ev.wait(realtime_s)
        finally:
            with self.cv: self.done = True; self.cv.notify_all()

def make_handler(sim):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/": return self._send(200, viewer_html(), "text/html; charset=utf-8")
            if p == "/info": return self._send(200, json.dumps(sim.info), "application/json")
            if p == "/snapshot": return self._send(200 if sim.latest else 503, json.dumps(sim.latest or {}), "application/json")
            if p != "/stream": return self._send(404, "not found", "text/plain")
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            self.close_connection = True; seen = 0
            try:
                self.wfile.write(b"retry: 2000\n\n"); self.wfile.flush()
                while not sim.stop_ev.is_set():
                    with sim.cv:
                        sim.cv.wait_for(lambda: sim.seq != seen or sim.done or sim.stop_ev.is_set(), timeout=15)
                        seq, tk, done = sim.seq, sim.latest, sim.done
                    if seq != seen and tk is not None:    # slow clients skip frames: always send the latest tick
                        seen = seq; self.wfile.write(b"data: " + json.dumps(tk, separators=(",", ":")).encode() + b"\n\n")
                    elif done: self.wfile.write(b"event: end\ndata: {}\n\n"); self.wfile.flush(); break
                    else: self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
    return H

def viewer_html():
    R = lambda p: open(os.path.join(HERE, p)).read(); html = R("viewer_src/life_live_template.html")
    for key, val in [("/*THREE*/", R("vendor/three.min.js")), ("/*ORBIT*/", R("vendor/OrbitControls.js")), ("/*SHADERS*/", R("viewer_src/shaders.js")), ("/*APP*/", R("viewer_src/life_live.js"))]:
        html = html.replace(key, val.replace("</script", "<\\/script"), 1)
    return html

def serve(sim, port=8765, host="127.0.0.1", max_years=None, emit_every=1, realtime_s=.05):
    """Start sim + HTTP threads; returns (server, sim_thread). server.server_address[1] is the bound port (port=0 -> ephemeral)."""
    srv = ThreadingHTTPServer((host, port), make_handler(sim)); srv.daemon_threads = True
    th = threading.Thread(target=sim.loop, args=(max_years, emit_every, realtime_s), daemon=True); th.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, th

def shutdown(srv, sim):
    sim.stop_ev.set()
    with sim.cv: sim.cv.notify_all()
    srv.shutdown(); srv.server_close()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4); ap.add_argument("--dt", type=float, default=20.); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--emit-every", type=int, default=1); ap.add_argument("--realtime-s", type=float, default=.05); ap.add_argument("--max-years", type=float, default=0.)
    ap.add_argument("--plant", action="store_true"); ap.add_argument("--no-culture", dest="culture", action="store_false", default=True)
    ap.add_argument("--params", default=None); ap.add_argument("--config", default=None); ap.add_argument("--planet", default=None)
    ap.add_argument("--max-show", type=int, default=48); ap.add_argument("--rng", default="numpy"); ap.add_argument("--distance-pc", type=float, default=10.)
    a = ap.parse_args(argv)
    sim = LiveSim(a.seed, a.dt, a.plant, a.culture, a.params, a.config, a.distance_pc, a.planet, a.max_show, a.rng)
    srv, th = serve(sim, a.port, a.host, a.max_years or None, a.emit_every, a.realtime_s)
    print(f"live: {sim.info['title']} ({sim.info['ptype']}) -> http://{a.host}:{srv.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        while th.is_alive():
            th.join(5); tk = sim.latest
            if tk: print(f"  t_w={tk['t_w']:.0f} T={tk['climate']['T_K']:.0f}K biomass={tk['biomass']:.3f} n_alive={tk['n_alive']} species={tk['n_species']} tech={tk['tech']:.3f}", flush=True)
        print("max-years reached; still serving last tick (Ctrl+C to stop)", flush=True)
        while True: time.sleep(3600)
    except KeyboardInterrupt: print("stopping", flush=True)
    finally: shutdown(srv, sim)

if __name__ == "__main__": main()
