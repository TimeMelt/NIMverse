"""Mind leaf: optional pluggable 'intelligence' for individuals (default OFF -> zero effect, no extra rng draws).
Physics stays authoritative: a mind only emits SOFT, CLAMPED nudges that the individuals leaf applies.
Every `every` world-yr the mind sees up to `batch` living organisms (own rng subsample, never the sim rng) and returns actions;
actions persist for those ids until the next query (others / newborns act neutral).

obs:    {id, traits:{hardiness,metabolism_eff,mobility,reproduction_rate} (0..1 normalised), energy (0..1 of E_MAX), stress, food (per-capita
         food signal), density (1 = uniform), lat, lon, species, gen, culture?:{level,pollution,mitigation}}
action: {move:[dx,dy,dz] or null, forage:float~1, mate:float~1, rest:float~0}
clamps: move -> tangent of the sphere, |move| <= 1 (x MOVE_K x the organism's own random-walk step); forage in [.5,1.5] (multiplies
        the competitive food weight; uptake cap unchanged); mate in [.5,2] (candidate attractiveness in mate choice); rest in [0,1]
        (cuts basal burn by <= REST_K and intake by <= 2 REST_K).
experience (learn): {id, t0, t1, energy0, energy1, alive, obs, action}
Kinds (make_mind): none|null -> NullMind; heuristic -> HeuristicMind (built-in online linear policy, REINFORCE-lite);
circuit|inceptron -> CircuitMind (QWRK-structured PennyLane circuit: XY AngleEmbedding, RX/RY/RZ body, PauliX readout, 5 wires = HEADS);
http -> HttpMind (DEPRECATED for steering, back-compat only); script -> ScriptMind (user .py with decide / optional learn).
Talk (Ollama / OpenAI-compatible): optional `mind.talk = Talk(...)` narrates already-chosen actions after decide; logged only."""
import json, os, importlib.util, urllib.request, numpy as np
from .params import TRAITS, LIFE_SPACE

LO = np.array([LIFE_SPACE[t][0] for t in TRAITS]); SPAN = np.array([LIFE_SPACE[t][1] for t in TRAITS]) - LO
MOVE_K, REST_K = 1., .3
FORAGE_R, MATE_R, REST_R = (.5, 1.5), (.5, 2.), (0., 1.)
NEUTRAL = dict(move=None, forage=1., mate=1., rest=0.)

def _num(x, d):
    try: x = float(x); return x if np.isfinite(x) else d
    except (TypeError, ValueError): return d

def clamp_action(a):
    """Any dict (or junk) -> safe action. Missing/NaN fields fall back to neutral."""
    if not isinstance(a, dict): return dict(NEUTRAL)
    mv = a.get("move")
    try:
        mv = np.asarray(mv, float).reshape(3) if mv is not None else None
        if mv is not None and not np.isfinite(mv).all(): mv = None
    except (TypeError, ValueError): mv = None
    if mv is not None:
        nm = float(np.linalg.norm(mv)); mv = (mv/nm if nm > 1 else mv).tolist()
    return dict(move=mv, forage=float(np.clip(_num(a.get("forage"), 1.), *FORAGE_R)), mate=float(np.clip(_num(a.get("mate"), 1.), *MATE_R)),
                rest=float(np.clip(_num(a.get("rest"), 0.), *REST_R)))

class NullMind:
    """Default: never due, never acts."""
    kind = "null"; active = False; every = 20.; batch = 16; learn_enabled = False; trace_path = None
    def decide(self, obs): return [dict(NEUTRAL) for _ in obs]
    def learn(self, exps): pass
    def summary(self): return dict(kind=self.kind)

class Mind(NullMind):
    """Shared scheduler/state; subclasses implement decide (+ learn)."""
    active = True
    def __init__(self, every=20., batch=16, learn=False, trace_path=None, seed=0, select="random"):
        self.every, self.batch, self.learn_enabled, self.trace_path, self.select = float(every), int(batch), bool(learn), trace_path, select
        self.rng = np.random.default_rng([int(seed) % 2**31, 97]); self.last_t = -1e18; self.ids = np.zeros(0, int); self.act = None
        self.pending = None; self.culture = None; self.talk = None; self.n_queries = 0; self.n_learn = 0; self.n_fail = 0
    def _trace(self, rows):
        if not self.trace_path: return
        with open(self.trace_path, "a") as f:
            for r in rows: f.write(json.dumps(r, default=float) + "\n")
    def __deepcopy__(self, memo):      # live checkpoints: learner state is copied; module / transport handles are shared
        import copy; c = object.__new__(type(self)); memo[id(self)] = c
        for k, v in self.__dict__.items(): setattr(c, k, v if k in ("mod", "post", "culture_advise", "_q", "talk") else copy.deepcopy(v, memo))
        return c
    def summary(self): return dict(kind=self.kind, every=self.every, batch=self.batch, learn=self.learn_enabled, queries=self.n_queries, learn_calls=self.n_learn, failures=self.n_fail, **({"talk": self.talk.summary()} if self.talk is not None else {}))

def observe(pop, idx, stress, dens, food_pc, culture=None):
    from .individuals import E_MAX, latlon
    x = (pop.traits[idx] - LO)/SPAN; la, lo = latlon(pop.pos[idx]); out = []
    for j, i in enumerate(idx):
        o = dict(id=int(pop.id[i]), traits={t: round(float(v), 4) for t, v in zip(TRAITS, x[j])}, energy=round(float(pop.energy[i]/E_MAX), 4),
                 stress=round(float(stress[i]), 4), food=round(float(food_pc), 4), density=round(float(dens[i]), 4), lat=round(float(la[j]), 4),
                 lon=round(float(lo[j]), 4), species=int(pop.species[i]), gen=int(pop.gen[i]))
        if culture: o["culture"] = culture
        out.append(o)
    return out

def tick(mind, pop, stress, dens, food_pc):
    """Called by individuals._substep when the mind is active. Learns from the last batch, re-queries when due, and returns
    per-organism arrays aligned to pop (move[n,3], forage[n], mate[n], rest[n]). Uses only mind.rng."""
    n = len(pop)
    if pop.t - mind.last_t >= mind.every - 1e-9:
        if mind.pending is not None and (mind.learn_enabled or mind.trace_path):
            ids, e0, t0, obs, acts = mind.pending; pos = {int(k): i for i, k in enumerate(pop.id)}
            exps = [dict(id=int(k), t0=t0, t1=float(pop.t), energy0=float(e0[j]), energy1=float(pop.energy[pos[int(k)]]) if int(k) in pos else 0.,
                         alive=int(k) in pos, obs=obs[j], action=acts[j]) for j, k in enumerate(ids)]
            mind._trace(exps)
            if mind.learn_enabled:
                try: mind.learn(exps); mind.n_learn += 1
                except Exception: mind.n_fail += 1
        k = min(n, mind.batch)
        if mind.select == "energy": idx = np.sort(np.argsort(-pop.energy)[:k])
        else: idx = np.sort(mind.rng.choice(n, k, replace=False)) if k else np.zeros(0, int)
        obs = observe(pop, idx, stress, dens, food_pc, mind.culture)
        try: raw = mind.decide(obs); raw = raw if isinstance(raw, list) and len(raw) == len(obs) else [None]*len(obs)
        except Exception: raw = [None]*len(obs); mind.n_fail += 1
        acts = [clamp_action(a) for a in raw]; mind.n_queries += 1; mind.last_t = float(pop.t)
        mind.ids = pop.id[idx].copy()
        mind.act = (np.array([a["move"] or [0., 0., 0.] for a in acts]).reshape(-1, 3), np.array([a["forage"] for a in acts]),
                    np.array([a["mate"] for a in acts]), np.array([a["rest"] for a in acts]))
        mind.pending = (mind.ids, pop.energy[idx].copy(), float(pop.t), obs, acts)
        if getattr(mind, "talk", None) is not None:          # narration only: after actions are fixed, never read back
            try: mind.talk.explain(obs, acts, float(pop.t))
            except Exception: pass
    mv, fo, ma, re = np.zeros((n, 3)), np.ones(n), np.ones(n), np.zeros(n)
    if mind.act is not None and len(mind.ids) and n:
        o = np.argsort(mind.ids); s = mind.ids[o]; j = np.clip(np.searchsorted(s, pop.id), 0, len(s) - 1); hit = s[j] == pop.id; q = o[j[hit]]
        mv[hit], fo[hit], ma[hit], re[hit] = mind.act[0][q], mind.act[1][q], mind.act[2][q], mind.act[3][q]
    return mv, fo, ma, re

def tangent(pos, v):
    return v - (v*pos).sum(1, keepdims=True)*pos

# ---------------------------------------------------------------- built-in learner
FEATS = ("hardiness", "metabolism_eff", "mobility", "reproduction_rate", "energy", "stress", "food", "density", "abs_lat", "culture", "bias")
HEADS = ("north", "east", "forage", "mate", "rest")

def featurize(o):
    t = o["traits"]; c = o.get("culture") or {}
    return np.array([t[k] for k in TRAITS] + [o["energy"], np.tanh(o["stress"]), np.tanh(o["food"]), np.tanh(o["density"] - 1), abs(o["lat"])/(np.pi/2),
                     c.get("level", 0.), 1.])

def _frame(lat, lon):
    n = np.array([-np.sin(lat)*np.cos(lon), np.cos(lat), -np.sin(lat)*np.sin(lon)]); e = np.array([-np.sin(lon), 0., np.cos(lon)]); return n, e   # y-up (individuals.latlon)

class HeuristicMind(Mind):
    """Linear policy a = tanh(W f) + sigma eps over trait/energy/stress/food/density features. learn(): REINFORCE-lite
    W += lr (r - baseline) eps f^T / sigma with r = d(energy)/E_MAX - death_pen*(died); baseline = running mean. Deterministic per seed."""
    kind = "heuristic"
    def __init__(self, lr=.05, sigma=.3, death_pen=1., **kw):
        super().__init__(**kw); self.lr, self.sigma, self.death_pen = lr, sigma, death_pen
        self.W = np.zeros((len(HEADS), len(FEATS))); self.base = 0.; self.eps = {}
    def decide(self, obs):
        out = []
        for o in obs:
            f = featurize(o); eps = self.rng.normal(size=len(HEADS)); a = np.tanh(self.W @ f) + self.sigma*eps; self.eps[o["id"]] = (f, eps)
            nv, ev = _frame(o["lat"], o["lon"])
            out.append(dict(move=(a[0]*nv + a[1]*ev).tolist(), forage=1 + .5*a[2], mate=1 + .75*a[3], rest=.5 + .5*a[4]))
        return out
    def learn(self, exps):
        if not exps: return
        r = np.array([(e["energy1"] - e["energy0"])/12. - self.death_pen*(not e["alive"]) for e in exps])
        g = np.zeros_like(self.W); m = 0
        for e, ri in zip(exps, r):
            fe = self.eps.pop(e["id"], None)
            if fe is not None: g += (ri - self.base)*np.outer(fe[1], fe[0])/self.sigma; m += 1
        if m: self.W = np.clip(self.W + self.lr*g/m, -3, 3)
        self.base += .2*(float(r.mean()) - self.base); self.eps.clear()
    def summary(self): return dict(super().summary(), W=np.round(self.W, 4).tolist(), feats=FEATS, heads=HEADS, baseline=round(self.base, 5))


# ---------------------------------------------------------------- Inceptron circuit mind (hot path)
class CircuitMind(Mind):
    """QWRK-structured variational circuit (TimeMelt/QWRK QWRK.py motifs, as adapted in NIM-Drone-Autopilot drone/circuit.py arch="qwrk"),
    PennyLane default.qubit, backprop. Layout:
      wires   5 = one per HEAD (north, east, forage, mate, rest): the readout of wire q IS head q, so no classical decoder is needed.
      feats   all 11 FEATS enter via a trainable linear projection x = tanh(E f) * pi/2  (E: [5, 11], last FEAT is bias), i.e. the
              QWRK AngleEmbedding gets an n_wires-length vector built from every feature (option (a); tanh bounds angles to |x|<=pi/2,
              the range where XY prep + PauliX stays odd/monotonic for signed heads).
      prep    AngleEmbedding(x, rotation="X") then AngleEmbedding(x, rotation="Y")  (QWRK X,Y,Z prep minus the Z-on-|0> phase step).
      body    QWRK order: for each wire, for each layer: RX(W[l,q,0]) -> RY(W[l,q,1]) -> RZ(W[l,q,2]) (per-wire weights).
              entangle: "layer" (default, sparse: one CNOT ring after each layer's per-wire block), "off", or "dense"
              (QWRK-literal ring after every gate; hurts per-head locality).
      readout <PauliX_q> in [-1,1], sign-preserving (move north/east are signed). f=0 / x=0 -> |0> -> <X>=0, so the start is neutral-ish.
    Policy a = <X>(f) + sigma eps -> same action dict as HeuristicMind. learn(): REINFORCE-lite, reward r = dE/E_MAX - death_pen*died,
    ascent on mean (r-b) eps.<X>(f)/sigma, gradients for E and W by backprop. No QrnlOpt/KDE/RBM. Deterministic per seed (init +
    exploration from self.rng). Any circuit error -> neutral actions, n_fail++."""
    kind = "circuit"; ENTANGLE = ("layer", "off", "dense")
    def __init__(self, layers=2, lr=.05, sigma=.3, death_pen=1., entangle="layer", **kw):
        super().__init__(**kw); self.layers, self.lr, self.sigma, self.death_pen = int(layers), lr, sigma, death_pen
        if entangle not in self.ENTANGLE: raise ValueError(f"entangle must be one of {self.ENTANGLE}")
        nq, nf = len(HEADS), len(FEATS); self.n_wires, self.entangle = nq, entangle
        self.E = self.rng.normal(0, .3, (nq, nf)); self.W = self.rng.normal(0, .1, (self.layers, nq, 3))
        self.base = 0.; self.eps = {}; self._q = None
    def _qnode(self):
        if self._q is None:
            import pennylane as qml
            nq, L, ent = self.n_wires, self.layers, self.entangle; dev = qml.device("default.qubit", wires=nq)
            def ring():
                for q in range(nq): qml.CNOT(wires=[q, (q + 1) % nq])
            @qml.qnode(dev, interface="autograd", diff_method="backprop")
            def circ(f, E, W):
                x = qml.math.tanh(qml.math.tensordot(E, f, axes=[[1], [0]]))*(np.pi/2)
                qml.AngleEmbedding(x, wires=range(nq), rotation="X"); qml.AngleEmbedding(x, wires=range(nq), rotation="Y")
                for l in range(L):
                    for q in range(nq):
                        for g, op in enumerate((qml.RX, qml.RY, qml.RZ)):
                            op(W[l][q][g], wires=q)
                            if ent == "dense": ring()
                    if ent == "layer": ring()
                return [qml.expval(qml.PauliX(q)) for q in range(nq)]
            self._q = circ
        return self._q
    def z(self, f): return np.array(self._qnode()(np.asarray(f, float), self.E, self.W), dtype=float)
    def decide(self, obs):
        out = []
        try:
            for o in obs:
                f = featurize(o); eps = self.rng.normal(size=len(HEADS)); a = self.z(f) + self.sigma*eps; self.eps[o["id"]] = (f, eps)
                nv, ev = _frame(o["lat"], o["lon"])
                out.append(dict(move=(a[0]*nv + a[1]*ev).tolist(), forage=1 + .5*a[2], mate=1 + .75*a[3], rest=.5 + .5*a[4]))
            return out
        except Exception: self.n_fail += 1; return [dict(NEUTRAL) for _ in obs]
    def learn(self, exps):
        if not exps: return
        r = np.array([(e["energy1"] - e["energy0"])/12. - self.death_pen*(not e["alive"]) for e in exps])
        items = [(ri - self.base, self.eps.pop(e["id"])) for e, ri in zip(exps, r) if e["id"] in self.eps]
        if items:
            import pennylane as qml; from pennylane import numpy as pnp; circ = self._qnode()
            def obj(E, W):
                s = 0.
                for adv, (f, eps) in items: s = s + adv*qml.math.sum(qml.math.stack(circ(pnp.array(f, requires_grad=False), E, W))*eps)/self.sigma
                return s/len(items)
            gE, gW = qml.grad(obj, argnums=[0, 1])(pnp.array(self.E, requires_grad=True), pnp.array(self.W, requires_grad=True))
            self.E = np.clip(self.E + self.lr*np.asarray(gE), -3, 3); self.W = self.W + self.lr*np.asarray(gW)
        self.base += .2*(float(r.mean()) - self.base); self.eps.clear()
    def summary(self): return dict(super().summary(), arch="qwrk", wires=self.n_wires, layers=self.layers, entangle=self.entangle, readout="PauliX",
                                   heads=HEADS, feats=FEATS, baseline=round(self.base, 5), E=np.round(self.E, 4).tolist(), W=np.round(self.W, 4).tolist())

# ---------------------------------------------------------------- talk (Ollama / OpenAI-compatible) - narration only, never steers
TALK_SYS = ("You narrate a simulated ecosystem. The actions below were ALREADY chosen by a quantum-circuit policy; do not change them. "
            "Reply with 1-3 short sentences explaining plausibly why these organisms act this way given their observations.")

class Talk:
    """Post-decide natural-language rationale via an OpenAI-compatible /chat/completions (Ollama default). Output is only logged
    (JSONL rows {t,kind:"talk",ids,text}); it is never fed back into actions or rng. Errors -> None, n_fail++."""
    def __init__(self, url=None, model=None, timeout=10., api_key=None, post=None, path=None):
        self.url = (url or os.environ.get("NIM_MIND_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
        self.model = model or os.environ.get("NIM_MIND_MODEL") or "llama3.2"; self.timeout = float(timeout)
        self.api_key = api_key or os.environ.get("NIM_MIND_KEY"); self.post = post or _http_post; self.path = path
        self.n_calls = self.n_fail = 0; self.last = None
    def explain(self, obs, actions, t=None):
        if not obs: return None
        body = dict(model=self.model, temperature=0, messages=[dict(role="system", content=TALK_SYS), dict(role="user", content=json.dumps(
            dict(observations=obs, actions=actions), separators=(",", ":"), default=float))])
        h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
        self.n_calls += 1
        try: txt = str(self.post(self.url + "/chat/completions", body, h, self.timeout)["choices"][0]["message"]["content"]).strip()
        except Exception: self.n_fail += 1; return None
        self.last = txt
        if self.path:
            with open(self.path, "a") as fh: fh.write(json.dumps(dict(t=t, kind="talk", ids=[o.get("id") for o in obs], text=txt)) + "\n")
        return txt
    def __deepcopy__(self, memo): return self
    def summary(self): return dict(url=self.url, model=self.model, calls=self.n_calls, failures=self.n_fail)

def _http_post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

# ---------------------------------------------------------------- adapters
SYS = ("You steer simulated organisms. Reply ONLY with JSON {\"actions\":[...]} - one object per observation, same order, each "
       "{\"id\":int,\"move\":[dx,dy,dz] or null,\"forage\":0.5..1.5,\"mate\":0.5..2,\"rest\":0..1}. Goal: survive and keep energy high.")

class HttpMind(Mind):
    """DEPRECATED for steering (LLMs hallucinate ids; slow): use --mind circuit [--mind-talk]. Kept for back-compat. OpenAI-compatible chat endpoint (Ollama: http://127.0.0.1:11434/v1). Any error/timeout/bad JSON -> neutral actions for that batch.
    `post(url, body, headers, timeout) -> dict` is injectable for tests."""
    kind = "http"
    def __init__(self, url=None, model=None, timeout=10., api_key=None, post=None, **kw):
        super().__init__(**kw)
        self.url = (url or os.environ.get("NIM_MIND_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
        self.model = model or os.environ.get("NIM_MIND_MODEL") or "llama3.2"; self.timeout = float(timeout)
        self.api_key = api_key or os.environ.get("NIM_MIND_KEY"); self.post = post or self._post; self.memory = []
    def _post(self, url, body, headers, timeout):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    def decide(self, obs):
        if not obs: return []
        msg = dict(observations=obs, recent_outcomes=self.memory[-8:])
        body = dict(model=self.model, temperature=0, response_format={"type": "json_object"},
                    messages=[dict(role="system", content=SYS), dict(role="user", content=json.dumps(msg, separators=(",", ":")))])
        h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
        try:
            r = self.post(self.url + "/chat/completions", body, h, self.timeout); txt = r["choices"][0]["message"]["content"]
            txt = txt[txt.find("{"):txt.rfind("}") + 1]; acts = json.loads(txt); acts = acts.get("actions", acts) if isinstance(acts, dict) else acts
            by = {a.get("id"): a for a in acts if isinstance(a, dict)} if isinstance(acts, list) else {}
            if set(by) >= {o["id"] for o in obs}: return [by[o["id"]] for o in obs]
            if isinstance(acts, list) and len(acts) == len(obs): return list(acts)
            return [by.get(o["id"]) for o in obs]
        except Exception: self.n_fail += 1; return [dict(NEUTRAL) for _ in obs]
    def learn(self, exps):
        """In-context 'learning': keep a short memory of outcomes (mean d-energy / survival per action) sent with the next query."""
        if exps: self.memory.append(dict(t=exps[-1]["t1"], n=len(exps), survived=float(np.mean([e["alive"] for e in exps])),
                                         d_energy=round(float(np.mean([e["energy1"] - e["energy0"] for e in exps])), 4),
                                         mean_forage=round(float(np.mean([e["action"]["forage"] for e in exps])), 3)))

class ScriptMind(Mind):
    """User .py exposing decide(obs:list[dict])->list[dict] (required), learn(experiences) and culture_advise(state) (optional)."""
    kind = "script"
    def __init__(self, path=None, **kw):
        super().__init__(**kw)
        if not path or not os.path.isfile(path): raise FileNotFoundError(f"mind script not found: {path}")
        spec = importlib.util.spec_from_file_location("nim_user_mind", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if not callable(getattr(mod, "decide", None)): raise AttributeError(f"{path}: mind script must define decide(obs) -> list[dict]")
        self.mod = mod
        if callable(getattr(mod, "culture_advise", None)): self.culture_advise = mod.culture_advise
    def decide(self, obs): return self.mod.decide(obs)
    def learn(self, exps):
        f = getattr(self.mod, "learn", None)
        if callable(f): f(exps)

def culture_nudge(mind, c, dt):
    """Optional culture hook: mind.culture_advise(state)->{mitigation_boost} adds <= .002/yr mitigation (clamped; fossil_pref reserved)."""
    f = getattr(mind, "culture_advise", None)
    if not (mind is not None and getattr(mind, "active", False) and callable(f) and c.active): return c
    try: adv = f(dict(level=c.level, pollution=c.pollution, mitigation=c.mitigation, warm_K=c.warm_K, conflict=c.conflict)) or {}
    except Exception: return c
    c.mitigation = float(min(1., c.mitigation + np.clip(_num(adv.get("mitigation_boost"), 0.), 0., 1.)*.002*dt)); return c

KINDS = ("none", "null", "heuristic", "circuit", "inceptron", "http", "script")
def make_mind(kind="none", **kw):
    kind = (kind or "none").lower()
    if kind in ("none", "null"): return None
    if kind == "heuristic": return HeuristicMind(**kw)
    if kind in ("circuit", "inceptron"): return CircuitMind(**kw)
    if kind == "http": return HttpMind(**kw)
    if kind == "script": return ScriptMind(**kw)
    raise ValueError(f"unknown mind kind {kind!r}; use one of {KINDS}")

def add_args(ap):
    g = ap.add_argument_group("local mind / AI plug-in (default off)")
    g.add_argument("--mind", default="none", choices=["none", "heuristic", "circuit", "inceptron", "http", "script"],
                   help="circuit (alias inceptron): quantum-circuit steering; http is deprecated for steering - use circuit + --mind-talk"); g.add_argument("--mind-url", default=None, help="talk (and legacy http) OpenAI-compatible base, e.g. http://127.0.0.1:11434/v1")
    g.add_argument("--mind-talk", action="store_true", help="after each decide, ask Ollama (--mind-url/--mind-model) for a rationale; logged only, never steers")
    g.add_argument("--mind-talk-log", default=None, help="talk JSONL (default: --mind-trace)"); g.add_argument("--mind-layers", type=int, default=2)
    g.add_argument("--mind-entangle", default="layer", choices=["layer", "off", "dense"], help="circuit CNOT rings: one per layer (default), none, or after every gate (QWRK-literal)")
    g.add_argument("--mind-model", default=None); g.add_argument("--mind-script", default=None); g.add_argument("--mind-timeout", type=float, default=10.)
    g.add_argument("--mind-every", type=float, default=20., help="world-yr between queries"); g.add_argument("--mind-batch", type=int, default=16)
    g.add_argument("--mind-select", default="random", choices=["random", "energy"]); g.add_argument("--mind-learn", action="store_true")
    g.add_argument("--mind-trace", default=None, help="JSONL experience log")

def from_args(a, seed=0):
    kw = dict(every=a.mind_every, batch=a.mind_batch, learn=a.mind_learn, trace_path=a.mind_trace, seed=seed, select=a.mind_select)
    if a.mind == "http": kw.update(url=a.mind_url, model=a.mind_model, timeout=a.mind_timeout)
    if a.mind == "script": kw.update(path=a.mind_script)
    if a.mind in ("circuit", "inceptron"): kw.update(layers=getattr(a, "mind_layers", 2), entangle=getattr(a, "mind_entangle", "layer"))
    m = make_mind(a.mind, **kw)
    if m is not None and getattr(a, "mind_talk", False):
        m.talk = Talk(url=a.mind_url, model=a.mind_model, timeout=a.mind_timeout, path=getattr(a, "mind_talk_log", None) or a.mind_trace)
    return m
