"""solvers.foodweb -- N-guild, mass-conserving food-web ODE (out-of-leaf solver; HabitatRun(foodweb=True)).

Compartments (biomass in the leaves' capacity units; one currency = limiting-nutrient mass):
    Nut  dissolved nutrient (abiotic)            Det  detritus (abiotic)
    P    producers   H herbivores   C carnivores   X apex predators (eat C and H)   Dc decomposers (eat Det)
Fluxes (all per year, every flux moves mass between two compartments, so d(sum)/dt = external only):
    uptake      Nut -> P :  mu P Nut/(Nut+K_N) (1 - P/Kp)^+      (Monod nutrient x crowding by the habitat capacity K)
    grazing     prey -> pred : Beddington-DeAngelis  a R Z / (1 + a h R + i Z); a fraction e assimilated,
                (1-e) egested -> Det
    respiration B -> Nut (remineralised);   mortality B -> Det;   decomposition Det -> Dc (eff e_D, rest -> Nut)
    external    Nut <- supply: (K NUT_FAC - total)/TAU_SUPPLY  (weathering/upwelling; 0 if closed=True)
Integration: explicit flux-limited Euler substeps (no compartment can lose more than it holds in a substep) -> positivity and
exact mass conservation to round-off (sum of compartments changes only by the external flux and disturbance; tested).
Ledger 'loss' = all mass leaving the living compartments (respiration + mortality + egestion + disturbance), the same
meaning as leaves.biosphere's ledger for the C/N/P books; 'ext' = external nutrient flux.
Maps back onto leaves.biosphere.Biosphere (P = producers, C = H + C + X, D = decomposers) so organisms / culture /
biogeochem see the same API.  Fidelity: a textbook guild model (Holling II + interference), not a species network;
trophic efficiencies ~10-30 %, apex/predator/herbivore biomass pyramids emerge from parameters, not imposed."""
import numpy as np
from dataclasses import dataclass, field, asdict
from ..leaves import biosphere, biosign
from ..leaves.packets import NicheOffer, BioSignFlux, as_bus

GUILDS = ("P", "H", "C", "X", "Dc")
POOLS = ("Nut", "Det")
NUT_FAC = 1.6           # total nutrient mass supported per unit capacity K
TAU_SUPPLY = 200.       # yr relaxation of the nutrient inventory toward K*NUT_FAC
K_N = .15               # nutrient half-saturation (capacity units)
K_DET = .1
# per-guild rates in units of eco_growth (mu): (max ingestion a_max, half-sat prey, assimilation e, respiration, mortality, interference)
PRED = {"H": dict(prey={"P": 1.}, a=3.0, half=.30, e=.35, r=.15, m=.08, i=4.0),
        "C": dict(prey={"H": 1.}, a=2.5, half=.08, e=.30, r=.12, m=.06, i=6.0),
        "X": dict(prey={"C": .8, "H": .2}, a=2.0, half=.03, e=.25, r=.08, m=.05, i=10.0)}
P_RESP, P_MORT = .12, .10
DC = dict(a=4.0, e=.30, r=.30, m=.10)
SEED_FRAC = 1e-3


@dataclass
class FoodWeb:
    Nut: float = 0.; Det: float = 0.; P: float = 0.; H: float = 0.; C: float = 0.; X: float = 0.; Dc: float = 0.
    alive: bool = False; ext_in: float = 0.; cum_loss: float = 0.; closed: bool = False
    hist: list = field(default_factory=list)

    def living(self): return self.P + self.H + self.C + self.X + self.Dc
    def total(self): return self.Nut + self.Det + self.living()
    def to_dict(self):
        d = asdict(self); d.pop("hist"); return d

    def seed(self, K, amount=1e-3):
        self.Nut = max(self.Nut, K*NUT_FAC); self.P = max(self.P, amount); self.Dc = max(self.Dc, .1*amount); self.alive = True

    def sync_from(self, bio):
        """Absorb external edits of the Biosphere (creature grazing, culture draw): scale guilds to match; returns removed mass."""
        before = self.living()
        self.P = bio.P; self.Dc = bio.D; hc = self.H + self.C + self.X
        if hc > 0: f = bio.C/hc; self.H, self.C, self.X = self.H*f, self.C*f, self.X*f
        elif bio.C > 0: self.H = bio.C
        return before - self.living()

    def rates(self, mu, K):
        """Flux list [(src, dst, rate)] at the current state."""
        F = []; s = self
        Kp = max(K, 1e-9)
        up = mu*s.P*s.Nut/(s.Nut + K_N)*max(0., 1 - s.P/Kp)
        F.append(("Nut", "P", up)); F.append(("P", "Nut", mu*P_RESP*s.P)); F.append(("P", "Det", mu*P_MORT*s.P*(1 + s.P/Kp)))
        st = {g: getattr(s, g) for g in GUILDS}
        for z, q in PRED.items():
            Z = st[z]
            if Z <= 0: continue
            R = sum(w*st[p] for p, w in q["prey"].items())
            ing = mu*q["a"]*R*Z/(q["half"] + R + q["i"]*Z)     # Beddington-DeAngelis (Holling II + predator interference)
            for p, w in q["prey"].items():
                if R > 0 and st[p] > 0:
                    fl = ing*w*st[p]/R; F.append((p, z, q["e"]*fl)); F.append((p, "Det", (1 - q["e"])*fl))
            F.append((z, "Nut", mu*q["r"]*Z)); F.append((z, "Det", mu*q["m"]*Z))
        if s.Dc > 0:
            dec = mu*DC["a"]*s.Dc*s.Det/(s.Det + K_DET)
            F.append(("Det", "Dc", DC["e"]*dec)); F.append(("Det", "Nut", (1 - DC["e"])*dec))
            F.append(("Dc", "Nut", mu*DC["r"]*s.Dc)); F.append(("Dc", "Det", mu*DC["m"]*s.Dc*(1 + s.Dc/max(.2*Kp, 1e-9))))
        return F

    def step(self, K, mu, dt, disturbance=0., invade=True):
        """Advance dt (yr). Returns ledger dict(loss, ext, resp, mort, egest, uptake)."""
        led = dict(loss=0., ext=0., uptake=0.)
        if disturbance > 0:
            f = min(disturbance, 1.)
            for g in GUILDS:
                v = getattr(self, g); led["loss"] += v*f; self.Det += v*f; setattr(self, g, v*(1 - f))
        if not self.alive: return led
        if K <= 0:      # habitat gone: everything living dies into detritus
            for g in GUILDS:
                v = getattr(self, g); led["loss"] += v; self.Det += v; setattr(self, g, 0.)
            self.alive = False; return led
        if invade:      # immigration / emergence of higher guilds once their prey is established
            for z, q in PRED.items():
                if getattr(self, z) == 0. and sum(getattr(self, p) for p in q["prey"]) > .05*K:
                    take = SEED_FRAC*K; src = "P" if self.P > take else None
                    if src: self.P -= take; setattr(self, z, take)
            if self.Dc == 0. and self.Det > .01: m = min(self.Det, SEED_FRAC*K); self.Det -= m; self.Dc = m
        t = 0.
        while t < dt - 1e-12:
            F = self.rates(mu, K)
            out = {}
            for a, b, r in F: out[a] = out.get(a, 0.) + r
            hmax = min([.2*getattr(self, a)/r for a, r in out.items() if r > 0 and getattr(self, a) > 0] + [dt - t, 2.])
            h = max(hmax, min(1e-3, dt - t))
            lim = {a: min(1., getattr(self, a)/(r*h)) if r > 0 else 1. for a, r in out.items()}
            for a, b, r in F:
                m = r*h*lim[a]
                if m <= 0: continue
                setattr(self, a, getattr(self, a) - m); setattr(self, b, getattr(self, b) + m)
                if a in GUILDS and b in POOLS: led["loss"] += m
                if a == "Nut": led["uptake"] += m
            if not self.closed:
                ext = (K*NUT_FAC - self.total())/TAU_SUPPLY*h
                ext = max(ext, -self.Nut); self.Nut += ext; led["ext"] += ext; self.ext_in += ext
            for g in GUILDS:
                if getattr(self, g) < 1e-12: setattr(self, g, 0.)
            if self.P < 1e-6 and K > 0: d = 1e-6 - self.P; take = min(d, self.Nut); self.Nut -= take; self.P += take   # refuge
            t += h
        if self.living() < 1e-9: self.alive = False
        self.cum_loss += led["loss"]
        return led

    def summary(self):
        lv = self.living(); g = {k: float(getattr(self, k)) for k in GUILDS}
        tl = {"P": 1, "H": 2, "C": 3, "X": 4.2, "Dc": 1.5}
        mtl = sum(tl[k]*v for k, v in g.items())/lv if lv > 0 else 0.
        return dict(guilds=g, pools=dict(Nut=float(self.Nut), Det=float(self.Det)), total=float(self.total()), living=float(lv),
                    n_guilds_alive=int(sum(v > 1e-6 for v in g.values())), mean_trophic_level=float(mtl),
                    pyramid=[float(g[k]) for k in ("P", "H", "C", "X")], ext_in=float(self.ext_in), alive=bool(self.alive))


def step(fw, bio, climate, chem, LP=None, dt_w_yr=1., disturbance=0., growth_mult=1., ledger=None):
    """Drop-in for leaves.biosphere.step driven by a FoodWeb fw (bio is synced in/out).  Returns (bio, niche, flux)."""
    P_ = as_bus(LP); K = biosphere.capacity(climate, chem); st = climate.stressors or {}
    disturbance = 1 - (1 - min(disturbance, 1.))*(1 - st.get("impact_kill", 0.))*(1 - st.get("uv_kill", 0.))*(1 - st.get("tech_kill", 0.))
    if bio.alive and not fw.alive: fw.seed(K, max(bio.P, 1e-3)); fw.Dc = max(fw.Dc, bio.D)
    removed = fw.sync_from(bio) if fw.alive else 0.
    led = fw.step(K, P_["eco_growth"]*growth_mult, dt_w_yr, disturbance)
    b = biosphere.Biosphere(**bio.to_dict())
    b.P, b.C, b.D = fw.P, fw.H + fw.C + fw.X, fw.Dc; b.alive = bool(fw.alive and fw.living() > 1e-9)
    s = fw.summary(); b.diversity = max(b.diversity, float(s["n_guilds_alive"]))
    if not b.alive: b.P = b.C = b.D = 0.
    if ledger is not None: ledger["loss"] = ledger.get("loss", 0.) + float(led["loss"]); ledger["foodweb"] = dict(led, removed=float(removed))
    free = max(0., K - b.total())
    niche = NicheOffer(float(K), int(s["n_guilds_alive"]) if b.alive else 0, float(b.P*.5 + free*.1) if b.alive else 0.)
    flux = BioSignFlux(float(.21*b.P/(1 + b.P)), float(1e-3*b.D/(1 + b.D) + 5e-4*b.C/(1 + b.C)), float(b.P/(1 + b.P)), 0., biosign.h2o_column(climate))
    biosign.attach(flux)
    return b, niche, flux
