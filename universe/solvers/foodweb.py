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
    if hasattr(fw, "set_climate"): fw.set_climate(climate)
    if bio.alive and not fw.alive: fw.seed(K, max(bio.P, 1e-3)); fw.Dc = max(fw.Dc, bio.D)
    removed = fw.sync_from(bio) if fw.alive else 0.
    led = fw.step(K, P_["eco_growth"]*growth_mult, dt_w_yr, disturbance)
    b = biosphere.Biosphere(**bio.to_dict())
    b.P, b.C, b.D = fw.P, fw.H + fw.C + fw.X, fw.Dc; b.alive = bool(fw.alive and fw.living() > 1e-9)
    s = fw.summary(); b.diversity = max(b.diversity, float(s.get("shannon_eff", s["n_guilds_alive"])))
    if not b.alive: b.P = b.C = b.D = 0.
    if ledger is not None: ledger["loss"] = ledger.get("loss", 0.) + float(led["loss"]); ledger["foodweb"] = dict(led, removed=float(removed))
    free = max(0., K - b.total())
    niche = NicheOffer(float(K), int(s.get("n_species_alive", s["n_guilds_alive"])) if b.alive else 0, float(b.P*.5 + free*.1) if b.alive else 0.)
    flux = BioSignFlux(float(.21*b.P/(1 + b.P)), float(1e-3*b.D/(1 + b.D) + 5e-4*b.C/(1 + b.C)), float(b.P/(1 + b.P)), 0., biosign.h2o_column(climate))
    biosign.attach(flux)
    return b, niche, flux


# ----------------------------------------------------------------------------- N-species size-spectrum mode
N_SPECIES = 18
PPMR_LOG, PPMR_W = 2., 1.3       # preferred log10 predator/prey body-mass ratio and diet width (decades)
THERM_W = 8.                     # K thermal niche half-width
NICHE_W = 10.                    # K: competition overlap alpha_ij = exp(-(Topt_i - Topt_j)^2 / (4 NICHE_W^2))
SP = dict(a_P=1., a_C=2.6, half=.12, i=6., e_herb=.35, e_carn=.3, r_P=P_RESP, m_P=P_MORT, r_C=.12, m_C=.06)


class SpeciesWeb:
    """N-species size-structured food web (HabitatRun(foodweb="species")): same pools (Nut, Det), decomposers Dc and
    flux-limited mass-conserving integrator as FoodWeb, but living mass is split over N_SPECIES species with traits
    - log10 body mass (size spectrum, producers = the smallest sizes);
    - thermal optimum T_opt (thermal performance g(T) = mean over latitude bands of exp(-((T - T_opt)/THERM_W)^2),
      so species occupy climate niches; with the grid/lat T from the climate packet).
    Feeding: allometric (mass-specific rates ~ m^-1/4), diet kernel phi_ij = exp(-((log m_i - log m_j - PPMR_LOG)/PPMR_W)^2)
    (Brose et al. 2006 / Andersen size-spectrum style), Beddington-DeAngelis functional response.
    Producers compete for nutrient and space with niche overlap alpha_ij (thermal), the coexistence mechanism.
    Trophic levels are diagnosed from realised diets; the Biosphere API sees P = producers, C = consumers
    (properties H/C/X are the consumers binned by trophic level <2.5, 2.5-3.5, >3.5), D = decomposers.
    Exact mass conservation (all fluxes move mass between compartments; tested)."""
    def __init__(self, n=N_SPECIES, seed=0, closed=False):
        r = np.random.default_rng([int(seed) % 2**31, 4243])
        self.n = n; nP = max(3, n//3); self.nP = nP
        lm = np.linspace(-7., 3., n) + r.uniform(-.3, .3, n)          # size spectrum: ~0.6 decades apart, producers smallest
        self.logm = lm; self.prod = np.arange(n) < nP
        self.Topt = r.uniform(260., 310., n)
        cons = ~self.prod; A = np.ones(n)
        A[cons] = 10**(-.25*(lm[cons] - lm[cons].mean())); A[self.prod] = 10**(-.15*(lm[self.prod] - lm[self.prod].mean()))
        self.A = np.clip(A, .25, 4.)
        d = lm[:, None] - lm[None, :]
        self.phi = np.where(cons[:, None], np.exp(-((d - PPMR_LOG)/PPMR_W)**2), 0.); np.fill_diagonal(self.phi, 0.)
        self.phi[self.phi < 1e-3] = 0.
        self.alpha = np.exp(-(self.Topt[:, None] - self.Topt[None, :])**2/(4*NICHE_W**2))*(self.prod[:, None] & self.prod[None, :])
        self.B = np.zeros(n); self.Nut = 0.; self.Det = 0.; self.Dc = 0.
        self.alive = False; self.ext_in = 0.; self.cum_loss = 0.; self.closed = closed; self.perf = np.ones(n); self.hist = []
    # Biosphere-facing aggregates
    @property
    def P(self): return float(self.B[self.prod].sum())
    def _tl(self):
        TL = np.where(self.prod, 1., 2.)
        W = self.phi*self.B[None, :]; s = W.sum(1, keepdims=True); D = np.where(s > 0, W/np.where(s > 0, s, 1.), 0.)
        for _ in range(12): TL = np.where(self.prod, 1., 1. + (D*TL[None, :]).sum(1) + (s[:, 0] <= 0))
        return TL
    def _bin(self, lo, hi):
        TL = self._tl(); m = (~self.prod) & (TL >= lo) & (TL < hi); return float(self.B[m].sum())
    @property
    def H(self): return self._bin(0., 2.5)
    @property
    def C(self): return self._bin(2.5, 3.5)
    @property
    def X(self): return self._bin(3.5, 99.)
    def living(self): return float(self.B.sum() + self.Dc)
    def total(self): return self.Nut + self.Det + self.living()
    def to_dict(self):
        return dict(B=self.B.tolist(), Nut=self.Nut, Det=self.Det, Dc=self.Dc, alive=self.alive, ext_in=self.ext_in, cum_loss=self.cum_loss)
    def seed(self, K, amount=1e-3):
        self.Nut = max(self.Nut, K*NUT_FAC)
        w = self.perf*self.prod; w = w/w.sum() if w.sum() > 0 else self.prod/self.prod.sum()
        add = np.maximum(amount*w - self.B, 0.)*self.prod; self.B = self.B + add
        self.Dc = max(self.Dc, .1*amount); self.alive = True
    def sync_from(self, bio):
        before = self.living(); p = self.prod; Ps = self.B[p].sum(); Cs = self.B[~p].sum()
        if Ps > 0: self.B[p] *= bio.P/Ps
        if Cs > 0: self.B[~p] *= bio.C/Cs
        elif bio.C > 0: k = np.flatnonzero(~p); self.B[k[0]] = bio.C
        self.Dc = bio.D
        return before - self.living()
    def set_climate(self, climate):
        st = climate.stressors or {}; Tb = np.asarray(st.get("lat_T") or [climate.T_K], float)
        wat = np.asarray(st.get("lat_water") or [1.]*len(Tb), float)
        if len(wat) != len(Tb): wat = np.ones(len(Tb))
        w = wat/wat.sum() if wat.sum() > 0 else np.ones(len(Tb))/len(Tb)
        E = np.exp(-((Tb[None, :] - self.Topt[:, None])/THERM_W)**2); perf = (E*w[None, :]).sum(1)
        self.perf = perf/max(float(perf.max()), 1e-9)          # relative: absolute productivity is already in K
        # MacArthur utilisation overlap over the habitat's latitude bands: alpha_ij = sum w e_i e_j / sum w e_i^2,
        # capacity share k_i = sum w e_i (the habitat fraction species i can use) -> coexistence of thermal specialists
        Ew = E*w[None, :]; ii = (Ew*E).sum(1)
        self.alpha = np.where(self.prod[:, None] & self.prod[None, :], (Ew @ E.T)/np.maximum(ii[:, None], 1e-12), 0.)
        self.kshare = np.clip(perf/max(float(perf[self.prod].max()), 1e-9), 1e-6, 1.)
    def _fluxes(self, mu, K):
        """(src, dst, rate) arrays over the flat state [Nut, Det, Dc, B_0..B_n-1]."""
        n = self.n; B = self.B; g = self.perf; p = self.prod; src, dst, rate = [], [], []
        def add(s_, d_, r_):
            r_ = np.atleast_1d(np.asarray(r_, float)); src.append(np.broadcast_to(np.asarray(s_), r_.shape)); dst.append(np.broadcast_to(np.asarray(d_), r_.shape)); rate.append(r_)
        ip = np.flatnonzero(p) + 3; ic = np.flatnonzero(~p) + 3
        Kp = max(K, 1e-9)
        Ki = Kp*getattr(self, "kshare", np.ones(n))[p]
        crowd = np.maximum(0., 1 - (self.alpha[np.ix_(p, p)] @ B[p])/Ki)
        add(0, ip, mu*SP["a_P"]*self.A[p]*g[p]*B[p]*self.Nut/(self.Nut + K_N)*crowd)
        add(ip, 0, mu*SP["r_P"]*B[p]); add(ip, 1, mu*SP["m_P"]*B[p]*(1 + B[p].sum()/Kp))
        R = self.phi @ B                                                   # available prey per predator
        Bc = B[~p]; Rc = R[~p]; Ac = self.A[~p]; gc = .3 + .7*g[~p]/max(g.max(), 1e-9)
        ing = mu*SP["a_C"]*Ac*gc*Rc*Bc/(SP["half"] + Rc + SP["i"]*Bc)
        Wd = self.phi[~p]*B[None, :]; frac = np.where(Rc[:, None] > 0, Wd/np.where(Rc[:, None] > 0, Rc[:, None], 1.), 0.)
        fl = ing[:, None]*frac                                             # (n_cons, n) prey mass eaten
        e = np.where(p[None, :], SP["e_herb"], SP["e_carn"])*np.ones_like(fl)
        I, J = np.nonzero(fl > 0)
        if len(I):
            add(J + 3, ic[I], e[I, J]*fl[I, J]); add(J + 3, 1, (1 - e[I, J])*fl[I, J])
        add(ic, 0, mu*SP["r_C"]*Ac*Bc); add(ic, 1, mu*SP["m_C"]*Bc)
        if self.Dc > 0:
            dec = mu*DC["a"]*self.Dc*self.Det/(self.Det + K_DET)
            add(1, 2, DC["e"]*dec); add(1, 0, (1 - DC["e"])*dec)
            add(2, 0, mu*DC["r"]*self.Dc); add(2, 1, mu*DC["m"]*self.Dc*(1 + self.Dc/max(.2*Kp, 1e-9)))
        return np.concatenate(src).astype(int), np.concatenate(dst).astype(int), np.concatenate(rate)
    def step(self, K, mu, dt, disturbance=0., invade=True):
        led = dict(loss=0., ext=0., uptake=0.)
        x = np.concatenate([[self.Nut, self.Det, self.Dc], self.B])
        if disturbance > 0:
            f = min(disturbance, 1.); kill = x[2:]*f; led["loss"] += float(kill.sum()); x[1] += kill.sum(); x[2:] -= kill
        if not self.alive: self._put(x); return led
        if K <= 0:
            led["loss"] += float(x[2:].sum()); x[1] += x[2:].sum(); x[2:] = 0.; self._put(x); self.alive = False; return led
        self._put(x)
        if invade:                                   # immigration: consumers whose prey is established, producers whose niche is open
            p = self.prod; R = self.phi @ self.B; Ps = self.B[p].sum()
            for i in range(self.n):
                if self.B[i] > 0: continue
                if p[i] and self.perf[i] > .2 and self.Nut > SEED_FRAC*K: self.Nut -= SEED_FRAC*K; self.B[i] = SEED_FRAC*K
                elif not p[i] and R[i] > .02*K and Ps > 2*SEED_FRAC*K:
                    j = int(np.argmax(self.B*p)); m = min(SEED_FRAC*K, .5*self.B[j]); self.B[j] -= m; self.B[i] = m
            if self.Dc == 0. and self.Det > .01: m = min(self.Det, SEED_FRAC*K); self.Det -= m; self.Dc = m
        t = 0.
        while t < dt - 1e-12:
            x = np.concatenate([[self.Nut, self.Det, self.Dc], self.B])
            s, d, r = self._fluxes(mu, K)
            out = np.bincount(s, r, len(x))
            ok = (out > 0) & (x > 0)
            hmax = min(float((.2*x[ok]/out[ok]).min()) if ok.any() else 2., dt - t, 2.)
            h = max(hmax, min(1e-3, dt - t))
            lim = np.where(out > 0, np.minimum(1., x/np.maximum(out*h, 1e-300)), 1.)
            m = r*h*lim[s]
            x = x - np.bincount(s, m, len(x)) + np.bincount(d, m, len(x))
            led["loss"] += float(m[(s >= 2) & (d <= 1)].sum()); led["uptake"] += float(m[s == 0].sum())
            if not self.closed:
                ext = (K*NUT_FAC - x.sum())/TAU_SUPPLY*h; ext = max(ext, -x[0]); x[0] += ext; led["ext"] += ext; self.ext_in += ext
            x[2:] = np.where(x[2:] < 1e-12, 0., x[2:])
            self._put(x)
            if self.P < 1e-6 and K > 0:                                   # refuge: best-adapted producer
                i = int(np.argmax(np.where(self.prod, self.perf, -1))); dd = 1e-6 - self.P; take = min(dd, self.Nut); self.Nut -= take; self.B[i] += take
            t += h
        if self.living() < 1e-9: self.alive = False
        self.cum_loss += led["loss"]
        return led
    def _put(self, x):
        self.Nut, self.Det, self.Dc = float(x[0]), float(x[1]), float(x[2]); self.B = np.array(x[3:], float)
    def summary(self):
        lv = self.living(); TL = self._tl(); B = self.B; al = B > 1e-6
        pr = B[al]/B[al].sum() if al.any() else np.array([1.])
        sl = float(np.polyfit(self.logm[al], np.log10(B[al]), 1)[0]) if al.sum() >= 3 else None
        g = dict(P=self.P, H=self.H, C=self.C, X=self.X, Dc=float(self.Dc))
        mtl = float((TL*B).sum()/B.sum()) if B.sum() > 0 else 0.
        return dict(mode="species", guilds={k: float(v) for k, v in g.items()}, pools=dict(Nut=float(self.Nut), Det=float(self.Det)),
                    total=float(self.total()), living=float(lv), n_species=int(self.n), n_species_alive=int(al.sum()),
                    shannon_eff=float(np.exp(-(pr*np.log(pr)).sum())), size_spectrum_slope=sl, max_trophic_level=float(TL[al].max()) if al.any() else 0.,
                    n_guilds_alive=int(sum(v > 1e-6 for v in g.values())), mean_trophic_level=mtl,
                    pyramid=[float(g[k]) for k in ("P", "H", "C", "X")], ext_in=float(self.ext_in), alive=bool(self.alive))


def make(mode=True, seed=0):
    """FoodWeb (mode True/'guild', the default) or SpeciesWeb (mode 'species')."""
    return SpeciesWeb(seed=seed) if mode == "species" else FoodWeb()
