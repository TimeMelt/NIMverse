"""Biosphere: three-guild biomass ODEs (producers P, consumers C, decomposers D), logistic + LV coupling."""
import numpy as np
from dataclasses import dataclass, field, asdict
from .packets import NicheOffer, BioSignFlux, as_bus
from . import biosign

@dataclass
class Biosphere:
    P: float = 0.; C: float = 0.; D: float = 0.; alive: bool = False; diversity: float = 0.
    def total(self): return self.P + self.C + self.D
    def to_dict(self): return asdict(self)

def capacity(climate, chem):
    if climate.liquid_water_frac <= 0 or not chem.polymer_ready: return 0.
    return float(climate.liquid_water_frac*(1. + min(chem.species["organics"], 2.)))

def seed(bio, gate, amount=1e-3):
    if gate.fired and not bio.alive: bio.P, bio.D, bio.alive, bio.diversity = amount, amount*.1, True, 1.
    return bio

def step(bio, climate, chem, LP=None, dt_w_yr=1., disturbance=0.):
    """Returns (bio, NicheOffer, BioSignFlux). disturbance in [0,1] kills that biomass fraction."""
    P_ = as_bus(LP); K = capacity(climate, chem); b = Biosphere(**bio.to_dict()); st = climate.stressors or {}
    disturbance = 1 - (1-min(disturbance, 1.))*(1-st.get("impact_kill", 0.))*(1-st.get("uv_kill", 0.))*(1-st.get("tech_kill", 0.))   # ClimateStress shocks
    if disturbance: f = 1-min(disturbance, 1.); b.P, b.C, b.D = b.P*f, b.C*f, b.D*f
    if b.alive:
        g, s, dth = P_["eco_growth"], P_["eco_stiffness"], P_["eco_death"]
        n = max(1, int(np.ceil(dt_w_yr*g*4)))
        for _ in range(n):
            h = dt_w_yr/n; tot = b.P + b.C + b.D
            dP = (g*b.P*(1 - (tot/K)**s) if K > 0 else -b.P) - .2*b.C*b.P/(b.P+.5)
            dC = .1*b.C*b.P/(b.P+.5) - dth*b.C + (1e-4*b.P if b.C == 0 else 0)
            dD = .1*dth*(b.P+b.C) - dth*b.D + (1e-4*b.P if b.D == 0 else 0)
            if K <= 0: dP, dC, dD = -b.P, -b.C, -b.D               # no growth without water/chem
            b.P, b.C, b.D = max(0., b.P+h*dP), max(0., b.C+h*dC), max(0., b.D+h*dD)
            if K > 0 and b.P < 1e-6: b.P = 1e-6                    # refuge: producers persist while habitat holds
        if b.total() < 1e-9: b.alive, b.P, b.C, b.D = False, 0., 0., 0.
    free = max(0., K - b.total())
    niche = NicheOffer(float(K), int(b.alive)*3, float(b.P*.5 + free*.1) if b.alive else 0.)
    flux = BioSignFlux(float(.21*b.P/(1+b.P)), float(1e-3*b.D/(1+b.D) + 5e-4*b.C/(1+b.C)), float(b.P/(1+b.P)), 0., biosign.h2o_column(climate))
    biosign.attach(flux)
    return b, niche, flux

def apply_creatures(bio, pkt):
    """Feedback from organisms: grazing removes producers, diversity tracks cohort count."""
    b = Biosphere(**bio.to_dict()); b.P = max(0., b.P - .01*pkt.energy_budget); b.diversity = max(b.diversity, float(pkt.trait_means.get("cohorts", 0)))
    return b
