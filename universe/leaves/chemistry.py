"""Prebiotic chemistry: reduced network over C/H/O/N/P/S pools + an aggregated 'organics' pool.
Approximation: organics are built from C,H,O,N,P (stoichiometry CH2O.05N.005P per unit) so atoms are
conserved in synthesis/decay; outgassing/escape are ignored (closed box)."""
import numpy as np
from .packets import ChemInventory, AbiogenesisGate, as_bus

STOICH = dict(C=1., H=2., O=1., N=.05, P=.005)

def init_chem(h):
    v = h.volatiles; return ChemInventory(dict(C=1.*v, H=4.*v, O=2.*v, N=.5*v, P=.05, S=.1, organics=0.), redox=float(.5-v))

def step(chem, climate, hw, LP=None, dt_w_yr=1.):
    P = as_bus(LP); s = dict(chem.species); w = climate.liquid_water_frac
    temp_ok = np.exp(-((climate.T_K-300.)/60.)**2)
    rate = P["chem_rate"]*(1 + P["photo_boost"]*min(hw.uv_proxy + hw.flare_rate, 3.))*temp_ok*(.2 + w)
    lim = min(s[k]/STOICH[k] for k in STOICH)                    # limiting reagent
    make = min(lim, rate*lim*dt_w_yr*2e-2); decay = s["organics"]*1e-3*dt_w_yr*(1. + (climate.T_K > 373))
    d = make - decay
    for k, c in STOICH.items(): s[k] = max(0., s[k] - c*d)
    s["organics"] = max(0., s["organics"] + d)
    pr = bool(w > .05 and 250. < climate.T_K < 373. and s["organics"] > P["organic_thresh"])
    vr = bool(pr and s["P"] > .01 and w > .1)
    return ChemInventory(s, float(chem.redox - .1*d), pr, vr)

def abiogenesis_roll(chem, climate, LP, rng, t_w=0., dt_w_yr=1., gate=None):
    P = as_bus(LP); g = gate or AbiogenesisGate()
    if g.fired: return g
    ready = chem.polymer_ready and chem.vesicle_ready and climate.liquid_water_frac > 0
    p = float(1 - (1 - P["abio_p"]*climate.liquid_water_frac)**dt_w_yr) if ready else 0.
    fired = bool(p > 0 and rng.random() < p)
    return AbiogenesisGate(p, fired, float(t_w) if fired else -1.)

def planted_gate(t_w=0.): return AbiogenesisGate(1., True, float(t_w), planted=True)
