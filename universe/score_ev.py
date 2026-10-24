"""v4 objective = 0.4 * v3 structure/stability score + 0.6 * high-energy realism/visibility score."""
import numpy as np
from .score2 import score as score2
from .events import life_myr, T_G_MYR, V_KMS, SMBH
EW = {"sn_rate": .2, "bh_frac": .1, "kick": .1, "rem_bound": .1, "smbh_mass": .1, "agn": .05, "grb": .05, "visible": .15, "merger": .15}

def score_ev(Gx, S, gx, gv, sx, sv, gk, info, C):
    base = score2(dict(Gx, kind=np.where(Gx["kind"] == SMBH, 8, Gx["kind"])), S, gx, gv, sx, sv) if sx is not None else None
    L = info["log"]; sn = [e for e in L if e["type"] == "supernova"]; sm = Gx["smass"]; k0 = Gx["kind"]
    T = C["galaxy_steps"]*C["galaxy_dt"]*T_G_MYR
    # reference: real physics (M >= 8 Msun core collapse, lifetime ~ M^-2.5) on the same stars with a fresh age draw expectation
    st = k0 == 1; ref = float(np.sum(np.minimum(1, T/life_myr(sm[st], 2.5))*(sm[st] >= 8)))
    s = {}
    s["sn_rate"] = float(np.exp(-abs(np.log((len(sn)+.5)/(ref+.5)))))
    cc = [e for e in sn]; fbh = np.mean([e["remnant"] == "black hole" for e in cc]) if cc else 0.
    nbh_ref = float(np.sum(sm[st] >= 20)); fref = nbh_ref/max(1, float(np.sum(sm[st] >= 8)))
    s["bh_frac"] = float(np.exp(-((fbh-fref)/.12)**2)) if cc else .5
    ns = [e["kick_kms"] for e in cc if e["remnant"] != "black hole"]
    s["kick"] = float(np.exp(-abs(np.log((np.mean(ns)+20)/(265+20))))) if ns else .5
    # remnant still bound to its galaxy (within 3R of its galaxy's halo core)
    cores = np.where(k0 == 0)[0]; R = Gx["R"]; gid = Gx["gid"]; kf = gk[-1]
    rem = np.where((kf == 5) | (kf == 6))[0]
    s["rem_bound"] = float(np.mean(np.linalg.norm(gx[-1, rem]-gx[-1, cores[gid[rem]]], axis=1) < 3*R[gid[rem]])) if len(rem) else .5
    sb = info["smbh"]; frac = np.array([Gx["m"][b]/(Gx["m"][b]+Gx["m"][cores[gid[b]]]) for b in sb])
    s["smbh_mass"] = float(np.mean(np.exp(-np.abs(np.log(frac/.003))/1.)))   # ~ M_bh / M_bulge (Magorrian-like target)
    act = info["agn_frac"] if info.get("agn_frac") is not None else (np.mean(list(info["active"].values())) if info["active"] else 0.)   # v5: time-averaged duty
    s["agn"] = float(np.exp(-((act-.2)/.2)**2))
    nbh = sum(e["remnant"] == "black hole" for e in cc); ngrb = sum(e["grb"] for e in cc)
    s["grb"] = float(np.exp(-abs(ngrb/max(nbh, 1)-.02)/.05))
    tde = sum(e["type"] == "tidal disruption" for e in L)
    s["visible"] = float(.5*min(1, len(sn)/5) + .25*min(1, tde/2) + .25*np.exp(-max(0, tde-20)/20))
    # SMBH mergers should happen iff the galaxies actually merged (halo cores closer than half their summed radii at the end)
    cp = gx[-1, cores]; merged = any(np.linalg.norm(cp[a]-cp[b]) < .5*(R[a]+R[b]) for a in range(len(cores)) for b in range(a+1, len(cores)))
    nm = sum(e["type"] == "black hole merger" for e in L)
    s["merger"] = float(1. if (nm > 0) == merged else 0.)
    ev = sum(EW[k]*s[k] for k in EW); s["events"] = ev
    if base is not None: s["base"] = base["total"]; s["total"] = .4*base["total"] + .6*ev
    else: s["total"] = ev
    s["n_sn"], s["n_ref"], s["n_tde"], s["n_merge"], s["n_grb"], s["triggered"] = len(sn), ref, tde, nm, ngrb, info["triggered"]
    return s
