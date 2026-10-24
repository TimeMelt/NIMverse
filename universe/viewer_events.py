"""Viewer v4: the v3 viewer plus inceptron-driven high-energy events (supernova flashes + nebula shells, black holes with
accretion disks / lensing / jets, pulsars, TDEs, GRBs, SMBH mergers) and a clickable event timeline.
python -m universe.viewer_events --seed 7 [--run /workspace/nim_universe/out/universe_v4_seed7_events]"""
import argparse, json, os, numpy as np
from . import sim2
from .viewer import galaxy_block, system_block, gallery, HERE, OUT
from .nim_rng import NIMRandom
TYPES = {"supernova": "SN", "tidal disruption": "TDE", "black hole merger": "MERGER", "quasar ignition": "AGN"}
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--run"); ap.add_argument("--out")
    ap.add_argument("--stars-per-galaxy", type=int, default=1500); ap.add_argument("--max-tde", type=int, default=60)
    a = ap.parse_args(argv); stem = a.run or f"{OUT}/universe_v4_seed{a.seed}_events"
    J = json.load(open(stem+".json")); C = J["config"]; P = J["params"]; z = np.load(stem+".npz", allow_pickle=True)
    k0 = z["kind"]; kf = z["kind_frames"]; F = z["galaxy_frames"]; nF = len(F); smbh = np.where(k0 == 4)[0]
    evs = sorted(J["events"], key=lambda e: e["t"])
    tde = [e for e in evs if e["type"] == "tidal disruption"]
    if len(tde) > a.max_tde:  # keep the log readable: subsample TDEs evenly (all are counted in the header)
        keep = set(id(tde[int(i)]) for i in np.linspace(0, len(tde)-1, a.max_tde)); evs = [e for e in evs if e["type"] != "tidal disruption" or id(e) in keep]
    force = list(smbh) + [e["i"] for e in evs if e["type"] in ("supernova", "tidal disruption")]
    gal, ntot = galaxy_block(z, a.stars_per_galaxy, force=force); idx = gal.pop("_idx"); pos = {int(i): j for j, i in enumerate(idx)}
    r = NIMRandom(a.seed); sim2.make_galaxies(r, C, P); S = sim2.make_system(r, C, P); sysb, pl = system_block(S, z["system_frames"], a.seed)
    fv = lambda fr: float(min(fr, nF-1)*(sysb["nf"]-1)/(nF-1))
    # per kept particle: frame of death (viewer units) and what it became
    death = np.full(len(idx), 1e9); rem = np.zeros(len(idx), int)
    for j, i in enumerate(idx):
        ch = np.where(kf[:, i] != kf[0, i])[0]
        if len(ch): death[j] = fv(ch[0]); rem[j] = kf[ch[0], i]
    act = {int(k): v for k, v in J.get("active", {}).items()}; ign = {e["i"]: fv(e["frame"]) for e in evs if e["type"] in ("quasar ignition", "black hole merger")}
    iv = {}   # v5 duty cycle: per SMBH list of [on, off) quasar intervals in viewer frames
    for e in sorted([e for e in evs if e["type"] in ("quasar ignition", "black hole merger", "quasar fades")], key=lambda e: e["frame"]):
        L = iv.setdefault(int(e["i"]), [])
        if e["type"] == "quasar fades":
            if L and L[-1][1] >= 1e9: L[-1][1] = fv(e["frame"])
        elif not (L and L[-1][1] >= 1e9): L.append([fv(e["frame"]), 1e9])
    duty = any(e["type"] == "quasar fades" for e in evs) or "agn_duty_myr" in J.get("event_params", {})
    bhs = [dict(j=pos[int(b)], smbh=1, mass=float(z["m_final"][b]*1e8) if "m_final" in z else 0, on=(ign.get(int(b), 0.) if act.get(int(b)) else 1e9),
                gone=float(death[pos[int(b)]]) if rem[pos[int(b)]] == 9 else 1e9) for b in smbh]
    if duty:
        for b in bhs: b["iv"] = iv.get(next(int(k) for k in smbh if pos[int(k)] == b["j"]), [])
    out = []
    for e in evs:
        if e["type"] not in TYPES: continue
        d = dict(type=TYPES[e["type"]], f=fv(e["frame"]), g=e["g"], x=e["x"], mass=e["mass"], j=pos.get(e["i"], -1),
                 t_myr=e["t"]*sim2.T_G_MYR)
        if e["type"] == "supernova":
            d.update(rem=e["remnant"], kick=e["kick_kms"], grb=e["grb"], trig=e["triggered"], spin=e["spin"])
            if e["remnant"] == "black hole": bhs.append(dict(j=d["j"], smbh=0, mass=e["mass"]*.3, on=1e9, born=d["f"], gone=1e9))
        if e["type"] == "tidal disruption": d["bh"] = pos.get(e["b"], -1)
        if e["type"] == "black hole merger": d.update(gw=e["gw_energy"], g2=e["g2"])
        out.append(d)
    from collections import Counter
    cnt = Counter(TYPES[e["type"]] for e in J["events"] if e["type"] in TYPES); cnt["GRB"] = sum(1 for e in J["events"] if e.get("grb"))
    cnt["pulsar"] = sum(1 for e in J["events"] if e.get("remnant") == "pulsar")
    data = dict(title=f"NIM universe · seed {a.seed} · high-energy events", counts=dict(galaxy=ntot), gal=gal, sys=sysb, gallery=gallery(a.seed), tour=dict(planet=pl),
                ev=dict(list=out, bh=bhs, death=death.tolist(), rem=rem.tolist(), isBH=[int(k0[i] == 4) for i in idx], counts=dict(cnt),
                        params={k: round(v, 4) for k, v in J["event_params"].items()}, label=J["event_label"], qf=gal["q"], cores_g=list(range(gal["nG"]))))
    R = lambda p: open(os.path.join(HERE, p)).read()
    app = R("viewer_src/app.js").replace("// ---------------- loop + scripted API", R("viewer_src/events.js")+"\n// ---------------- loop + scripted API", 1)
    app = app.replace("if(fc.kind==='galcore')", "if(fc.kind==='evt') return EV.pos(fc.e,fc.rem); if(fc.kind==='galcore')", 1)
    app = app.replace("renderer.render(scenes[mode], camera);", "EV.update(); if(mode==='galaxy') EV.render(); else renderer.render(scenes[mode], camera);", 1)
    app = app.replace("window.NIM = {", "window.NIM = {events:D.ev.list, viewEvent:(k,dist,dir,df,rem)=>EV.view(k,dist,dir,df,rem), ", 1)
    html = R("viewer_src/template.html").replace("</style>", R("viewer_src/events.css")+"</style>", 1)
    for key, val in [("/*THREE*/", R("vendor/three.min.js")), ("/*ORBIT*/", R("vendor/OrbitControls.js")), ("/*DATA*/", json.dumps(data, separators=(",", ":"))),
                     ("/*SHADERS*/", R("viewer_src/shaders.js")+"\n"+R("viewer_src/shaders_events.js")), ("/*APP*/", app)]:
        html = html.replace(key, val.replace("</script", "<\\/script"), 1)
    o = a.out or f"{OUT}/viewer_events_seed{a.seed}.html"; open(o, "w").write(html)
    print(f"wrote {o} ({os.path.getsize(o)/1e6:.1f} MB); events shown {len(out)} of {len(J['events'])}: {dict(cnt)}")
if __name__ == "__main__": main()
