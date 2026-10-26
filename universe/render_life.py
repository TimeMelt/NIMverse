"""Record per-step life snapshots (via life_loop record callback) and render dashboard / planet card / evolution mp4.
python -m universe.render_life --seed 4 --years 20000 --dt 20 [--json results/life_demo_seed4.json] [--no-rerun]"""
import argparse, json, os, shutil, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from .run_life import common_args, load_lp
OUT = "/workspace/nim_universe/out"; RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black", "text.color": "white",
                     "axes.labelcolor": "white", "xtick.color": "#bbb", "ytick.color": "#bbb", "axes.edgecolor": "#555", "font.size": 10})
TC = {"lava": "#ff5a1f", "gas giant": "#e0b070", "ocean": "#3fa9ff", "ice": "#bfe8ff", "rocky": "#a08060", "terran": "#3fd06b"}
BIO = ["o2", "ch4", "pigments", "tech"]; BC = ["#7fdbff", "#ffb347", "#5cff7a", "#ff4fd8"]

def rerun(a):
    from .sim2 import make_system, DEFAULTS
    from .leaves import life_loop
    C = json.load(open(a.config)); C = {**C, "asteroids": C.get("eval", {}).get("asteroids", 40)}
    S = make_system(np.random.default_rng(a.seed), C, DEFAULTS); rec = []
    rep = life_loop(S, a.seed, load_lp(a.params), a.years, a.dt, a.plant, culture=a.culture, distance_pc=a.distance_pc, record=rec.append)
    return rep, rec

def dashboard(rep, path):
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 2, hspace=.35, wspace=.25)
    pl = rep["planets"]; h0 = pl[0]["habitat"]; life = set(rep["with_life"]); tech = set(rep["with_tech"])
    ax = fig.add_subplot(gs[0, 0]); ax.axvspan(h0["hz_in"], h0["hz_out"], color="#1f6f3f", alpha=.35, label="habitable zone")
    ax.scatter([0.15], [0], s=900, c="#fff6c0", edgecolors="#ffd040", zorder=3); ax.text(.15, .3, f"star\n{rep['host']['M_star']:.2f} M☉, {rep['host']['T_eff']:.0f} K", ha="center", va="bottom", fontsize=8)
    for j, p in enumerate(pl):
        h = p["habitat"]; sz = 60 + 400*np.cbrt(h["m"]/1e-5)/3
        ax.scatter([h["a"]], [0], s=sz, c=TC.get(h["ptype"], "#999"), zorder=3, edgecolors="white" if h["name"] in life else "none", linewidths=2)
        lbl = f"{h['name']}\n{h['ptype']}\n{h['a']:.2f} AU"
        if h["name"] in life: lbl += "\n★ LIFE" + (" + TECH" if h["name"] in tech else "")
        ax.text(h["a"], .25 if j % 2 else -.75, lbl, ha="center", fontsize=8, color="#5cff7a" if h["name"] in life else "white")
    ax.set_xscale("log"); ax.set_xlim(.1, 25); ax.set_ylim(-1, 1.2); ax.set_yticks([]); ax.set_xlabel("semi-major axis (AU)")
    ax.set_title(f"A · System seed {rep['seed']}", loc="left"); ax.legend(loc="lower right", facecolor="black", labelcolor="white", fontsize=8)
    lp = [p for p in pl if p["habitat"]["name"] in life] or [p for p in pl if "climate" in p]
    ax = fig.add_subplot(gs[0, 1])
    if lp:
        p = lp[0]; c = p["climate"]; b = p["biosign"]
        vals = [b[k] for k in BIO]; ax.barh(BIO, vals, color=BC); ax.barh(["biomass"], [p["biomass"]], color="#3fd06b")
        for i, v in enumerate(vals + [p["biomass"]]): ax.text(v + .02, i, f"{v:.3g}", va="center", fontsize=9)
        ax.set_xlim(0, 1.4); ax.set_title(f"B · {p['habitat']['name']} ({p['habitat']['ptype']}): T={c['T_K']:.1f} K, liquid water={c['liquid_water_frac']*100:.0f}%, {c['climate_class']}", loc="left")
    ax = fig.add_subplot(gs[1, 0]); rad = rep["radiation"]; fr = rad.get("first", {})
    for i, (k, v) in enumerate(fr.items()):
        ax.plot([v["t_emit"], v["t_arr"]], [i, i], color="#ffd040", lw=3); ax.scatter([v["t_emit"]], [i], c="#5cff7a", s=60, zorder=3); ax.scatter([v["t_arr"]], [i], c="#ff4fd8", s=60, marker="D", zorder=3)
        ax.text(v["t_arr"] + 300, i, f"emit {v['t_emit']:.0f} → arrive {v['t_arr']:.0f} yr", va="center", fontsize=8)
    ax.set_yticks(range(len(fr))); ax.set_yticklabels(list(fr)); ax.set_xlim(0, rep["years"]); ax.set_ylim(-.7, len(fr) - .3); ax.set_xlabel("world time (yr)")
    obs = rad["observers"][0]; ax.set_title(f"C · First detections at {obs['name']} ({obs['distance_pc']:.0f} pc, delay {list(rad['delay_yr'].values())[0]:.1f} yr) — {rad['n_detections']} detections / {rad['n_arrived']} signals", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[1, 1]); hf = rep["host_flares"]; sim = [p for p in pl if "impacts" in p]
    names = [p["habitat"]["name"] for p in sim]; ev = [p["impacts"]["events"] for p in sim]; ch = [p["climate_history"]["impacts_total"] for p in sim]
    x = np.arange(len(names)); ax.bar(x - .2, ev, .4, color="#ff7f50", label="impacts → biosphere"); ax.bar(x + .2, ch, .4, color="#888", label="impacts (climate hist.)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(facecolor="black", labelcolor="white", fontsize=8); ax.set_ylabel("impact events")
    ax.set_title(f"D · Host flares: {hf['n_flares']} over {rep['years']:.0f} yr, UV mean {hf['uv_mean']:.3f} / max {hf['uv_max']:.3f}, dM★={hf['dM_star']:.1e}", loc="left", fontsize=9)
    fig.suptitle("nim-universe · life dashboard", fontsize=16, color="#5cff7a"); fig.savefig(path, dpi=110); plt.close(fig)

def card(rep, path):
    p = next(p for p in rep["planets"] if p["habitat"]["name"] in rep["with_life"]); h, c, b = p["habitat"], p["climate"], p["biosign"]
    fig = plt.figure(figsize=(8, 10)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1.25); ax.axis("off")
    rs = np.random.default_rng(1); ax.scatter(rs.random(250), rs.random(250)*1.25, s=rs.random(250)*3, c="white", alpha=.6)
    for r, al in [(.30, .08), (.27, .15), (.25, 1)]: ax.add_patch(plt.Circle((.5, .88), r, color=TC["ocean"] if al == 1 else "#5cff7a", alpha=al))
    for k in range(7): ax.add_patch(plt.Circle((.5 + rs.normal(0, .1), .88 + rs.normal(0, .1)), .03 + .03*rs.random(), color="#2e8b57", alpha=.8))
    ax.add_patch(matplotlib.patches.Wedge((.5, .88), .25, 90, 270, color="black", alpha=.35))
    ax.text(.5, .55, h["name"].upper(), ha="center", fontsize=26, weight="bold"); ax.text(.5, .51, f"{h['ptype']} world · {h['a']:.2f} AU · in habitable zone", ha="center", color="#aaa")
    rows = [("Temperature", f"{c['T_K']:.1f} K ({c['T_K']-273.15:.1f} °C) · {c['climate_class']}"), ("Liquid water", f"{c['liquid_water_frac']*100:.0f}% of surface"),
            ("Life began", f"t = {p['gate']['t_w']:.0f} yr (abiogenesis, not planted)"), ("Biomass", f"{p['biomass']:.2f}  · diversity {p['biosphere']['diversity']:.0f}"),
            ("Creatures", f"n ≈ {p['creatures']['n']:.2f}" if p.get("creatures") else "—"), ("Technology", f"level {p['tech']:.2f} (culture since t={p['culture']['t_start']:.0f} yr)" if p.get("culture") else f"{p['tech']:.2f}")]
    cu = p.get("culture") or {}
    if "energy_mix" in cu: rows.append(("Tech climate", f"net {cu['warm_K']:+.2f} K (GHG {cu['greenhouse_K']:+.2f}, aerosol −{cu['aerosol_K']:.2f}) · acid {cu['acidification']:.2f} · mitig {cu['mitigation']:.2f}\n"
                                                     f"fossil {cu['energy_mix']['fossil']*100:.0f}% · {len(cu['polities'])} polities · conflict {cu['conflict']:.2f}"))
    for i, (k, v) in enumerate(rows): ax.text(.08, .45 - i*.04, k, color="#5cff7a", fontsize=12); ax.text(.36, .45 - i*.04, v, fontsize=12 if "\n" not in v else 9, va="top" if "\n" in v else "baseline")
    ax.text(.08, .17, "Biosignatures", color="#5cff7a", fontsize=12)
    for i, k in enumerate(BIO):
        y = .13 - i*.03; ax.add_patch(plt.Rectangle((.3, y), .55, .018, color="#222")); ax.add_patch(plt.Rectangle((.3, y), .55*min(b[k], 1), .018, color=BC[i]))
        ax.text(.1, y, k, fontsize=10); ax.text(.87, y, f"{b[k]:.3g}", fontsize=9)
    fig.savefig(path, dpi=110); plt.close(fig)

def video(rec, path, secs=18, fps=30):
    names = sorted({r["planet"] for r in rec}); tl = {n: [r for r in rec if r["planet"] == n] for n in names}
    live = [n for n in names if any(r["alive"] for r in tl[n])] or names; n = live[0]; R = tl[n]
    t = np.array([r["t_w"] for r in R]); keys = ["T", "water", "biomass", "o2", "ch4", "pigments", "tech", "n_org"]; D = {k: np.array([r[k] for r in R]) for k in keys}
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2)); fig.suptitle(f"{n} · life evolution (seed 4)", color="#5cff7a", fontsize=15)
    specs = [(axs[0, 0], ["T"], ["#ff7f50"], "Temperature (K)"), (axs[0, 1], ["water", "biomass"], ["#3fa9ff", "#3fd06b"], "water frac / biomass"),
             (axs[1, 0], BIO, BC, "biosignature flux"), (axs[1, 1], ["n_org"], ["#ffd040"], "organisms n")]
    lines = []
    for ax, ks, cs, ttl in specs:
        for k, c in zip(ks, cs):
            ln, = ax.plot([], [], color=c, lw=2, label=k); lines.append((ln, k))
        ax.set_xlim(0, t[-1]); lo = min(D[k].min() for k in ks); hi = max(D[k].max() for k in ks); pad = (hi - lo)*.1 + 1e-3
        ax.set_ylim(lo - pad, hi + pad); ax.set_title(ttl, loc="left"); ax.legend(facecolor="black", labelcolor="white", fontsize=8, loc="upper left")
    txt = fig.text(.5, .01, "", ha="center", fontsize=12)
    nf = secs*fps; idx = np.linspace(1, len(t), nf).astype(int)
    def upd(f):
        i = idx[f]
        for ln, k in lines: ln.set_data(t[:i], D[k][:i])
        r = R[i - 1]; ev = "  ★ LIFE" if r["alive"] else ""; ev += "  ⚙ TECH" if r["tech"] > 0 else ""
        txt.set_text(f"t = {r['t_w']:6.0f} yr   T={r['T']:.1f} K   water={r['water']:.2f}   biomass={r['biomass']:.2f}{ev}"); return []
    animation.FuncAnimation(fig, upd, frames=nf).save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000)); plt.close(fig)
    return n

def main(argv=None):
    ap = argparse.ArgumentParser(); common_args(ap); ap.add_argument("--json", default=None); ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args(argv); os.makedirs(OUT, exist_ok=True); s = a.seed
    rep, rec = (json.load(open(a.json)), None) if a.no_rerun else rerun(a)
    if a.json and not a.no_rerun: rep = json.load(open(a.json))  # dashboard from the canonical JSON
    outs = [f"life_dashboard_seed{s}.png", f"life_planet_card_seed{s}.png"]
    dashboard(rep, os.path.join(OUT, outs[0])); card(rep, os.path.join(OUT, outs[1]))
    if rec:
        json.dump(rec, open(os.path.join(OUT, f"life_record_seed{s}.json"), "w")); outs.append(f"life_record_seed{s}.json")
        video(rec, os.path.join(OUT, f"life_evolution_seed{s}.mp4")); outs.append(f"life_evolution_seed{s}.mp4")
    for o in outs: shutil.copy(os.path.join(OUT, o), os.path.join(RES, o)); print("wrote", os.path.join(OUT, o))

if __name__ == "__main__": main()
