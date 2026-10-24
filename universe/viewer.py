"""Build a single self-contained HTML viewer for a simulated NIM universe.
python -m universe.viewer --seed 7 --params /workspace/nim_universe/out/learned_params_v3.json
Reuses the saved fly-by run (npz/json) when present for that seed; procedural body detail comes from bodies.py (NIM RNG)."""
import argparse, base64, json, os, sys, numpy as np
from . import sim2, bodies
from .nim_rng import NIMRandom
HERE = os.path.dirname(__file__); OUT = "/workspace/nim_universe/out"
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def star_rgb(T):  # rough blackbody -> sRGB
    t = T/100.
    r = 1. if t <= 66 else np.clip(329.7*(t-60)**-.1332/255, 0, 1)
    g = np.clip((99.47*np.log(t)-161.1)/255, 0, 1) if t <= 66 else np.clip(288.1*(t-60)**-.0755/255, 0, 1)
    b = 1. if t >= 66 else (0. if t <= 19 else np.clip((138.5*np.log(t-10)-305)/255, 0, 1))
    return [float(r), float(g), float(b)]

def galaxy_block(z, per=1500, step=2, force=()):
    k, gid, sm, F = z["kind"], z["gid"], z["smass"], z["galaxy_frames"][::step]
    cores = np.where(k == 0)[0]; nG = len(cores); rs = np.random.default_rng(1)
    keep = [cores]; R, M = [], []
    for g in range(nG):
        st = np.where((k == 1) & (gid == g))[0]; d = np.linalg.norm(F[0, st]-F[0, cores[g]], axis=1)
        R.append(float(np.percentile(d, 90))); M.append(float(sm[st].sum()*100))
        keep.append(np.sort(rs.choice(st, min(per, len(st)), replace=False)))
    idx = np.unique(np.concatenate(keep+[np.asarray(force, int)])); idx = np.concatenate([cores, idx[~np.isin(idx, cores)]]); X = F[:, idx]; ext = float(np.abs(X).max()); q = ext/32000
    # host star: galaxy-0 star nearest 0.55 R at t=0
    st0 = idx[(k[idx] == 1) & (gid[idx] == 0)]; d = np.linalg.norm(F[0, st0]-F[0, cores[0]], axis=1); host = int(np.where(idx == st0[np.argmin(abs(d-.55*R[0]))])[0][0])
    col = np.zeros((len(idx), 3)); size = np.zeros(len(idx))
    hue = np.array([[1, .85, .7], [.75, .85, 1], [1, .75, .85], [.8, 1, .85], [.95, .9, 1]])
    for j, i in enumerate(idx):
        g = gid[i]
        if k[i] == 0: col[j] = [1, .95, .8]; size[j] = 6; continue
        rr = np.linalg.norm(F[0, i]-F[0, cores[g]])/R[g]; w = np.exp(-rr*3)
        col[j] = (np.array([1, .8, .55])*w + np.array([.65, .78, 1])*(1-w))*hue[g % 5]*(.6+.4*rs.random()); size[j] = .5+.8*rs.random()**3
    centre = (F[0, cores].mean(0)); ext = float(np.linalg.norm(F[0, cores]-centre, axis=1).max()+max(R))
    return dict(n=len(idx), nf=len(F), q=q, pos=b64(np.round(X/q).astype(np.int16)), col=b64((np.clip(col, 0, 1)*255).astype(np.uint8)),
                size=b64((np.clip(size, 0, 6)*40).astype(np.uint8)), gid=gid[idx].tolist(), isCore=(k[idx] == 0).astype(int).tolist(),
                cores=list(range(nG)), R=R, M=M, nG=nG, host=host, ext=ext, hazeSize=float(np.mean(R)*.08*1.0),
                haze=(hue*.5).tolist(), _idx=idx, desc=["disk galaxy, simulated self-gravity (Barnes–Hut)"]*nG), int(len(k))

def system_block(S, frames, seed):
    k, m = S["kind"], S["m"]; det = bodies.system_details(S, seed); n = len(k)
    stars = np.where(k == 0)[0]; ab = float(np.linalg.norm(frames[0, stars[0]]-frames[0, stars[-1]])) if len(stars) > 1 else 1
    rvis = np.zeros(n); parent = -np.ones(n, int); moonK = np.ones(n)
    cur = -1
    for i in range(n):
        if k[i] == 1: cur = i
        if k[i] == 2: parent[i] = cur
    for i, d in det.items():
        if d["type"] == "star": d["rgb"] = star_rgb(d["T"]); rvis[i] = min(.04*d["radius_rel"], .3*ab)
        elif d["type"] == "moon": rvis[i] = .0045
        else: rvis[i] = .02*d["radius_rel"]/(2.2 if "giant" in d["type"] else 1)
    for p in np.where(k == 1)[0]:
        ms = np.where(parent == p)[0]
        if len(ms): moonK[p] = max(1, 3.5*rvis[p]/np.linalg.norm(frames[0, ms]-frames[0, p], axis=1).min())
    amax = float(np.linalg.norm(frames[0, k == 1], axis=1).max())
    terr = [i for i in det if det[i]["type"] in ("terran", "ocean")]; pl = terr[0] if terr else int(np.where(k == 1)[0][0])
    return dict(n=n, nf=len(frames), pos=b64(frames.astype(np.float32)), kind=k.tolist(), names=[str(s) for s in S["name"]],
                detail={int(i): d for i, d in det.items()}, rvis=rvis.tolist(), parent=parent.tolist(), moonK=moonK.tolist(),
                amax=amax, binary=bool(S["binary"]), belt=bool(S["belt"] is not None), nPl=int((k == 1).sum()), nMoon=int((k == 2).sum())), pl

def gallery(seed):  # one of each surface type, forced via synthetic orbits around a 1 Msun star
    out = []
    for gid, (me, a) in enumerate([(1.0, 1.0), (1.0, 1.0), (300, 1.5), (40, 9.), (1.5, .3), (.8, .8), (2., 4.)]):
        bid = 1000+gid
        for tries in range(40):
            d = bodies.planet_detail(seed, bid, me*bodies.EARTH, a, 1.0)
            want = ["terran", "terran", "gas giant", "ice giant", "lava", "desert", "ice"][gid]
            if d["type"] == want: break
            bid += 100
        out.append(d)
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--params", default=None)
    ap.add_argument("--run", default=None, help="npz/json stem of a saved run (default: universe_v3_seed{seed}_learned_flyby)")
    ap.add_argument("--out", default=None); ap.add_argument("--stars-per-galaxy", type=int, default=1500)
    a = ap.parse_args(argv); stem = a.run or f"{OUT}/universe_v3_seed{a.seed}_learned_flyby"
    if not os.path.exists(stem+".npz"):
        sys.exit(f"no saved run {stem}.npz; create it first with python -m universe.run3 --seed {a.seed} --params ... (the viewer replays simulated frames)")
    J = json.load(open(stem+".json")); C = J["config"]; P = json.load(open(a.params)) if a.params else J["params"]
    P = P.get("params", P); z = np.load(stem+".npz")
    r = NIMRandom(a.seed); sim2.make_galaxies(r, C, P); S = sim2.make_system(r, C, P)
    gal, ntot = galaxy_block(z, a.stars_per_galaxy); gal.pop("_idx"); sysb, pl = system_block(S, z["system_frames"], a.seed)
    data = dict(title=f"NIM universe · seed {a.seed}", counts=dict(galaxy=ntot), gal=gal, sys=sysb, gallery=gallery(a.seed), tour=dict(planet=pl))
    R = lambda p: open(os.path.join(HERE, p)).read()
    html = R("viewer_src/template.html")
    for key, val in [("/*THREE*/", R("vendor/three.min.js")), ("/*ORBIT*/", R("vendor/OrbitControls.js")), ("/*DATA*/", json.dumps(data, separators=(",", ":"))),
                     ("/*SHADERS*/", R("viewer_src/shaders.js")), ("/*APP*/", R("viewer_src/app.js"))]:
        html = html.replace(key, val.replace("</script", "<\\/script"), 1)
    out = a.out or f"{OUT}/viewer_seed{a.seed}.html"; open(out, "w").write(html)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB); tour planet {sysb['names'][pl]}; gallery {[g['type'] for g in data['gallery']]}")

if __name__ == "__main__": main()
