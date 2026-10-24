"""Random 3D universe generator + leapfrog N-body simulator + renderer, using NIM as the RNG."""
import json, sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from universe.nim_rng import NIMRandom

def rand_rot(r):
    q = r.normal(size=4); q = q/np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])

def make_galaxies(r, C):
    P, V, M, K, G = [], [], [], [], []
    for g in range(C["n_galaxies"]):
        c = r.uniform(-C["galaxy_spread"], C["galaxy_spread"], 3)*C.get("spread_mult", 1.0); v0 = r.normal(0, C.get("vel_disp", 0.3), 3)
        Mg = r.uniform(500, 2000)*C.get("gal_mass_mult", 1.0); R = r.uniform(4, 10)*C.get("gal_radius_mult", 1.0); Rot = rand_rot(r)
        P.append(c); V.append(v0); M.append(Mg); K.append(0); G.append(g)   # core: SMBH+halo point mass
        n = C["stars_per_galaxy"]; arms = r.integers(2, 5); spiral = r.random() > .3
        rad = R*np.sqrt(r.random(n)) + 0.5
        th = r.uniform(0, 2*np.pi, n)
        if spiral: th = np.floor(r.random(n)*arms)*2*np.pi/arms + rad*0.6 + r.normal(0, .25, n)
        lp = np.c_[rad*np.cos(th), rad*np.sin(th), r.normal(0, 0.03*R, n)]
        vc = np.sqrt(Mg/np.sqrt(rad**2 + C["softening"]**2))
        lv = np.c_[-vc*np.sin(th), vc*np.cos(th), np.zeros(n)]
        P += list(c + lp@Rot.T); V += list(v0 + lv@Rot.T); M += [1e-3]*n; K += [1]*n; G += [g]*n
    return np.array(P), np.array(V), np.array(M), np.array(K), np.array(G)

def make_system(r, C):
    names, P, V, M, K = ["star"], [np.zeros(3)], [np.zeros(3)], [r.uniform(0.7, 1.5)], [1]
    Ms = M[0]; a = r.uniform(0.3, 0.5)
    for p in range(r.integers(*C["planets_range"])):
        a *= r.uniform(1.4, 1.9)*C.get("spacing_mult", 1.0); m = 10**r.uniform(-6, -3); inc = r.normal(0, 0.03); th = r.uniform(0, 2*np.pi)
        pp = a*np.array([np.cos(th), np.sin(th)*np.cos(inc), np.sin(th)*np.sin(inc)])
        vv = np.sqrt(Ms/a)*np.array([-np.sin(th), np.cos(th)*np.cos(inc), np.cos(th)*np.sin(inc)])
        names.append(f"planet{p}"); P.append(pp); V.append(vv); M.append(m); K.append(2)
        if r.random() < C["moon_prob"]:
            hill = a*(m/(3*Ms))**(1/3)
            for k in range(r.integers(1, C["max_moons"]+1)):
                d = hill*r.uniform(0.15, 0.4)*C.get("moon_dist_mult", 1.0); ph = r.uniform(0, 2*np.pi)
                names.append(f"moon{p}.{k}"); P.append(pp + d*np.array([np.cos(ph), np.sin(ph), 0]))
                V.append(vv + np.sqrt(m/d)*np.array([-np.sin(ph), np.cos(ph), 0])); M.append(m*1e-3); K.append(3)
    return names, np.array(P, float), np.array(V, float), np.array(M), np.array(K)

def accel(x, m, eps, src=None):
    s = np.arange(len(m)) if src is None else src
    d = x[s][None] - x[:, None]
    r2 = (d**2).sum(-1) + eps**2
    return (d*(m[s]/r2**1.5)[..., None]).sum(1)

def leapfrog(x, v, m, dt, steps, nframes, eps, src=None):
    x, v = x.copy(), v.copy(); fr, every = [], max(1, steps//nframes); a = accel(x, m, eps, src)
    for i in range(steps):
        v += .5*dt*a; x += dt*v; a = accel(x, m, eps, src); v += .5*dt*a
        if i % every == 0: fr.append(x.copy())
    return np.array(fr)

def render(fg, kind, gid, fs, sk, out, seed, nG, video=True):
    cm = plt.get_cmap("tab10"); col = [cm(g % 10) for g in gid]; sz = np.where(kind == 0, 40, 1.5)
    S = {1: (120, "gold"), 2: (25, "deepskyblue"), 3: (6, "lightgray")}
    ss, sc = [S[k][0] for k in sk], [S[k][1] for k in sk]
    lg = np.percentile(np.abs(fg[0] - fg[0].mean(0)), 99)*1.05; ls = np.abs(fs[0]).max()*1.1
    fig = plt.figure(figsize=(14, 7), facecolor="black")
    a1 = fig.add_subplot(121, projection="3d"); a2 = fig.add_subplot(122, projection="3d")
    def draw(t):
        for ax, L in ((a1, lg), (a2, ls)):
            ax.cla(); ax.set_facecolor("black"); ax.set_axis_off()
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
        a1.scatter(*fg[t].T, s=sz, c=col, alpha=.85, depthshade=False, linewidths=0)
        a1.set_title(f"NIM universe seed={seed}: {nG} galaxies  frame {t}", color="w")
        tr = fs[max(0, t-30):t+1]
        for i, k in enumerate(sk):
            if k == 2: a2.plot(*tr[:, i].T, c="deepskyblue", lw=.6, alpha=.5)
        a2.scatter(*fs[t].T, s=ss, c=sc, depthshade=False)
        a2.set_title("featured star system: star / planets / moons", color="w")
        a1.view_init(25, 30+t*.75); a2.view_init(35, 30+t*.75)
    draw(len(fg)-1); fig.savefig(f"{out}/universe_seed{seed}.png", dpi=110, facecolor="black"); print("png saved")
    if not video: return
    anim = animation.FuncAnimation(fig, draw, frames=len(fg))
    anim.save(f"{out}/universe_seed{seed}.mp4", writer=animation.FFMpegWriter(fps=20), dpi=80, savefig_kwargs={"facecolor": "black"}); print("mp4 saved")
    os.system(f"ffmpeg -loglevel error -y -i {out}/universe_seed{seed}.mp4 -vf 'fps=12,scale=700:-1' {out}/universe_seed{seed}.gif"); print("gif saved")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_config.json"))
    p.add_argument("--seed", type=int); p.add_argument("--galaxies", type=int); p.add_argument("--stars", type=int)
    p.add_argument("--rng", choices=["nim", "numpy"]); p.add_argument("--params", help="learned params json (from learn.py)"); p.add_argument("--tag", default=""); p.add_argument("--no-video", action="store_true")
    a = p.parse_args(); C = json.load(open(a.config))
    for k, v in (("seed", a.seed), ("n_galaxies", a.galaxies), ("stars_per_galaxy", a.stars), ("rng", a.rng)):
        if v is not None: C[k] = v
    if a.params: C.update(json.load(open(a.params))["params"])
    r = NIMRandom(C["seed"]) if C["rng"] == "nim" else np.random.default_rng(C["seed"])
    if C["rng"] == "nim": print("NIM free-think inceptron expvals:", np.round(r.expvals, 4))
    x, v, m, kind, gid = make_galaxies(r, C)
    cores = np.where(kind == 0)[0]   # restricted N-body: everything feels the galaxy cores; cores feel each other
    fg = leapfrog(x, v, m, C["galaxy_dt"], C["galaxy_steps"], C["frames"], C["softening"], src=cores)
    names, sx, sv, sm, sk = make_system(r, C)
    fs = leapfrog(sx, sv, sm, C["system_dt"], C["system_steps"], C["frames"], 1e-4)   # full N-body
    n = min(len(fg), len(fs)); fg, fs = fg[:n], fs[:n]
    o = C["out_dir"]; os.makedirs(o, exist_ok=True); s = f'{C["seed"]}{a.tag}'
    np.savez_compressed(f"{o}/universe_seed{s}.npz", galaxy_frames=fg, kind=kind, gid=gid, mass=m,
                        system_frames=fs, system_names=np.array(names), system_mass=sm)
    json.dump({"config": C, "system_bodies": names}, open(f"{o}/universe_seed{s}.json", "w"), indent=1)
    print(f"simulated {len(m)} galactic bodies + {len(sm)} system bodies: {names}")
    render(fg, kind, gid, fs, sk, o, s, C["n_galaxies"], not a.no_video)

if __name__ == "__main__": main()
