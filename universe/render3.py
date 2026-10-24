"""v3 renderer: stateless frames (trails rebuilt from past frames) -> rendered in parallel worker processes;
numba splats, float32, low-res glow passes; small palette GIF."""
import numpy as np, subprocess, os, numba as nb, multiprocessing as mp
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image, ImageDraw
from .render2 import bb_rgb, cam, project, tonemap, FONT, SMALL

@nb.njit(cache=True, fastmath=True)
def splat(img, u, v, col, w):
    H, W = img.shape[0], img.shape[1]
    for i in range(u.shape[0]):
        if w[i] == 0.: continue
        iu = int(u[i]); iv = int(v[i])
        if 0 <= iu < W and 0 <= iv < H and u[i] >= 0 and v[i] >= 0:
            for c in range(3): img[iv, iu, c] += col[i, c]*w[i]

def blur_lowres(img, s, sigma):
    H, W = img.shape[:2]; sm = img[::s, ::s]*0; sm = img.reshape(H//s, s, W//s, s, 3).sum((1, 3)) if H % s == 0 and W % s == 0 else img[::s, ::s]*s*s
    sm = gaussian_filter(sm, (sigma/s, sigma/s, 0)); return zoom(sm, (H/sm.shape[0], W/sm.shape[1], 1), order=1)[:H, :W]/(s*s)

_G = {}
def _init(state): _G.update(state)

def _frame(t):
    g = _G; Gx, S, host, W, H = g["Gx"], g["S"], g["host"], g["W"], g["H"]; gx, sx, nfr = g["gx"], g["sx"], len(g["gx"])
    center, ext = g["center"][t], g["ext"][t]
    R = cam(.6+2*np.pi*.35*t/nfr, .45+.15*np.sin(2*np.pi*t/nfr)); dist = 2.4*ext; f = .9*W
    img = np.zeros((H, W, 3), np.float32); sk = g["sky"]
    su, sv, _, _ = project(sk*1e6, np.zeros(3), R, 0., f, W, H); ok = (sk@R.T)[:, 2] > 0
    splat(img, su, sv, g["sky_c"], g["sky_b"]*ok)
    pts = np.zeros_like(img); trail = np.zeros_like(img)
    for j in range(6):                              # trails = decayed splats of the previous 5 frames (stateless)
        if t-j < 0: break
        u, v, z, ok = project(gx[t-j], center, R, dist, f, W, H); w = g["lum"]*(dist/np.maximum(z, 1e-3))**2*ok
        if j == 0: splat(pts, u, v, g["col"], w); u0, v0, z0, ok0, w0 = u, v, z, ok, w
        else: splat(trail, u, v, g["col"], w*.5*.82**j)
    u, v, z, ok, w = u0, v0, z0, ok0, w0; cores = g["cores"]; st = Gx["kind"] == 1
    img += pts + np.minimum(trail, pts.max()*0+.5) + gaussian_filter(pts, (1.2, 1.2, 0))*2.5
    bg = np.zeros_like(img); splat(bg, u, v, g["col"], w*g["big"]); img += blur_lowres(bg, 2, 4)*6
    hz = np.zeros_like(img); splat(hz, u, v, g["haze_col"], .012*ok*st*(dist/np.maximum(z, 1e-3))**2); img += blur_lowres(hz, 4, 20)*16
    cb = np.zeros_like(img); splat(cb, u[cores], v[cores], np.tile(np.float32([1., .85, .6]), (len(cores), 1)), 40*ok[cores]*(dist/np.maximum(z[cores], 1e-3)))
    img += blur_lowres(cb, 2, 6) + blur_lowres(cb, 4, 18)*3
    out = Image.fromarray(tonemap(img, 1.3)); d = ImageDraw.Draw(out)
    hu, hv = u[host], v[host]; d.ellipse([hu-14, hv-14, hu+14, hv+14], outline=(120, 255, 160), width=2)
    out.paste(_inset(t, sx), (W-460, H-460)); bx = (W-460, H-460)
    d.line([(hu+14, hv), bx], fill=(120, 255, 160), width=1); d.rectangle([bx, (W-20, H-20)], outline=(120, 255, 160), width=2)
    d.text((20, 16), g["title"], fill=(230, 230, 230), font=FONT); d.text((20, 44), f"frame {t+1}/{nfr}", fill=(160, 160, 160), font=SMALL)
    d.text((W-452, H-455), g["inset_title"], fill=(120, 255, 160), font=SMALL)
    return np.asarray(out).tobytes()

def _inset(t, sx):
    g = _G; S = g["S"]; n = 440; k = S["kind"]; pl = k <= 2; s_col, s_w = g["s_col"], g["s_w"]
    sc = np.abs(S["x"][k == 1]).max()*1.15 if (k == 1).any() else 1
    R = cam(.3+.6*t/240, .9); c = sx[t][k == 0].mean(0)
    img = np.zeros((n, n, 3), np.float32); tr = np.zeros_like(img)
    for j in range(1, 90, 2):                       # orbit trails from the last ~90 frames
        if t-j < 0: break
        u, v, _, ok = project(sx[t-j][pl], c, R, 4*sc, .45*n*4, n, n); splat(tr, u, v, s_col[pl], .25*(1-j/90)*s_w[pl]*ok)
    u, v, _, ok = project(sx[t], c, R, 4*sc, .45*n*4, n, n)
    p = np.zeros_like(img); splat(p, u[pl], v[pl], s_col[pl], s_w[pl]*ok[pl])
    a = np.zeros_like(img); splat(a, u[~pl], v[~pl], s_col[~pl], s_w[~pl]*ok[~pl])
    img += np.clip(tr, 0, .4) + gaussian_filter(p, (1, 1, 0))*4 + a + gaussian_filter(a, (1, 1, 0))
    s = np.zeros_like(img); splat(s, u[k == 0], v[k == 0], s_col[k == 0], 30*np.ones((k == 0).sum(), np.float32))
    img += gaussian_filter(s, (3, 3, 0)) + gaussian_filter(s, (10, 10, 0))*2
    return Image.fromarray(tonemap(img, 1.5))

def render_movie(Gx, S, gx, sx, host, out_prefix, W, H, title, fps=30, focus="all", workers=6, inset_title="featured system (in galaxy 0)"):
    st = Gx["kind"] == 1; m = np.where(st, Gx["smass"], 1.); cores = np.where(Gx["kind"] == 0)[0]
    lum = np.where(st, np.clip(m, .1, 30)**1.3, 0.); lum = .25*lum/np.percentile(lum[st], 90)
    rg = np.random.default_rng(0); sky = rg.normal(size=(2500, 3)); sky /= np.linalg.norm(sky, axis=1)[:, None]
    pal = np.array([[.35, .25, .75], [.75, .3, .5], [.3, .5, .8], [.6, .35, .7], [.4, .6, .9]])
    k = S["kind"]; sT = 5800*np.clip(S["m"], .1, 60)**.55
    sc = [[(bb_rgb(sT[i]), 6.), ([.35, .65, 1.], 1.5), ([.8, .8, .8], .5), ([.6, .45, .3], .25)][kk] for i, kk in enumerate(k)]
    # camera track: centre on all cores, or on the interacting pair (focus="encounter"), smoothed extent
    sel = cores if focus == "all" or Gx["encounter"] is None else cores[list(Gx["encounter"])]
    ids = np.isin(Gx["gid"], Gx["gid"][sel])
    C = np.array([np.median(x[sel], 0) if len(sel) > 2 else x[sel].mean(0) for x in gx])
    E = np.array([np.percentile(np.linalg.norm(x[ids]-c, axis=1), 92 if focus == "all" else 80) for x, c in zip(gx, C)])
    for i in range(1, len(E)): E[i] = .97*E[i-1]+.03*E[i]
    for i in range(1, len(C)): C[i] = .9*C[i-1]+.1*C[i]
    state = dict(Gx=Gx, S=S, host=host, W=W, H=H, gx=gx.astype(np.float32), sx=sx, title=title, center=C, ext=E, cores=cores,
                 col=bb_rgb(np.where(st, 5800*np.clip(m, .1, 60)**.55, 6000)).astype(np.float32), lum=lum.astype(np.float32),
                 big=(lum > np.percentile(lum[st], 97)).astype(np.float32), haze_col=pal[Gx["gid"] % 5].astype(np.float32),
                 sky=sky, sky_b=(rg.random(2500)**4*.6).astype(np.float32), sky_c=bb_rgb(rg.uniform(3000, 15000, 2500)).astype(np.float32),
                 s_col=np.array([c for c, _ in sc], np.float32), s_w=np.array([w for _, w in sc], np.float32), inset_title=inset_title)
    _init(state); _frame(0)                            # compile numba in parent before fork
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_prefix+".mp4"], stdin=subprocess.PIPE)
    n = len(gx); still = None
    with mp.get_context("fork").Pool(workers, _init, (state,)) as pool:
        for t, fr in enumerate(pool.imap(_frame, range(n), chunksize=2)):
            ff.stdin.write(fr)
            if t == int(n*.75): still = np.frombuffer(fr, np.uint8).reshape(H, W, 3)
    ff.stdin.close(); ff.wait(); Image.fromarray(still).save(out_prefix+".png")
    os.system(f"ffmpeg -loglevel error -y -i {out_prefix}.mp4 -vf 'fps=12,scale=640:-1:flags=lanczos,split[a][b];"
              f"[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle' {out_prefix}.gif")
