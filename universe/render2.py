"""numpy splat renderer: blackbody-coloured stars, mass-scaled glow, nebula haze, motion trails, star-system inset."""
import numpy as np, subprocess, os
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image, ImageDraw, ImageFont
try: FONT, SMALL = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
except Exception: FONT = SMALL = None

def bb_rgb(T):  # blackbody colour approximation (Tanner Helland), T in K -> rgb in [0,1]
    t = np.clip(T, 1000, 40000)/100.
    with np.errstate(all='ignore'): return _bb(t)
def _bb(t):
    r = np.where(t <= 66, 255, 329.7*(t-60)**-.1332); g = np.where(t <= 66, 99.47*np.log(t)-161.1, 288.1*(t-60)**-.0755)
    b = np.where(t >= 66, 255, np.where(t <= 19, 0, 138.5*np.log(np.maximum(t-10, 1e-3))-305.0))
    return np.clip(np.stack([r, g, b], -1)/255, 0, 1)

def cam(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx@Ry@np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])   # z-up world -> camera

def project(x, center, R, dist, f, Wd, Hd):
    p = (x-center)@R.T; z = p[:, 2]+dist; ok = z > .1*dist
    return f*p[:, 0]/np.maximum(z, 1e-6)+Wd/2, -f*p[:, 1]/np.maximum(z, 1e-6)+Hd/2, z, ok

def splat(img, u, v, col, w):
    H, W = img.shape[:2]; iu, iv = u.astype(int), v.astype(int); k = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    np.add.at(img, (iv[k], iu[k]), col[k]*w[k, None])

def tonemap(img, exp=1.):
    return (np.clip(1-np.exp(-img*exp), 0, 1)**(1/2.2)*255).astype(np.uint8)

class Renderer:
    def __init__(self, Gx, S, host, W=1600, H=900, seed=0, title=""):
        self.Gx, self.S, self.host, self.W, self.H, self.title = Gx, S, host, W, H, title
        st = Gx["kind"] == 1; m = np.where(st, Gx["smass"], 1.)
        T = np.where(st, 5800*np.clip(m, .1, 60)**.55, 6000); self.col = bb_rgb(T)
        self.lum = np.where(st, np.clip(m, .1, 30)**1.3, 0.); self.lum = .25*self.lum/np.percentile(self.lum[st], 90)
        self.big = self.lum > np.percentile(self.lum[st], 97)
        g = np.random.default_rng(seed); rng_cols = [np.array(c) for c in ([.35, .25, .75], [.75, .3, .5], [.3, .5, .8], [.6, .35, .7], [.4, .6, .9])]
        self.haze_col = np.array([rng_cols[i % 5] for i in Gx["gid"]])
        self.sky = g.normal(size=(2500, 3)); self.sky /= np.linalg.norm(self.sky, axis=1)[:, None]
        self.sky_b = g.random(2500)**4*.6; self.sky_c = bb_rgb(g.uniform(3000, 15000, 2500))
        self.trail = np.zeros((H, W, 3)); self.inset_trail = None
        k = S["kind"]; self.s_col = np.zeros((len(k), 3)); self.s_w = np.zeros(len(k))
        sT = 5800*np.clip(S["m"], .1, 60)**.55
        for i, kk in enumerate(k):
            self.s_col[i], self.s_w[i] = [(bb_rgb(sT[i]), 6.), ([.35, .65, 1.], 1.5), ([.8, .8, .8], .5), ([.6, .45, .3], .25)][kk]
        self.ext0 = None

    def frame(self, t, nfr, gx, sx):
        W, H = self.W, self.H; x = gx; cores = np.where(self.Gx["kind"] == 0)[0]
        center = np.median(x[cores], 0) if len(cores) > 2 else x[cores].mean(0)
        ext = np.percentile(np.linalg.norm(x-center, axis=1), 92); self.ext0 = ext if self.ext0 is None else .97*self.ext0+.03*ext
        R = cam(.6+2*np.pi*.35*t/nfr, .45+.15*np.sin(2*np.pi*t/nfr)); dist = 2.4*self.ext0; f = .9*W
        img = np.zeros((H, W, 3))
        su, sv, _, _ = project(self.sky*1e6, np.zeros(3), R, 0., f, W, H); ok = (self.sky@R.T)[:, 2] > 0
        splat(img, su[ok], sv[ok], self.sky_c[ok], self.sky_b[ok])
        u, v, z, ok = project(x, center, R, dist, f, W, H); w = self.lum*(dist/np.maximum(z, 1e-3))**2*ok
        pts = np.zeros((H, W, 3)); splat(pts, u, v, self.col, w)
        self.trail = np.maximum(self.trail*.82, pts*.5); img += pts + self.trail
        img += gaussian_filter(pts, (1.2, 1.2, 0))*2.5
        bg = np.zeros((H, W, 3)); splat(bg, u, v, self.col, w*self.big); img += gaussian_filter(bg, (4, 4, 0))*6
        hz = np.zeros((H//4+1, W//4+1, 3)); st = self.Gx["kind"] == 1
        splat(hz, u[st]/4, v[st]/4, self.haze_col[st], (.012*ok[st]*(dist/np.maximum(z[st], 1e-3))**2))
        hz = gaussian_filter(hz, (5, 5, 0)); img += zoom(hz, (H/hz.shape[0], W/hz.shape[1], 1), order=1)[:H, :W]
        cb = np.zeros((H, W, 3)); splat(cb, u[cores], v[cores], np.tile([1., .85, .6], (len(cores), 1)), 40*ok[cores]*(dist/np.maximum(z[cores], 1e-3)))
        img += gaussian_filter(cb, (6, 6, 0)) + gaussian_filter(cb, (18, 18, 0))*3
        out = Image.fromarray(tonemap(img, 1.3)); d = ImageDraw.Draw(out)
        hu, hv = u[self.host], v[self.host]; d.ellipse([hu-14, hv-14, hu+14, hv+14], outline=(120, 255, 160), width=2)
        out.paste(self.inset(t, sx), (W-460, H-460)); bx = (W-460, H-460)
        d.line([(hu+14, hv), bx], fill=(120, 255, 160), width=1); d.rectangle([bx, (W-20, H-20)], outline=(120, 255, 160), width=2)
        d.text((20, 16), self.title, fill=(230, 230, 230), font=FONT); d.text((20, 44), f"frame {t+1}/{nfr}", fill=(160, 160, 160), font=SMALL)
        d.text((W-452, H-455), "featured system (in galaxy 0)", fill=(120, 255, 160), font=SMALL)
        return np.asarray(out)

    def inset(self, t, sx):
        n = 440; k = self.S["kind"]; pl = k <= 2
        sc = np.abs(self.S["x"][k == 1]).max()*1.15 if (k == 1).any() else 1
        R = cam(.3+.6*t/240, .9); u, v, _, ok = project(sx, sx[k == 0].mean(0), R, 4*sc, .45*n*4, n, n)
        if self.inset_trail is None: self.inset_trail = np.zeros((n, n, 3))
        img = np.zeros((n, n, 3)); p = np.zeros((n, n, 3)); splat(p, u[pl], v[pl], self.s_col[pl], self.s_w[pl]*ok[pl])
        self.inset_trail = np.maximum(self.inset_trail*.985, np.clip(p, 0, .4)*.6)
        a = np.zeros((n, n, 3)); splat(a, u[~pl], v[~pl], self.s_col[~pl], self.s_w[~pl]*ok[~pl])
        img += self.inset_trail + gaussian_filter(p, (1, 1, 0))*4 + a + gaussian_filter(a, (1, 1, 0))
        s = np.zeros((n, n, 3)); splat(s, u[k == 0], v[k == 0], self.s_col[k == 0], 30*np.ones((k == 0).sum()))
        img += gaussian_filter(s, (3, 3, 0)) + gaussian_filter(s, (10, 10, 0))*2
        return Image.fromarray(tonemap(img, 1.5))

def render_movie(Gx, S, gx, sx, host, out_prefix, W, H, title, fps=30):
    R = Renderer(Gx, S, host, W, H, title=title); n = len(gx)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_prefix+".mp4"], stdin=subprocess.PIPE)
    still = None
    for t in range(n):
        fr = R.frame(t, n, gx[t], sx[t]); ff.stdin.write(fr.tobytes())
        if t == int(n*.75): still = fr
    ff.stdin.close(); ff.wait(); Image.fromarray(still).save(out_prefix+".png")
    os.system(f"ffmpeg -loglevel error -y -i {out_prefix}.mp4 -vf 'fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse' {out_prefix}.gif")
