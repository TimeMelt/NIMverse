"""v2 universe: self-gravitating galaxies (numba softened direct-sum), encounters/tidal tails,
featured star system (binaries, planets, moons, asteroid belt) hosted by a star in galaxy 0.
All randomness comes from `r` (NIMRandom or numpy Generator); generation settings from `P`."""
import numpy as np, numba as nb

# learnable generation settings: name -> (lo, hi); hand-picked default = midpoint
SPACE = {"spread_mult": (.6, 1.4), "gal_mass_mult": (.5, 1.5), "gal_radius_mult": (.6, 1.4), "vel_disp": (0., .6),
         "disk_frac": (.05, .35), "thickness": (.01, .09), "arms_mean": (2., 5.), "spiral_prob": (.4, 1.),
         "imf_slope": (1.85, 2.85), "encounter_prob": (0., .6), "halo_core": (.3, .9), "planets_mean": (2., 8.),
         "spacing_mult": (.8, 1.2), "moon_dist_mult": (.5, 1.5), "binary_prob": (0., .6), "belt_prob": (.2, 1.)}
DEFAULTS = {k: (lo+hi)/2 for k, (lo, hi) in SPACE.items()}

def U(r, a=0., b=1., n=None): return a + (b-a)*(r.random() if n is None else np.asarray(r.random(n)))
def N(r, n): return np.asarray(r.normal(0., 1., n), float).reshape(-1)
def rot(r):
    q = N(r, 4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])

@nb.njit(parallel=True, fastmath=True, cache=True)
def accel(x, m, e2):
    n = x.shape[0]; a = np.zeros_like(x)
    for i in nb.prange(n):
        ax = ay = az = 0.
        for j in range(n):
            if j == i or m[j] == 0.: continue
            dx = x[j, 0]-x[i, 0]; dy = x[j, 1]-x[i, 1]; dz = x[j, 2]-x[i, 2]
            r2 = dx*dx+dy*dy+dz*dz + .5*(e2[i]+e2[j]); f = m[j]/(r2*np.sqrt(r2))
            ax += f*dx; ay += f*dy; az += f*dz
        a[i, 0] = ax; a[i, 1] = ay; a[i, 2] = az
    return a

def leapfrog(x, v, m, e2, dt, steps, nframes):
    x, v = x.copy(), v.copy(); every = max(1, steps//nframes); fx, fv = [], []
    a = accel(x, m, e2)
    for i in range(steps):
        v += .5*dt*a; x += dt*v; a = accel(x, m, e2); v += .5*dt*a
        if i % every == 0: fx.append(x.copy()); fv.append(v.copy())
    return np.array(fx), np.array(fv)

def imf(r, n, slope):  # power-law IMF on [0.1, 30] Msun, dN/dm ~ m^-slope
    u = U(r, 0, 1, n); a = 1-slope; lo, hi = .1**a, 30.**a
    return (lo + u*(hi-lo))**(1/a)

def make_galaxies(r, C, P):
    G = []
    for g in range(C["n_galaxies"]):
        c = U(r, -1, 1, 3)*C["galaxy_spread"]*P["spread_mult"]; v0 = N(r, 3)*P["vel_disp"]
        M = U(r, 500, 2000)*P["gal_mass_mult"]; R = U(r, 4, 10)*P["gal_radius_mult"]
        G.append(dict(c=c, v0=v0, M=M, R=R, Rot=rot(r), spiral=r.random() < P["spiral_prob"],
                      arms=int(np.clip(round(P["arms_mean"] + N(r, 1)[0]), 1, 6))))
    enc = None
    want = r.random() < P["encounter_prob"]
    if len(G) > 1 and (want or C.get("force_encounter", False)):   # galaxy 1 on a parabolic fly-by of galaxy 0
        g0, g1 = G[0], G[1]; s = g0["R"]+g1["R"]; D = C.get("enc_D", 6)*s; b = C.get("enc_b", 1.5)*s
        u = N(r, 3); u /= np.linalg.norm(u); w = np.cross(u, N(r, 3)); w /= np.linalg.norm(w)
        if C.get("enc_prograde", False):   # orbit in galaxy-0's disk plane, same sense as its spin -> strong tidal tails
            ex, ey = g0["Rot"][:, 0], g0["Rot"][:, 1]; ph = U(r, 0, 2*np.pi)
            u = np.cos(ph)*ex + np.sin(ph)*ey; w = np.cos(ph+np.pi/2)*ex + np.sin(ph+np.pi/2)*ey
            # incoming along -u with offset +w: L = (D u + b w) x (-u) = b (u x w) = +b z  -> prograde
        g1["c"] = g0["c"] + D*u + b*w; g1["v0"] = g0["v0"] - np.sqrt(2*(g0["M"]+g1["M"])/np.hypot(D, b))*u; enc = (0, 1)
    X, V, Mm, E2, K, GID, SM = [], [], [], [], [], [], []
    for g, d in enumerate(G):
        n = C["stars_per_galaxy"]; a = P["halo_core"]*d["R"]; Md = P["disk_frac"]*d["M"]; Mc = d["M"]-Md
        X.append(d["c"]); V.append(d["v0"]); Mm.append(Mc); E2.append(a*a); K.append(0); GID.append(g); SM.append(0.)
        ms = imf(r, n, P["imf_slope"])
        rad = np.minimum(-np.log(np.maximum(U(r, 0, 1, n)*U(r, 0, 1, n), 1e-9))*d["R"]/3.5, 1.6*d["R"]) + .3
        th = U(r, 0, 2*np.pi, n)
        if d["spiral"]:
            th = np.floor(U(r, 0, 1, n)*d["arms"])*2*np.pi/d["arms"] + 2.2*np.log(rad) + N(r, n)*.3
        lp = np.c_[rad*np.cos(th), rad*np.sin(th), N(r, n)*P["thickness"]*d["R"]]
        ae2 = .5*(a*a + C["star_soft"]**2)
        order = np.argsort(rad); menc = np.empty(n); menc[order] = np.cumsum(np.ones(n))*Md/n
        vc = np.sqrt(Mc*rad**2/(rad**2+ae2)**1.5 + menc/np.sqrt(rad**2+C["star_soft"]**2))
        if C.get("equilibrium", False):
            # vertical equilibrium: sigma_z = nu*h, nu^2 = core vertical frequency + thin-disk term 4*pi*G*rho_mid
            h = P["thickness"]*d["R"]; Sig = Md/(2*np.pi*(d["R"]/3.5)**2)*np.exp(-rad*3.5/d["R"])
            nu2 = Mc/(rad**2+ae2)**1.5 + menc/(rad**2+C["star_soft"]**2)**1.5
            sz = np.sqrt(nu2)*h; sr = np.minimum(.08*vc, 1.5*sz)    # mild radial dispersion, asymmetric drift corr.
            vphi = np.sqrt(np.maximum(vc**2 - 2*sr**2*rad/(d["R"]/3.5) , .5*vc**2))
            lv = np.c_[-vphi*np.sin(th) + N(r, n)*sr*np.cos(th), vphi*np.cos(th) + N(r, n)*sr*np.sin(th), N(r, n)*sz]
        else:
            lv = np.c_[-vc*np.sin(th), vc*np.cos(th), np.zeros(n)] + N(r, 3*n).reshape(n, 3)*.05*vc[:, None]
        X += list(d["c"] + lp@d["Rot"].T); V += list(d["v0"] + lv@d["Rot"].T)
        Mm += list(np.full(n, Md/n) if C.get("equal_mass", False) else Md*ms/ms.sum()); E2 += [C["star_soft"]**2]*n; K += [1]*n; GID += [g]*n; SM += list(ms)
    return dict(x=np.array(X), v=np.array(V), m=np.array(Mm), e2=np.array(E2), kind=np.array(K), gid=np.array(GID),
                smass=np.array(SM), R=np.array([d["R"] for d in G]), encounter=enc)

def make_system(r, C, P):
    B = dict(name=[], x=[], v=[], m=[], kind=[])   # kind: 0 star,1 planet,2 moon,3 asteroid
    def add(n, x, v, m, k): B["name"].append(n); B["x"].append(np.asarray(x, float)); B["v"].append(np.asarray(v, float)); B["m"].append(m); B["kind"].append(k)
    M1 = U(r, .7, 1.5); binary = r.random() < P["binary_prob"]
    if binary:
        M2 = M1*U(r, .3, 1.); ab = U(r, .05, .15); Mt = M1+M2; vrel = np.sqrt(Mt/ab)
        add("starA", [ab*M2/Mt, 0, 0], [0, vrel*M2/Mt, 0], M1, 0); add("starB", [-ab*M1/Mt, 0, 0], [0, -vrel*M1/Mt, 0], M2, 0)
        a = max(3.5*ab, U(r, .3, .5))
    else:
        Mt = M1; add("star", [0, 0, 0], [0, 0, 0], M1, 0); a = U(r, .3, .5)
    npl = int(np.clip(round(P["planets_mean"] + N(r, 1)[0]), 1, 9)); radii = []
    for p in range(npl):
        a *= U(r, 1.4, 1.9)*P["spacing_mult"]; radii.append(a)
        m = 10**U(r, -6, -3); inc = N(r, 1)[0]*.03; th = U(r, 0, 2*np.pi)
        pp = a*np.array([np.cos(th), np.sin(th)*np.cos(inc), np.sin(th)*np.sin(inc)])
        vv = np.sqrt(Mt/a)*np.array([-np.sin(th), np.cos(th)*np.cos(inc), np.cos(th)*np.sin(inc)])
        add(f"planet{p}", pp, vv, m, 1)
        if r.random() < .6:
            hill = a*(m/(3*Mt))**(1/3)
            for k in range(int(1+U(r)*3)):
                d = hill*U(r, .15, .4)*P["moon_dist_mult"]; ph = U(r, 0, 2*np.pi)
                add(f"moon{p}.{k}", pp + d*np.array([np.cos(ph), np.sin(ph), 0]), vv + np.sqrt(m/d)*np.array([-np.sin(ph), np.cos(ph), 0]), m*1e-3, 2)
    belt = None
    if npl >= 2 and r.random() < P["belt_prob"]:
        k = int(U(r)*(npl-1)); r0, r1 = radii[k], radii[k+1]; lo, hi = r0+.2*(r1-r0), r1-.2*(r1-r0); belt = (lo, hi)
        na = C["asteroids"]; ra = U(r, lo, hi, na); th = U(r, 0, 2*np.pi, na); z = N(r, na)*.01*ra; vk = np.sqrt(Mt/ra)
        for i in range(na): add(f"ast{i}", [ra[i]*np.cos(th[i]), ra[i]*np.sin(th[i]), z[i]], [-vk[i]*np.sin(th[i]), vk[i]*np.cos(th[i]), 0], 0., 3)
    S = {k: np.array(v) for k, v in B.items()}; S.update(binary=binary, belt=belt, Mt=Mt); return S

def host_star(Gx, C):
    """Star in galaxy 0 nearest 0.55 R from its core: hosts the featured system."""
    st = np.where((Gx["gid"] == 0) & (Gx["kind"] == 1))[0]
    d = np.linalg.norm(Gx["x"][st]-Gx["x"][0], axis=1); return int(st[np.argmin(np.abs(d-.55*Gx["R"][0]))])

def leapfrog_bh(x, v, m, e2, kind, dt, steps, nframes, theta=.6):
    from .bh import accel_bh
    cores = np.where(kind == 0)[0]; stars = np.where(kind != 0)[0]; se2 = float(e2[stars[0]])
    x, v = x.copy(), v.copy(); every = max(1, steps//nframes); fx, fv = [], []
    a = accel_bh(x, m, e2, cores, stars, se2, theta)
    for i in range(steps):
        v += .5*dt*a; x += dt*v; a = accel_bh(x, m, e2, cores, stars, se2, theta); v += .5*dt*a
        if i % every == 0: fx.append(x.copy()); fv.append(v.copy())
    return np.array(fx), np.array(fv)

# unit bridge for galactic tides: galaxy units (1 kpc, 1e8 Msun, G=1) -> t_g ~ 47 Myr;
# system units (1 AU, 1 Msun, G=1) -> t_s = 1 yr/2pi.  Tidal tensor scales as (t_s/t_g)^2.
_KPC_KMS = 3.0857e16/3.15576e13   # 1 kpc/(km/s) in Myr ~ 977.8
T_G_MYR = _KPC_KMS/np.sqrt(4.30091e-6*1e8)        # ~47.2 Myr
TIDE_PHYS = ((1/(2*np.pi))/(T_G_MYR*1e6))**2       # ~1.1e-17

def tidal_tensors(gx, m, e2, host, h=.05):
    """Galactic tidal tensor T_ij = d a_i / d x_j at the host star for every frame (central differences)."""
    from .bh import direct_src
    src = np.arange(len(m)) != host; Ts = np.zeros((len(gx), 3, 3))
    for k, X in enumerate(gx):
        xs, ms, es = np.ascontiguousarray(X[src]), m[src], e2[src]
        pts = X[host] + h*np.vstack([np.eye(3), -np.eye(3)]); a = direct_src(pts, np.full(6, e2[host]), xs, ms, es)
        T = (a[:3]-a[3:]).T/(2*h); Ts[k] = .5*(T+T.T)
    return Ts

def leapfrog_tidal(x, v, m, dt, steps, nframes, Ts, scale):
    """System integrator with external tidal acceleration a = scale*T(t)·x (host-centred frame)."""
    x, v = x.copy(), v.copy(); every = max(1, steps//nframes); fx, fv = [], []; e2 = np.full(len(m), 1e-8)
    def acc(x, i): return accel(x, m, e2) + scale*x@Ts[min(i//every, len(Ts)-1)].T
    a = acc(x, 0)
    for i in range(steps):
        v += .5*dt*a; x += dt*v; a = acc(x, i); v += .5*dt*a
        if i % every == 0: fx.append(x.copy()); fv.append(v.copy())
    return np.array(fx), np.array(fv)

def cosmo_factors(z_start, dt, steps, unit_myr=None, n_sub=8):
    """CosmoDrive (opt-in, C["cosmo_drive"] = {"z_start": z}): comoving leapfrog factors for the galaxy N-body.
    Comoving x with a~ = a/a_start (so x = physical at the start), canonical momentum p = a~^2 dx/dt:
        dx/dt = p/a~^2,   dp/dt = -(1/a~) [grad Phi(x) - (4 pi G/3) rho_m x]   (G = 1; rho_m comoving mean matter density)
    i.e. the KDK step uses kick = int dt/a~ and drift = int dt/a~^2 (GADGET-style) and the background term makes an
    isolated overdense region obey the exact physical equation r'' = -GM/r^2 + (Lambda/3) r: bound galaxies decouple
    from the expansion, the group's outskirts feel the Hubble/Lambda flow.  Returns (k1, d, k2, bg, a_end) per step."""
    from .leaves import cosmology as cz
    um = T_G_MYR if unit_myr is None else unit_myr; ug = um/1e3
    a0 = 1/(1 + float(z_start)); t0 = cz.t_of_a(a0)
    at = lambda t: cz.a_of_t(t)/a0
    def integ(ta, tb, p):
        tt = np.linspace(ta, tb, n_sub + 1); y = np.array([at(t)**-p for t in tt]); return float(np.trapezoid(y, tt))/ug
    # rho_m (comoving, sim units 1e8 Msun / kpc^3): Om rho_crit0 / a_start^3 ; rho_crit0 = 3 H0^2/(8 pi G)
    rho_c0 = 277.5*(cz.H0/100.)**2/1e8; bg = 4*np.pi/3*cz.OM*rho_c0/a0**3
    k1 = np.empty(steps); d = np.empty(steps); k2 = np.empty(steps)
    for i in range(steps):
        ta, tm, tb = t0 + i*dt*ug, t0 + (i + .5)*dt*ug, t0 + (i + 1)*dt*ug
        k1[i] = integ(ta, tm, 1); d[i] = integ(ta, tb, 2); k2[i] = integ(tm, tb, 1)
    return k1, d, k2, bg, at(t0 + steps*dt*ug), np.array([at(t0 + i*dt*ug) for i in range(steps + 1)])

def leapfrog_cosmo(x, v, m, e2, dt, steps, nframes, z_start, accf=None):
    """Comoving KDK (see cosmo_factors); frames are physical positions a~ x and peculiar velocities p/a~."""
    k1, dd, k2, bg, a_end, aa = cosmo_factors(z_start, dt, steps)
    acc = (lambda y: accel(y, m, e2)) if accf is None else accf
    x, p = x.copy(), v.copy(); every = max(1, steps//nframes); fx, fv = [], []
    g = acc(x) + bg*x
    for i in range(steps):
        p += k1[i]*g; x += dd[i]*p; g = acc(x) + bg*x; p += k2[i]*g
        if i % every == 0: fx.append(aa[i + 1]*x); fv.append(p/aa[i + 1])
    return np.array(fx), np.array(fv), dict(z_start=float(z_start), a_end_rel=float(a_end), z_end=float((1 + z_start)/a_end - 1),
                                             t_gyr=float(steps*dt*T_G_MYR/1e3))

def simulate(r, C, P, nframes):
    Gx = make_galaxies(r, C, P); S = make_system(r, C, P)
    cd = C.get("cosmo_drive")
    if cd:                                                              # opt-in; default path below is untouched
        if C.get("gravity", "direct") == "bh":
            from .bh import accel_bh
            cores = np.where(Gx["kind"] == 0)[0]; stars = np.where(Gx["kind"] != 0)[0]; se2 = float(Gx["e2"][stars[0]])
            af = lambda y: accel_bh(y, Gx["m"], Gx["e2"], cores, stars, se2, C.get("bh_theta", .6))
        else: af = None
        gx, gv, Gx["cosmo"] = leapfrog_cosmo(Gx["x"], Gx["v"], Gx["m"], Gx["e2"], C["galaxy_dt"], C["galaxy_steps"], nframes,
                                             float(cd.get("z_start", 1.) if isinstance(cd, dict) else 1.), af)
    elif C.get("gravity", "direct") == "bh":
        gx, gv = leapfrog_bh(Gx["x"], Gx["v"], Gx["m"], Gx["e2"], Gx["kind"], C["galaxy_dt"], C["galaxy_steps"], nframes, C.get("bh_theta", .6))
    else:
        gx, gv = leapfrog(Gx["x"], Gx["v"], Gx["m"], Gx["e2"], C["galaxy_dt"], C["galaxy_steps"], nframes)
    host = host_star(Gx, C)
    if C.get("tides", False):
        Ts = tidal_tensors(gx, Gx["m"], Gx["e2"], host); scale = TIDE_PHYS*C.get("tide_boost", 1.)
        sx, sv = leapfrog_tidal(S["x"], S["v"], S["m"], C["system_dt"], C["system_steps"], nframes, Ts, scale)
        S["tide"] = dict(scale=scale, boost=C.get("tide_boost", 1.), Tmax=float(np.abs(np.linalg.eigvalsh(Ts)).max()))
    else:
        sx, sv = leapfrog(S["x"], S["v"], S["m"], np.full(len(S["m"]), 1e-8), C["system_dt"], C["system_steps"], nframes)
    return Gx, S, gx, gv, sx, sv, host
