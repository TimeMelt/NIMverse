"""v4 high-energy events: stellar lifetimes, core-collapse supernovae (mass loss, kicks, remnant shells that trigger
star formation), neutron stars/pulsars, stellar black holes, supermassive black holes in every galaxy core (gravitating
point masses with accretion + capture radius), tidal disruption events, SMBH mergers in galaxy mergers, AGN/quasar
jets, long GRBs from collapsars.  Settings come from a dedicated inceptron "event node" (ESPACE); individual
events are rolled with the NIM quantum RNG.  Units: kpc, 1e8 Msun, G=1, t_unit = T_G_MYR (~47 Myr)."""
import numpy as np
from .sim2 import make_galaxies, make_system, leapfrog, host_star, tidal_tensors, leapfrog_tidal, TIDE_PHYS, accel, T_G_MYR
from .nim_rng import NIMRandom
from .leaves import strong_field as sf

# Strong-field fidelity (improvement item 6).  Unit mapping: sim mass unit = 1e8 Msun -> SMBH M_sun = m*MASS_UNIT_MSUN;
# stellar BHs use their physical remnant mass (0.3*smass, Msun); disrupted star m_star = smass (Msun), R_star = R_sun*m^0.8 (AU).
# Lengths compared in AU (strong_field units), independent of the kpc capture radius tde_r which only selects candidates.
MASS_UNIT_MSUN = 1e8
def star_radius_au(m_star): return float(.00465*max(m_star, .08)**.8)
def bh_fidelity(M_bh_msun, m_star=1., chi=0.):
    """Additive strong-field fields for an event log entry (AU / dimensionless)."""
    Rs = star_radius_au(m_star); rs = sf.r_s(M_bh_msun); rt = sf.r_tde(M_bh_msun, m_star, Rs); ri = sf.r_isco(M_bh_msun, chi)
    return dict(M_bh_msun=float(M_bh_msun), r_s=float(rs), r_tde=float(rt), r_isco=float(ri), visible=bool(rt > rs),
                gamma=float(sf.gamma_orbit(max(rt, ri), M_bh_msun)), eta=float(1-np.sqrt(1-rs/(3*ri))))   # eta = 1-E_isco (2GM/3c^2 r_isco); Schwarzschild 0.057
def agn_fid(m_sim, chi=0.):
    """AGN strong-field report: r_s / r_isco (AU), thin-disk efficiency eta, L_edd (erg/s); disk exists outside ISCO."""
    M = float(m_sim)*MASS_UNIT_MSUN; f = bh_fidelity(M, 1., chi)
    return dict(M_bh_msun=M, r_s=f["r_s"], r_isco=f["r_isco"], eta=f["eta"], L_edd=1.26e38*M, gamma_isco=float(sf.gamma_orbit(f["r_isco"], M)))

# v5 (alternating training) adds two learnable fixes; absent keys -> v4 behaviour, so v4 results stay reproducible
ESPACE5_EXTRA = {"tde_eff": (0., .1), "agn_duty_myr": (5., 200.)}
ESPACE = {"sn_mass_min": (5., 14.), "bh_mass_min": (15., 40.), "life_exp": (1.8, 3.2), "kick_kms": (0., 600.),
          "smbh_frac": (.0005, .02), "agn_prob": (0., 1.), "tde_r": (.02, .5), "smbh_merge_r": (.1, 3.), "grb_prob": (0., .3)}
EDEFAULTS = {k: (lo+hi)/2 for k, (lo, hi) in ESPACE.items()}
V_KMS = 977.8/T_G_MYR               # 1 velocity unit in km/s (~20.7)
SMBH, SBH, NS, WD, GONE = 4, 5, 6, 7, 9   # particle kinds (0 core halo, 1 star)

def life_myr(M, ex):
    with np.errstate(divide="ignore"): return 1e4*np.asarray(M, float)**-ex
def ev_rng(seed): return NIMRandom(int((seed*2654435761 + 99991) % 2**31), n_wires=6, shots=1024)

def add_smbh(Gx, EP):
    """One SMBH per galaxy core (mass smbh_frac of the core), split off the core mass."""
    cores = np.where(Gx["kind"] == 0)[0]; k = len(cores)
    mb = EP["smbh_frac"]*Gx["m"][cores]; Gx["m"][cores] -= mb
    for f, v in [("x", Gx["x"][cores]), ("v", Gx["v"][cores]), ("m", mb), ("e2", np.full(k, .05**2)), ("kind", np.full(k, SMBH)),
                 ("gid", Gx["gid"][cores]), ("smass", np.zeros(k))]:
        Gx[f] = np.concatenate([Gx[f], v])
    return Gx

def run_events(Gx, C, EP, seed, nframes, gravity):
    """Leapfrog (BH tree or direct) with an event pass every `ev_every` steps.  Returns frames + event log."""
    er = ev_rng(seed); x, v, m, e2, kind = Gx["x"].copy(), Gx["v"].copy(), Gx["m"].copy(), Gx["e2"].copy(), Gx["kind"].copy()
    sm, gid, R = Gx["smass"], Gx["gid"], Gx["R"]; n = len(m); st = np.where(kind == 1)[0]
    dt, steps = C["galaxy_dt"], C["galaxy_steps"]; every = max(1, steps//nframes); evk = C.get("ev_every", 5)
    life = life_myr(sm, EP["life_exp"]); tdie = np.full(n, np.inf); tdie[st] = life[st]*(1-er.random(len(st)))/T_G_MYR  # steady-state ages
    smbh = list(np.where(kind == SMBH)[0]); active = {int(i): False for i in smbh}; accreted = {int(i): 0. for i in smbh}
    duty = EP["agn_duty_myr"]/T_G_MYR if "agn_duty_myr" in EP else 0.; fresh = {int(i): False for i in smbh}; until = {int(i): 0. for i in smbh}; on_time = {int(i): 0. for i in smbh}
    # strong-field TDE gate: fidelity fields are always logged; "tag" (default) keeps type/AGN fuel unchanged (learned scores
    # stay put: default SMBHs ~1e8.5-1e9 Msun are above the Hills mass for sun-like stars, so a hard gate swallows nearly all);
    # "strict"/True relabels R_tde<r_s as "swallowed whole" (not luminous, no AGN fuel); False/"off" = legacy (no fields).
    sfg = EP.get("sf_gate", C.get("sf_gate", "tag")); sfg = {True: "strict", False: "off"}.get(sfg, sfg); bhM = {}
    # opt-in universe.solvers.gr (config/EP "gr_solver": true): extra gr_* fields (Kerr ISCO/photon/horizon, eta, capture flag,
    # chirp mass) on TDE / quasar / merger entries.  Logging only: dynamics, RNG draws and the default log are unchanged.
    # gr_solver="dynamics" additionally switches on (i) a Kerr-ISCO-matched Paczynski-Wiita pull of SMBHs on stars and
    # (ii) a Kerr marginally-bound capture gate (stars whose tidal radius lies inside r_mb are swallowed whole: no flare,
    # no AGN fuel).  Spin of every SMBH = C["gr_spin"] (default 0.7).  Off (default) -> RNG streams and log are untouched.
    grf = None; grdyn = False; grspin = float(C.get("gr_spin", .7))
    _grm = EP.get("gr_solver", C.get("gr_solver", False))
    if _grm:
        from .solvers import gr as _gr; grf = _gr.event_fields; grdyn = _grm == "dynamics"
    RG_KPC = 1476.625*MASS_UNIT_MSUN/3.0857e19          # GM/c^2 of one sim mass unit, in kpc
    se2 = float(C["star_soft"]**2); log = []; triggered = 0; Mref = float(np.median(m[smbh])) if smbh else 1.
    def srcs(): return np.where((kind == 0) | (kind == SMBH) | (kind == SBH))[0]
    def acc():
        if gravity == "bh":
            from .bh import accel_bh
            s = srcs(); stars = np.where((kind != 0) & (kind != SMBH) & (kind != SBH))[0]
            a_ = accel_bh(x, m, e2, s, stars, se2, C.get("bh_theta", .6))
        else: a_ = accel(x, m, e2)
        if grdyn:
            rs = _gr.pw_rs(grspin)*RG_KPC; sl = np.where(kind == 1)[0]
            for b in np.where(kind == SMBH)[0]:
                a_[sl] += _gr.pw_correction(x[sl] - x[b], m[b], rs*m[b])
        return a_
    a = acc(); fx, fv, fk = [], [], []
    for i in range(steps):
        v += .5*dt*a; x += dt*v
        t = (i+1)*dt; fr = (i+1)//every
        if i % evk == 0:
            # 1) stellar deaths
            dead = np.where((kind == 1) & (tdie <= t))[0]
            for j in dead:
                M = sm[j]
                if M < EP["sn_mass_min"]: kind[j] = WD; continue
                bh = M >= EP["bh_mass_min"]; rem = (.3*M if bh else 1.4); fb = .7 if bh else 0.
                kick = er.normal(0., 1., 3).ravel()*EP["kick_kms"]/V_KMS/np.sqrt(3)*(1-fb)
                m[j] *= rem/M; v[j] += kick; kind[j] = SBH if bh else NS
                if bh: e2[j] = .25*se2; bhM[int(j)] = .3*float(M)
                grb = bool(bh and er.random() < EP["grb_prob"]); pulsar = bool(not bh and er.random() < .7)
                d = np.linalg.norm(x[st]-x[j], axis=1); near = st[(d < .35) & (kind[st] == 1)]   # shell-triggered star formation
                trig = near[er.random(len(near)) < .3] if len(near) else near
                tdie[trig] = t + life[trig]/T_G_MYR; triggered += len(trig)
                log.append(dict(type="supernova", t=t, frame=fr, i=int(j), g=int(gid[j]), x=x[j].tolist(), mass=float(M),
                                remnant="black hole" if bh else ("pulsar" if pulsar else "neutron star"), kick_kms=float(np.linalg.norm(kick)*V_KMS),
                                grb=grb, triggered=int(len(trig)), spin=float(.5+er.random()*3)))
            # 2) tidal disruption / capture by black holes (radius ~ M^(1/3))
            for b in np.where((kind == SMBH) | (kind == SBH))[0]:
                rt = EP["tde_r"]*(m[b]/Mref)**(1/3); live = np.where(kind == 1)[0]
                d = np.linalg.norm(x[live]-x[b], axis=1); hit = live[d < rt]
                if "tde_eff" in EP and len(hit):   # v5: only stars bound to the hole (loss cone), each disrupted with prob tde_eff per pass
                    dv = np.linalg.norm(v[hit]-v[b], axis=1); dh = np.maximum(d[d < rt], 1e-4)
                    hit = hit[(dv**2 < 2*m[b]/dh) & (er.random(len(hit)) < EP["tde_eff"])]
                for j in hit:
                    Mb = float(m[b])*MASS_UNIT_MSUN if kind[b] == SMBH else bhM.get(int(b), float(m[b])*MASS_UNIT_MSUN)
                    fid = bh_fidelity(Mb, float(sm[j])) if sfg != "off" else {}; lum = fid.get("visible", True) or sfg != "strict"   # dynamics identical either way
                    if grdyn:   # Kerr capture: tidal radius inside the marginally bound orbit -> swallowed whole
                        cap = bh_fidelity(Mb, float(sm[j]))["r_tde"] < _gr.capture_radius(grspin)*Mb*_gr.M_SUN_AU
                        lum = lum and not cap; fid = dict(fid, gr_kerr_captured=bool(cap))
                    p = m[b]*v[b] + .5*m[j]*v[j]; m[b] += .5*m[j]; v[b] = p/m[b]; kind[j] = GONE; m[j] = 0.
                    if kind[b] == SMBH: accreted[int(b)] += 1; fresh[int(b)] = fresh[int(b)] or lum   # swallowed whole: no disk -> no AGN fuel
                    log.append(dict(type="tidal disruption" if lum else "swallowed whole", **({"swallowed": not fid["visible"]} if fid else {}), t=t, frame=fr, i=int(j), b=int(b),
                                    g=int(gid[b]), x=x[b].tolist(), mass=float(sm[j]), **fid,
                                    **(grf(Mb, grspin if grdyn else 0., r_peri_au=bh_fidelity(Mb, float(sm[j]))["r_tde"], m2_msun=float(sm[j])) if grf else {})))
            # 3) SMBH mergers
            sb = [b for b in smbh if kind[b] == SMBH]
            for p_ in range(len(sb)):
                for q_ in range(p_+1, len(sb)):
                    b1, b2 = sb[p_], sb[q_]
                    if kind[b1] != SMBH or kind[b2] != SMBH: continue
                    if np.linalg.norm(x[b1]-x[b2]) < EP["smbh_merge_r"]:
                        M1, M2 = m[b1], m[b2]; v[b1] = (M1*v[b1]+M2*v[b2])/(M1+M2); x[b1] = (M1*x[b1]+M2*x[b2])/(M1+M2)
                        m[b1] = .95*(M1+M2); m[b2] = 0.; kind[b2] = GONE; active[int(b1)] = True; until[int(b1)] = t + duty
                        log.append(dict(type="black hole merger", t=t, frame=fr, i=int(b1), b=int(b2), g=int(gid[b1]), g2=int(gid[b2]), x=x[b1].tolist(),
                                        mass=float(M1+M2), gw_energy=.05*float(min(M1, M2)),
                                        **(dict(grf(float(M1+M2)*MASS_UNIT_MSUN), gr_chirp_msun=float((M1*M2)**.6/(M1+M2)**.2)*MASS_UNIT_MSUN,
                                                  **_gr.ringdown_fields(float(M1)*MASS_UNIT_MSUN, float(M2)*MASS_UNIT_MSUN, grspin if grdyn else 0., grspin if grdyn else 0.)) if grf else {})))
            # 4) AGN switch-on: fed SMBHs light up with probability agn_prob
            if duty:   # v5: a fresh feeding episode ignites a quasar with prob agn_prob; it fades after agn_duty_myr
                for b in smbh:
                    ib = int(b)
                    if kind[b] != SMBH: continue
                    if active[ib] and t >= until[ib]:
                        active[ib] = False; log.append(dict(type="quasar fades", t=t, frame=fr, i=ib, g=int(gid[b]), x=x[b].tolist(), mass=float(m[b])))
                    if fresh[ib] and not active[ib] and er.random() < EP["agn_prob"]:
                        active[ib] = True; until[ib] = t + duty; log.append(dict(type="quasar ignition", t=t, frame=fr, i=ib, g=int(gid[b]), x=x[b].tolist(), mass=float(m[b]), **(agn_fid(m[b]) if sfg != "off" else {}), **(grf(float(m[b])*MASS_UNIT_MSUN) if grf else {})))
                    fresh[ib] = False; on_time[ib] += evk*dt*active[ib]
            for b in ([] if duty else smbh):
                if kind[b] == SMBH and not active[int(b)] and accreted[int(b)] > 0 and er.random() < EP["agn_prob"]:
                    active[int(b)] = True; log.append(dict(type="quasar ignition", t=t, frame=fr, i=int(b), g=int(gid[b]), x=x[b].tolist(), mass=float(m[b]), **(agn_fid(m[b]) if sfg != "off" else {}), **(grf(float(m[b])*MASS_UNIT_MSUN) if grf else {})))
        a = acc(); v += .5*dt*a
        if i % every == 0: fx.append(x.copy()); fv.append(v.copy()); fk.append(kind.copy())
    for b in ([] if duty else smbh):   # initial AGN roll for unfed SMBHs (quiescent vs active nuclei)
        if kind[b] == SMBH and not active[int(b)] and er.random() < EP["agn_prob"]*.5: active[int(b)] = True
    info = dict(agn_frac=(float(np.mean([on_time[k]/(steps*dt) for k in on_time])) if duty and on_time else None), log=log, triggered=triggered, active={k: bool(v_) for k, v_ in active.items()}, m_final=m, kind_final=kind,
                smbh=smbh, tdie=tdie, n_massive=int(((kind == 1) | (kind >= 4)).sum()))
    return np.array(fx), np.array(fv), np.array(fk), info

def simulate_ev(r, C, P, EP, nframes, seed, system=True):
    Gx = make_galaxies(r, C, P); S = make_system(r, C, P); Gx = add_smbh(Gx, EP)
    gx, gv, gk, info = run_events(Gx, C, EP, seed, nframes, C.get("gravity", "direct"))
    host = host_star(Gx, C)
    if not system: return Gx, S, gx, gv, None, None, host, gk, info
    if C.get("tides", False):
        Ts = tidal_tensors(gx, info["m_final"], Gx["e2"], host); scale = TIDE_PHYS*C.get("tide_boost", 1.)
        sx, sv = leapfrog_tidal(S["x"], S["v"], S["m"], C["system_dt"], C["system_steps"], nframes, Ts, scale)
    else:
        sx, sv = leapfrog(S["x"], S["v"], S["m"], np.full(len(S["m"]), 1e-8), C["system_dt"], C["system_steps"], nframes)
    return Gx, S, gx, gv, sx, sv, host, gk, info
