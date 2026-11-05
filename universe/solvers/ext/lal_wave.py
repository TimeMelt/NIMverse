"""LALSuite / lalsimulation GR waveforms (opt-in when gr_solver is on).

Attaches real IMRPhenomD / IMRPhenomXPHM tags (f_merge, f_ring, peak amp, optional downsampled h+/hx)
onto event-log fields. Falls back silently so solvers.gr.imr_tags PhenomA path remains available
when lalsimulation import fails.
"""
from __future__ import annotations
import numpy as np

MSUN_SI = 1.9884099021470415e30
PC_SI = 3.085677581491367e16
WAVEFORM_CAP = 4096  # max samples stored in an event log


def lalsim_available():
    try:
        import lalsimulation  # noqa: F401
        import lal  # noqa: F401
        return True
    except Exception:
        return False


def _approx(name="IMRPhenomD"):
    import lalsimulation as lalsim
    name = str(name)
    if hasattr(lalsim, name): return getattr(lalsim, name)
    # fallbacks
    for cand in ("IMRPhenomD", "IMRPhenomXPHM", "IMRPhenomPv2", "IMRPhenomC"):
        if hasattr(lalsim, cand): return getattr(lalsim, cand)
    raise RuntimeError("no Phenom approximant in lalsimulation")


def peak_freq_hz(m1, m2, chi1=0., chi2=0., approx="IMRPhenomD"):
    """Merger / peak GW frequency (Hz) for a BBH."""
    import lalsimulation as lalsim
    if approx == "IMRPhenomD" and hasattr(lalsim, "IMRPhenomDGetPeakFreq"):
        return float(lalsim.IMRPhenomDGetPeakFreq(float(m1), float(m2), float(chi1), float(chi2)))
    # estimate from FD amplitude peak
    wf = fd_waveform(m1, m2, chi1, chi2, approx=approx, f_min=20., f_max=2048., delta_f=.25)
    return float(wf["f_peak_hz"])


def fd_waveform(m1, m2, chi1=0., chi2=0., dist_mpc=410., f_min=20., f_max=1024., delta_f=.125,
                approx="IMRPhenomD", inclination=0.):
    """Frequency-domain strain via SimInspiralChooseFDWaveform. Returns dict with f, hp, hc, tags."""
    import lalsimulation as lalsim, lal
    m1si = float(m1) * lal.MSUN_SI; m2si = float(m2) * lal.MSUN_SI
    dist = float(dist_mpc) * 1e6 * lal.PC_SI
    apx = _approx(approx)
    s1x = s1y = s2x = s2y = 0.; s1z = float(chi1); s2z = float(chi2)
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1si, m2si, s1x, s1y, s1z, s2x, s2y, s2z, dist, float(inclination), 0., 0., 0., 0.,
        float(delta_f), float(f_min), float(f_max), float(f_min), None, apx)
    hp_d = np.array(hp.data.data); hc_d = np.array(hc.data.data)
    f = np.arange(len(hp_d)) * float(delta_f)
    amp = np.abs(hp_d)
    band = f >= f_min
    # physical peak often near merger; prefer GetPeakFreq when available
    try: f_peak = peak_freq_hz(m1, m2, chi1, chi2, approx=approx if approx == "IMRPhenomD" else "IMRPhenomD")
    except Exception:
        i0 = int(np.argmax(amp[band])) if band.any() else int(np.argmax(amp))
        f_peak = float(f[band][i0]) if band.any() else float(f[i0])
    # ringdown ~ remnant QNM; use PhenomA-ish ratio as tag if no QNM helper
    f_ring = float(f_peak * 1.2)
    return dict(f=f, hp=hp_d, hc=hc_d, amp=amp, f_peak_hz=f_peak, f_ring_hz=f_ring,
                peak_amp=float(np.nanmax(amp[band])) if band.any() else float(np.nanmax(amp)),
                approx=str(approx), f_min=float(f_min), f_max=float(f_max), delta_f=float(delta_f),
                m1=float(m1), m2=float(m2), dist_mpc=float(dist_mpc))


def td_waveform(m1, m2, chi1=0., chi2=0., dist_mpc=410., f_min=20., delta_t=1 / 4096.,
                approx="IMRPhenomD", inclination=0.):
    """Time-domain strain via SimInspiralChooseTDWaveform."""
    import lalsimulation as lalsim, lal
    m1si = float(m1) * lal.MSUN_SI; m2si = float(m2) * lal.MSUN_SI
    dist = float(dist_mpc) * 1e6 * lal.PC_SI
    apx = _approx(approx)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1si, m2si, 0., 0., float(chi1), 0., 0., float(chi2), dist, float(inclination), 0., 0., 0., 0.,
        float(delta_t), float(f_min), float(f_min), None, apx)
    hp_d = np.array(hp.data.data); hc_d = np.array(hc.data.data)
    t = np.arange(len(hp_d)) * float(delta_t)
    amp = np.sqrt(hp_d ** 2 + hc_d ** 2); ipeak = int(np.argmax(amp))
    return dict(t=t, hp=hp_d, hc=hc_d, amp=amp, t_peak=float(t[ipeak]), peak_amp=float(amp[ipeak]),
                approx=str(approx), delta_t=float(delta_t), m1=float(m1), m2=float(m2))


def downsample_polarizations(hp, hc, f=None, t=None, cap=WAVEFORM_CAP):
    """Store a size-capped copy of h+/hx (and f or t) for event logs."""
    n = len(hp); step = max(1, int(np.ceil(n / cap)))
    out = dict(hp=np.asarray(hp[::step], float).tolist(), hc=np.asarray(hc[::step], float).tolist(),
               n_full=int(n), stride=int(step))
    if f is not None: out["f_hz"] = np.asarray(f[::step], float).tolist()
    if t is not None: out["t_s"] = np.asarray(t[::step], float).tolist()
    return out


def enrich_imr_tags(base, m1, m2, chi1=0., chi2=0., approx="IMRPhenomD", store_waveform=False,
                    f_lo_hz=10., domain="fd"):
    """Merge LAL Phenom tags into a solvers.gr.imr_tags dict. Returns base unchanged if LAL fails."""
    if not lalsim_available(): return base
    try:
        # Prefer XPHM when requested and present, else PhenomD
        import lalsimulation as lalsim
        if approx == "auto":
            approx = "IMRPhenomXPHM" if hasattr(lalsim, "IMRPhenomXPHM") else "IMRPhenomD"
        if domain == "td":
            wf = td_waveform(m1, m2, chi1, chi2, approx=approx, f_min=max(f_lo_hz, 10.))
            f_merge = peak_freq_hz(m1, m2, chi1, chi2)
            f_ring = f_merge * 1.2
            peak_amp = wf["peak_amp"]
            extra_wf = downsample_polarizations(wf["hp"], wf["hc"], t=wf["t"]) if store_waveform else None
        else:
            wf = fd_waveform(m1, m2, chi1, chi2, approx=approx, f_min=max(f_lo_hz, 10.))
            f_merge = wf["f_peak_hz"]; f_ring = wf["f_ring_hz"]; peak_amp = wf["peak_amp"]
            extra_wf = downsample_polarizations(wf["hp"], wf["hc"], f=wf["f"]) if store_waveform else None
        out = dict(base)
        # Keep PhenomA / caller primary tags; attach LAL as parallel fields (tests assert PhenomA bands).
        out.update(gr_imr_lal=True, gr_imr_lal_approx=str(approx),
                   gr_imr_lal_f_merge_hz=float(f_merge), gr_imr_lal_f_ring_hz=float(f_ring),
                   gr_imr_lal_peak_amp=float(peak_amp),
                   gr_imr_model=str(base.get("gr_imr_model", "PhenomA-tags")) + f"+LAL-{approx}")
        if extra_wf is not None: out["gr_imr_waveform"] = extra_wf
        return out
    except Exception:
        return base


def gw150914_like_fd(approx="IMRPhenomD"):
    """GW150914-ish FD waveform helper for tests (36+29 Msun, ~410 Mpc)."""
    return fd_waveform(36., 29., 0., 0., dist_mpc=410., f_min=20., f_max=1024., delta_f=.125, approx=approx)
