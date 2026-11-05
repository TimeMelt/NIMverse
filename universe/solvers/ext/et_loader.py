"""Numerical-relativity product loader (SpEC/SXS / Einstein Toolkit offline waveforms).

Load ASCII or HDF5 h+/hx when --nr-waveform PATH (or config nr_waveform) is set.
Otherwise callers fall through to LALSuite (ext.lal_wave) or solvers.gr PhenomA tags.
ET / SpEC themselves are not run here — document as offline producers.
"""
from __future__ import annotations
import os
import numpy as np

from . import UnavailableError

EXAMPLE_WAVE = os.path.join(os.path.dirname(__file__), "..", "data", "nr_examples", "example_bbh_strain.npz")


def available(path=None):
    p = path or os.environ.get("NR_WAVEFORM")
    return bool(p) and os.path.isfile(p)


def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    def get(*names):
        for n in names:
            if n in keys: return np.asarray(d[n])
        raise KeyError(names)
    try: t = get("t", "time", "t_s")
    except KeyError: t = None
    try: f = get("f", "freq", "f_hz")
    except KeyError: f = None
    hp = get("hp", "hplus", "h+")
    try: hc = get("hc", "hx", "hcross", "h×")
    except KeyError: hc = np.zeros_like(hp)
    meta = {k: (str(d[k]) if np.ndim(d[k]) == 0 else d[k]) for k in keys if k not in ("t", "time", "t_s", "f", "freq", "f_hz", "hp", "hplus", "h+", "hc", "hx", "hcross", "h×")}
    return dict(t=None if t is None else np.asarray(t, float),
                f=None if f is None else np.asarray(f, float),
                hp=np.asarray(hp, float), hc=np.asarray(hc, float),
                source=str(d["source"]) if "source" in keys else os.path.basename(path), meta=meta)


def _load_ascii(path):
    """Whitespace ASCII: columns t, hp, hc  OR  f, Re(hp), Im(hp), Re(hc), Im(hc)."""
    rows = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("%"): continue
        rows.append([float(x) for x in s.replace(",", " ").split()])
    if not rows: raise ValueError(f"empty NR ascii: {path}")
    arr = np.asarray(rows, float)
    if arr.shape[1] >= 3:
        # guess: if col0 spans many seconds / negative -> time domain
        c0 = arr[:, 0]
        if c0.min() < 0 or (c0.max() - c0.min()) > 1. or np.median(np.diff(c0)) < 0:
            return dict(t=c0, f=None, hp=arr[:, 1], hc=arr[:, 2], source=os.path.basename(path), meta={})
        return dict(t=c0, f=None, hp=arr[:, 1], hc=arr[:, 2], source=os.path.basename(path), meta={})
    raise ValueError(f"need >=3 columns in {path}")


def _load_hdf5(path):
    try:
        import h5py
    except ImportError as e:
        raise UnavailableError(f"h5py not installed; convert the NR waveform to npz/ascii ({e})") from e
    with h5py.File(path, "r") as f:
        # SXS-ish: Extrapolated_N2.dir/Y_l2_m2.dat or strain
        def find(node, names):
            for n in names:
                if n in node: return node[n]
            return None
        # try common layouts
        for cand in ("hp", "hplus", "strain", "rhOverM"):
            if cand in f:
                hp = np.asarray(f[cand])
                hc = np.asarray(f["hc"]) if "hc" in f else np.zeros(len(hp))
                t = np.asarray(f["t"]) if "t" in f else np.arange(len(hp))
                return dict(t=t, f=None, hp=hp.real if np.iscomplexobj(hp) else hp,
                            hc=hc.real if np.iscomplexobj(hc) else hc, source=os.path.basename(path), meta={})
        # walk for first dataset with shape (N,2) or (N,3)
        for k in f.keys():
            node = f[k]
            if hasattr(node, "shape") and len(getattr(node, "shape", ())) == 2 and node.shape[1] >= 2:
                arr = np.asarray(node)
                return dict(t=arr[:, 0], f=None, hp=arr[:, 1],
                            hc=arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr)),
                            source=os.path.basename(path), meta={"dataset": k})
    raise UnavailableError(f"unrecognised HDF5 NR layout: {path}")


def load_nr_waveform(path):
    """Load SpEC/SXS/ET product -> {t|f, hp, hc, source}."""
    path = os.path.abspath(path)
    if not os.path.isfile(path): raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith(".npz"): return _load_npz(path)
    if low.endswith(".h5") or low.endswith(".hdf5"): return _load_hdf5(path)
    if low.endswith(".txt") or low.endswith(".dat") or low.endswith(".asc"): return _load_ascii(path)
    # sniff
    try: return _load_npz(path)
    except Exception:
        return _load_ascii(path)


def tags_from_waveform(wf, f_lo_hz=10.):
    """Derive coarse f_merge / f_ring / peak_amp tags from an NR strain."""
    hp, hc = wf["hp"], wf["hc"]
    amp = np.sqrt(np.asarray(hp) ** 2 + np.asarray(hc) ** 2)
    ipeak = int(np.argmax(amp)); peak_amp = float(amp[ipeak])
    f_merge = f_ring = None
    if wf.get("f") is not None:
        f = np.asarray(wf["f"], float); band = f >= f_lo_hz
        f_merge = float(f[band][np.argmax(amp[band])]) if band.any() else float(f[ipeak])
        f_ring = f_merge * 1.2
    elif wf.get("t") is not None:
        t = np.asarray(wf["t"], float)
        # instantaneous freq near peak via phase derivative of complex strain
        h = np.asarray(hp) + 1j * np.asarray(hc)
        ph = np.unwrap(np.angle(h))
        # smooth finite difference around peak
        i0 = max(1, ipeak - 5); i1 = min(len(t) - 1, ipeak + 5)
        if i1 > i0 and t[i1] != t[i0]:
            f_inst = abs((ph[i1] - ph[i0]) / (2 * np.pi * (t[i1] - t[i0])))
            f_merge = float(f_inst); f_ring = f_merge * 1.2
    return dict(gr_imr_model=f"NR:{wf.get('source', 'file')}", gr_imr_f_merge_hz=f_merge,
                gr_imr_f_ring_hz=f_ring, gr_imr_peak_amp=peak_amp, gr_imr_nr=True,
                gr_imr_source=wf.get("source"))


def build_example_waveform(path=EXAMPLE_WAVE, m1=36., m2=29.):
    """Tiny synthetic TD chirp for tests (not real NR). Prefer LAL when available."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        from . import lal_wave
        if lal_wave.lalsim_available():
            wf = lal_wave.td_waveform(m1, m2, delta_t=1 / 2048., f_min=30.)
            # downsample for ship size
            step = max(1, len(wf["t"]) // 2048)
            np.savez_compressed(path, t=wf["t"][::step], hp=wf["hp"][::step], hc=wf["hc"][::step],
                                source="lal_IMRPhenomD_example", note="stand-in; real SpEC/ET products via --nr-waveform")
            return path
    except Exception:
        pass
    # analytic toy chirp
    t = np.linspace(-0.5, 0.05, 2048)
    f0, k = 40., 80.
    phase = 2 * np.pi * (f0 * t + .5 * k * t ** 2)
    env = np.exp(-((t - 0.) / .05) ** 2) * (t < 0.02)
    hp = env * np.cos(phase); hc = env * np.sin(phase)
    np.savez_compressed(path, t=t, hp=hp, hc=hc, source="synthetic_chirp", note="NOT NR; SpEC/ET offline")
    return path


def ensure_example():
    if not os.path.isfile(EXAMPLE_WAVE): build_example_waveform()
    return EXAMPLE_WAVE
