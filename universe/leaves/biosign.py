"""Spectral biosignatures (toy, physically directed; no radiative transfer).
Turns BioSignFlux scalars (o2, ch4, pigments, h2o, tech) into a short list of absorption/emission features with
rest wavelengths (nm).  Each line: {name, species, rest_nm, kind, strength} where strength = scalar * weight, so a
line is detectable iff its species scalar * weight clears thresh[species] * weight (legacy scalar parity)."""
import math

# name, species, rest_nm, kind, weight (relative depth within its species)
LINES = [
    ("o2_760",   "o2",       760.,   "absorption", 1.0),    # O2 A-band
    ("o2_690",   "o2",       690.,   "absorption", .4),     # O2 B-band
    ("ch4_1650", "ch4",      1650.,  "absorption", 1.0),
    ("ch4_2300", "ch4",      2300.,  "absorption", .7),
    ("h2o_1400", "h2o",      1400.,  "absorption", 1.0),
    ("h2o_1900", "h2o",      1900.,  "absorption", .8),
    ("redge_720","pigments", 720.,   "continuum_step", 1.0),  # vegetation red edge
    ("tech_589", "tech",     589.,   "emission", 1.0),     # Na-lamp city-lights marker
    ("tech_10um","tech",     10000., "ir_excess", .6),     # waste-heat IR excess
]

def h2o_column(climate):
    """Water-vapour proxy: liquid-water fraction x smooth temperature (evaporation) factor, in [0, ~.6]."""
    if climate is None: return 0.
    w = float(getattr(climate, "liquid_water_frac", 0.) or 0.); T = float(getattr(climate, "T_K", 288.) or 0.)
    return float(.5*w*(1/(1 + math.exp(-(T - 260.)/12.))))

def build_lines(flux, min_strength=0.):
    f = flux.to_dict() if hasattr(flux, "to_dict") else dict(flux)
    out = []
    for name, sp, nm, kind, w in LINES:
        s = float(f.get(sp, 0.) or 0.)*w
        if s > min_strength: out.append(dict(name=name, species=sp, rest_nm=nm, kind=kind, weight=w, strength=s))
    return out

def attach(flux):
    """Refresh flux.lines from its current scalars (call after tech is set)."""
    flux.lines = build_lines(flux); return flux

def observed(lines, z, dim=1.):
    return [dict(l, obs_nm=float(l["rest_nm"]*(1 + z)), strength=float(l["strength"]*dim)) for l in lines]
