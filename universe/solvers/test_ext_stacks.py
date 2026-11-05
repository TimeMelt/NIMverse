"""External science-stack adapters (universe.solvers.ext). Opt-in; skips when MESA/ExoPlaSim absent."""
import os, numpy as np, pytest

from universe.solvers.ext import (
    UnavailableError, mist_tracks, mesa_runner, rebound_orbit, lal_wave,
    exoplasim_climate, rt_tables, et_loader,
)
from universe.solvers import gr, stellar_track as st


def test_rebound_earth_sun_period():
    P = rebound_orbit.two_body_period_yr(M_star=1., a=1., m=3e-6, integrator="whfast", n_orbits=3)
    assert abs(P - 1.) < 2e-3


def test_lal_gw150914_like_peak_freq():
    assert lal_wave.lalsim_available()
    wf = lal_wave.gw150914_like_fd()
    # IMRPhenomDGetPeakFreq for 36+29 ~ 242 Hz; accept a broad physical band
    assert 100. < wf["f_peak_hz"] < 400.
    assert wf["peak_amp"] > 0 and np.isfinite(wf["peak_amp"])
    # enrichment keeps PhenomA primary bands and adds LAL fields
    tags = gr.imr_tags(36., 29., 0., 0., 3.5, f_lo_hz=10.)
    assert 80. < tags["gr_imr_f_merge_hz"] < 200. < tags["gr_imr_f_ring_hz"] < 400.
    assert tags.get("gr_imr_lal") is True
    assert 100. < tags["gr_imr_lal_f_merge_hz"] < 400.


def test_mist_proxy_sun_track():
    mist_tracks.ensure_proxy()
    d = mist_tracks.solve_track(1., FeH=0.)
    assert d["mist_source"] in ("proxy", "mist")
    assert 8. < d["t_ms"] < 12.
    # L(4.57)/L_ZAMS ≈ 1/0.72
    ms = np.array([str(p) == "ms" for p in d["phase"]])
    assert ms.any()
    t_zams = float(d["t"][ms][0]); Lz = float(np.interp(t_zams, d["t"], d["L"]))
    L457 = float(np.interp(4.57, d["t"], d["L"]))
    assert abs((L457 / Lz) / (1 / 0.72) - 1) < .08
    # MS luminosity roughly monotonic rising
    Lms = d["L"][ms]
    assert Lms[-1] > Lms[0]
    # HostWeather API surface
    api = mist_tracks.api()
    L, T, R = api.track(1., 1., 5772., 4.57, 4.57)
    assert abs(L - 1.) < 1e-6 and R > 0


def test_rt_table_roundtrip():
    path = rt_tables.ensure_example()
    tab = rt_tables.load_rt_table(path)
    assert tab["olr"].ndim == 4 and tab["sw"].ndim == 5
    prev = rt_tables.install_rt_table(path)
    from universe.solvers import rt
    got = rt.climate_tables()
    assert np.allclose(got["olr"], tab["olr"]) and "rt_table" in got
    rt_tables.uninstall_rt_table(prev)


def test_climate_table_drives_finite_T():
    path = exoplasim_climate.build_example_table()
    tab = exoplasim_climate.load_climate_table(path)
    clim = exoplasim_climate.TableClimate(tab)
    class H: e = 0.; tilt = .4; a = 1.; m = 3e-6; name = "p"; ptype = "rocky"; volatiles = .5; planet_id = 2
    class HW: L_star = 1.; T_eff = 5772.
    c = clim.step(H(), HW(), None, 10., {}, 0.)
    assert np.isfinite(c.T_K) and 200. < c.T_K < 330.
    assert c.stressors["climate_model"] == "climate_table"


def test_mesa_runner_skips_without_mesa_dir():
    assert not mesa_runner.available()
    with pytest.raises(UnavailableError):
        mesa_runner.solve_track(1.)


def test_exoplasim_skips_without_binary():
    assert not exoplasim_climate.exoplasim_binary_ok()
    with pytest.raises(UnavailableError):
        exoplasim_climate.run_exoplasim()


def test_et_loader_example():
    path = et_loader.ensure_example()
    wf = et_loader.load_nr_waveform(path)
    assert len(wf["hp"]) > 10 and np.isfinite(wf["hp"]).all()
    tags = et_loader.tags_from_waveform(wf)
    assert tags["gr_imr_nr"] and tags["gr_imr_peak_amp"] > 0


def test_stellar_api_routes_mist():
    api = st.api("mist", 0.0142)
    assert api.t_ms_Gyr(1.) > 8.
    with pytest.raises(UnavailableError):
        st.api("mesa", 0.0142)


def test_flags_accept_ext():
    import argparse
    from universe.solvers import add_args, from_args
    ap = argparse.ArgumentParser(); add_args(ap)
    a = ap.parse_args(["--stellar-solver", "mist", "--orbit-solver", "rebound",
                       "--climate-model", "climate_table", "--climate-table",
                       exoplasim_climate.EXAMPLE_TABLE])
    cm, sv = from_args(a)
    assert cm == "climate_table" and sv["stellar_solver"] == "mist" and sv["orbit_solver"] == "rebound"
