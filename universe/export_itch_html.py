"""Record the local life sim and export a self-contained itch.io browser replay.

PYTHONPATH=.:/workspace/NIM python -m universe.export_itch_html
No simulation, circuit evaluation, fetch or Python server is needed at playback.
"""
import argparse
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def export_demo(info, ticks, output, interval=.2):
    """Embed an existing recording and viewer sources; return (folder, zip_path)."""
    if len(ticks) < 2:
        raise ValueError("Browser replay requires at least two ticks")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(info=info, ticks=ticks, interval=interval), separators=(",", ":"), allow_nan=False)
    html = (HERE / "viewer_src/life_live_template.html").read_text()
    html = html.replace("<title>NIM life LIVE</title>", '<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>NIMverse · Browser demo</title>')
    html = html.replace("<script>/*THREE*/", "<script>window.NIM_DEMO=" + data.replace("</", "<\\/") + ";</script><script>/*THREE*/", 1)
    app = "\n".join((HERE / p).read_text() for p in ["viewer_src/life_replay.js", "viewer_src/creature_mesh.js", "viewer_src/life_live.js"])
    for marker, content in [("/*THREE*/", (HERE / "vendor/three.min.js").read_text()),
                            ("/*ORBIT*/", (HERE / "vendor/OrbitControls.js").read_text()),
                            ("/*SHADERS*/", (HERE / "viewer_src/shaders.js").read_text()), ("/*APP*/", app)]:
        html = html.replace(marker, content.replace("</script", "<\\/script"), 1)
    html = html.replace("</body>", '<div id="credit">NIMverse · NIM + QWRK · github.com/TimeMelt/NIMverse</div></body>')
    (output / "index.html").write_text(html)
    manifest = dict(project="NIMverse", seed=info.get("seed"), frames=len(ticks),
                    start_year=ticks[0]["t_w"], end_year=ticks[-1]["t_w"],
                    playback_seconds=len(ticks)*interval, runtime="recorded replay",
                    mind=info.get("mind"), recording=info.get("recording"), embed_size=[1280, 720])
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Preserve license notices for the app and bundled third-party runtime.
    (output / "LICENSE.txt").write_text((HERE.parent / "LICENSE").read_text() +
        "\nBundled Three.js and OrbitControls:\n" +
        (HERE.parent / "LICENSE").read_text().replace("Copyright (c) 2026 TimeMelt", "Copyright 2010-2021 Three.js Authors"))
    archive = output.parent / "nimverse_html_demo.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in ["index.html", "demo_manifest.json", "LICENSE.txt"]:
            z.write(output / name, name)
    return output, archive


def record_demo(seed=4, frames=240, warmup=2000., dt=20., max_show=32, mind="circuit"):
    """Run the actual LiveSim locally, skipping warm-up and saving bounded ticks."""
    from .run_life_live import LiveSim
    from .leaves.mind import CircuitMind
    policy = (lambda s: CircuitMind(seed=s, every=100., batch=8)) if mind == "circuit" else None
    sim = LiveSim(seed=seed, dt=dt, plant=True, max_show=max_show, history=1,
                  ckpt_every=0, trail_max=8, mind=policy)
    for _ in range(round(warmup/dt)):
        sim._advance(1)
    ticks = []
    for i in range(frames):
        sim._advance(1)
        tick = sim._emit()
        tick["seq"] = i+1
        ticks.append(tick)
        if (i+1) % 40 == 0:
            print(f"Recorded {i+1}/{frames}: year {tick['t_w']:.0f}, {tick['n_alive']} living", flush=True)
    info = dict(sim.info, title=f"NIMverse · seed {seed} · recorded simulation",
                recording=dict(dt=dt, warmup=warmup, frames=frames, max_show=max_show, plant=True))
    if policy:
        info["mind"] = sim.run.mind.summary()
        if info["mind"]["failures"] or not info["mind"]["queries"]:
            raise RuntimeError(f"Circuit recording did not complete cleanly: {info['mind']}")
    if not any(tick.get("creatures") for tick in ticks):
        raise RuntimeError("Recording has no creatures; increase warm-up or adjust seed")
    return info, ticks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--warmup", type=float, default=2000.)
    ap.add_argument("--dt", type=float, default=20.)
    ap.add_argument("--max-show", type=int, default=32)
    ap.add_argument("--mind", choices=["circuit", "none"], default="circuit")
    ap.add_argument("--interval", type=float, default=.2)
    ap.add_argument("--out", type=Path, default=Path("/workspace/itch/html_demo"))
    a = ap.parse_args(argv)
    if a.frames < 2 or a.dt <= 0 or a.interval <= 0 or a.warmup < 0 or a.max_show < 1:
        ap.error("frames >= 2, dt/interval/max-show > 0, warmup >= 0 required")
    info, ticks = record_demo(a.seed, a.frames, a.warmup, a.dt, a.max_show, a.mind)
    root, archive = export_demo(info, ticks, a.out, a.interval)
    print(f"Exported {root}/index.html; zip {archive} ({archive.stat().st_size:,} bytes)")
    print(json.dumps(info.get("mind", {}), indent=2))


if __name__ == "__main__":
    main()
