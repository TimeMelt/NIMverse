"""Packaging contract for the offline itch export (no full sim needed)."""
import importlib.util
import json
from pathlib import Path
import zipfile


def test_exporter_exists():
    assert importlib.util.find_spec("universe.export_itch_html") is not None


def test_export_embeds_data_and_vendored_runtime(tmp_path):
    from universe.export_itch_html import export_demo
    info = {"seed": 4, "planet": "planet2", "title": "test </script>"}
    ticks = [{"seq": 1, "t_w": 20}, {"seq": 2, "t_w": 40}]
    root, archive = export_demo(info, ticks, tmp_path / "html_demo")
    html = (root / "index.html").read_text()
    assert "window.NIM_DEMO=" in html
    assert "test <\\/script>" in html
    assert "NIMverse · NIM + QWRK · github.com/TimeMelt/NIMverse" in html
    assert "/*THREE*/" not in html and "/*APP*/" not in html
    assert "OrbitControls" in html and "NIMReplay" in html
    assert 'name="viewport"' in html
    with zipfile.ZipFile(archive) as z:
        assert "index.html" in z.namelist()
        assert len(z.namelist()) < 1000
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in z.namelist())
    assert archive.stat().st_size < 50_000_000
    manifest = json.loads((root / "demo_manifest.json").read_text())
    assert manifest["frames"] == 2 and manifest["runtime"] == "recorded replay"


def test_export_rejects_empty_timeline(tmp_path):
    from universe.export_itch_html import export_demo
    import pytest
    with pytest.raises(ValueError, match="two"):
        export_demo({}, [], tmp_path / "demo")
