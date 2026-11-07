"""Headless smoke test and screenshots for the offline HTML export.
Requires playwright with Chromium installed. Starts/stops its own static server.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import urlsplit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", type=Path, default=Path("/workspace/itch/html_demo"))
    ap.add_argument("--shots", type=Path, default=Path("/workspace/itch/demo_shots"))
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    a.shots.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (a.shots / "http_server.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(a.demo)], stdout=log, stderr=log)
    report = dict(console_errors=[], page_errors=[], warnings=[], external_requests=[], requests=[], checks=[])
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--enable-unsafe-swiftshader", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
            page.on("console", lambda m: report["console_errors" if m.type == "error" else "warnings"].append(m.text) if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: report["page_errors"].append(str(e)))
            def request(req):
                report["requests"].append(req.url)
                if urlsplit(req.url).scheme in ("http", "https") and urlsplit(req.url).hostname != "127.0.0.1":
                    report["external_requests"].append(req.url)
            page.on("request", request)
            page.route("**/*", lambda route: route.continue_() if urlsplit(route.request.url).hostname == "127.0.0.1" else route.abort())
            # Opaque-origin sandbox deliberately disallows storage, as a strict itch-like iframe.
            page.set_content(f'<style>html,body{{margin:0;height:100%;overflow:hidden}}iframe{{border:0;width:100%;height:100%}}</style><iframe sandbox="allow-scripts" src="http://127.0.0.1:{port}/index.html"></iframe>')
            frame = page.frames[1]
            frame.wait_for_function("window.NIM && NIM.ready && NIM.n()>0", timeout=90000)
            frame.evaluate("NIM.control({action:'pause'})")
            page.wait_for_timeout(1200)
            report["checks"].append("opaque-origin sandboxed iframe initialized with creatures")
            before = frame.evaluate("NIM.raw().seq")
            page.wait_for_timeout(500)
            assert frame.evaluate("NIM.raw().seq") == before
            report["checks"].append("pause holds recorded tick")
            for view, filename in [("orbit", "01_orbit.png"), ("surface", "02_surface.png"), ("creature", "03_creature.png")]:
                frame.locator(f'[data-view="{view}"]').click()
                page.wait_for_timeout(900)
                page.screenshot(path=str(a.shots / filename))
            frame.locator("#lookbtn").click()
            page.wait_for_timeout(1200)
            assert frame.evaluate("NIM.look()") == "photo"
            page.screenshot(path=str(a.shots / "04_photoreal.png"))
            report["checks"].append("orbit / surface / creature buttons and photoreal toggle")
            frame.locator("#scrub").evaluate("e=>{e.value=120;e.dispatchEvent(new Event('input'));e.dispatchEvent(new Event('change'));}")
            assert frame.evaluate("NIM.raw().seq") == 120
            report["checks"].append("timeline scrub seeks and pauses")
            frame.evaluate("NIM.control({action:'seek',seq:NIM_DEMO.ticks.length}); NIM.control({action:'resume'});")
            frame.wait_for_function("NIM.raw().seq < 10", timeout=15000)
            frame.evaluate("NIM.control({action:'pause'})")
            report["checks"].append("recorded timeline loops and resumes")
            page.set_viewport_size({"width": 640, "height": 480})
            page.wait_for_timeout(800)
            assert frame.evaluate("document.querySelector('canvas').clientWidth") == 640
            assert frame.evaluate("document.documentElement.scrollWidth<=innerWidth")
            page.screenshot(path=str(a.shots / "05_responsive.png"))
            report["checks"].append("responsive 640x480 canvas without horizontal overflow")
            # Real key events dispatched at the canvas avoid the focused range input.
            frame.locator("canvas").click(position={"x": 320, "y": 240})
            page.keyboard.press("1")
            page.keyboard.press("Space")
            frame.wait_for_function("!NIM.raw().paused", timeout=5000)
            page.keyboard.press("Space")
            assert frame.evaluate("NIM.raw().paused")
            report["checks"].append("keyboard views and pause/resume")
            report["summary"] = frame.evaluate("({frames:NIM_DEMO.ticks.length, displayed:NIM.n(),look:NIM.look(),focus:NIM.focus()})")
            browser.close()
        assert not report["console_errors"], report["console_errors"]
        assert not report["page_errors"], report["page_errors"]
        assert not report["external_requests"], report["external_requests"]
        print(json.dumps(report, indent=2))
    finally:
        server.terminate()
        server.wait(timeout=10)
        log.close()
        report["server_stopped"] = server.poll() is not None
        (a.shots / "verification.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
