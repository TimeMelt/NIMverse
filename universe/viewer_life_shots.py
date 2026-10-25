"""Stills + optional tour for the life viewer. usage: python -m universe.viewer_life_shots [--seed 4] [--tour 72]"""
import argparse, asyncio, os, subprocess, shutil
from playwright.async_api import async_playwright
OUT = "/workspace/nim_universe/out"; RES = os.path.join(os.path.dirname(__file__), "..", "results")
async def main(a):
    html = f"{OUT}/viewer_life_seed{a.seed}.html"; pre = f"{OUT}/life_surface_seed{a.seed}"
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720}); errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://"+html); await pg.wait_for_function("window.NIM && NIM.ready", timeout=120000); await pg.evaluate("NIM.still()")
        shots = [("01_orbit", "NIM.setTime(3);NIM.view('orbit',0,{dist:2.6})"), ("02_surface", "NIM.setTime(3);NIM.view('surface',0,{dist:.2})"),
                 ("03_surface_herd2", "NIM.setTime(5);NIM.view('surface',a.seedB,{dist:.16,az:1})".replace("a.seedB", str(a.herd2))),
                 ("04_creature", f"NIM.setTime(4);NIM.view('creature',{a.c1},{{dist:.028}})"), ("05_creature2", f"NIM.setTime(4);NIM.view('creature',{a.c2},{{dist:.03,az:.8}})"),
                 ("06_tech_lights", "NIM.setTime(2);NIM.view('surface',0,{dist:.14,az:2.2})")]
        for n, js in shots:
            await pg.evaluate(js); await pg.evaluate("NIM.render()"); await pg.screenshot(path=f"{pre}_{n}.png"); print("shot", n, flush=True)
        if a.tour:
            td = f"{OUT}/life_tour_seed{a.seed}"; os.makedirs(td, exist_ok=True); N = a.tour
            for i in range(N):
                s = i/(N-1); d = 2.8*(1-s)**2 + .12  # orbit in, then ride along creature 0's herd
                await pg.evaluate(f"NIM.setTime({3+i*.12});NIM.view('surface',0,{{dist:{d},az:{s*2.5}}})"); await pg.screenshot(path=f"{td}/f{i:04d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"{OUT}/life_creatures_seed{a.seed}.mp4"], check=True)
        print("errors:", len(errs), errs[:5]); await b.close()
    for n, _ in shots: shutil.copy(f"{pre}_{n}.png", RES)       # all stills (01-06) -> results/
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4); ap.add_argument("--tour", type=int, default=0)
    ap.add_argument("--herd2", type=int, default=30); ap.add_argument("--c1", type=int, default=0); ap.add_argument("--c2", type=int, default=40)
    asyncio.run(main(ap.parse_args()))
