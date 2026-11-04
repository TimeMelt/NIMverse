"""Stills + optional tour for the life viewer.
usage: python -m universe.viewer_life_shots [--seed 4] [--tour 72] [--look stylized|photo|both] [--tour-look stylized|photo|both]
stylized stills keep the Phase A names (life_surface_seed4_NN_*.png); photo stills are life_surface_seed4_photo_NN_*.png;
tour: life_creatures_seed4.mp4 (stylized) / life_creatures_seed4_photo.mp4 (photo)."""
import argparse, asyncio, os, subprocess, shutil, time
from playwright.async_api import async_playwright
OUT = "/workspace/nim_universe/out"; RES = os.path.join(os.path.dirname(__file__), "..", "results")
async def main(a):
    html = f"{OUT}/viewer_life_seed{a.seed}.html"; base = f"{OUT}/life_surface_seed{a.seed}"; done = []
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720}); errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://"+html); await pg.wait_for_function("window.NIM && NIM.ready", timeout=120000); await pg.evaluate("NIM.still()")
        shots = [("01_orbit", "NIM.setTime(3);NIM.view('orbit',0,{dist:2.6})"), ("02_surface", "NIM.setTime(3);NIM.view('surface',0,{dist:.2})"),
                 ("03_surface_herd2", "NIM.setTime(5);NIM.view('surface',a.seedB,{dist:.16,az:1})".replace("a.seedB", str(a.herd2))),
                 ("04_creature", f"NIM.setTime(4);NIM.view('creature',{a.c1},{{dist:.028}})"), ("05_creature2", f"NIM.setTime(4);NIM.view('creature',{a.c2},{{dist:.03,az:.8}})"),
                 ("06_tech_lights", "NIM.setTime(2);NIM.view('surface',0,{dist:.14,az:2.2})")]
        plans = await pg.evaluate("(()=>{const o={}; NIM.creatures.forEach((c,i)=>{if(!(c.plan in o))o[c.plan]=i;}); return o;})()")
        shots += [(f"07_{pl}_closeup", f"NIM.setTime(4.3);NIM.view('creature',{i},{{dist:.022,az:.6}})") for pl, i in sorted(plans.items())]   # one hero closeup per body plan
        shots += [(f"08_{pl}_macro", f"NIM.setTime(4.3);NIM.view('creature',{i},{{dist:.011,az:1.1}})") for pl, i in sorted(plans.items())]   # tight macro: skin/material detail
        looks = ["stylized", "photo"] if a.look == "both" else [a.look]
        for lk in looks:
            await pg.evaluate(f"NIM.setLook('{lk}')"); pre = base + ("_photo" if lk == "photo" else "")
            print(lk, "triangles (all creatures):", await pg.evaluate("NIM.tris?NIM.tris():-1"), flush=True)
            for n, js in shots:
                t0 = time.time(); await pg.evaluate(js); await pg.evaluate("NIM.render()"); await pg.screenshot(path=f"{pre}_{n}.png"); done.append(f"{pre}_{n}.png"); print("shot", lk, n, f"{time.time()-t0:.1f}s", flush=True)
        for lk in (["stylized", "photo"] if a.tour_look == "both" else [a.tour_look]) if a.tour else []:
            await pg.evaluate(f"NIM.setLook('{lk}')"); sfx = "_photo" if lk == "photo" else ""
            td = f"{OUT}/life_tour_seed{a.seed}{sfx}"; os.makedirs(td, exist_ok=True); N = a.tour
            for i in range(N):
                s = i/(N-1); d = 2.8*(1-s)**2 + .12  # orbit in, then ride along creature 0's herd
                await pg.evaluate(f"NIM.setTime({3+i*.12});NIM.view('surface',0,{{dist:{d},az:{s*2.5}}})"); await pg.screenshot(path=f"{td}/f{i:04d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"{OUT}/life_creatures_seed{a.seed}{sfx}.mp4"], check=True); done.append(f"{OUT}/life_creatures_seed{a.seed}{sfx}.mp4")
        print("errors:", len(errs), errs[:5]); await b.close()
    for f in done: shutil.copy(f, RES); print("wrote", f)       # all stills + mp4s -> results/
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4); ap.add_argument("--tour", type=int, default=0)
    ap.add_argument("--look", choices=["stylized", "photo", "both"], default="both"); ap.add_argument("--tour-look", dest="tour_look", choices=["stylized", "photo", "both"], default="stylized")
    ap.add_argument("--herd2", type=int, default=30); ap.add_argument("--c1", type=int, default=0); ap.add_argument("--c2", type=int, default=40)
    asyncio.run(main(ap.parse_args()))
