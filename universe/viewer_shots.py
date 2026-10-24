"""Headless verification + stills + fly-through for viewer HTML. usage: python viewer_shots.py html outdir [--tour N]"""
import sys, asyncio, os, subprocess
from playwright.async_api import async_playwright
html, out = sys.argv[1], sys.argv[2]; NT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720}); errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type in ("error", "warning") else None); pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://"+html); await pg.wait_for_function("window.NIM && NIM.ready", timeout=120000); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{out}/00_initial.png")
        await pg.evaluate("NIM.still()")
        if NT == 0:
            shots = [("01_universe", "NIM.view('galaxy',-1,GX*2.4,[.4,.6,1])"), ("02_galaxy0", "NIM.view('galaxy',0,G0*3,[.2,.9,.6])"), ("03_galaxy0_edge", "NIM.view('galaxy',0,G0*2.2,[1,.12,.3])"),
                     ("04_system", "NIM.view('system',-1,AM*2.6,[.2,.75,1])"), ("05_starA", "NIM.view('system',0,RV[0]*4,[.3,.3,1])"),
                     ("06_gasgiant_planet1", "NIM.view('system',6,RV[6]*2.6,[.4,.3,1])"), ("07_ocean_planet4", "NIM.view('system',15,RV[15]*2.4,[.5,.3,1])"),
                     ("08_desert_planet2", "NIM.view('system',10,RV[10]*2.4,[.5,.3,1])"), ("09_lava_planet0", "NIM.view('system',2,RV[2]*2.4,[.5,.3,1])"), ("10_moon", "NIM.view('system',16,RV[16]*2.6,[.5,.3,1])"),
                     ("11_terran_closeup", "NIM.view('gallery',0,1.25,[.55,.35,1])"), ("12_terran_surface", "NIM.view('gallery',0,.56,[.75,.25,.6])"), ("13_terran2", "NIM.view('gallery',1,1.3,[.5,.3,1])"),
                     ("14_gasgiant_closeup", "NIM.view('gallery',2,1.25,[.5,.25,1])"), ("15_icegiant", "NIM.view('gallery',3,1.6,[.5,.3,1])"), ("16_lava", "NIM.view('gallery',4,1.2,[.5,.3,1])"), ("17_ice", "NIM.view('gallery',6,1.2,[.5,.3,1])")]
            pre = "var GX=NIMDATA.gal.ext,G0=NIMDATA.gal.R[0],AM=NIMDATA.sys.amax,RV=NIMDATA.sys.rvis;"
            for n, js in shots:
                await pg.evaluate(pre+"NIM.setFrame(80);"+js); await pg.evaluate("NIM.render()"); await pg.screenshot(path=f"{out}/{n}.png")
        else:
            os.makedirs(f"{out}/tour", exist_ok=True)
            for i in range(int(os.environ.get("START",0)), NT):
                await pg.evaluate(f"NIM.tour({i/(NT-1)})"); await pg.screenshot(path=f"{out}/tour/f{i:04d}.png", timeout=180000)
                if i % 30 == 0: print("frame", i, flush=True)
        print("console errors/warnings:", len(errs), [e[:300] for e in errs[:5]]); await b.close()
asyncio.run(main())
