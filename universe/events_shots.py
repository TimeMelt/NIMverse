"""Headless check + stills + highlight reel for the events viewer.  python -m universe.events_shots html outdir [--reel N_PER_EVENT]"""
import sys, asyncio, os, json
from playwright.async_api import async_playwright
html, out = sys.argv[1], sys.argv[2]; NR = int(sys.argv[4]) if len(sys.argv) > 4 else 0; os.makedirs(out, exist_ok=True)
PICK = """(()=>{const L=NIM.events, B=NIMDATA.ev.bh, f=t=>L.findIndex(t);
 const sn=f(e=>e.type==='SN'&&e.rem==='black hole')>=0?f(e=>e.type==='SN'&&e.rem==='black hole'):f(e=>e.type==='SN');
 const sn2=f((e,k)=>e.type==='SN'&&k!==sn&&e.trig>0), psr=f(e=>e.type==='SN'&&e.rem==='pulsar'&&e.j>=0), agn=f(e=>e.type==='AGN'), mrg=f(e=>e.type==='MERGER'), tde=f(e=>e.type==='TDE');
 return {sn, sn2, psr, agn, mrg, tde};})()"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720}); errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type in ("error", "warning") else None); pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://"+html); await pg.wait_for_function("window.NIM && NIM.ready", timeout=120000); await pg.wait_for_timeout(2000)
        await pg.screenshot(path=f"{out}/00_initial.png"); k = await pg.evaluate(PICK); print("picked", k, flush=True)
        # UI test: click a timeline entry (fly-to), let it animate
        await pg.click("#evlog .e >> nth=0"); await pg.wait_for_timeout(2500); await pg.screenshot(path=f"{out}/01_timeline_click.png")
        await pg.evaluate("NIM.still()"); await pg.add_style_tag(content="body.clean #panel,body.clean #evlog,body.clean #info,body.clean #bar,body.clean #help,body.clean #evstrip,body.clean #mode{display:none}")
        shots = []
        if k["sn"] >= 0: shots += [("02_supernova_flash", k["sn"], 2.5, [.35, .55, 1], 2), ("03_supernova_shell", k["sn"], 2.8, [.35, .55, 1], 45), ("04_supernova_shell_late", k["sn"], 3.5, [.6, .3, 1], 110)]
        if k["sn2"] >= 0: shots += [("05_supernova2_shell", k["sn2"], 2.6, [.2, .7, 1], 30)]
        if k["agn"] >= 0: shots += [("06_quasar_jets", k["agn"], 2.2, [1, .25, .4], 10), ("07_blackhole_closeup", k["agn"], .9, [1, .18, .35], 10)]
        if k["psr"] >= 0: shots += [("08_pulsar", k["psr"], .6, [.4, .4, 1], 30, 1)]
        if k["mrg"] >= 0: shots += [("09_smbh_merger", k["mrg"], 6, [.3, .8, 1], 4)]
        if k["tde"] >= 0: shots += [("10_tde", k["tde"], 1.2, [.3, .5, 1], 2, 0)]
        await pg.evaluate("document.body.classList.add('clean')")
        for sh in shots:
            n, e, d, dr, df = sh[:5]; rm = sh[5] if len(sh) > 5 else 0
            await pg.evaluate(f"NIM.setTime({df*.1});NIM.viewEvent({e},{d},{json.dumps(dr)},{df},{rm})"); await pg.screenshot(path=f"{out}/{n}.png", timeout=180000); print(n, flush=True)
        if NR:
            os.makedirs(f"{out}/reel", exist_ok=True); seq = [(k[x], x) for x in ("sn", "sn2", "agn", "psr", "mrg", "tde") if k[x] >= 0]; i = 0
            for e, x in seq:
                for s in range(NR):
                    u = s/(NR-1); df = -3+u*(70 if x.startswith("sn") else 30); d = {"sn": 4.5-2*u, "sn2": 3.5, "agn": 3-2*u, "psr": .8-.3*u, "mrg": 7, "tde": 1.5}[x]
                    import math; a = .6*u-.3; dr = [math.sin(a)+.3, .5, math.cos(a)]
                    await pg.evaluate(f"NIM.setTime({i/30});NIM.viewEvent({e},{d},{json.dumps(dr)},{df},{int(x=='psr')})"); await pg.screenshot(path=f"{out}/reel/f{i:04d}.png", timeout=180000); i += 1
                print("reel", x, i, flush=True)
        print("console errors/warnings:", len(errs), [e[:300] for e in errs[:8]]); await b.close()
asyncio.run(main())
