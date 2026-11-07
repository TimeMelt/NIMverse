"""Run against a STATIC server. Exercises nested itch-like iframe, never a backend.
PYTHONPATH=/workspace/itch/python-tools /workspace/nimenv/bin/python web/live/verify_browser.py
"""
import json,time,sys,statistics
from pathlib import Path
sys.path.insert(0,'/workspace/itch/python-tools')
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/itch/live_shots');ROOT.mkdir(exist_ok=True)
def wait(frame,expr,timeout=90):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if frame.evaluate(expr):return
        frame.wait_for_timeout(100)
    raise AssertionError('Timed out: '+expr)

def compare(native,browser):
    # Climate uses different BLAS backends; compare physically meaningful fields.
    out={}
    for k,a,b,tol in [('T_K',native['climate']['T_K'],browser['climate']['T_K'],1e-8),('water',native['climate']['water'],browser['climate']['water'],1e-8),('biomass',native['biomass'],browser['biomass'],1e-8)]:
        out[k]=abs(a-b);assert out[k]<tol,(k,a,b)
    assert native['n_alive']==browser['n_alive'],(native['n_alive'],browser['n_alive'])
    assert native['births']==browser['births']
    assert native['n_species']==browser['n_species']
    return out

with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    ctx=b.new_context(viewport={'width':1440,'height':900},device_scale_factor=1)
    errors=[];external=[];bad=[];consoleErrors=[]
    ctx.on('request',lambda r:external.append(r.url) if not r.url.startswith(('http://127.0.0.1:8874/','blob:','data:')) else None)
    ctx.on('response',lambda r:bad.append((r.status,r.url)) if r.status>=400 else None)
    page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:consoleErrors.append(m.text) if m.type=='error' else None)
    page.goto('http://127.0.0.1:8874/')
    page.set_content('<iframe id="game" sandbox="allow-scripts allow-same-origin" allow="fullscreen" allowfullscreen src="/live_universe/" style="border:0;width:100vw;height:100vh;position:fixed;inset:0"></iframe>')
    game=page.frames[1];wait(game,'Boolean(window.NIM?.workerPerf)')
    result={'boot':game.evaluate('NIM.workerPerf'),'screenshots':[],'parity':[]}
    def shot(name):
        print('Capturing',name,flush=True);path=ROOT/(name+'.png');page.screenshot(path=str(path));result['screenshots'].append(str(path))
    shot('01_galaxy')
    wait(game,'NIM.universe.events.some(e=>e.type==="supernova")')
    game.evaluate('NIM.focusEvent(NIM.universe.events.find(e=>e.type==="supernova"))')
    page.wait_for_timeout(300);shot('02_supernova')
    game.locator('#galaxy').click();game.locator('#pause').click()
    # Exercise the actual point raycaster, rather than only the Explore shortcut.
    pos=game.evaluate('''(()=>{for(const b of NIM.universe.bodies){if(b.kind!==1||b.gid!==1)continue;const p=NIM.projectStar(b);if(p.x>500&&p.x<1000&&p.y>200&&p.y<650)return p;}return null;})()''')
    assert pos
    page.mouse.click(pos['x'],pos['y']);wait(game,'NIM.mode==="system"')
    result['clicked_system']=game.evaluate('NIM.system.id');shot('03_clicked_star_system')
    game.locator('[data-world="2"]').click();shot('04_generated_planet')
    # Legacy habitat establishes a known living reference; no recording is used.
    game.locator('#galaxy').click();game.locator('#pause').click();game.locator('#legacy').click()
    wait(game,'Boolean(NIM.lastTick?.step>=10)')
    surface=game.child_frames[0];wait(surface,'Boolean(window.NIM?.ready)')
    wait(surface,'NIM.n()>0')
    result['live_initial']=game.evaluate('({step:NIM.lastTick.step,n:NIM.lastTick.n_alive,species:NIM.lastTick.n_species,mind:NIM.lastTick.mind,memoryBytes:NIM.lastTick.memoryBytes,tickMs:NIM.lastTick.tickMs})')
    surface.locator('[data-view="surface"]').click();page.wait_for_timeout(600);shot('05_live_surface')
    game.locator('#photo').click();wait(surface,'NIM.look()==="photo"');shot('06_photoreal')
    surface.locator('[data-view="creature"]').click();page.wait_for_timeout(400);shot('07_live_creature')
    # Backpressure and sustained tick timing, across real evolving populations.
    page.wait_for_timeout(2500)
    samples=[]
    for _ in range(10):
        samples.append(game.evaluate('NIM.lastTick.tickMs'));page.wait_for_timeout(100)
    result['performance']={'detailedTickMedianMs':statistics.median(samples),'detailedTickRangeMs':[min(samples),max(samples)],'surface':surface.evaluate('NIM.stats()'),'main':game.evaluate('NIM.stats()')}
    result['performance']['mainJsHeapBytes']=game.evaluate('performance.memory?.usedJSHeapSize||null')
    game.locator('#pause').click();page.wait_for_timeout(500);st=game.evaluate('NIM.lastTick.step');page.wait_for_timeout(400);assert game.evaluate('NIM.lastTick.step')==st
    game.locator('#pause').click()
    game.locator('#quantum').uncheck();wait(game,'NIM.lastTick.quantum_drive.every(z=>z===0)');result['quantum_toggle']=True
    game.locator('#quantum').check();wait(game,'NIM.lastTick.quantum_drive.some(z=>Math.abs(z)>1e-8)')
    game.locator('#galaxy').click();game.locator('#explore').click();game.locator('[data-world="2"]').click();game.locator('#land').click()
    wait(game,'Boolean(NIM.lastTick?.planet?.startsWith("World"))');surface=game.child_frames[0];wait(surface,'Boolean(window.NIM?.ready)');shot('08_second_generated_world')
    result['second_world']=game.evaluate('({id:NIM.planet.id,type:NIM.lastTick.ptype,T:NIM.lastTick.climate.T_K,water:NIM.lastTick.climate.water,n:NIM.lastTick.n_alive,mind:NIM.lastTick.mind})')
    # Freeze UI simulation while exact worker-vs-native reference is measured.
    game.locator('#pause').click();page.wait_for_timeout(500)
    ref=json.loads(Path('/workspace/itch/native_live_reference.json').read_text())
    game.evaluate('(s)=>NIM.rpc("create",{spec:s})',ref['spec'])
    for n in ref['ticks']:
        tk=game.evaluate('()=>NIM.rpc("parity",{key:"parity",n:1})')
        result['parity'].append(compare(n,tk))
    # Fullscreen requires a real click user gesture inside the sandbox.
    game.locator('#full').click();wait(game,'Boolean(document.fullscreenElement)',timeout=10)
    result['fullscreen']=True;game.evaluate('document.exitFullscreen()')
    game.locator('#seed').fill('22');game.locator('#generate').click();wait(game,'NIM.universe.seed===22&&NIM.mode==="galaxy"');page.wait_for_timeout(300)
    game.locator('#explore').click();game.locator('#reset').click();wait(game,'NIM.mode==="galaxy"');page.wait_for_timeout(300)
    result['seed_reset']=True
    result.update(pageErrors=errors,consoleErrors=consoleErrors,externalRequests=external,httpErrors=bad)
    Path('/workspace/itch/live_browser_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    assert not errors and not consoleErrors and not external and not bad
    b.close()
