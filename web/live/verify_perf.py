"""Measure service time AND wall-clock tick throughput, with software WebGL."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'/workspace/itch/python-tools')
from playwright.sync_api import sync_playwright

def wait(f,expr):
    end=time.monotonic()+90
    while time.monotonic()<end:
        if f.evaluate(expr):return
        f.wait_for_timeout(100)
    raise AssertionError(expr)
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    pg=b.new_page(viewport={'width':1280,'height':720});pg.goto('http://127.0.0.1:8874/live_universe/')
    pg.locator('#legacy').click();wait(pg,'Boolean(window.NIM?.lastTick?.step>8)')
    f=pg.frames[1];wait(f,'Boolean(window.NIM?.ready)');report={}
    for name in ['orbit','photoreal_surface']:
        if name!='orbit':f.locator('[data-view="surface"]').click();pg.locator('#photo').click()
        pg.wait_for_timeout(3000)
        start=pg.evaluate('({step:NIM.lastTick.step,t:performance.now()})')
        pg.wait_for_timeout(6000)
        end=pg.evaluate('({step:NIM.lastTick.step,t:performance.now(),ms:NIM.lastTick.tickMs,mem:NIM.lastTick.memoryBytes,heap:performance.memory?.usedJSHeapSize})')
        report[name]=dict(ticksPerSecond=(end['step']-start['step'])/((end['t']-start['t'])/1000),workerServiceMs=end['ms'],wasmBytes=end['mem'],mainJsHeapBytes=end['heap'],render=f.evaluate('NIM.stats()'))
        print(name,report[name],flush=True)
    Path('/workspace/itch/live_performance_report.json').write_text(json.dumps(report,indent=2));b.close()
    assert all(5<=v['ticksPerSecond']<=20 for v in report.values()),report
