"""Rerunnable static sandboxed-iframe feasibility probe of the final build."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/itch/python-tools')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    page=b.new_page();errors=[];external=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:external.append(r.url) if not r.url.startswith(('http://127.0.0.1:8874/','data:','blob:')) else None)
    page.goto('http://127.0.0.1:8874/')
    page.set_content('<iframe sandbox="allow-scripts allow-same-origin" src="/live_universe/" width="1280" height="720"></iframe>')
    f=page.frames[1]
    end=time.monotonic()+90
    while time.monotonic()<end:
        if f.evaluate('Boolean(window.NIM?.workerPerf)'):break
        page.wait_for_timeout(100)
    boot=f.evaluate('()=>NIM.rpc("boot")')
    spec=json.loads(Path('.build-live/seed4.json').read_text());spec['id']='gate';spec['circuit']=False
    f.evaluate('(s)=>NIM.rpc("create",{spec:s})',spec)
    tick=f.evaluate('()=>NIM.rpc("parity",{key:"gate",n:5})')
    forbidden=json.loads(f.evaluate('(code)=>NIM.rpc("eval",{code})', "__import__('json').dumps([n for n in ('numba','pennylane','torch','jax','jaxlib') if n in sys.modules])"))
    assert forbidden==[]
    result=dict(boot=boot,tick=tick,errors=errors,external=external,forbiddenRuntimeImports=forbidden)
    Path('/workspace/itch/live_gate_final.json').write_text(json.dumps(result,indent=2))
    assert len(tick['climate']['lat_T'])==24 and tick['n_alive']>0
    assert not errors and not external
    print(json.dumps(dict(boot=boot,fiveTicksMs=tick['tickMs'],n=tick['n_alive'],errors=errors,external=external)))
    b.close()
