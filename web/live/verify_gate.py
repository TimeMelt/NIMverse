"""Static sandboxed-iframe gate; no special HTTP headers or backend."""
import json,sys,time
sys.path.insert(0,'/workspace/itch/python-tools')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    page=b.new_page(); errors=[];external=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:external.append(r.url) if not r.url.startswith(('http://127.0.0.1:8874','data:','blob:')) else None)
    page.goto('http://127.0.0.1:8874/')
    page.set_content('<iframe sandbox="allow-scripts allow-same-origin" src="/live_universe/" width="1280" height="720"></iframe>')
    frame=page.frames[1]
    frame.wait_for_function('window.gate || window.gateError',timeout=180000)
    result=frame.evaluate('({gate:window.gate,error:window.gateError})')
    result.update(errors=errors,external=external)
    print(json.dumps(result,indent=2));open('/workspace/itch/live_gate.json','w').write(json.dumps(result,indent=2))
    assert result.get('gate') and not errors and not external
    b.close()
