"""Generate native reference for browser-worker parity (same leaf entry point)."""
import importlib.util,json
from pathlib import Path
sp=importlib.util.spec_from_file_location('bridge',Path(__file__).with_name('bridge.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=json.loads(Path('.build-live/seed4.json').read_text());s['id']='parity';s['circuit']=False
m.create(s);ticks=[]
for _ in range(5): ticks.append(json.loads(m.advance('parity')))
Path('/workspace/itch/native_live_reference.json').write_text(json.dumps(dict(spec=s,ticks=ticks),indent=2))
print('Native parity reference:',ticks[-1]['climate']['T_K'],ticks[-1]['n_alive'])
