"""Reproducible offline live-browser build. Downloads only at build time.
PYTHONPATH=.:/workspace/NIM /workspace/nimenv/bin/python -m universe.export_itch_live
"""
import argparse, hashlib, json, shutil, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
WEB=ROOT/'web/live'
CACHE=ROOT/'.build-live'
VERSION='314.0.7'
BASE=f'https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/'

def get(name,sha=None):
    p=CACHE/'pyodide'/name; p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(BASE+name,timeout=120) as r: p.write_bytes(r.read())
    if sha and hashlib.sha256(p.read_bytes()).hexdigest()!=sha: raise ValueError('Checksum mismatch: '+name)
    return p

def runtime(out):
    lock=json.loads(get('pyodide-lock.json').read_text())
    names=['pyodide.mjs','pyodide.asm.mjs','pyodide.asm.wasm','python_stdlib.zip','pyodide-lock.json']
    deps=set()
    def add(k):
        if k in deps:return
        deps.add(k)
        for d in lock['packages'][k]['depends']:add(d)
    for k in ['numpy','scipy','rebound']:add(k)
    for k in sorted(deps):
        p=lock['packages'][k];get(p['file_name'],p['sha256']);names.append(p['file_name'])
    dest=out/'pyodide';dest.mkdir(exist_ok=True)
    for n in names:shutil.copy2(get(n),dest/n)
    (out/'runtime_manifest.json').write_text(json.dumps(dict(version=VERSION,files={n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in names}),indent=2))

def seed4_spec():
    import numpy as np
    from .run_life_live import LiveSim
    s=LiveSim(seed=4,plant=True,history=1,ckpt_every=0)
    h=s.run.h; st=s.run.hw
    from .bodies import planet_detail
    legacy_look=planet_detail(4,h.planet_id,h.m,h.a,h.L_star,rng='numpy')
    if legacy_look['type']!=h.ptype:
        from .bodies import PAL,ATM
        legacy_look.update(type=h.ptype,colors=PAL[h.ptype],atm=ATM[h.ptype])
    return dict(id='legacy-4-planet2',seed=4,look=legacy_look,habitat=h.to_dict(),star=dict(mass=st.M_star,luminosity=st.L_star,temperature=st.T_eff,age=st.age_prior_Gyr),plant=True)

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    runtime(out)
    spec=seed4_spec();(CACHE/'seed4.json').write_text(json.dumps(spec));(out/'seed4.json').write_text(json.dumps(spec))
    # Pack pure Python into one archive: itch sees tens of files, not thousands.
    with zipfile.ZipFile(out/'leaves.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'universe').rglob('*.py'):
            if 'test_' not in p.name and '__pycache__' not in str(p):z.write(p,str(p.relative_to(ROOT)))
        z.write(WEB/'bridge.py','live_bridge.py')
    for p in WEB.iterdir():
        if p.suffix in ['.js','.html','.css','.json']:shutil.copy2(p,out/p.name)
    for name in ['three.min.js','OrbitControls.js']:
        shutil.copy2(ROOT/'universe/vendor'/name,out/name)
    shutil.copy2(ROOT/'universe/viewer_src/shaders.js',out/'shaders.js')
    # Reuse the existing live surface renderer; swap ONLY its network transport.
    html=(ROOT/'universe/viewer_src/life_live_template.html').read_text()
    html=html.replace('<title>NIM life LIVE</title>','<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>NIMverse live surface</title>')
    html=html.replace('</style>','#rw,#fw,#lv,#rp,#scrub{display:none!important}</style>')
    html=html.replace('<script>/*THREE*/</script><script>/*ORBIT*/</script><script>/*SHADERS*/</script><script>/*APP*/</script>', '<script src="surface-transport.js"></script><script src="three.min.js"></script><script src="OrbitControls.js"></script><script src="shaders.js"></script><script src="creature_mesh.js"></script><script src="surface.js"></script>')
    app=(ROOT/'universe/viewer_src/life_live.js').read_text()
    app=app.replace('const replay=window.NIM_DEMO?new NIMReplay(window.NIM_DEMO):null;','const replay=await window.NIM_TRANSPORT;')
    app=app.replace('Offline · recorded simulation · loops','● Live Python + QWRK · fully offline')
    app=app.replace('NIMverse · Browser demo','NIMverse · Live world')
    app=app.replace("REPLAY'} #${raw.seq} / ${hi}","LIVE'} tick ${raw.seq}")
    app=app.replace('←/→ step · scrub timeline · T trails','T trails · real simulation, no recording')
    (out/'surface.html').write_text(html)
    (out/'surface.js').write_text(app)
    shutil.copy2(ROOT/'universe/viewer_src/creature_mesh.js',out/'creature_mesh.js')
    shutil.copy2(ROOT/'LICENSE',out/'LICENSE.txt')
    archive=out.parent/'nimverse_live_universe.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())<1000
        assert archive.stat().st_size<100_000_000
        print(f'ZIP {archive} {archive.stat().st_size} bytes; {len(z.namelist())} files')
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',default='/workspace/itch/live_universe');args=a.parse_args();build(args.out)
