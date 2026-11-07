"""Reproducible offline live-browser build. Downloads only at build time.
PYTHONPATH=.:/workspace/NIM /workspace/nimenv/bin/python -m universe.export_itch_live
"""
import argparse, hashlib, io, json, shutil, tarfile, urllib.request, zipfile
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

def notices(out):
    third=out/'third_party';third.mkdir(exist_ok=True)
    shutil.copy2(WEB/'licenses/MPL-2.0.txt',third/'MPL-2.0.txt')
    shutil.copy2(WEB/'THIRD_PARTY.txt',out/'THIRD_PARTY.txt')
    pylicense=CACHE/'CPYTHON-LICENSE.txt'
    if not pylicense.exists():
        data=urllib.request.urlopen('https://www.python.org/ftp/python/3.14.2/Python-3.14.2.tar.xz',timeout=120).read()
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:xz') as t:
            pylicense.write_bytes(t.extractfile('Python-3.14.2/LICENSE').read())
    shutil.copy2(pylicense,third/'CPYTHON-LICENSE.txt')

    # GPL binary accompanies the corresponding upstream source, not just a link.
    src=CACHE/'rebound-4.4.7-source.tar.gz'
    if not src.exists():
        url='https://files.pythonhosted.org/packages/d5/4d/d8813ca16f8c3791401007b0fdec79c523be9918eb3d2c3d605670dfd5e2/rebound-4.4.7.tar.gz'
        src.write_bytes(urllib.request.urlopen(url,timeout=120).read())
    assert hashlib.sha256(src.read_bytes()).hexdigest()=='35a47f25ff7a913402b326a8e0c43d1d0f74965d7d3f03822063cb7d9b8c7175'
    shutil.copy2(src,third/src.name)
    for wheel in (out/'pyodide').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if '.dist-info/' in name and name.rsplit('/',1)[-1] in ('LICENSE','LICENSE.txt'):
                    package=wheel.name.split('-')[0]
                    if name.count('/')<=3:(third/(package+'-LICENSE.txt')).write_bytes(z.read(name))
    mit=(ROOT/'LICENSE').read_text().replace('Copyright (c) 2026 TimeMelt','Copyright 2010-2021 Three.js Authors')
    (third/'THREE-MIT.txt').write_text(mit)

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
    for name in ['gate.html','test_universe.js','test_system.js','test_quantum.js','package.json','circuit_fixtures.json','pyodide/pyodide.js']:
        (out/name).unlink(missing_ok=True)
    runtime(out)
    notices(out)
    spec=seed4_spec();(CACHE/'seed4.json').write_text(json.dumps(spec));(out/'seed4.json').write_text(json.dumps(spec))
    # Pack pure Python into one archive: itch sees tens of files, not thousands.
    with zipfile.ZipFile(out/'leaves.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'universe').rglob('*.py'):
            if 'test_' not in p.name and '__pycache__' not in str(p):z.write(p,str(p.relative_to(ROOT)))
        z.write(WEB/'bridge.py','live_bridge.py')
    for p in WEB.iterdir():
        if p.suffix in ['.js','.html','.css','.json'] and not p.name.startswith(('test_','gate.')) and p.name not in ['package.json','circuit_fixtures.json']:shutil.copy2(p,out/p.name)
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
    app=app.replace('window.NIM={ready:true,', 'window.NIM={stats:()=>({fps:qualityFps,pixelRatio:renderer.getPixelRatio(),renderCap,drawCalls:renderer.info.render.calls,triangles:renderer.info.render.triangles}),ready:true,')
    app=app.replace('let replayTime=performance.now();', 'let replayTime=performance.now(),qualityTime=performance.now(),qualityFrames=0,qualityFps=0,renderCap=8;')
    app=app.replace('function animate(now){requestAnimationFrame(animate);', 'function animate(now){requestAnimationFrame(animate);qualityFrames++;if(now-qualityTime>2000){qualityFps=qualityFrames*1000/(now-qualityTime);if(qualityFps<25)renderCap=4;qualityFrames=0;qualityTime=now;if(qualityFps<27&&renderer.getPixelRatio()>.4){renderer.setPixelRatio(Math.max(.4,renderer.getPixelRatio()*.8));renderer.setSize(W(),H());for(const m of [core,cloud])if(m?.material.uniforms.uOct)m.material.uniforms.uOct.value=Math.max(3,m.material.uniforms.uOct.value-1);}}')
    app=app.replace('sphere=new THREE.SphereGeometry(1,192,128)', 'sphere=new THREE.SphereGeometry(1,128,80)')
    app=app.replace('sync(tk.creatures||[])','sync((tk.creatures||[]).slice(0,renderCap))')
    # 20-year ticks turn over a displayed cohort quickly; long replay fades create
    # hundreds of stale mesh draws. Keep a short visual-only birth/death fade.
    app=app.replace('(now-cr.dying)/800','(now-cr.dying)/120').replace('(now-cr.born)/600','(now-cr.born)/180')
    app=app.replace('uOct:{value:9}', 'uOct:{value:6}')
    app=app.replace('window.NIM={stats:', 'window.NIM={stats:')
    app=app.replace('select:showSel};','select:showSel};parent.postMessage({nim:"surface-online"},"*");')
    (out/'surface.js').write_text(app)
    shutil.copy2(ROOT/'universe/viewer_src/creature_mesh.js',out/'creature_mesh.js')
    shutil.copy2(ROOT/'LICENSE',out/'LICENSE.txt')
    (out/'README-LIVE.txt').write_text((ROOT/'docs/live-browser-plan.md').read_text())
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
