// NIM universe viewer: galaxy scene (simulated particles) <-> star-system scene (shader bodies) <-> planet gallery
(function(){
const D = window.NIMDATA, V3 = THREE.Vector3;
function b64(s, T){const b=atob(s), u=new Uint8Array(b.length); for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i); return new T(u.buffer);}
const W=()=>innerWidth, H=()=>innerHeight;
const renderer = new THREE.WebGLRenderer({antialias:true, logarithmicDepthBuffer:true, preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(W(),H()); document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(50, W()/H(), 1e-7, 1e6);
const controls = new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=.12; controls.zoomSpeed=1.4;
addEventListener('resize',()=>{camera.aspect=W()/H(); camera.updateProjectionMatrix(); renderer.setSize(W(),H());});
function glowTex(c0='rgba(255,240,210,1)', c1='rgba(255,200,140,.25)'){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d');
 const gr=g.createRadialGradient(64,64,0,64,64,64); gr.addColorStop(0,c0); gr.addColorStop(.2,c1); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
function sprite(tex, col, s){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:tex, color:col, blending:THREE.AdditiveBlending, depthWrite:false, transparent:true})); m.scale.set(s,s,1); return m;}
function skyPoints(n, R, seed){let x=seed; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;}; const p=new Float32Array(n*3), c=new Float32Array(n*3), s=new Float32Array(n);
 for(let i=0;i<n;i++){const u=rnd()*2-1, t=rnd()*6.283, r=Math.sqrt(1-u*u); p.set([R*r*Math.cos(t),R*u,R*r*Math.sin(t)],3*i); const b=.3+.7*Math.pow(rnd(),3); c.set([b,b*(.85+.15*rnd()),b*(.8+.3*rnd())],3*i); s[i]=1+2*Math.pow(rnd(),4);}
 const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3)); g.setAttribute('size',new THREE.BufferAttribute(s,1));
 return new THREE.Points(g,new THREE.ShaderMaterial({uniforms:{uScale:{value:R*1.5},uAlpha:{value:1}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false}));}

// ---------------- galaxy scene ----------------
const G = D.gal, gS = new THREE.Scene(); gS.background=new THREE.Color(0x000004);
const gPos = b64(G.pos, Int16Array), gN=G.n, gNF=G.nf, gq=G.q;
const gGeo = new THREE.BufferGeometry(), gCur=new Float32Array(gN*3);
gGeo.setAttribute('position', new THREE.BufferAttribute(gCur,3));
const gcol=b64(G.col,Uint8Array), gsz=b64(G.size,Uint8Array), gc=new Float32Array(gN*3), gs=new Float32Array(gN), hz=new Float32Array(gN*3), hs=new Float32Array(gN);
for(let i=0;i<gN;i++){for(let k=0;k<3;k++){gc[3*i+k]=gcol[3*i+k]/255; hz[3*i+k]=G.haze[G.gid[i]%G.haze.length][k];} gs[i]=gsz[i]/40; hs[i]=G.isCore[i]?0:G.hazeSize;}
gGeo.setAttribute('color', new THREE.BufferAttribute(gc,3)); gGeo.setAttribute('size', new THREE.BufferAttribute(gs,1));
const hGeo = new THREE.BufferGeometry(); hGeo.setAttribute('position', gGeo.getAttribute('position')); hGeo.setAttribute('color', new THREE.BufferAttribute(hz,3)); hGeo.setAttribute('size', new THREE.BufferAttribute(hs,1));
const ptsMat = new THREE.ShaderMaterial({uniforms:{uScale:{value:H()*.9},uAlpha:{value:1.}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true});
const hazeMat = new THREE.ShaderMaterial({uniforms:{uScale:{value:H()*.9},uAlpha:{value:.06}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true});
gS.add(new THREE.Points(hGeo, hazeMat)); gS.add(new THREE.Points(gGeo, ptsMat)); gS.add(skyPoints(3000, 5000, 7));
const coreTex = glowTex(), coreSpr = G.cores.map((c,g)=>{const s=sprite(coreTex, 0xffe0b0, G.R[g]*1.6); gS.add(s); return s;});
const hostSpr = sprite(glowTex('rgba(140,255,180,1)','rgba(80,255,140,.3)'), 0xffffff, G.R[0]*.12); gS.add(hostSpr);
function gFrame(f){const fi=Math.min(f*(gNF-1)/(D.sys.nf-1), gNF-1.0001), i0=Math.floor(fi), t=fi-i0, o0=i0*gN*3, o1=o0+gN*3;
 for(let j=0;j<gN*3;j++) gCur[j]=(gPos[o0+j]*(1-t)+gPos[o1+j]*t)*gq; gGeo.getAttribute('position').needsUpdate=true;
 G.cores.forEach((c,g)=>coreSpr[g].position.set(gCur[3*c],gCur[3*c+1],gCur[3*c+2])); const h=G.host; hostSpr.position.set(gCur[3*h],gCur[3*h+1],gCur[3*h+2]);}
const gp=(i)=>new V3(gCur[3*i],gCur[3*i+1],gCur[3*i+2]);
function gCenter(){const c=new V3(); G.cores.forEach(i=>c.add(gp(i))); return c.multiplyScalar(1/G.cores.length);}

// ---------------- system scene ----------------
const S = D.sys, sS = new THREE.Scene(); sS.background=new THREE.Color(0x000002); sS.add(skyPoints(4000, 3000, 11));
const sPos = b64(S.pos, Float32Array), sN=S.n;
const sun = new V3(), sunCol=new THREE.Color(1,1,1), bodies = [], meshes=[], tmpM=new THREE.Matrix3();
const sphere = new THREE.SphereGeometry(1, 128, 96);
function U(o){const u={uOct:{value:6},uSun:{value:sun},uSunCol:{value:sunCol},uOff:{value:new V3(...(o.noise_off||[0,0,0]))},uTime:{value:0},uRot:{value:new THREE.Matrix3()}}; return u;}
function mat(fs, u, extra={}){return new THREE.ShaderMaterial(Object.assign({uniforms:u, vertexShader:VS_BODY, fragmentShader:fs}, extra));}
function makeBody(d, r){ // returns THREE.Group with .userData
 const grp=new THREE.Group(), u=U(d); let core;
 if(d.type==='star'){u.uCol={value:new THREE.Color(...d.rgb)}; u.uGran={value:d.gran}; u.uSpots={value:d.spots};
  core=new THREE.Mesh(sphere, mat(FS_STAR,u)); grp.add(core); const cor=sprite(glowTex('rgba(255,255,255,1)','rgba(255,220,180,.35)'), new THREE.Color(...d.rgb), r*7); grp.add(cor);}
 else if(d.type==='moon'){u.uCr={value:d.craters}; u.uDepth={value:d.crater_depth}; u.uCol={value:new V3(...d.color)}; core=new THREE.Mesh(sphere, mat(FS_MOON,u)); grp.add(core);}
 else if(d.type.indexOf('giant')>=0){u.uBands={value:d.bands}; u.uTurb={value:d.turb}; ['uC0','uC1','uC2'].forEach((k,i)=>u[k]={value:new V3(...d.colors[i])}); u.uStorm={value:new V3(...d.storm)};
  core=new THREE.Mesh(sphere, mat(FS_GAS,u)); grp.add(core); const a={uAtm:{value:new V3(...d.colors[2])},uSun:{value:sun}}; const am=new THREE.Mesh(sphere, mat(FS_ATM,a,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); am.scale.setScalar(1.03); grp.add(am);}
 else {Object.assign(u,{uSea:{value:d.sea},uRough:{value:d.rough},uCS:{value:d.cont_scale},uMtn:{value:d.mountains},uMoist:{value:d.moisture},uCap:{value:d.cap},uLava:{value:d.type==='lava'?1:0}});
  d.colors.forEach((c,i)=>u['uC'+i]={value:new V3(...c)}); core=new THREE.Mesh(sphere, mat(FS_ROCKY,u)); grp.add(core);
  if(d.clouds>.02){const cu=U(d); cu.uOff.value.add(new V3(31,7,13)); cu.uCover={value:d.clouds*.6}; const cm=new THREE.Mesh(sphere, mat(FS_CLOUD,cu,{transparent:true,depthWrite:false})); cm.scale.setScalar(1.012); grp.add(cm); grp.userData.cloud=cm;}
  const a={uAtm:{value:new V3(...d.atm)},uSun:{value:sun}}; const am=new THREE.Mesh(sphere, mat(FS_ATM,a,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); am.scale.setScalar(1.05); grp.add(am);}
 core.userData.grp=grp; grp.userData.core=core; grp.userData.d=d; grp.userData.r=r; grp.scale.setScalar(r); grp.rotation.z=d.tilt||0; return grp;}
for(let i=0;i<sN;i++){if(S.kind[i]===3) continue; const d=S.detail[i]; const g=makeBody(d, S.rvis[i]); g.userData.i=i; sS.add(g); bodies[i]=g; meshes.push(g.userData.core);}
const ast=[...S.kind.keys()].filter(i=>S.kind[i]===3), aGeo=new THREE.BufferGeometry(), aCur=new Float32Array(ast.length*3);
aGeo.setAttribute('position',new THREE.BufferAttribute(aCur,3)); aGeo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(ast.length*3).fill(.55),3)); aGeo.setAttribute('size',new THREE.BufferAttribute(new Float32Array(ast.length).fill(.004),1));
const aMat=new THREE.ShaderMaterial({uniforms:{uScale:{value:H()},uAlpha:{value:1}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false}); sS.add(new THREE.Points(aGeo,aMat));
function sRaw(f,i){const o=(f*sN+i)*3; return new V3(sPos[o],sPos[o+1],sPos[o+2]);}
function sAt(f,i){const f0=Math.floor(f), t=f-f0, f1=Math.min(f0+1,S.nf-1); return sRaw(f0,i).lerp(sRaw(f1,i),t);}
function sBodyPos(f,i){const p=sAt(f,i); const par=S.parent[i]; if(par>=0){const pp=sAt(f,par); p.sub(pp).multiplyScalar(S.moonK[par]).add(pp);} return p;}
for(let i=0;i<sN;i++) if(S.kind[i]===1){const pts=[]; for(let f=0;f<S.nf;f+=1) pts.push(sRaw(f,i)); const l=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:0x3a6a9a,transparent:true,opacity:.35})); sS.add(l);}
function sFrame(f, tsec){const st=[...S.kind.keys()].filter(i=>S.kind[i]===0); sun.set(0,0,0); let lw=0; st.forEach(i=>{const w=S.detail[i].L; sun.add(sBodyPos(f,i).multiplyScalar(w)); lw+=w;}); sun.multiplyScalar(1/lw);
 const c0=new THREE.Color(...S.detail[st[0]].rgb); sunCol.setRGB(Math.min(1.2,c0.r*1.2),Math.min(1.2,c0.g*1.2),Math.min(1.2,c0.b*1.2));
 bodies.forEach((g,i)=>{if(!g)return; g.position.copy(sBodyPos(f,i)); const d=g.userData.d; g.rotation.y=tsec*(d.spin||.2)*.15; if(g.userData.cloud) g.userData.cloud.rotation.y=tsec*.01;});
 ast.forEach((i,k)=>{const p=sAt(f,i); aCur.set([p.x,p.y,p.z],3*k);}); aGeo.getAttribute('position').needsUpdate=true;}

// ---------------- planet gallery scene ----------------
const yS = new THREE.Scene(); yS.background=new THREE.Color(0x000002); yS.add(skyPoints(3000, 3000, 5)); const gal=[]; const ySun=new V3(-40,8,30);
D.gallery.forEach((d,k)=>{const g=makeBody(d, .5*Math.min(d.radius_rel,2.5)/ (d.type.indexOf('giant')>=0?2.5:1)); g.position.set((k-(D.gallery.length-1)/2)*3,0,0); g.userData.gallery=k; yS.add(g); gal.push(g); meshes.push(g.userData.core);});

// ---------------- state, LOD, camera following ----------------
let mode='galaxy', frame=S.nf*.25, playing=true, speed=1, focus=null, tsec=0;
const scenes={galaxy:gS, system:sS, gallery:yS};
function focusPos(fc){ if(!fc) return new V3(); if(fc.kind==='galcore') return gp(G.cores[fc.g]); if(fc.kind==='host') return gp(G.host); if(fc.kind==='galall') return gCenter();
 if(fc.kind==='sysall') return sun.clone(); if(fc.kind==='body') return (mode==='gallery'?fc.g.position.clone():bodies[fc.i].position.clone()); return new V3();}
function lod(){const cs=[...meshes]; cs.forEach(m=>{const g=m.userData.grp; if(!g.visible)return; const wp=g.getWorldPosition(new V3()); const r=g.userData.r;
 const dist=Math.max(camera.position.distanceTo(wp)-r, r*1e-4); const oct=Math.max(4, Math.min(13, 5+1.25*Math.log2(r/dist*4)));
 g.traverse(o=>{if(o.material&&o.material.uniforms){const u=o.material.uniforms; if(u.uOct)u.uOct.value=oct; if(u.uTime)u.uTime.value=tsec; if(u.uRot){tmpM.setFromMatrix4(o.matrixWorld); const e=tmpM.elements; for(let c=0;c<3;c++){const l=Math.hypot(e[3*c],e[3*c+1],e[3*c+2]); e[3*c]/=l;e[3*c+1]/=l;e[3*c+2]/=l;} u.uRot.value.copy(tmpM);}}});});
 if(mode==='gallery'){sun.copy(ySun); sunCol.setRGB(1.1,1.05,1);} }
let fly=null, lastFP=null;
function flyTo(fc, dist, dir, dur=1.6){focus=fc; const p=focusPos(fc); dir=dir||camera.position.clone().sub(controls.target).normalize(); if(dir.lengthSq()<1e-9) dir=new V3(.3,.5,1).normalize();
 fly={t0:performance.now(), dur:dur*1000, cp0:camera.position.clone(), tp0:controls.target.clone(), dist, dir:dir.clone().normalize(), d0:camera.position.distanceTo(controls.target)}; lastFP=p;}
function placeCam(fc, dist, dir){focus=fc; const p=focusPos(fc); controls.target.copy(p); camera.position.copy(p.clone().add(dir.clone().normalize().multiplyScalar(dist))); lastFP=p; fly=null;}
function follow(){if(!focus) return; const p=focusPos(focus);
 if(fly){const t=Math.min((performance.now()-fly.t0)/fly.dur,1), e=t*t*(3-2*t); const tgt=fly.tp0.clone().lerp(p,e); const d=Math.exp(Math.log(fly.d0)*(1-e)+Math.log(fly.dist)*e);
  const dir0=fly.cp0.clone().sub(fly.tp0).normalize(); const dir=dir0.lerp(fly.dir,e).normalize(); controls.target.copy(tgt); camera.position.copy(tgt.clone().add(dir.multiplyScalar(d))); if(t>=1) fly=null;}
 else if(lastFP){const dl=p.clone().sub(lastFP); camera.position.add(dl); controls.target.add(dl);} lastFP=p;}
function setMode(m){mode=m; ui.mode.textContent={galaxy:'Galaxy view (simulated particles)',system:'Featured star system (in galaxy 0)',gallery:'Planet gallery (same seed, not in this system)'}[m];
 const sc=mode==='galaxy'?1:.0;}
// ---------------- UI ----------------
const ui={}; const panel=document.getElementById('panel'); ui.mode=document.getElementById('mode'); ui.info=document.getElementById('info');
function btn(label, fn, cls=''){const b=document.createElement('button'); b.textContent=label; b.className=cls; b.onclick=fn; panel.appendChild(b); return b;}
function hdr(t){const h=document.createElement('div'); h.className='hdr'; h.textContent=t; panel.appendChild(h);}
function fade(fn){const o=document.getElementById('fade'); o.style.opacity=1; setTimeout(()=>{fn(); o.style.opacity=0;},350);}
function info(html){ui.info.innerHTML=html;}
const typeIcon={terran:'🌍',ocean:'🌊',desert:'🏜',ice:'🧊',lava:'🌋','gas giant':'🪐','ice giant':'🔵',moon:'🌑',star:'☀'};
function goGalaxyAll(){fade(()=>{setMode('galaxy'); const c=gCenter(); placeCam({kind:'galall'}, G.ext*2.4, new V3(.4,.6,1)); info(`<b>${D.title}</b><br>${G.nG} galaxies · ${D.counts.galaxy} simulated bodies (${gN} shown)`);});}
function goGal(g){const f=()=>{flyTo({kind:'galcore',g}, G.R[g]*3.2, new V3(.2,.9,.6)); info(`<b>Galaxy ${g}</b>${g===0?' (hosts the featured system)':''}<br>radius ${G.R[g].toFixed(1)} kpc · mass ${G.M[g].toFixed(0)}×10⁸ M☉<br>${G.desc[g]}`);}; if(mode!=='galaxy') fade(()=>{setMode('galaxy'); placeCam({kind:'galcore',g}, G.R[g]*8, new V3(.3,.6,1)); f();}); else f();}
function goSystem(){const go=()=>fade(()=>{setMode('system'); placeCam({kind:'sysall'}, S.amax*2.6, new V3(.2,.75,1)); info(`<b>Featured system</b> · ${S.binary?'binary star':'single star'}${S.belt?' · asteroid belt':''}<br>${S.nPl} planets, ${S.nMoon} moons, ${ast.length} asteroids. Sizes and moon distances are exaggerated for visibility.`);});
 if(mode==='galaxy'){flyTo({kind:'host'}, G.R[0]*.04, null, 1.4); setTimeout(go, 1450);} else go();}
function bodyInfo(d,i){const n=S.names?S.names[i]:''; if(d.type==='star') return `<b>${n}</b> ☀ star<br>${d.mass.toFixed(2)} M☉ · T ≈ ${d.T.toFixed(0)} K · L ≈ ${d.L.toFixed(2)} L☉`;
 if(d.type==='moon') return `<b>${n}</b> 🌑 cratered moon`; return `<b>${n||'gallery planet'}</b> ${typeIcon[d.type]||''} ${d.type}<br>${d.mass_earth.toFixed(2)} M⊕${d.a?` · a = ${d.a.toFixed(2)} AU`:''}${d.sea!==undefined?` · sea level ${d.sea.toFixed(2)} · clouds ${d.clouds.toFixed(2)}`:''}`;}
function goBody(i, dist=3.2, dir){const f=()=>{flyTo({kind:'body',i}, S.rvis[i]*dist, dir||new V3(.3,.35,1)); info(bodyInfo(S.detail[i],i));}; if(mode!=='system') fade(()=>{setMode('system'); placeCam({kind:'sysall'}, S.amax*2.6, new V3(.2,.75,1)); f();}); else f();}
function goGallery(k, dist=3.2, dir){const g=gal[k]; const f=()=>{flyTo({kind:'body',g}, g.userData.r*dist, dir||new V3(.3,.35,1)); info(bodyInfo(g.userData.d,-1));}; if(mode!=='gallery') fade(()=>{setMode('gallery'); placeCam({kind:'body',g}, 12, new V3(0,.3,1)); f();}); else f();}
hdr('Universe'); btn('Whole universe', goGalaxyAll); for(let g=0;g<G.nG;g++) btn(`Galaxy ${g}${g===0?' ★':''}`, ()=>goGal(g));
hdr('Featured system'); btn('System overview', goSystem);
for(let i=0;i<sN;i++){if(S.kind[i]>2) continue; const d=S.detail[i]; btn(`${typeIcon[d.type]||''} ${S.names[i]}${d.type!=='moon'&&d.type!=='star'?' · '+d.type:''}`, ()=>goBody(i), S.kind[i]===2?'sub':'');}
hdr('Planet gallery'); D.gallery.forEach((d,k)=>btn(`${typeIcon[d.type]} ${d.type} #${k}`, ()=>goGallery(k)));
const sl=document.getElementById('time'), pb=document.getElementById('play'), sp=document.getElementById('speed'); sl.max=S.nf-1;
sl.oninput=()=>{frame=+sl.value;}; pb.onclick=()=>{playing=!playing; pb.textContent=playing?'❚❚':'▶';}; sp.oninput=()=>{speed=+sp.value;};
const ray=new THREE.Raycaster(); renderer.domElement.addEventListener('dblclick',e=>{const m=new THREE.Vector2(e.clientX/W()*2-1,-e.clientY/H()*2+1); ray.setFromCamera(m,camera);
 if(mode==='galaxy'){let best=-1,bd=.05; G.cores.forEach((c,g)=>{const p=gp(c).project(camera); const d=Math.hypot(p.x-m.x,p.y-m.y); if(d<bd){bd=d;best=g;}}); const hp=gp(G.host).project(camera); if(Math.hypot(hp.x-m.x,hp.y-m.y)<.03) return goSystem(); if(best>=0) goGal(best); return;}
 const hit=ray.intersectObjects(meshes.filter(o=>o.parent.parent===scenes[mode]))[0]; if(hit){const g=hit.object.userData.grp; if(mode==='gallery') goGallery(g.userData.gallery); else goBody(g.userData.i);}});
// ---------------- loop + scripted API (used for headless tours/screenshots) ----------------
function update(){gFrame(frame); sFrame(frame, tsec); follow(); controls.update(); lod(); ptsMat.uniforms.uScale.value=hazeMat.uniforms.uScale.value=H()*.9;
 const dz=camera.position.distanceTo(controls.target); camera.near=Math.max(dz*1e-4,1e-8); camera.far=dz*1e5+1e4; camera.updateProjectionMatrix();
 renderer.render(scenes[mode], camera); sl.value=Math.floor(frame); document.getElementById('fr').textContent=`frame ${Math.floor(frame)+1}/${S.nf}`;}
let last=performance.now(), auto=true;
function loop(now){const dt=Math.min((now-last)/1000,.1); last=now; if(auto){tsec+=dt; if(playing){frame+=dt*15*speed; if(frame>S.nf-1) frame=0;} update();} requestAnimationFrame(loop);}
setMode('galaxy'); gFrame(frame); sFrame(frame,0); placeCam({kind:'galall'}, G.ext*2.4, new V3(.4,.6,1)); info(`<b>${D.title}</b><br>Double-click a galaxy, the green host star or a body; or pick from the list.`);
requestAnimationFrame(loop);
const TP = D.tour;  // tour: universe -> galaxy 0 -> host star -> system -> planet surface
function lerpLog(a,b,e){return Math.exp(Math.log(a)*(1-e)+Math.log(b)*e);} function sm(e){e=Math.min(Math.max(e,0),1); return e*e*(3-2*e);}
window.NIM = {ready:true, bodies:S.names, gallery:D.gallery.map(d=>d.type),
 still(){auto=false; playing=false;}, setFrame(f){frame=f;}, setTime(t){tsec=t;}, render(){update(); return true;},
 view(kind, arg, dist, dir){auto=false; fly=null; dir=new V3(...(dir||[.3,.35,1]));
  if(kind==='galaxy'){setMode('galaxy'); placeCam(arg<0?{kind:'galall'}:{kind:'galcore',g:arg}, dist, dir);}
  else if(kind==='system'){setMode('system'); sFrame(frame,tsec); placeCam(arg<0?{kind:'sysall'}:{kind:'body',i:arg}, dist, dir); if(arg>=0) info(bodyInfo(S.detail[arg],arg));}
  else {setMode('gallery'); lod(); placeCam({kind:'body',g:gal[arg]}, dist, dir); info(bodyInfo(gal[arg].userData.d,-1));} update(); return true;},
 tour(u){auto=false; fly=null; frame=S.nf*.2+u*S.nf*.6; tsec=u*60; gFrame(frame); sFrame(frame,tsec); const pi=TP.planet, r=S.rvis[pi];
  if(u<.36){setMode('galaxy'); const c=gCenter(), c0=gp(G.cores[0]), h=gp(G.host);
   if(u<.18){const e=sm(u/.18); const tgt=c.clone().lerp(c0,e); const dist=lerpLog(G.ext*2.4,G.R[0]*3,e); const dir=new V3(.4,.6,1).normalize().lerp(new V3(.2,.9,.6).normalize(),e).normalize(); placeCam({kind:'galcore',g:0},1,dir); controls.target.copy(tgt); camera.position.copy(tgt.clone().add(dir.multiplyScalar(dist)));}
   else {const e=sm((u-.18)/.18); const tgt=c0.clone().lerp(h,e); const dist=lerpLog(G.R[0]*3,G.R[0]*.03,e); const dir=new V3(.2,.9,.6).normalize(); placeCam({kind:'host'},1,dir); controls.target.copy(tgt); camera.position.copy(tgt.clone().add(dir.multiplyScalar(dist)));}
   info('<b>Zooming in:</b> universe → galaxy 0 → host star');}
  else if(u<.62){setMode('system'); const e=sm((u-.36)/.26); const s0=sun.clone(), p=bodies[pi].position.clone(); const tgt=s0.lerp(p,e); const dist=lerpLog(S.amax*2.6, r*5, e);
   const dir=new V3(.2,.75,1).normalize().lerp(new V3(.6,.25,1).normalize(),e).normalize(); controls.target.copy(tgt); camera.position.copy(tgt.clone().add(dir.multiplyScalar(dist))); focus=null;
   info(`<b>Featured system</b> → ${S.names[pi]} (${S.detail[pi].type})`);}
  else {setMode('system'); const e=sm((u-.62)/.38); const p=bodies[pi].position.clone(); const sd=sun.clone().sub(p).normalize();
   const up=new V3(0,1,0); const side=new V3().crossVectors(sd,up).normalize(); const dir=sd.clone().multiplyScalar(.55).add(side.multiplyScalar(.8)).add(up.clone().multiplyScalar(.35*(1-e)+.05)).normalize();
   const dist=lerpLog(r*5, r*1.08, e); const tgt=p.clone(); const tilt=side.clone().multiplyScalar(-r*.6*e); controls.target.copy(tgt.add(tilt)); camera.position.copy(p.clone().add(dir.multiplyScalar(dist))); focus=null;
   info(bodyInfo(S.detail[pi],pi));}
  update(); return true;}};
})();
