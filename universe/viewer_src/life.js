// NIM life surface: living planet (FS_ROCKY/FS_CLOUD/FS_ATM from shaders.js, same as gallery closeups) + low-poly creatures from cohort phenotypes
(function(){
const D=window.NIMDATA, V3=THREE.Vector3, W=()=>innerWidth, H=()=>innerHeight;
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(W(),H()); document.body.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(50,W()/H(),.0005,5000); const controls=new THREE.OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.dampingFactor=.12;
addEventListener('resize',()=>{camera.aspect=W()/H(); camera.updateProjectionMatrix(); renderer.setSize(W(),H());});
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x000002);
// sky
{let x=7; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;}; const n=4000,R=1500,p=new Float32Array(n*3),c=new Float32Array(n*3),s=new Float32Array(n);
 for(let i=0;i<n;i++){const u=rnd()*2-1,t=rnd()*6.283,r=Math.sqrt(1-u*u); p.set([R*r*Math.cos(t),R*u,R*r*Math.sin(t)],3*i); const b=.3+.7*Math.pow(rnd(),3); c.set([b,b*(.85+.15*rnd()),b*(.8+.3*rnd())],3*i); s[i]=1+2*Math.pow(rnd(),4);}
 const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3)); g.setAttribute('size',new THREE.BufferAttribute(s,1));
 scene.add(new THREE.Points(g,new THREE.ShaderMaterial({uniforms:{uScale:{value:R*1.5},uAlpha:{value:1}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false})));}
const sun=new V3(60,25,40), sunCol=new THREE.Color(1,.97,.92), sphere=new THREE.SphereGeometry(1,192,128);
function U(o){return {uOct:{value:9},uSun:{value:sun},uSunCol:{value:sunCol},uOff:{value:new V3(...(o.noise_off||[0,0,0]))},uTime:{value:0},uRot:{value:new THREE.Matrix3()}};}
function mat(fs,u,ex={}){return new THREE.ShaderMaterial(Object.assign({uniforms:u,vertexShader:VS_BODY,fragmentShader:fs},ex));}
const d=D.planet, planet=new THREE.Group(), u=U(d);
Object.assign(u,{uSea:{value:d.sea},uRough:{value:d.rough},uCS:{value:d.cont_scale},uMtn:{value:d.mountains},uMoist:{value:d.moisture},uCap:{value:d.cap},uLava:{value:0}});
d.colors.forEach((c,i)=>u['uC'+i]={value:new V3(...c)}); const core=new THREE.Mesh(sphere,mat(FS_ROCKY,u)); planet.add(core);
let cloud=null; if(d.clouds>.02){const cu=U(d); cu.uOff.value.add(new V3(31,7,13)); cu.uCover={value:d.clouds*.45}; cloud=new THREE.Mesh(sphere,mat(FS_CLOUD,cu,{transparent:true,depthWrite:false})); cloud.scale.setScalar(1.035); planet.add(cloud);}
{const a={uAtm:{value:new V3(...d.atm)},uSun:{value:sun}}; const am=new THREE.Mesh(sphere,mat(FS_ATM,a,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); am.scale.setScalar(1.06); planet.add(am);}
scene.add(planet);
const sunL=NIMC.lights(scene,renderer,sun); u.uWet={value:1};   // lighting pass (creature_mesh.js): ACES, PMREM sky env, key+rim+hemi; ocean wet sheen
// sun glow sprite
function glowTex(c0,c1){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d'); const gr=g.createRadialGradient(64,64,0,64,64,64); gr.addColorStop(0,c0); gr.addColorStop(.25,c1); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
const GT=glowTex('rgba(255,255,255,1)','rgba(255,255,255,.3)');
function spr(col,s,op=1){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:GT,color:col,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op})); m.scale.set(s,s,1); return m;}
{const s=spr(new THREE.Color(1,.95,.85),60); s.position.copy(sun).multiplyScalar(8); scene.add(s);}
// ---- creatures ----
const SC=.0045, C=D.creatures, crs=[], pickables=[]; let tris=0;
function creature(k){const c=C[k], g=new THREE.Group(), S=c.size; const B=NIMC.build(c), body=B.body, limbs=B.limbs; g.add(body); g.add(B.shadow); tris+=B.tris;
 body.traverse(o=>{if(o.isMesh){o.userData.k=k; pickables.push(o);}});
 g.scale.setScalar(SC*S); scene.add(g);
 const p0=new V3(Math.cos(c.lat)*Math.cos(c.lon),Math.sin(c.lat),Math.cos(c.lat)*Math.sin(c.lon)); let e=new V3(0,1,0).cross(p0).normalize(), n=p0.clone().cross(e);
 const t0=e.multiplyScalar(Math.cos(c.heading)).add(n.multiplyScalar(Math.sin(c.heading))).normalize();
 const alt={swimmer:1.0045,crawler:1.006,flyer:1.04,floater:1.012}[c.plan];
 let path=null, line=null;                                     // recorded path (oldest -> newest lat/lon samples): replayed, drawn as a fading trail
 if(c.trail&&c.trail.length>=2){path=c.trail.map(([la,lo])=>new V3(Math.cos(la)*Math.cos(lo),Math.sin(la),Math.cos(la)*Math.sin(lo)));
  const pts=[], cols=[]; const col=new THREE.Color(...c.accent);
  for(let i=0;i<path.length-1;i++) for(let j=0;j<6;j++){const u=(i+j/6)/(path.length-1); pts.push(pathAt(path,u).multiplyScalar(alt*1.0015)); const f=.15+.85*u; cols.push(col.r*f,col.g*f,col.b*f);}
  pts.push(path[path.length-1].clone().multiplyScalar(alt*1.0015)); cols.push(col.r,col.g,col.b);
  const gg=new THREE.BufferGeometry().setFromPoints(pts); gg.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  line=new THREE.Line(gg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.85,depthWrite:false})); scene.add(line);}
 return {g,body,limbs,c,p0,t0,alt,w:c.speed*.012,path,line,shadow:B.shadow};}
function pathAt(P,u){u=Math.min(1,Math.max(0,u))*(P.length-1); const i=Math.min(P.length-2,Math.floor(u)), f=u-i; return P[i].clone().multiplyScalar(1-f).add(P[i+1].clone().multiplyScalar(f)).normalize();}
const REPLAY=10;   // time units for one replay of the recorded path (oldest sample -> current position), then hold
function replayU(t){return Math.min(1,Math.max(0,(t-1)/REPLAY));}
for(let k=0;k<C.length;k++) crs.push(creature(k));
// look toggle (stylized default | photo): rebuild every creature body in place with the shared builder
function setLook(l){NIMC.setLook(l); NIMC.applyLookLights(renderer); tris=0; pickables.length=0;
 crs.forEach((cr,k)=>{cr.g.remove(cr.body); cr.g.remove(cr.shadow); cr.body.traverse(o=>{if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose();});
  const B=NIMC.build(cr.c); cr.body=B.body; cr.limbs=B.limbs; cr.shadow=B.shadow; cr.g.add(B.body); cr.g.add(B.shadow); tris+=B.tris; B.body.traverse(o=>{if(o.isMesh){o.userData.k=k; pickables.push(o);}});});
 setTime(T); if(lookBtn) lookBtn.refresh(); return NIMC.getLook();}
let lookBtn=null; NIMC.applyLookLights(renderer);
// tech: lit structures + city lights near herds of the most mobile cohort
const tech=[]; if(D.tech>0){let x=99; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;}; const base=crs[0].p0;
 for(let i=0;i<Math.round(14*D.tech);i++){const p=base.clone().add(new V3(rnd()-.5,rnd()-.5,rnd()-.5).multiplyScalar(.6)).normalize();
  const g=NIMC.tower(rnd); g.position.copy(p).multiplyScalar(1.0025); g.quaternion.setFromUnitVectors(new V3(0,1,0),p); g.scale.setScalar(SC); scene.add(g); tech.push(g);}}
// ---- time / animation ----
let T=0, mode='orbit', sel=-1; const up=new V3(), tmp=new V3();
function place(cr,t){const c=cr.c; let p, fw, mean=.25*Math.sin(t*.3+c.wobble);
 if(cr.path){const u=replayU(t), du=.004; p=pathAt(cr.path,u); const q=pathAt(cr.path,Math.min(1,u+du)), q0=pathAt(cr.path,Math.max(0,u-du));   // replay recorded path
  fw=q.sub(q0); fw.sub(p.clone().multiplyScalar(fw.dot(p))); if(fw.lengthSq()<1e-12) fw=cr.t0.clone(); fw.normalize(); mean*=.3;
  if(u>=1){const a=.01*Math.sin(cr.w*t*6+c.phase); p.add(fw.clone().multiplyScalar(a)).normalize();}}           // arrived: small local wander
 else{const a=c.real_pos?.025*Math.sin(cr.w*t*6+c.phase):cr.w*t;   // no trail: local wander around simulated lat/lon (legacy: herd orbit)
  p=cr.p0.clone().multiplyScalar(Math.cos(a)).add(cr.t0.clone().multiplyScalar(Math.sin(a))); fw=cr.t0.clone().multiplyScalar(Math.cos(a)).sub(cr.p0.clone().multiplyScalar(Math.sin(a)));}
 fw.applyAxisAngle(p,mean).normalize(); let h=cr.alt; const ph=t*c.speed*6+c.phase;
 if(c.plan==='crawler'&&c.traits.mobility>.5) h+=.004*Math.max(0,Math.sin(ph*.5)); if(c.plan==='flyer') h+=.01*Math.sin(t*.7+c.phase); if(c.plan==='floater') h+=.004*Math.sin(t*1.3+c.phase);
 cr.g.position.copy(p).multiplyScalar(h); const m=new THREE.Matrix4(); const xx=p.clone().cross(fw).normalize(); m.makeBasis(xx.negate(),p,fw); cr.g.quaternion.setFromRotationMatrix(m);
 NIMC.animate(cr,ph,t); NIMC.placeShadow(cr,h,SC*c.size);
 cr.body.rotation.z=c.plan==='swimmer'?.08*Math.sin(ph*.5):c.plan==='flyer'?.12*Math.sin(t*.7+c.phase):0;}
function setTime(t){T=t; crs.forEach(c=>place(c,t)); [core,cloud].forEach(m=>{if(m)m.material.uniforms.uTime.value=t;});}
// ---- UI ----
const info=document.getElementById('info'), selE=document.getElementById('sel'), modeE=document.getElementById('mode');
const plans={}; C.forEach(c=>plans[c.plan]=(plans[c.plan]||0)+1); const f=(x,n=2)=>Number(x).toFixed(n);
info.innerHTML=`<b>${D.name}</b> · ${D.ptype} world<br>biomass ${f(D.biomass,3)} · ${D.individual_mode?`<b>${D.n_alive}</b> living individuals · ${D.cohorts.length} species · showing <b>${C.length}</b> real ids`:`${D.cohorts.length} cohorts · <b>${C.length}</b> creatures`}<br>`+Object.entries(plans).map(([k,v])=>`${v} ${k}s`).join(' · ')+
 `<br>T ${f(D.climate.T_K,0)} K (${D.climate.cls}) · water ${f(D.climate.water*100,0)}%<br>O₂ ${f(D.biosign.o2,3)} · pigments ${f(D.biosign.pigments,2)} · tech ${f(D.tech,2)}<br><span style="color:#789">${D.title}</span>`;
function showSel(k){sel=k; if(k<0){selE.style.display='none';return;} const c=C[k]; selE.style.display='block';
 selE.innerHTML=`${c.id!==undefined?`<b>individual #${c.id}</b> · species ${c.species_id} · gen ${c.generation}<br>parent ${c.parent_id??'founder'}${c.parent2_id!=null?` × ${c.parent2_id} (sexual)`:''} · age ${f(c.age,0)} yr · energy ${f(c.energy)}<br>`:`<b>creature #${k}</b> · cohort ${c.cohort}<br>`}plan <b>${c.plan}</b> · size ${f(c.size)} · limbs ${c.limbs} · ${c.shell?'shelled':'soft-bodied'}${c.glow>.05?' · bioluminescent':''}<br>speed ${f(c.speed)}<hr style="border-color:#234">`+Object.entries(c.traits).map(([a,b])=>`${a} <b>${f(b,3)}</b>`).join('<br>');}
const ray=new THREE.Raycaster(), mv=new THREE.Vector2();
renderer.domElement.addEventListener('click',e=>{mv.set(e.clientX/W()*2-1,-e.clientY/H()*2+1); ray.setFromCamera(mv,camera); const h=ray.intersectObjects(pickables,false); if(h.length) view('creature',h[0].object.userData.k); else showSel(-1);});
addEventListener('keydown',e=>{if(e.key==='1')view('orbit'); if(e.key==='2')view('surface',Math.max(sel,0)); if(e.key==='3')view('creature',(sel+1)%C.length);});
function view(m,i=0,o={}){mode=m; const labels={orbit:'Life surface · orbit',surface:'Life surface · ground level',creature:'Life surface · creature'}; modeE.textContent=labels[m];
 if(m==='orbit'){showSel(-1); const p=crs[i]?crs[i].p0:new V3(1,0,0); controls.target.set(0,0,0); camera.position.copy(p).multiplyScalar(o.dist||3).add(new V3(0,.6,0)); controls.minDistance=1.05;}
 else{const cr=crs[i]; showSel(m==='creature'?i:-1); const p=cr.g.position.clone(), n=p.clone().normalize(); const side=new V3(0,1,0).cross(n).normalize(); if(o.az) side.applyAxisAngle(n,o.az);
  const dist=m==='creature'?(o.dist||.03)*cr.c.size:(o.dist||.1); controls.target.copy(m==='creature'?p:n.clone().multiplyScalar(1.0));
  camera.position.copy(p).add(n.clone().multiplyScalar(dist*(m==='creature'?.55:.55))).add(side.multiplyScalar(dist)); controls.minDistance=.002;}
 controls.update(); render();}
function follow(){if(mode==='creature'&&sel>=0){const p=crs[sel].g.position; const dlt=p.clone().sub(controls.target); controls.target.add(dlt); camera.position.add(dlt);}}
function render(){follow(); controls.update(); const dist=camera.position.length()-1; camera.near=Math.max(.0003,dist*.02); camera.updateProjectionMatrix(); renderer.render(scene,camera);}
let playing=true, last=performance.now();
function loop(now){requestAnimationFrame(loop); if(playing){setTime(T+Math.min(.05,(now-last)/1000));} last=now; render();}
setTime(0); view('orbit'); lookBtn=NIMC.lookUI(()=>setLook(NIMC.getLook())); requestAnimationFrame(loop);
window.NIM={ready:true, view, setLook, look:()=>NIMC.getLook(), setTime, render, still(){playing=false;}, play(){playing=true;}, creatures:C, tris:()=>tris, pos:i=>crs[i].g.position.toArray(), cam:(p,t)=>{camera.position.set(...p); controls.target.set(...t); render();}};
})();
