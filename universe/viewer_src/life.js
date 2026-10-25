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
const sunL=new THREE.DirectionalLight(0xfff6ea,1.5); sunL.position.copy(sun); scene.add(sunL); scene.add(new THREE.AmbientLight(0x223344,.55)); scene.add(new THREE.HemisphereLight(0x6688aa,0x081018,.35));
// sun glow sprite
function glowTex(c0,c1){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d'); const gr=g.createRadialGradient(64,64,0,64,64,64); gr.addColorStop(0,c0); gr.addColorStop(.25,c1); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
const GT=glowTex('rgba(255,255,255,1)','rgba(255,255,255,.3)');
function spr(col,s,op=1){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:GT,color:col,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op})); m.scale.set(s,s,1); return m;}
{const s=spr(new THREE.Color(1,.95,.85),60); s.position.copy(sun).multiplyScalar(8); scene.add(s);}
// ---- creatures ----
const SC=.0045, C=D.creatures, crs=[], pickables=[];
const sm=(c,ex={})=>new THREE.MeshStandardMaterial(Object.assign({color:new THREE.Color(...c),flatShading:true,roughness:.6,metalness:.05},ex));
function creature(k){const c=C[k], g=new THREE.Group(), body=new THREE.Group(); g.add(body);
 const glow=c.glow>.05, mb=sm(c.color, glow?{emissive:new THREE.Color(...c.accent),emissiveIntensity:.35*c.glow}:{}), ma=sm(c.accent,{emissive:new THREE.Color(...c.accent),emissiveIntensity:glow?.9*c.glow:.15});
 const limbs=[], S=c.size;
 const ball=(r,x,y,z,sx=1,sy=1,sz=1,m=mb,det=1)=>{const o=new THREE.Mesh(new THREE.IcosahedronGeometry(r,det),m); o.position.set(x,y,z); o.scale.set(sx,sy,sz); body.add(o); return o;};
 if(c.plan==='swimmer'){ball(1,0,0,0,.7,.55,1.7); const tail=new THREE.Mesh(new THREE.ConeGeometry(.6,1.2,4),ma); tail.rotation.x=-Math.PI/2; tail.position.z=-2.1; tail.scale.set(1,1,.3); body.add(tail); limbs.push({o:tail,k:'tail'});
  for(let i=0;i<c.limbs;i++){const s=i%2?1:-1, f=new THREE.Mesh(new THREE.ConeGeometry(.35,1.1,3),ma); f.position.set(s*.75,-.1,.6-.5*Math.floor(i/2)); f.rotation.z=s*Math.PI/2; body.add(f); limbs.push({o:f,k:'fin',s});}
  ball(.18,.35,.25,1.35,1,1,1,ma,0); ball(.18,-.35,.25,1.35,1,1,1,ma,0);}
 else if(c.plan==='crawler'){for(let i=0;i<c.segments;i++) ball(.7-.06*i,0,.1,1.1-1.0*i,1,.75,1.1);
  for(let i=0;i<c.limbs;i++){const s=i%2?1:-1, l=new THREE.Mesh(new THREE.CylinderGeometry(.09,.06,1.4,4),mb); l.position.set(s*.85,-.35,1.1-(c.segments-1)*Math.floor(i/2)/Math.max(1,c.limbs/2-1)); l.rotation.z=s*1.0; body.add(l); limbs.push({o:l,k:'leg',s,i});}
  ball(.15,.3,.45,1.65,1,1,1,ma,0); ball(.15,-.3,.45,1.65,1,1,1,ma,0); const ant=new THREE.Mesh(new THREE.ConeGeometry(.05,1.0,3),ma); ant.position.set(0,.8,1.9); ant.rotation.x=.7; body.add(ant);}
 else if(c.plan==='flyer'){ball(.6,0,0,0,.8,.7,1.6); const wg=new THREE.BoxGeometry(2.6,.06,1.1);
  for(const s of [-1,1]){const w=new THREE.Group(), m=new THREE.Mesh(wg,ma); m.position.x=s*1.35; w.add(m); w.position.set(s*.4,.2,.1); body.add(w); limbs.push({o:w,k:'wing',s});}
  const t=new THREE.Mesh(new THREE.ConeGeometry(.3,1.4,4),mb); t.rotation.x=-Math.PI/2; t.position.z=-1.5; body.add(t); ball(.12,.2,.25,.9,1,1,1,ma,0); ball(.12,-.2,.25,.9,1,1,1,ma,0);}
 else {const bell=new THREE.Mesh(new THREE.SphereGeometry(1,10,6,0,6.283,0,1.6),new THREE.MeshStandardMaterial({color:new THREE.Color(...c.color),emissive:new THREE.Color(...c.accent),emissiveIntensity:.25+.8*c.glow,transparent:true,opacity:.8,flatShading:true,side:THREE.DoubleSide})); bell.position.y=.6; body.add(bell); limbs.push({o:bell,k:'bell'});
  for(let i=0;i<6+c.segments;i++){const a=i/(6+c.segments)*6.283, t=new THREE.Mesh(new THREE.CylinderGeometry(.05,.02,1.8,3),ma); t.position.set(.7*Math.cos(a),-.4,.7*Math.sin(a)); body.add(t); limbs.push({o:t,k:'tent',a});}}
 if(c.shell){const sh=new THREE.Mesh(new THREE.SphereGeometry(1,8,5,0,6.283,0,1.3),sm(c.accent.map(v=>v*.6),{roughness:.35,metalness:.3})); sh.scale.set(.85,.6,c.plan==='crawler'?.6*c.segments:1.4); sh.position.set(0,.25,c.plan==='crawler'?.6-.5*c.segments:0); body.add(sh);
  for(let i=0;i<Math.round(3+4*c.shell_k);i++){const sp=new THREE.Mesh(new THREE.ConeGeometry(.08,.35,3),sm(c.accent)); const a=i*.9; sp.position.set(.5*Math.sin(a),.75,.6-i*.35); body.add(sp);}}
 if(glow){const s=spr(new THREE.Color(...c.accent),4.5*c.glow+1.5,.8); s.position.y=.3; body.add(s);}
 body.traverse(o=>{if(o.isMesh){o.userData.k=k; pickables.push(o);}});
 g.scale.setScalar(SC*S); scene.add(g);
 const p0=new V3(Math.cos(c.lat)*Math.cos(c.lon),Math.sin(c.lat),Math.cos(c.lat)*Math.sin(c.lon)); let e=new V3(0,1,0).cross(p0).normalize(), n=p0.clone().cross(e);
 const t0=e.multiplyScalar(Math.cos(c.heading)).add(n.multiplyScalar(Math.sin(c.heading))).normalize();
 const alt={swimmer:1.0045,crawler:1.006,flyer:1.04,floater:1.012}[c.plan];
 return {g,body,limbs,c,p0,t0,alt,w:c.speed*.012};}
for(let k=0;k<C.length;k++) crs.push(creature(k));
// tech: lit structures + city lights near herds of the most mobile cohort
const tech=[]; if(D.tech>0){let x=99; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;}; const base=crs[0].p0;
 for(let i=0;i<Math.round(14*D.tech);i++){const p=base.clone().add(new V3(rnd()-.5,rnd()-.5,rnd()-.5).multiplyScalar(.6)).normalize(); const g=new THREE.Group();
  const tw=new THREE.Mesh(new THREE.CylinderGeometry(.18,.35,2.4+3*rnd(),6),new THREE.MeshStandardMaterial({color:0x8899aa,metalness:.6,roughness:.3,flatShading:true})); tw.position.y=1.2; g.add(tw);
  const l=spr(new THREE.Color(1,.8,.45),4,1); l.position.y=2.8; g.add(l); g.position.copy(p).multiplyScalar(1.004); g.quaternion.setFromUnitVectors(new V3(0,1,0),p); g.scale.setScalar(SC); scene.add(g); tech.push(g);}}
// ---- time / animation ----
let T=0, mode='orbit', sel=-1; const up=new V3(), tmp=new V3();
function place(cr,t){const c=cr.c, a=c.real_pos?.025*Math.sin(cr.w*t*6+c.phase):cr.w*t,   // real positions: small local wander around the simulated lat/lon
 mean=.25*Math.sin(t*.3+c.wobble);
 const p=cr.p0.clone().multiplyScalar(Math.cos(a)).add(cr.t0.clone().multiplyScalar(Math.sin(a))); const fw=cr.t0.clone().multiplyScalar(Math.cos(a)).sub(cr.p0.clone().multiplyScalar(Math.sin(a)));
 fw.applyAxisAngle(p,mean).normalize(); let h=cr.alt; const ph=t*c.speed*6+c.phase;
 if(c.plan==='crawler'&&c.traits.mobility>.5) h+=.004*Math.max(0,Math.sin(ph*.5)); if(c.plan==='flyer') h+=.01*Math.sin(t*.7+c.phase); if(c.plan==='floater') h+=.004*Math.sin(t*1.3+c.phase);
 cr.g.position.copy(p).multiplyScalar(h); const m=new THREE.Matrix4(); const xx=p.clone().cross(fw).normalize(); m.makeBasis(xx.negate(),p,fw); cr.g.quaternion.setFromRotationMatrix(m);
 cr.limbs.forEach(L=>{if(L.k==='tail')L.o.rotation.y=.5*Math.sin(ph); else if(L.k==='fin')L.o.rotation.x=.6*Math.sin(ph+L.s); else if(L.k==='leg')L.o.rotation.x=.6*Math.sin(ph+(L.i%4)*1.57);
  else if(L.k==='wing')L.o.rotation.z=L.s*.7*Math.sin(ph*1.5); else if(L.k==='bell')L.o.scale.set(1+.12*Math.sin(ph),1-.1*Math.sin(ph),1+.12*Math.sin(ph)); else if(L.k==='tent')L.o.rotation.x=.3*Math.sin(ph+L.a);});
 cr.body.rotation.z=c.plan==='swimmer'?.15*Math.sin(ph):0;}
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
setTime(0); view('orbit'); requestAnimationFrame(loop);
window.NIM={ready:true, view, setTime, render, still(){playing=false;}, play(){playing=true;}, creatures:C, pos:i=>crs[i].g.position.toArray(), cam:(p,t)=>{camera.position.set(...p); controls.target.set(...t); render();}};
})();
