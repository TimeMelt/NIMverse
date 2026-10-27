// NIM life LIVE: same planet shaders / creature style as life.js, but fed by Server-Sent Events from run_life_live (/info + /stream).
(async function(){
const V3=THREE.Vector3, W=()=>innerWidth, H=()=>innerHeight, f=(x,n=2)=>Number(x).toFixed(n);
const info=await (await fetch('info')).json();
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(W(),H()); document.body.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(50,W()/H(),.0005,5000); const controls=new THREE.OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.dampingFactor=.12;
addEventListener('resize',()=>{camera.aspect=W()/H(); camera.updateProjectionMatrix(); renderer.setSize(W(),H());});
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x000002);
{let x=7; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;}; const n=4000,R=1500,p=new Float32Array(n*3),c=new Float32Array(n*3),s=new Float32Array(n);
 for(let i=0;i<n;i++){const u=rnd()*2-1,t=rnd()*6.283,r=Math.sqrt(1-u*u); p.set([R*r*Math.cos(t),R*u,R*r*Math.sin(t)],3*i); const b=.3+.7*Math.pow(rnd(),3); c.set([b,b*(.85+.15*rnd()),b*(.8+.3*rnd())],3*i); s[i]=1+2*Math.pow(rnd(),4);}
 const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3)); g.setAttribute('size',new THREE.BufferAttribute(s,1));
 scene.add(new THREE.Points(g,new THREE.ShaderMaterial({uniforms:{uScale:{value:R*1.5},uAlpha:{value:1}},vertexShader:VS_PTS,fragmentShader:FS_PTS,blending:THREE.AdditiveBlending,depthWrite:false})));}
const sun=new V3(60,25,40), sunCol=new THREE.Color(1,.97,.92), sphere=new THREE.SphereGeometry(1,192,128);
function U(o){return {uOct:{value:9},uSun:{value:sun},uSunCol:{value:sunCol},uOff:{value:new V3(...(o.noise_off||[0,0,0]))},uTime:{value:0},uRot:{value:new THREE.Matrix3()}};}
function mat(fs,u,ex={}){return new THREE.ShaderMaterial(Object.assign({uniforms:u,vertexShader:VS_BODY,fragmentShader:fs},ex));}
const d=info.look, planet=new THREE.Group(), u=U(d);
Object.assign(u,{uSea:{value:d.sea},uRough:{value:d.rough},uCS:{value:d.cont_scale},uMtn:{value:d.mountains},uMoist:{value:d.moisture},uCap:{value:d.cap},uLava:{value:0}});
d.colors.forEach((c,i)=>u['uC'+i]={value:new V3(...c)}); const core=new THREE.Mesh(sphere,mat(FS_ROCKY,u)); planet.add(core);
let cloud=null; if(d.clouds>.02){const cu=U(d); cu.uOff.value.add(new V3(31,7,13)); cu.uCover={value:d.clouds*.45}; cloud=new THREE.Mesh(sphere,mat(FS_CLOUD,cu,{transparent:true,depthWrite:false})); cloud.scale.setScalar(1.035); planet.add(cloud);}
{const a={uAtm:{value:new V3(...d.atm)},uSun:{value:sun}}; const am=new THREE.Mesh(sphere,mat(FS_ATM,a,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); am.scale.setScalar(1.06); planet.add(am);}
scene.add(planet);
const sunL=new THREE.DirectionalLight(0xfff6ea,1.5); sunL.position.copy(sun); scene.add(sunL); scene.add(new THREE.AmbientLight(0x223344,.55)); scene.add(new THREE.HemisphereLight(0x6688aa,0x081018,.35));
function glowTex(c0,c1){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d'); const gr=g.createRadialGradient(64,64,0,64,64,64); gr.addColorStop(0,c0); gr.addColorStop(.25,c1); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
const GT=glowTex('rgba(255,255,255,1)','rgba(255,255,255,.3)');
function spr(col,s,op=1){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:GT,color:col,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op})); m.scale.set(s,s,1); return m;}
{const s=spr(new THREE.Color(1,.95,.85),60); s.position.copy(sun).multiplyScalar(8); scene.add(s);}
// ---- creatures (keyed by individual id) ----
const SC=.0045, crs=new Map(), sm=(c,ex={})=>new THREE.MeshStandardMaterial(Object.assign({color:new THREE.Color(...c),flatShading:true,roughness:.6,metalness:.05},ex));
const ALT={swimmer:1.0045,crawler:1.006,flyer:1.04,floater:1.012}, ll=(la,lo)=>new V3(Math.cos(la)*Math.cos(lo),Math.sin(la),Math.cos(la)*Math.sin(lo));
function build(c){const g=new THREE.Group(), body=new THREE.Group(); g.add(body); const limbs=[];
 const glow=c.glow>.05, mb=sm(c.color, glow?{emissive:new THREE.Color(...c.accent),emissiveIntensity:.35*c.glow}:{}), ma=sm(c.accent,{emissive:new THREE.Color(...c.accent),emissiveIntensity:glow?.9*c.glow:.15});
 const ball=(r,x,y,z,sx=1,sy=1,sz=1,m=mb,det=1)=>{const o=new THREE.Mesh(new THREE.IcosahedronGeometry(r,det),m); o.position.set(x,y,z); o.scale.set(sx,sy,sz); body.add(o); return o;};
 if(c.plan==='swimmer'){ball(1,0,0,0,.7,.55,1.7); const tail=new THREE.Mesh(new THREE.ConeGeometry(.6,1.2,4),ma); tail.rotation.x=-Math.PI/2; tail.position.z=-2.1; tail.scale.set(1,1,.3); body.add(tail); limbs.push({o:tail,k:'tail'});
  for(let i=0;i<c.limbs;i++){const s=i%2?1:-1, fn=new THREE.Mesh(new THREE.ConeGeometry(.35,1.1,3),ma); fn.position.set(s*.75,-.1,.6-.5*Math.floor(i/2)); fn.rotation.z=s*Math.PI/2; body.add(fn); limbs.push({o:fn,k:'fin',s});}}
 else if(c.plan==='crawler'){for(let i=0;i<c.segments;i++) ball(.7-.06*i,0,.1,1.1-1.0*i,1,.75,1.1);
  for(let i=0;i<c.limbs;i++){const s=i%2?1:-1, l=new THREE.Mesh(new THREE.CylinderGeometry(.09,.06,1.4,4),mb); l.position.set(s*.85,-.35,1.1-(c.segments-1)*Math.floor(i/2)/Math.max(1,c.limbs/2-1)); l.rotation.z=s; body.add(l); limbs.push({o:l,k:'leg',s,i});}}
 else if(c.plan==='flyer'){ball(.6,0,0,0,.8,.7,1.6); const wg=new THREE.BoxGeometry(2.6,.06,1.1);
  for(const s of [-1,1]){const w=new THREE.Group(), m=new THREE.Mesh(wg,ma); m.position.x=s*1.35; w.add(m); w.position.set(s*.4,.2,.1); body.add(w); limbs.push({o:w,k:'wing',s});}}
 else {const bell=new THREE.Mesh(new THREE.SphereGeometry(1,10,6,0,6.283,0,1.6),new THREE.MeshStandardMaterial({color:new THREE.Color(...c.color),emissive:new THREE.Color(...c.accent),emissiveIntensity:.25+.8*c.glow,transparent:true,opacity:.8,flatShading:true,side:THREE.DoubleSide})); bell.position.y=.6; body.add(bell); limbs.push({o:bell,k:'bell'});
  for(let i=0;i<6+c.segments;i++){const a=i/(6+c.segments)*6.283, t=new THREE.Mesh(new THREE.CylinderGeometry(.05,.02,1.8,3),ma); t.position.set(.7*Math.cos(a),-.4,.7*Math.sin(a)); body.add(t); limbs.push({o:t,k:'tent',a});}}
 if(c.shell){const sh=new THREE.Mesh(new THREE.SphereGeometry(1,8,5,0,6.283,0,1.3),sm(c.accent.map(v=>v*.6),{roughness:.35,metalness:.3})); sh.scale.set(.85,.6,c.plan==='crawler'?.6*c.segments:1.4); sh.position.set(0,.25,c.plan==='crawler'?.6-.5*c.segments:0); body.add(sh);}
 if(glow){const s=spr(new THREE.Color(...c.accent),4.5*c.glow+1.5,.8); s.position.y=.3; body.add(s);}
 body.traverse(o=>{if(o.isMesh) o.userData.id=c.id;}); g.scale.setScalar(SC*c.size*.01); scene.add(g);
 const tgt=ll(c.lat,c.lon), par=crs.get(c.parent_id)||crs.get(c.parent2_id), p=par?par.p.clone():tgt.clone(); return {g,body,limbs,c,p,target:tgt,fw:new V3(0,1,0).cross(p).normalize(),born:performance.now(),dying:0};}
function dispose(cr){scene.remove(cr.g); cr.g.traverse(o=>{if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose();});}
function sync(list){const seen=new Set();
 for(const c of list){seen.add(c.id); let cr=crs.get(c.id); if(!cr){cr=build(c); crs.set(c.id,cr);} else {cr.c=Object.assign(cr.c,{age:c.age,energy:c.energy,traits:c.traits,species_id:c.species_id}); const t=ll(c.lat,c.lon); if(t.distanceTo(cr.target)>1e-6){const mv=t.clone().sub(cr.p); mv.sub(cr.p.clone().multiplyScalar(mv.dot(cr.p))); if(mv.lengthSq()>1e-10) cr.fw=mv.normalize(); cr.target=t;} cr.dying=0;}}
 for(const [id,cr] of crs) if(!seen.has(id)&&!cr.dying) cr.dying=performance.now();}
// tech lights near the first creature cluster
const tech=[]; function techLights(level,anchor){const want=Math.round(14*Math.min(1,level)); while(tech.length>want){const g=tech.pop(); scene.remove(g);}
 let x=99+tech.length*7; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;};
 while(tech.length<want&&anchor){const p=anchor.clone().add(new V3(rnd()-.5,rnd()-.5,rnd()-.5).multiplyScalar(.6)).normalize(), g=new THREE.Group();
  const tw=new THREE.Mesh(new THREE.CylinderGeometry(.18,.35,2.4+3*rnd(),6),new THREE.MeshStandardMaterial({color:0x8899aa,metalness:.6,roughness:.3,flatShading:true})); tw.position.y=1.2; g.add(tw);
  const l=spr(new THREE.Color(1,.8,.45),4,1); l.position.y=2.8; g.add(l); g.position.copy(p).multiplyScalar(1.004); g.quaternion.setFromUnitVectors(new V3(0,1,0),p); g.scale.setScalar(SC); scene.add(g); tech.push(g);}}
// ---- HUD + stream ----
const infoE=document.getElementById('info'), selE=document.getElementById('sel'), modeE=document.getElementById('mode'), connE=document.getElementById('conn');
modeE.textContent='Life surface · LIVE'; let last=null, nTicks=0, paused=false, sel=null, t0=performance.now();
function hud(tk){const cu=tk.culture||{}, plans={}; let nShow=0; for(const cr of crs.values()) if(!cr.dying){nShow++; plans[cr.c.plan]=(plans[cr.c.plan]||0)+1;}
 infoE.innerHTML=`<b>${tk.planet}</b> · ${tk.ptype} world · <b>year ${Math.round(tk.t_w).toLocaleString()}</b><br>T ${f(tk.climate.T_K,0)} K (${tk.climate.cls}) · water ${f(tk.climate.water*100,0)}%<br>biomass <b>${f(tk.biomass,3)}</b> · ${tk.alive?'alive':'<span style="color:#f88">no life yet</span>'}<br>`+
  `<b>${tk.n_alive}</b> living individuals · ${tk.n_species} species · births ${tk.births}<br>showing ${nShow} · `+Object.entries(plans).map(([k,v])=>`${v} ${k}s`).join(' · ')+
  `<br>O₂ ${f(tk.biosign.o2,3)} · CH₄ ${f(tk.biosign.ch4,3)} · pigments ${f(tk.biosign.pigments,2)}<br>tech <b>${f(tk.tech,2)}</b>${cu.active?` · culture active · polities ${cu.polities_n} · pollution ${f(cu.pollution,2)} · warm ${f(cu.warm_K,2)} K`:''}<br>impacts ${tk.impacts}<br><span style="color:#789">${info.title}</span>`;}
function onTick(tk){last=tk; nTicks++; if(paused) return; sync(tk.creatures||[]); const first=crs.values().next().value; techLights(tk.tech,first&&first.p); hud(tk); if(sel!=null) showSel(sel);}
function connect(){const es=new EventSource('stream'); es.onopen=()=>{connE.textContent='● live'; connE.style.color='#8f8';};
 es.onmessage=e=>{try{onTick(JSON.parse(e.data));}catch(err){console.error(err);}};
 es.addEventListener('end',()=>{connE.textContent='■ sim ended (max-years)'; connE.style.color='#fb6'; es.close();});
 es.onerror=()=>{connE.textContent='○ reconnecting…'; connE.style.color='#f88';};}
connect(); setInterval(()=>{if(last&&connE.textContent.startsWith('●')) connE.textContent=`● live · year ${Math.round(last.t_w)} · ${nTicks} ticks${paused?' · view paused':''}`;},500);
function showSel(id){sel=id; const cr=crs.get(id); if(!cr){selE.style.display='none'; sel=null; return;} const c=cr.c; selE.style.display='block';
 selE.innerHTML=`<b>individual #${c.id}</b> · species ${c.species_id} · gen ${c.generation}<br>parent ${c.parent_id??'founder'}${c.parent2_id!=null?` × ${c.parent2_id}`:''} · age ${f(c.age,0)} yr · energy ${f(c.energy)}<br>plan <b>${c.plan}</b> · size ${f(c.size)}<hr style="border-color:#234">`+Object.entries(c.traits).map(([a,b])=>`${a} <b>${f(b,3)}</b>`).join('<br>');}
const ray=new THREE.Raycaster(), mv=new THREE.Vector2();
renderer.domElement.addEventListener('click',e=>{mv.set(e.clientX/W()*2-1,-e.clientY/H()*2+1); ray.setFromCamera(mv,camera); const ps=[]; for(const cr of crs.values()) cr.g.traverse(o=>{if(o.isMesh)ps.push(o);}); const h=ray.intersectObjects(ps,false); if(h.length) showSel(h[0].object.userData.id); else showSel(null);});
addEventListener('keydown',e=>{if(e.key==='1'){sel=null; showSel(null); camera.position.set(0,.8,3); controls.target.set(0,0,0);} if(e.key==='3'){const ids=[...crs.keys()]; if(ids.length) showSel(ids[(ids.indexOf(sel)+1)%ids.length]);} if(e.key===' ') paused=!paused;});
camera.position.set(0,.8,3); controls.minDistance=1.02;
function animate(now){requestAnimationFrame(animate); const t=(now-t0)/1000;
 for(const [id,cr] of crs){const c=cr.c, k=Math.min(1,.04); cr.p.lerp(cr.target,k).normalize(); const p=cr.p, ph=t*c.speed*6+c.phase;
  let fw=cr.fw.clone(); fw.sub(p.clone().multiplyScalar(fw.dot(p))); if(fw.lengthSq()<1e-10) fw=new V3(0,1,0).cross(p); fw.normalize();
  let h=ALT[c.plan]||1.005; if(c.plan==='flyer') h+=.01*Math.sin(t*.7+c.phase); if(c.plan==='floater') h+=.004*Math.sin(t*1.3+c.phase);
  let s=Math.min(1,(now-cr.born)/600); if(cr.dying){s=1-(now-cr.dying)/800; if(s<=0){dispose(cr); crs.delete(id); continue;}}
  cr.g.scale.setScalar(SC*c.size*Math.max(.01,s)); cr.g.position.copy(p).multiplyScalar(h); const m=new THREE.Matrix4(); const xx=p.clone().cross(fw).normalize(); m.makeBasis(xx.negate(),p,fw); cr.g.quaternion.setFromRotationMatrix(m);
  cr.limbs.forEach(L=>{if(L.k==='tail')L.o.rotation.y=.5*Math.sin(ph); else if(L.k==='fin')L.o.rotation.x=.6*Math.sin(ph+L.s); else if(L.k==='leg')L.o.rotation.x=.6*Math.sin(ph+(L.i%4)*1.57); else if(L.k==='wing')L.o.rotation.z=L.s*.7*Math.sin(ph*1.5); else if(L.k==='bell')L.o.scale.set(1+.12*Math.sin(ph),1-.1*Math.sin(ph),1+.12*Math.sin(ph)); else if(L.k==='tent')L.o.rotation.x=.3*Math.sin(ph+L.a);});}
 if(sel!=null&&crs.get(sel)){const p=crs.get(sel).g.position, dl=p.clone().sub(controls.target); controls.target.add(dl); camera.position.add(dl); controls.minDistance=.002;}
 [core,cloud].forEach(m=>{if(m)m.material.uniforms.uTime.value=t;}); controls.update(); const dist=camera.position.length()-1; camera.near=Math.max(.0003,dist*.02); camera.updateProjectionMatrix(); renderer.render(scene,camera);}
requestAnimationFrame(animate);
window.NIM={ready:true, last:()=>last, ticks:()=>nTicks, n:()=>crs.size, cam:(p,t)=>{camera.position.set(...p); controls.target.set(...t);}, select:showSel};
})();
