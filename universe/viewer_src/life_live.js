// NIM life LIVE: same planet shaders / creature style as life.js, but fed by Server-Sent Events from run_life_live (/info + /stream).
(async function(){
const V3=THREE.Vector3, W=()=>innerWidth, H=()=>innerHeight, f=(x,n=2)=>Number(x).toFixed(n);
// auth: token from ?token= in the page URL (or a prompt on 401, remembered in sessionStorage); sent as Bearer on fetch, ?token= on EventSource
const replay=window.NIM_DEMO?new NIMReplay(window.NIM_DEMO):null;
let TOKEN=replay?'':new URLSearchParams(location.search).get('token')||sessionStorage.getItem('nim_token')||'';
const AH=()=>TOKEN?{Authorization:'Bearer '+TOKEN}:{}; const TQ=u=>TOKEN?u+(u.includes('?')?'&':'?')+'token='+encodeURIComponent(TOKEN):u;
async function afetch(u,o={}){for(let i=0;i<3;i++){const r=await fetch(u,Object.assign({},o,{headers:Object.assign({},o.headers||{},AH())})); if(r.status!==401) return r;
 TOKEN=prompt('NIM live: access token')||''; sessionStorage.setItem('nim_token',TOKEN);} throw new Error('unauthorized');}
const info=replay?replay.info:await (await afetch('info')).json();
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
let planet=null, core=null, cloud=null;
function buildPlanet(d){if(planet){scene.remove(planet); planet.traverse(o=>{if(o.geometry&&o.geometry!==sphere)o.geometry.dispose(); if(o.material)o.material.dispose();});}
 planet=new THREE.Group(); const u=U(d);
 Object.assign(u,{uSea:{value:d.sea},uRough:{value:d.rough},uCS:{value:d.cont_scale},uMtn:{value:d.mountains},uMoist:{value:d.moisture},uCap:{value:d.cap},uLava:{value:0}});
 d.colors.forEach((c,i)=>u['uC'+i]={value:new V3(...c)}); u.uWet={value:1}; core=new THREE.Mesh(sphere,mat(FS_ROCKY,u)); planet.add(core);
 cloud=null; if(d.clouds>.02){const cu=U(d); cu.uOff.value.add(new V3(31,7,13)); cu.uCover={value:d.clouds*.45}; cloud=new THREE.Mesh(sphere,mat(FS_CLOUD,cu,{transparent:true,depthWrite:false})); cloud.scale.setScalar(1.035); planet.add(cloud);}
 {const a={uAtm:{value:new V3(...d.atm)},uSun:{value:sun}}; const am=new THREE.Mesh(sphere,mat(FS_ATM,a,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); am.scale.setScalar(1.06); planet.add(am);}
 scene.add(planet);}
const planets=info.planets||[{planet:info.planet,ptype:info.ptype,look:info.look}]; let focus=info.focus||info.planet;
buildPlanet((planets.find(q=>q.planet===focus)||planets[0]).look);
const sunL=NIMC.lights(scene,renderer,sun);   // shared lighting pass (creature_mesh.js)
function glowTex(c0,c1){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d'); const gr=g.createRadialGradient(64,64,0,64,64,64); gr.addColorStop(0,c0); gr.addColorStop(.25,c1); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
const GT=glowTex('rgba(255,255,255,1)','rgba(255,255,255,.3)');
function spr(col,s,op=1){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:GT,color:col,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op})); m.scale.set(s,s,1); return m;}
{const s=spr(new THREE.Color(1,.95,.85),60); s.position.copy(sun).multiplyScalar(8); scene.add(s);}
// ---- creatures (keyed by individual id) ----
const SC=.0045, crs=new Map(), sm=(c,ex={})=>new THREE.MeshStandardMaterial(Object.assign({color:new THREE.Color(...c),flatShading:true,roughness:.6,metalness:.05},ex));
const ALT={swimmer:1.0045,crawler:1.006,flyer:1.04,floater:1.012}, ll=(la,lo)=>new V3(Math.cos(la)*Math.cos(lo),Math.sin(la),Math.cos(la)*Math.sin(lo));
function build(c){const g=new THREE.Group(), B=NIMC.build(c), body=B.body, limbs=B.limbs; g.add(body); g.add(B.shadow);   // shared builder: creature_mesh.js
 body.traverse(o=>{if(o.isMesh) o.userData.id=c.id;}); g.scale.setScalar(SC*c.size*.01); scene.add(g);
 const tgt=ll(c.lat,c.lon), par=crs.get(c.parent_id)||crs.get(c.parent2_id), p=par?par.p.clone():tgt.clone(); return {g,body,limbs,c,p,target:tgt,shadow:B.shadow,fw:new V3(0,1,0).cross(p).normalize(),born:performance.now(),dying:0};}
function dispose(cr){scene.remove(cr.g); if(cr.trail){scene.remove(cr.trail); cr.trail.geometry.dispose(); cr.trail.material.dispose();} cr.g.traverse(o=>{if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose();});}
function sync(list){const seen=new Set();
 for(const c of list){seen.add(c.id); let cr=crs.get(c.id); if(!cr){cr=build(c); crs.set(c.id,cr);} else {cr.c=Object.assign(cr.c,{age:c.age,energy:c.energy,traits:c.traits,species_id:c.species_id}); const t=ll(c.lat,c.lon); if(t.distanceTo(cr.target)>1e-6){const mv=t.clone().sub(cr.p); mv.sub(cr.p.clone().multiplyScalar(mv.dot(cr.p))); if(mv.lengthSq()>1e-10) cr.fw=mv.normalize(); cr.target=t;} cr.dying=0;}
  if(showTrails) pushTrail(cr,c);}
 for(const [id,cr] of crs) if(!seen.has(id)&&!cr.dying) cr.dying=performance.now();}
// trails: server-side c.trail ([lat,lon] oldest->newest, from the individual's stored path) when present; else last TRAIL_N streamed positions
const TRAIL_N=12; let showTrails=true;
function pushTrail(cr,c){const h=(ALT[cr.c.plan]||1.005)*1.001;
 if(c&&c.trail&&c.trail.length>=2) cr.tp=c.trail.map(q=>ll(q[0],q[1]).multiplyScalar(h));
 else {(cr.tp=cr.tp||[]).push(cr.target.clone().multiplyScalar(h)); if(cr.tp.length>TRAIL_N) cr.tp.shift();}
 if(cr.tp.length<2) return; if(!cr.trail){cr.trail=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:new THREE.Color(...cr.c.accent),transparent:true,opacity:.55})); scene.add(cr.trail);}
 cr.trail.geometry.setFromPoints(cr.tp);}
function clearTrails(){for(const cr of crs.values()){cr.tp=[]; if(cr.trail){scene.remove(cr.trail); cr.trail.geometry.dispose(); cr.trail.material.dispose(); cr.trail=null;}}}
function clearCreatures(){for(const cr of crs.values()) dispose(cr); crs.clear(); sel=null; selE.style.display='none';}
// tech lights near the first creature cluster
const tech=[]; function techLights(level,anchor){const want=Math.round(14*Math.min(1,level)); while(tech.length>want){const g=tech.pop(); scene.remove(g);}
 let x=99+tech.length*7; const rnd=()=>{x=(x*16807)%2147483647; return x/2147483647;};
 while(tech.length<want&&anchor){const p=anchor.clone().add(new V3(rnd()-.5,rnd()-.5,rnd()-.5).multiplyScalar(.6)).normalize();
  const g=NIMC.tower(rnd); g.position.copy(p).multiplyScalar(1.0025); g.quaternion.setFromUnitVectors(new V3(0,1,0),p); g.scale.setScalar(SC); scene.add(g); tech.push(g);}}
// ---- HUD + stream ----
const infoE=document.getElementById('info'), selE=document.getElementById('sel'), modeE=document.getElementById('mode'), connE=document.getElementById('conn');
let last=null, lastRaw=null, nTicks=0, frozen=false, sel=null, t0=performance.now(), st={paused:false,mode:'live',realtime_s:0};
const ctlE=document.getElementById('ctl'), plE=document.getElementById('planets');
const pick=tk=>tk.planets?(tk.planets.find(q=>q.planet===focus)||tk.planets[0]):tk;
function hud(tk){const cu=tk.culture||{}, plans={}; let nShow=0; for(const cr of crs.values()) if(!cr.dying){nShow++; plans[cr.c.plan]=(plans[cr.c.plan]||0)+1;}
 infoE.innerHTML=`<b>${tk.planet}</b> · ${tk.ptype} world · <b>year ${Math.round(tk.t_w).toLocaleString()}</b><br>T ${f(tk.climate.T_K,0)} K (${tk.climate.cls}) · water ${f(tk.climate.water*100,0)}%<br>biomass <b>${f(tk.biomass,3)}</b> · ${tk.alive?'alive':'<span style="color:#f88">no life yet</span>'}<br>`+
  `<b>${tk.n_alive}</b> living individuals · ${tk.n_species} species · births ${tk.births}<br>showing ${nShow} · `+Object.entries(plans).map(([k,v])=>`${v} ${k}s`).join(' · ')+
  `<br>O₂ ${f(tk.biosign.o2,3)} · CH₄ ${f(tk.biosign.ch4,3)} · pigments ${f(tk.biosign.pigments,2)}<br>tech <b>${f(tk.tech,2)}</b>${cu.active?` · culture active · polities ${cu.polities_n} · pollution ${f(cu.pollution,2)} · warm ${f(cu.warm_K,2)} K`:''}<br>impacts ${tk.impacts}<br><span style="color:#789">${info.title}</span>`;}
function planetsHud(raw){if(!raw.planets){plE.style.display='none'; return;} plE.style.display='block';
 plE.innerHTML='<b>planets</b> <span style="color:#789">[ ] switch</span><br>'+raw.planets.map(q=>`<div class="pl${q.planet===focus?' on':''}" data-p="${q.planet}">${q.planet===focus?'▶ ':''}${q.planet} · ${q.ptype} · ${q.alive?q.n_alive+' alive':'lifeless'} · tech ${f(q.tech,2)} · ${f(q.climate.T_K,0)} K</div>`).join('');
 plE.querySelectorAll('.pl').forEach(e=>e.onclick=()=>setFocus(e.dataset.p));}
function ctlHud(raw){const rep=raw.mode==='replay', lo=raw.hist_from||0, hi=raw.hist_to||0;
 document.getElementById('badge').textContent=replay?`${raw.paused?'PAUSED':'REPLAY'} #${raw.seq} / ${hi}`:rep?`REPLAY #${raw.seq} / ${raw.live_seq}`:(raw.paused?'LIVE · PAUSED':'LIVE'); document.getElementById('badge').className=rep?'rep':(raw.paused?'pau':'live');
 document.getElementById('bp').textContent=raw.paused?'▶ Resume':'❚❚ Pause'; const sl=document.getElementById('scrub'); sl.min=lo; sl.max=hi; if(!scrubbing) sl.value=raw.seq||hi;
 document.getElementById('spd').textContent=raw.realtime_s>0?`${f(raw.realtime_s,3)} s/tick`:'max speed';}
function render(){if(!lastRaw) return; const tk=pick(lastRaw); last=tk; sync(tk.creatures||[]); const first=crs.values().next().value; techLights(tk.tech,first&&first.p); hud(tk); planetsHud(lastRaw); if(sel!=null) showSel(sel);}
function onTick(raw){st=raw; nTicks++; ctlHud(raw); if(frozen) return; const jump=lastRaw&&raw.seq!==lastRaw.seq+1; if(jump) clearTrails(); lastRaw=raw; render();}
function setFocus(name){if(name===focus) return; focus=name; clearCreatures(); techLights(0,null); buildPlanet((planets.find(q=>q.planet===name)||planets[0]).look); render();}
function cyclePlanet(d){const i=planets.findIndex(q=>q.planet===focus); setFocus(planets[(i+d+planets.length)%planets.length].planet);}
async function control(body){if(replay) return replay.control(body); try{const r=await afetch('control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); Object.assign(st,j); return j;}catch(e){console.error(e);}}
let scrubbing=false;
ctlE.innerHTML=`<span id="badge" class="live">LIVE</span> <button id="bp">❚❚ Pause</button> <button data-s="0.25">0.25×</button><button data-s="1">1×</button><button data-s="4">4×</button><button data-s="0">max</button> <span id="spd"></span><br>
 <button id="rw">◀ step</button><input id="scrub" type="range" min="0" max="0" value="0" style="width:180px;vertical-align:middle"><button id="fw">step ▶</button> <button id="lv">⏭ Return to live</button> <button id="rp" title="restore physics checkpoint at this seq and branch (R / Alt+scrub)">⟲ Restore physics here</button> <label><input id="tr" type="checkbox" checked> trails</label>`;
document.getElementById('bp').onclick=()=>control({action:st.paused?'resume':'pause'});
ctlE.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>control({action:'set_speed',speed:+b.dataset.s}));
document.getElementById('rw').onclick=()=>control({action:'rewind',steps:1}); document.getElementById('fw').onclick=()=>control({action:'forward',steps:1});
document.getElementById('lv').onclick=()=>control({action:'live'});
let altDown=false; addEventListener('keydown',e=>{if(e.key==='Alt')altDown=true;}); addEventListener('keyup',e=>{if(e.key==='Alt')altDown=false;}); addEventListener('blur',()=>altDown=false);
const restoreHere=()=>{const q=st.replay_seq!=null?st.replay_seq:(st.mode==='replay'&&lastRaw?lastRaw.seq:null); if(q!=null) control({action:'restore',seq:q});};
document.getElementById('rp').onclick=restoreHere;
if(replay){document.getElementById('rp').style.display='none'; document.getElementById('lv').textContent='↻ Restart'; ctlE.querySelector('[data-s=\"0\"]').textContent='8×';}
document.getElementById('tr').onchange=e=>{showTrails=e.target.checked; if(!showTrails) clearTrails();};
{const sl=document.getElementById('scrub'); sl.oninput=e=>{scrubbing=true; control({action:'seek',seq:+sl.value,physics:altDown});}; sl.onchange=()=>{scrubbing=false;};}
function connect(){const es=new EventSource(TQ('stream')); es.onopen=()=>{connE.textContent='● live'; connE.style.color='#8f8';};
 es.onmessage=e=>{try{onTick(JSON.parse(e.data));}catch(err){console.error(err);}};
 es.addEventListener('end',()=>{connE.textContent='■ sim ended (max-years) · history still browsable'; connE.style.color='#fb6';});
 es.onerror=()=>{connE.textContent='○ reconnecting…'; connE.style.color='#f88';};}
if(replay){replay.start(onTick); connE.textContent='Offline · recorded simulation · loops'; connE.style.color='#8fc';}
else connect(); setInterval(()=>{if(last&&connE.textContent.startsWith('●')) connE.textContent=`● connected · year ${Math.round(last.t_w)} · ${nTicks} msgs${frozen?' · view frozen':''}`;},500);
modeE.textContent=(replay?'NIMverse · Browser demo':'Life surface · LIVE')+(planets.length>1?` · ${planets.length} planets`:'');
function showSel(id){sel=id; const cr=crs.get(id); if(!cr){selE.style.display='none'; sel=null; return;} const c=cr.c; selE.style.display='block';
 selE.innerHTML=`<b>individual #${c.id}</b> · species ${c.species_id} · gen ${c.generation}<br>parent ${c.parent_id??'founder'}${c.parent2_id!=null?` × ${c.parent2_id}`:''} · age ${f(c.age,0)} yr · energy ${f(c.energy)}<br>plan <b>${c.plan}</b> · size ${f(c.size)}<hr style="border-color:#234">`+Object.entries(c.traits).map(([a,b])=>`${a} <b>${f(b,3)}</b>`).join('<br>');}
const ray=new THREE.Raycaster(), mv=new THREE.Vector2();
renderer.domElement.addEventListener('click',e=>{mv.set(e.clientX/W()*2-1,-e.clientY/H()*2+1); ray.setFromCamera(mv,camera); const ps=[]; for(const cr of crs.values()) cr.g.traverse(o=>{if(o.isMesh&&!o.userData.noPick)ps.push(o);}); const h=ray.intersectObjects(ps,false); if(h.length) showSel(h[0].object.userData.id); else showSel(null);});
function view(mode){
 if(mode==='orbit'){showSel(null); camera.up.set(0,1,0); controls.minDistance=1.02; camera.position.set(0,.8,3); controls.target.set(0,0,0); return;}
 const ids=[...crs.keys()].filter(id=>!crs.get(id).dying); if(!ids.length) return;
 // Prefer the sunlit hemisphere so the first close-up is readable.
 const bright=ids.filter(id=>crs.get(id).p.dot(sun)>0); const pool=bright.length?bright:ids;
 const id=mode==='creature'?pool[(pool.indexOf(sel)+1)%pool.length]:pool[0], cr=crs.get(id);
 const up=cr.p.clone().normalize(), tangent=new V3(0,1,0).cross(up).normalize();
 const p=up.clone().multiplyScalar(ALT[cr.c.plan]||1.005);
 camera.up.copy(up); controls.minDistance=.002; controls.target.copy(p);
 camera.position.copy(p).addScaledVector(up,mode==='surface'?.065:.012).addScaledVector(tangent,mode==='surface'?.12:.03);
 showSel(mode==='creature'?id:null);
}
const views=document.createElement('div'); views.id='views';
views.innerHTML='<button data-view="orbit">1 Orbit</button><button data-view="surface">2 Surface</button><button data-view="creature">3 Creature</button>';
views.querySelectorAll('button').forEach(b=>b.onclick=()=>view(b.dataset.view)); document.body.appendChild(views);
if(replay) document.getElementById('help').textContent='Drag orbit · scroll zoom · 1/2/3 views · P photoreal · Space pause · ←/→ step · scrub timeline · T trails';
addEventListener('keydown',e=>{if(e.target.matches('input,select,textarea')) return; if(['ArrowLeft','ArrowRight',' '].includes(e.key)) e.preventDefault(); if(e.key==='1') view('orbit'); if(e.key==='2') view('surface'); if(e.key==='3') view('creature'); if(e.key===' '){e.preventDefault(); if(e.shiftKey) frozen=!frozen; else control({action:'toggle'});}
 if(e.key==='ArrowLeft') control({action:'rewind',steps:e.shiftKey?10:1}); if(e.key==='ArrowRight') control({action:'forward',steps:e.shiftKey?10:1});
 if(e.key==='l'||e.key==='L') control({action:'live'}); if(e.key==='r'||e.key==='R') restoreHere(); if(e.key==='[') cyclePlanet(-1); if(e.key===']') cyclePlanet(1);
 if(e.key==='t'||e.key==='T'){showTrails=!showTrails; document.getElementById('tr').checked=showTrails; if(!showTrails) clearTrails();}});
camera.position.set(0,.8,3); controls.minDistance=1.02;
let replayTime=performance.now();
function animate(now){requestAnimationFrame(animate); if(replay){replay.advance((now-replayTime)/1000); replayTime=now;} const t=(now-t0)/1000;
 for(const [id,cr] of crs){const c=cr.c, k=Math.min(1,.04); cr.p.lerp(cr.target,k).normalize(); const p=cr.p, ph=t*c.speed*6+c.phase;
  let fw=cr.fw.clone(); fw.sub(p.clone().multiplyScalar(fw.dot(p))); if(fw.lengthSq()<1e-10) fw=new V3(0,1,0).cross(p); fw.normalize();
  let h=ALT[c.plan]||1.005; if(c.plan==='flyer') h+=.01*Math.sin(t*.7+c.phase); if(c.plan==='floater') h+=.004*Math.sin(t*1.3+c.phase);
  let s=Math.min(1,(now-cr.born)/600); if(cr.dying){s=1-(now-cr.dying)/800; if(s<=0){dispose(cr); crs.delete(id); continue;}}
  cr.g.scale.setScalar(SC*c.size*Math.max(.01,s)); cr.g.position.copy(p).multiplyScalar(h); const m=new THREE.Matrix4(); const xx=p.clone().cross(fw).normalize(); m.makeBasis(xx.negate(),p,fw); cr.g.quaternion.setFromRotationMatrix(m);
  NIMC.animate(cr,ph,t); NIMC.placeShadow(cr,h,SC*c.size*Math.max(.01,s));}
 if(sel!=null&&crs.get(sel)){const p=crs.get(sel).g.position, dl=p.clone().sub(controls.target); controls.target.add(dl); camera.position.add(dl); controls.minDistance=.002;}
 [core,cloud].forEach(m=>{if(m)m.material.uniforms.uTime.value=t;}); controls.update(); const dist=camera.position.length()-1; camera.near=Math.max(.0003,dist*.02); camera.updateProjectionMatrix(); renderer.render(scene,camera);}
// look toggle (?look=photo, key P, HUD button): rebuild live creature bodies in place
function setLook(l){NIMC.setLook(l); NIMC.applyLookLights(renderer); for(const cr of crs.values()){cr.g.remove(cr.body); cr.g.remove(cr.shadow); cr.body.traverse(o=>{if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose();});
  const B=NIMC.build(cr.c); cr.body=B.body; cr.limbs=B.limbs; cr.shadow=B.shadow; cr.g.add(B.body); cr.g.add(B.shadow); B.body.traverse(o=>{if(o.isMesh) o.userData.id=cr.c.id;});} lookBtn.refresh(); return NIMC.getLook();}
NIMC.applyLookLights(renderer); const lookBtn=NIMC.lookUI(()=>setLook(NIMC.getLook()));
requestAnimationFrame(animate);
window.NIM={ready:true, view, setLook, look:()=>NIMC.getLook(), last:()=>last, raw:()=>lastRaw, focus:()=>focus, setFocus, control, ticks:()=>nTicks, n:()=>crs.size, cam:(p,t)=>{camera.position.set(...p); controls.target.set(...t);}, select:showSel};
})();
