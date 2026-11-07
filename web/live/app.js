import {Universe,hash,TAU} from './universe.js';
import {makeSystem,position,lifeSpec,look} from './system.js';
const $=id=>document.getElementById(id),V3=THREE.Vector3;
const preset=await(await fetch('preset.json')).json();$('preset').textContent='NIM · offline-trained preset';
let seed=4,u,mode='galaxy',sys=null,planet=null,paused=false,speed=1,year=0,photo=false,quantum=true;
let frameTimes=[],lastFrame=performance.now(),lastG=0,fps=0,steps=0;
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setSize(innerWidth,innerHeight);$('viewport').appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.001,10000),controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.1;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x02040d);const galGroup=new THREE.Group(),sysGroup=new THREE.Group(),worldGroup=new THREE.Group(),fxGroup=new THREE.Group();scene.add(galGroup,sysGroup,worldGroup,fxGroup);
let worldListKey="",eventListKey="";
let gPoints,gGeo,systemBodies=[],worldMesh,selectedStar=null;
const sun=new V3(0,0,0),sunCol=new THREE.Color(1,.95,.85),sphere=new THREE.SphereGeometry(1,64,40);
function material(fs,uniforms,ex={}){return new THREE.ShaderMaterial({vertexShader:VS_BODY,fragmentShader:fs,uniforms,...ex});}
function makeBody(d,r=1,localSun=sun){const g=new THREE.Group(),un={uOct:{value:5},uSun:{value:localSun},uSunCol:{value:sunCol},uOff:{value:new V3(...(d.noise_off||[0,0,0]))},uTime:{value:0},uRot:{value:new THREE.Matrix3()}};let core;
 if(d.type==='star'){Object.assign(un,{uCol:{value:new THREE.Color(1,.86,.6)},uGran:{value:25},uSpots:{value:.2}});core=new THREE.Mesh(sphere,material(FS_STAR,un));}
 else if(d.type.includes('giant')){Object.assign(un,{uBands:{value:d.bands},uTurb:{value:d.turb},uStorm:{value:new V3(...d.storm)}});d.colors.forEach((c,i)=>un['uC'+i]={value:new V3(...c)});core=new THREE.Mesh(sphere,material(FS_GAS,un));}
 else{Object.assign(un,{uSea:{value:d.sea},uRough:{value:d.rough},uCS:{value:d.cont_scale},uMtn:{value:d.mountains},uMoist:{value:d.moisture},uCap:{value:d.cap},uLava:{value:d.type==='lava'?1:0},uWet:{value:1}});d.colors.forEach((c,i)=>un['uC'+i]={value:new V3(...c)});core=new THREE.Mesh(sphere,material(FS_ROCKY,un));
  const au={uAtm:{value:new V3(...d.atm)},uSun:{value:localSun}};let at=new THREE.Mesh(sphere,material(FS_ATM,au,{transparent:true,blending:THREE.AdditiveBlending,depthWrite:false}));at.scale.setScalar(1.055);g.add(at);
  if(d.clouds>.02){let cu={...un,uCover:{value:d.clouds*.5}};const cl=new THREE.Mesh(sphere,material(FS_CLOUD,cu,{transparent:true,depthWrite:false}));cl.scale.setScalar(1.02);g.add(cl);}
 }
 g.add(core);g.userData.core=core;g.userData.u=un;g.scale.setScalar(r);return g;
}
function clear(group){while(group.children.length){let c=group.children[0];group.remove(c);c.traverse(x=>{if(x.geometry&&x.geometry!==sphere)x.geometry.dispose();if(x.material){if(x.material.map)x.material.map.dispose();x.material.dispose();}});}}
function buildGalaxy(){clear(galGroup);clear(fxGroup);const p=new Float32Array(u.bodies.length*3),c=new Float32Array(p.length),sz=new Float32Array(u.bodies.length);for(let i=0;i<u.bodies.length;i++){let b=u.bodies[i];p.set(b.p,i*3);c.set(b.kind===0?[.03,.03,.03]:b.kind===4?[.8,.6,1]:b.color,i*3);sz[i]=b.kind===0?0:b.kind===4?1.6:.12+Math.min(2,b.smass)*.14;}
 gGeo=new THREE.BufferGeometry();gGeo.setAttribute('position',new THREE.BufferAttribute(p,3));gGeo.setAttribute('color',new THREE.BufferAttribute(c,3));gGeo.setAttribute('size',new THREE.BufferAttribute(sz,1));
 gPoints=new THREE.Points(gGeo,new THREE.ShaderMaterial({uniforms:{uScale:{value:innerHeight},uAlpha:{value:1}},vertexShader:VS_PTS,fragmentShader:FS_PTS,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false}));galGroup.add(gPoints);
 // Haze uses the same live particle positions, no duplicate simulation/history.
 const hg=new THREE.BufferGeometry();hg.setAttribute('position',gGeo.getAttribute('position'));hg.setAttribute('color',gGeo.getAttribute('color'));hg.setAttribute('size',new THREE.BufferAttribute(new Float32Array(u.bodies.length).fill(1.6),1));galGroup.add(new THREE.Points(hg,new THREE.ShaderMaterial({uniforms:{uScale:{value:innerHeight},uAlpha:{value:.055}},vertexShader:VS_PTS,fragmentShader:FS_PTS,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})));
}
function seedUniverse(s){seed=s>>>0;u=new Universe(seed,preset.params,preset.events);sys=null;planet=null;year=0;steps=0;buildGalaxy();showGalaxy();}
function setMode(m){mode=m;galGroup.visible=m==='galaxy';sysGroup.visible=m==='system';worldGroup.visible=m==='planet';fxGroup.visible=m==='galaxy';$('surface').hidden=m!=='surface';$('intro').hidden=m!=='galaxy';$('land').hidden=m!=='planet'||planet.look.type.includes('giant');$('photo').hidden=m!=='surface';renderer.domElement.style.visibility=m==='surface'?'hidden':'visible';$('system').disabled=!sys;$('planet').disabled=!planet;}
function showGalaxy(){setMode('galaxy');camera.up.set(0,1,0);camera.position.set(0,55,100);controls.target.set(0,0,0);controls.minDistance=.1;controls.maxDistance=600;$('panelTitle').textContent='OBSERVATORY';updatePanel();}
function openSystem(star){selectedStar=star;sys=makeSystem(seed,star,preset.params);year=0;planet=null;clear(sysGroup);systemBodies=[];
 const starMesh=makeBody({type:'star'},.17);sysGroup.add(starMesh);
 for(const p of sys.planets){const r=.035+.023*Math.min(4,p.look.radius_rel),g=makeBody(p.look,r);g.userData.planet=p;g.userData.core.userData.planet=p;sysGroup.add(g);systemBodies.push(g);
  const pts=[];for(let j=0;j<=160;j++){const a=position(p,sys.star.mass,j/160*Math.sqrt(p.a**3/sys.star.mass));pts.push(new V3(...a).divideScalar(sys.scale));}sysGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x325968,transparent:true,opacity:.45})));
  for(const moon of p.moons){const mg=new THREE.Mesh(new THREE.SphereGeometry(.007,12,8),new THREE.MeshBasicMaterial({color:0x9caaa9}));mg.userData.moon=moon;mg.userData.host=g;sysGroup.add(mg);}
 }
 const hz=new THREE.Mesh(new THREE.RingGeometry(.95,1.67,128),new THREE.MeshBasicMaterial({color:0x56cda2,side:THREE.DoubleSide,transparent:true,opacity:.075,depthWrite:false}));hz.rotation.x=-Math.PI/2;sysGroup.add(hz);showSystem();
}
function showSystem(){if(!sys)return;setMode('system');camera.up.set(0,1,0);const r=Math.min(12,sys.planets.at(-1).a/sys.scale*1.4);camera.position.set(0,r*.65,r*1.25);controls.target.set(0,0,0);controls.minDistance=.1;controls.maxDistance=100;$('system').textContent=sys.star.id;$('panelTitle').textContent='GENERATED SYSTEM';updatePanel();}
function openPlanet(p){planet=p;clear(worldGroup);worldMesh=makeBody(p.look,1,new V3(60,25,40));worldGroup.add(worldMesh);setMode('planet');camera.up.set(0,1,0);camera.position.set(0,.5,3.3);controls.target.set(0,0,0);controls.minDistance=1.1;controls.maxDistance=10;$('planet').textContent=p.name;$('panelTitle').textContent='WORLD SURVEY';updatePanel();}
function updatePanel(){
 if(mode==='galaxy'){worldListKey='';$('details').innerHTML=`<b>${u.bodies.filter(b=>b.kind===1).length}</b> live star tracers<br>3 self-gravitating galaxies<br>kpc · 10⁸ solar masses · G=1<br>Gravity ${u.time.toFixed(3)} units<br>Event clock ${u.stellarMyr.toFixed(1)} Myr<br><br>Click a star or use Explore.<br>Drag to orbit · scroll to travel.`;$('worlds').innerHTML='';}
 else if(sys){const st=sys.star;$('details').innerHTML=`<b>${st.mass.toFixed(3)} M☉</b> · ${Math.round(st.temperature)} K<br>${st.luminosity.toPrecision(3)} L☉ · age ${st.age.toFixed(3)} Gyr<br>Metallicity Z ${st.Z.toFixed(4)}<br>System year ${year.toFixed(2)}<br><span style="color:#6d8b98">AU orbits; body sizes exaggerated</span>`;
  const wk=sys.id+'/'+(planet?planet.id:'');if(worldListKey!==wk){worldListKey=wk;
  $('worlds').innerHTML=sys.planets.map((p,i)=>`<button data-world="${i}" class="${planet===p?'on':''}">${p.name} · ${p.look.type}<br>${p.a.toPrecision(3)} AU · ${(p.m/3e-6).toFixed(2)} M⊕</button>`).join('');$('worlds').querySelectorAll('button').forEach(b=>b.onclick=()=>openPlanet(sys.planets[+b.dataset.world]));}
  if(planet){$('details').innerHTML+=`<br><b>${planet.name}</b> · ${(planet.a/sys.scale>=.95&&planet.a/sys.scale<=1.67)?'in habitable zone':'outside habitable zone'}<br>e ${planet.e.toFixed(3)} · ${planet.moons.length} moon(s)`;}
 }
 const ek=seed+'/'+u.events.map(e=>e.id+e.t).join(',');if(eventListKey!==ek){eventListKey=ek;
 $('eventList').innerHTML='<div class="eyebrow">LIVE EVENTS</div>'+u.events.slice(-4).reverse().map((e,i)=>`<button data-event="${u.events.length-1-i}">${e.type} · ${e.id}<br>${e.t.toFixed(1)} Myr</button>`).join('');$('eventList').querySelectorAll('button').forEach(b=>b.onclick=()=>focusEvent(u.events[+b.dataset.event]));}
}
const eventsSeen=new Set();let effectMeshes=[];
function drawEvents(now){for(const e of u.events){const key=e.id+e.t+e.type;if(eventsSeen.has(key))continue;eventsSeen.add(key);if(eventsSeen.size>128)eventsSeen.delete(eventsSeen.values().next().value);const color=e.type==='supernova'?0xffb75d:e.type==='GRB'?0x85dfff:0xad88ff;let m=new THREE.Mesh(new THREE.SphereGeometry(1,24,12),new THREE.MeshBasicMaterial({color,wireframe:true,transparent:true,opacity:.8,depthWrite:false}));m.position.set(...e.p);fxGroup.add(m);effectMeshes.push({m,born:now});}
 for(const fx of effectMeshes){const age=(now-fx.born)/1000;fx.m.scale.setScalar(.1+age*.8);fx.m.material.opacity=Math.max(0,.8-age*.1);}
 effectMeshes=effectMeshes.filter(fx=>{if(now-fx.born<8000)return true;fxGroup.remove(fx.m);fx.m.geometry.dispose();fx.m.material.dispose();return false;});}
function focusEvent(e){showGalaxy();controls.target.set(...e.p);camera.position.copy(controls.target).add(new V3(0,5,9));}
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();ray.params.Points.threshold=.3;let down;
renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);renderer.domElement.addEventListener('click',e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;mouse.set(e.clientX/innerWidth*2-1,-e.clientY/innerHeight*2+1);ray.setFromCamera(mouse,camera);if(mode==='galaxy'){const hit=ray.intersectObject(gPoints);const h=hit.find(h=>u.bodies[h.index].kind===1);if(h)openSystem(u.bodies[h.index]);}else if(mode==='system'){const h=ray.intersectObjects(systemBodies.map(g=>g.userData.core));if(h.length)openPlanet(h[0].object.userData.planet);}});
$('galaxy').onclick=showGalaxy;$('system').onclick=showSystem;$('planet').onclick=()=>planet&&openPlanet(planet);
$('explore').onclick=()=>{const stars=u.bodies.filter(b=>b.kind===1&&b.smass>.65&&b.smass<1.4);openSystem(stars[hash(seed,'explore')%stars.length]||u.bodies.find(b=>b.kind===1));};
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'▶ Resume':'Ⅱ Pause';};$('speed').onchange=()=>speed=+$('speed').value;$('quantum').onchange=()=>quantum=$('quantum').checked;
$('generate').onclick=()=>seedUniverse(Number($('seed').value));$('reset').onclick=()=>seedUniverse(seed);$('full').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(e=>$('status').textContent='Fullscreen unavailable: '+e.message);};
$('land').onclick=()=>{$('status').textContent='Live surface adapter is initializing…';};$('legacy').onclick=()=>{$('explore').click();};
addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();gPoints.material.uniforms.uScale.value=innerHeight;});
window.NIM={get universe(){return u},get system(){return sys},get planet(){return planet},get mode(){return mode},openSystem,openPlanet,showGalaxy,showSystem,focusEvent,lifeSpec,stats:()=>({fps,frames:frameTimes.length,gravitySteps:steps})};
seedUniverse(4);
function animate(now){requestAnimationFrame(animate);const dt=Math.min(.06,(now-lastFrame)/1000);lastFrame=now;frameTimes.push(dt);if(frameTimes.length>120)frameTimes.shift();
 if(!paused){year+=dt*.12*speed;if(now-lastG>80/speed){u.step(.003,.35);steps++;lastG=now;const pos=gGeo.getAttribute('position'),col=gGeo.getAttribute('color');for(let i=0;i<u.bodies.length;i++){let b=u.bodies[i];pos.setXYZ(i,...b.p);if(b.kind!==1&&b.kind!==0&&b.kind!==4)col.setXYZ(i,...(b.kind===9?[0,0,0]:b.kind===5?[.2,.1,.3]:[.5,.8,1]));}pos.needsUpdate=true;col.needsUpdate=true;updatePanel();}}
 for(const g of systemBodies)g.position.set(...position(g.userData.planet,sys.star.mass,year).map(v=>v/sys.scale));
 for(const g of sysGroup.children)if(g.userData.moon){const h=g.userData.host,m=g.userData.moon,ph=m.phase+year*TAU*Math.sqrt(h.userData.planet.m/m.a**3);g.position.copy(h.position).add(new V3(Math.cos(ph),0,Math.sin(ph)).multiplyScalar(Math.max(.08,m.a/sys.scale*3)));}
 if(worldMesh)worldMesh.rotation.y+=dt*.04;
 drawEvents(now);controls.update();renderer.render(scene,camera);
 if(Math.round(now)%500<20){fps=1/(frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length);$('stats').textContent=`${fps.toFixed(0)} FPS · ${mode.toUpperCase()} · ${quantum?'QUANTUM DRIVE ON':'CLASSICAL ONLY'}`;}
}
requestAnimationFrame(animate);$('status').textContent='Live generation · entirely local · no backend';
