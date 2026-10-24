// ---------------- v4: inceptron-driven high-energy events ----------------
const EV = (function(){
 const E=D.ev, L=E.list, add=THREE.AdditiveBlending, fr=()=>frame;
 function gpAt(i,f){const fi=Math.min(Math.max(f,0)*(gNF-1)/(S.nf-1), gNF-1.0001), i0=Math.floor(fi), t=fi-i0, o0=i0*gN*3+3*i, o1=o0+gN*3;
  return new V3((gPos[o0]*(1-t)+gPos[o1]*t)*gq,(gPos[o0+1]*(1-t)+gPos[o1+1]*t)*gq,(gPos[o0+2]*(1-t)+gPos[o1+2]*t)*gq);}
 // event anchor: offset from its galaxy core at the event time, carried along with the core afterwards
 L.forEach(e=>{const c=G.cores[e.g]; e.off=new V3(...e.x).sub(gpAt(c,e.f)); e.col={SN:0xbfd8ff,TDE:0xff9ad0,MERGER:0xa8c4ff,AGN:0x9fe8ff}[e.type];});
 function pos(k,rem){const e=L[k]; if(rem&&e.j>=0) return gp(e.j); if(e.type==='SN') return gp(G.cores[e.g]).add(e.off); return gp(G.cores[e.g]).add(e.type==='SN'?e.off:(e.type==='TDE'&&e.bh>=0?gp(e.bh).sub(gp(G.cores[e.g])):new V3()));}
 const flashTex=glowTex('rgba(255,255,255,1)','rgba(200,220,255,.35)'), tdeTex=glowTex('rgba(255,240,250,1)','rgba(255,120,200,.3)');
 const shellGeo=new THREE.IcosahedronGeometry(1,5), cone=new THREE.CylinderGeometry(0,1,1,32,1,true); cone.translate(0,.5,0);
 const ringGeo=new THREE.RingGeometry(.0,1,128,1); const diskGeo=new THREE.RingGeometry(.18,1,160,8);
 const objs=[]; // per-event transient visuals
 L.forEach((e,k)=>{const o={k};
  if(e.type==='SN'){o.flash=sprite(flashTex,e.rem==='black hole'?0xd8c8ff:0xcfe2ff,1); o.shell=new THREE.Mesh(shellGeo,new THREE.ShaderMaterial({uniforms:{uAge:{value:0},uSeed:{value:new V3(k*1.7,k*3.1,k*.7)},uAlpha:{value:1}},vertexShader:VS_EV,fragmentShader:FS_SHELL,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide}));
   gS.add(o.flash,o.shell); if(e.grb){o.grb=[0,1].map(s=>{const m=new THREE.Mesh(cone,new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uCol:{value:new THREE.Color(1,.95,.8)},uAlpha:{value:1}},vertexShader:VS_EV,fragmentShader:FS_JET,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide})); if(s) m.rotation.x=Math.PI; gS.add(m); return m;}); o.gax=new V3(Math.sin(k),1,Math.cos(k*2)).normalize();}}
  else if(e.type==='TDE'){o.flash=sprite(tdeTex,0xffffff,1); gS.add(o.flash);}
  else if(e.type==='MERGER'){o.flash=sprite(flashTex,0xbfd0ff,1); o.ring=new THREE.Mesh(ringGeo,new THREE.ShaderMaterial({uniforms:{uAge:{value:0}},vertexShader:VS_EV,fragmentShader:FS_RING,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide})); gS.add(o.flash,o.ring);}
  objs.push(o);});
 // black holes (SMBHs + stellar BHs from supernovae): shadow sphere + photon-ring glow + accretion disk + jets when active
 const ringTex=glowTex('rgba(255,230,190,.0)','rgba(255,200,150,.6)');
 const bhs=E.bh.map((b,n)=>{const grp=new THREE.Group(), R=b.smbh?.06:.012; grp.userData.R=R;
  const hole=new THREE.Mesh(new THREE.SphereGeometry(1,48,32),new THREE.MeshBasicMaterial({color:0x000000})); hole.scale.setScalar(R); hole.renderOrder=2;
  const glow=sprite(ringTex,0xffd9b0,R*6);
  const disk=new THREE.Mesh(diskGeo,new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uIn:{value:.18},uAct:{value:0}},vertexShader:VS_EV,fragmentShader:FS_DISK,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide}));
  disk.scale.setScalar(R*6); const tilt=new THREE.Euler(.55+.25*Math.sin(n*2.3),.3*Math.cos(n*1.7),n*1.3); disk.rotation.copy(tilt);
  const ax=new V3(0,0,1).applyEuler(tilt); const jets=[1,-1].map(s=>{const m=new THREE.Mesh(cone,new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uCol:{value:new THREE.Color(.55,.75,1)},uAlpha:{value:1}},vertexShader:VS_EV,fragmentShader:FS_JET,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide}));
   m.quaternion.setFromUnitVectors(new V3(0,1,0),ax.clone().multiplyScalar(s)); m.scale.set(R*1.6,R*(b.smbh?90:40),R*1.6); grp.add(m); return m;});
  grp.add(glow,disk,hole); gS.add(grp); return {b,grp,disk,jets,hole,glow};});
 // pulsars: neutron-star spark + two rotating beams (spin axis tilted from the magnetic axis)
 const psr=L.filter(e=>e.type==='SN'&&e.rem==='pulsar'&&e.j>=0).map((e,n)=>{const grp=new THREE.Group(), spark=sprite(flashTex,0xbfe0ff,.05); grp.add(spark);
  const beams=[1,-1].map(s=>{const m=new THREE.Mesh(cone,new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uCol:{value:new THREE.Color(.7,.85,1)},uAlpha:{value:.9}},vertexShader:VS_EV,fragmentShader:FS_JET,blending:add,transparent:true,depthWrite:false,side:THREE.DoubleSide})); m.scale.set(.04,.9,.04); if(s<0) m.rotation.x=Math.PI; const h=new THREE.Group(); h.add(m); h.rotation.z=.5; return h;});
  const spin=new THREE.Group(); beams.forEach(b=>spin.add(b)); grp.add(spin); gS.add(grp); return {e,grp,spin,spark};});
 // remnant point sizes: stars that died shrink to a faint dot; disrupted/merged ones vanish
 const s0=Float32Array.from(gs);
 function evUpdate(){if(mode!=='galaxy') return; const f=fr(), T=tsec, sz=gGeo.getAttribute('size');
  for(let j=0;j<gN;j++){let v=s0[j]; if(E.isBH[j]) v=0; else if(f>=E.death[j]){const r=E.rem[j]; v=r===9?0:(r===7?s0[j]*.3:s0[j]*.15);} gs[j]=v;} sz.needsUpdate=true;
  const cam=camera.position, cd=cam.distanceTo(controls.target); ptsMat.uniforms.uScale.value=H()*.9*Math.min(1,Math.max(.12,cd/10)); hazeMat.uniforms.uScale.value=H()*.9*Math.min(1,Math.max(.3,cd/10)); coreSpr.forEach(c=>c.material.opacity=Math.min(1,Math.max(.08,cd/12)));
  objs.forEach(o=>{const e=L[o.k], a=f-e.f, p=pos(o.k), vis=a>=0;
   if(o.flash){const life=e.type==='TDE'?14:(e.type==='MERGER'?20:24), on=vis&&a<life; o.flash.visible=on; if(on){const k=a/life, s=(e.type==='SN'?1.6:e.type==='MERGER'?3:.3)*Math.pow(1-k,1.5)*(.4+Math.min(a*.8,1)); o.flash.position.copy(p); o.flash.scale.set(s,s,1);}}
   if(o.shell){const on=vis&&a<150; o.shell.visible=on; if(on){const k=a/150, r=.04+1.1*Math.pow(k,.45); o.shell.position.copy(gp(G.cores[e.g]).add(e.off)); o.shell.scale.setScalar(r); o.shell.material.uniforms.uAge.value=k; o.shell.material.uniforms.uAlpha.value=Math.min(1,a/2)*(1-k);}}
   if(o.grb){const on=vis&&a<6; o.grb.forEach((m,s)=>{m.visible=on; if(on){m.position.copy(p); m.quaternion.setFromUnitVectors(new V3(0,1,0),o.gax.clone().multiplyScalar(s?-1:1)); m.scale.set(.03,4*(1-a/6)+.5,.03); m.material.uniforms.uTime.value=T; m.material.uniforms.uAlpha.value=1-a/6;}});}
   if(o.ring){const on=vis&&a<40; o.ring.visible=on; if(on){o.ring.position.copy(p); o.ring.lookAt(cam); o.ring.scale.setScalar(.2+a*.25); o.ring.material.uniforms.uAge.value=a/40;}}});
  bhs.forEach(h=>{const b=h.b, on=f>=(b.born||0)&&f<b.gone&&b.j>=0; h.grp.visible=on; if(!on) return; h.grp.position.copy(gp(b.j));
   const act=(b.iv?b.iv.some(v=>f>=v[0]&&f<v[1]):f>=b.on)?1:0, d=cam.distanceTo(h.grp.position), R=h.grp.userData.R, vs=Math.max(1,d*.004/R); h.grp.scale.setScalar(vs);
   h.disk.material.uniforms.uTime.value=T; h.disk.material.uniforms.uAct.value=act; h.jets.forEach(j=>{j.visible=!!act; j.material.uniforms.uTime.value=T;});});
  psr.forEach(p=>{const on=f>=p.e.f; p.grp.visible=on; if(!on) return; p.grp.position.copy(gp(p.e.j)); const d=cam.distanceTo(p.grp.position); p.grp.scale.setScalar(Math.max(1,d*.02)); p.spin.rotation.y=T*p.e.spin*2;
   p.spin.children.forEach(h=>h.children[0].material.uniforms.uTime.value=T);});
  // timeline highlight
  if(tl.length){let cur=-1; L.forEach((e,k)=>{if(e.f<=f) cur=k;}); if(cur!==tlCur){if(tlCur>=0) tl[tlCur].classList.remove('now'); if(cur>=0) tl[cur].classList.add('now'); tlCur=cur;}}}
 // lensing post-process for the galaxy scene
 const rt=new THREE.WebGLRenderTarget(W()*renderer.getPixelRatio(),H()*renderer.getPixelRatio());
 const lu={tDiffuse:{value:rt.texture},uL:{value:Array.from({length:8},()=>new THREE.Vector4())},uNL:{value:0},uAsp:{value:W()/H()}};
 const qS=new THREE.Scene(), qC=new THREE.OrthographicCamera(-1,1,1,-1,0,1); qS.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.ShaderMaterial({uniforms:lu,vertexShader:VS_Q,fragmentShader:FS_LENS,depthTest:false,depthWrite:false})));
 addEventListener('resize',()=>{rt.setSize(W()*renderer.getPixelRatio(),H()*renderer.getPixelRatio()); lu.uAsp.value=W()/H();});
 function render(){const ls=[]; bhs.forEach(h=>{if(!h.grp.visible) return; const p=h.grp.position.clone(), d=camera.position.distanceTo(p); const s=p.clone().project(camera); if(s.z>1||Math.abs(s.x)>1.3||Math.abs(s.y)>1.3) return;
   const rw=h.grp.userData.R*h.grp.scale.x*2.6, re=rw/(d*Math.tan(camera.fov*Math.PI/360))*.5; if(re<.002) return; ls.push([s.x*.5+.5,s.y*.5+.5,Math.min(re,.35)*(H()/W())*W()/H(),1,re]);});
  ls.sort((a,b)=>b[4]-a[4]); lu.uNL.value=Math.min(ls.length,8); ls.slice(0,8).forEach((l,i)=>lu.uL.value[i].set(l[0],l[1],l[2]*W()/H()/(W()/H()),l[3]));
  if(!ls.length){renderer.setRenderTarget(null); renderer.render(gS,camera); return;}
  renderer.setRenderTarget(rt); renderer.render(gS,camera); renderer.setRenderTarget(null); renderer.render(qS,qC);}
 // ---- timeline / log panel + strip on the time bar ----
 const icon={SN:'💥',TDE:'🩸',MERGER:'〰',AGN:'🔆'}, tl=[]; let tlCur=-1;
 function label(e){if(e.type==='SN') return `${icon.SN} Supernova (${e.mass.toFixed(1)} M☉) → ${e.rem}${e.grb?' + GRB':''}${e.trig?` · triggered ${e.trig} stars`:''}`;
  if(e.type==='TDE') return `${icon.TDE} Tidal disruption of a ${e.mass.toFixed(1)} M☉ star${e.bh>=0&&!E.bh.find(b=>b.j===e.bh&&b.smbh)?' (stellar BH)':''}`;
  if(e.type==='MERGER') return `${icon.MERGER} SMBH merger: galaxies ${e.g}+${e.g2} (${(e.mass*1e8).toExponential(1)} M☉)`; return `${icon.AGN} Quasar ignites in galaxy ${e.g}`;}
 const box=document.createElement('div'); box.id='evlog'; document.body.appendChild(box);
 const c=E.counts; box.innerHTML=`<div id="evhdr">High-energy events (inceptron event node: ${E.label})<br><span style="color:#9ab;font-weight:400">${c.SN||0} supernovae · ${c.pulsar||0} pulsars · ${c.GRB||0} GRBs · ${c.TDE||0} TDEs · ${c.MERGER||0} SMBH mergers · ${c.AGN||0} quasar ignitions</span></div>`;
 L.forEach((e,k)=>{const d=document.createElement('div'); d.className='e'; d.innerHTML=`<span class="t">${(e.t_myr).toFixed(0)} Myr</span><span>${label(e)}</span>`; d.onclick=()=>go(k); box.appendChild(d); tl.push(d);});
 const strip=document.createElement('canvas'); strip.id='evstrip'; document.body.appendChild(strip);
 function drawStrip(){strip.width=strip.clientWidth; strip.height=8; const g=strip.getContext('2d'); g.clearRect(0,0,strip.width,8); L.forEach(e=>{g.fillStyle='#'+e.col.toString(16).padStart(6,'0'); g.fillRect(e.f/(S.nf-1)*strip.width-1,0,e.type==='TDE'?1:3,8);});}
 drawStrip(); addEventListener('resize',drawStrip);
 strip.onclick=ev=>{const f=ev.offsetX/strip.width*(S.nf-1); let best=0,bd=1e9; L.forEach((e,k)=>{const d=Math.abs(e.f-f); if(d<bd){bd=d;best=k;}}); go(best);};
 function dist(e){return e.type==='SN'?3.5:e.type==='MERGER'?G.R[e.g]*1.2:e.type==='AGN'?2.5:1.2;}
 function go(k){const e=L[k]; const doIt=()=>{frame=Math.max(0,e.f-2); playing=true; speed=.5; flyTo({kind:'evt',e:k}, dist(e), new V3(.35,.55,1));
   info(`<b>${label(e)}</b><br>t = ${e.t_myr.toFixed(0)} Myr · galaxy ${e.g}${e.kick!==undefined?` · kick ${e.kick.toFixed(0)} km/s`:''}<br><span style="color:#9ab">Rates set by the inceptron event node; individual events rolled with the NIM RNG.</span>`);};
  if(mode!=='galaxy') fade(()=>{setMode('galaxy'); placeCam({kind:'evt',e:k}, dist(e)*4, new V3(.35,.55,1)); doIt();}); else doIt();}
 function view(k,d,dir,df,rem){auto=false; fly=null; setMode('galaxy'); frame=L[k].f+(df||0); gFrame(frame); placeCam({kind:'evt',e:k,rem:!!rem}, d||dist(L[k]), new V3(...(dir||[.35,.55,1]))); update(); return true;}
 return {update:evUpdate, render, pos, go, view};})();
