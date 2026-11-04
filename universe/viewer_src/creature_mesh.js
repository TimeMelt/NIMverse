// NIM shared creature builder + lighting pass (used by life.js AND life_live.js; viewer-only, sim untouched).
// Stylized-realistic, still procedural + trait-driven: smooth lathe bodies, jointed limbs, eyes/mouth, plated shells,
// plan-specific materials (wet swimmers, dusty crawlers, membranous flyers, translucent floaters), secondary motion, blob contact shadows.
window.NIMC=(function(){
const V3=THREE.Vector3, TAU=Math.PI*2, col=a=>new THREE.Color(...a);
let GT=null, ST=null, LOOK=((typeof location!=='undefined'&&new URLSearchParams(location.search).get('look'))==='photo')?'photo':'stylized', Q=1, MO={}, DT={};
// morph defaults (old reports without c.morph render exactly as Phase A-ish proportions)
const MD={elong:1,girth:1,head:1,eye:1,fin:1,seg_space:1,seg_taper:.05,wing_span:1,wing_aspect:1,tail:1,shell_ridges:.5,shell_dome:1,spines:1,tent_len:1,bell_flat:1,stripe:0,stripe_freq:6,spot:0,skin_detail:.5};
const q=n=>Math.max(3,Math.round(n*Q));
// --- photo mode: procedural detail maps generated on the fly in a canvas (height field -> tangent-space normal + roughness) ---
function detailTex(kind){if(DT[kind]) return DT[kind]; const N=256, h=new Float32Array(N*N), hs=(x,y)=>{const v=Math.sin(x*127.1+y*311.7)*43758.5453; return v-Math.floor(v);};
 const vn=(x,y,f)=>{x*=f; y*=f; const xi=Math.floor(x), yi=Math.floor(y), xf=x-xi, yf=y-yi, w=t=>t*t*(3-2*t), m=f|0, H=(a,b)=>hs(((a%m)+m)%m,((b%m)+m)%m);
  return H(xi,yi)+(H(xi+1,yi)-H(xi,yi))*w(xf)+(H(xi,yi+1)-H(xi,yi))*w(yf)+(H(xi,yi)-H(xi+1,yi)-H(xi,yi+1)+H(xi+1,yi+1))*w(xf)*w(yf);};
 for(let y=0;y<N;y++) for(let x=0;x<N;x++){const u=x/N, v=y/N; let z=0;
  if(kind==='scale'){const f=14, cx=u*f, cy=v*f+((Math.floor(u*f)%2)*.5), dx=cx-Math.floor(cx)-.5, dy=cy-Math.floor(cy)-.5; z=.75*Math.max(0,1-Math.sqrt(dx*dx+dy*dy)*1.9)+.25*vn(u,v,32);}
  else if(kind==='vein'){z=Math.abs(Math.sin((u*9+.6*vn(u,v,6))*Math.PI))*.35+Math.pow(1-Math.abs(Math.sin((v*5+.4*vn(u,v,4))*Math.PI)),6)*.5+.15*vn(u,v,48);}
  else {z=.5*vn(u,v,8)+.3*vn(u,v,24)+.2*vn(u,v,64);}   // 'skin': fine pores / wrinkles
  h[y*N+x]=z;}
 const mk=f=>{const c=document.createElement('canvas'); c.width=c.height=N; const g=c.getContext('2d'), im=g.createImageData(N,N); for(let i=0;i<N*N;i++){const o=f(i%N,(i/N)|0); im.data.set([o[0],o[1],o[2],255],4*i);} g.putImageData(im,0,0);
  const t=new THREE.CanvasTexture(c); t.wrapS=t.wrapT=THREE.RepeatWrapping; return t;};
 const H=(x,y)=>h[((y+N)%N)*N+((x+N)%N)], s=kind==='scale'?5:3;
 const nrm=mk((x,y)=>{const dx=(H(x+1,y)-H(x-1,y))*s, dy=(H(x,y+1)-H(x,y-1))*s, l=Math.hypot(dx,dy,1); return [128-127*dx/l,128-127*dy/l,128+127/l];});
 const rgh=mk((x,y)=>{const v=200-120*H(x,y); return [v,v,v];});
 const rp={scale:[3,3],vein:[2,2],skin:[4,2]}[kind]; [nrm,rgh].forEach(t=>t.repeat.set(...rp)); return DT[kind]={nrm,rgh};}
function radTex(stops){const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d'), gr=g.createRadialGradient(64,64,0,64,64,64); stops.forEach(([o,s])=>gr.addColorStop(o,s)); g.fillStyle=gr; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c);}
function glowTex(){return GT||(GT=radTex([[0,'rgba(255,255,255,1)'],[.18,'rgba(255,255,255,.55)'],[.45,'rgba(255,255,255,.12)'],[1,'rgba(0,0,0,0)']]));}
function shadowTex(){return ST||(ST=radTex([[0,'rgba(0,0,0,.85)'],[.5,'rgba(0,0,0,.45)'],[1,'rgba(0,0,0,0)']]));}
function spr(c,s,op=1){const m=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTex(),color:c,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true,opacity:op})); m.scale.set(s,s,1); return m;}
// --- geometry helpers ---
// lathe along +Z from a radius profile r(u), u in [0,1] tail->nose; belly/back countershading via vertex colours
function spindle(len,rad,prof,seg=18,rs=14,flatY=1,base,belly){const pts=[]; for(let i=0;i<=seg;i++){const u=i/seg; pts.push(new THREE.Vector2(Math.max(1e-3,rad*prof(u)),(u-.5)*len));}
 const g=new THREE.LatheGeometry(pts,rs); g.rotateX(Math.PI/2); g.scale(1,flatY,1); shade(g,base,belly); return g;}
function shade(g,base,belly,dark=.55){const p=g.attributes.position, n=p.count, c=new Float32Array(n*3); g.computeBoundingBox(); const y0=g.boundingBox.min.y, y1=g.boundingBox.max.y;
 for(let i=0;i<n;i++){const t=(p.getY(i)-y0)/Math.max(1e-6,y1-y0), k=THREE.MathUtils.smoothstep(t,.25,.75); const f=dark+(1-dark)*k; // back darker -> lighter? countershade: back=base*f, belly=belly colour
  let pf=1; if(MO.stripe>0&&k>.3){const z=p.getZ(i)+(g.userData.z0||0); pf-=.38*MO.stripe*THREE.MathUtils.smoothstep(Math.sin(z*MO.stripe_freq*1.3+p.getX(i)*1.5),.15,.55)*k;}   // dorsal banding
  if(MO.spot>0&&k>.35){const v=Math.sin(p.getX(i)*23.1+p.getZ(i)*17.3)*Math.sin(p.getY(i)*19.7-p.getZ(i)*11.9); if(v>.55) pf-=.45*MO.spot*k;}     // mottling
  const r=(belly[0]+(base[0]*f-belly[0])*k)*pf, gg=(belly[1]+(base[1]*f-belly[1])*k)*pf, b=(belly[2]+(base[2]*f-belly[2])*k)*pf; c.set([r,gg,b],3*i);}
 g.setAttribute('color',new THREE.BufferAttribute(c,3)); g.computeVertexNormals(); return g;}
function flatShape(pts,thick=0){const s=new THREE.Shape(); s.moveTo(pts[0][0],pts[0][1]); for(let i=1;i<pts.length;i+=2){const a=pts[i], b=pts[i+1]||pts[0]; s.quadraticCurveTo(a[0],a[1],b[0],b[1]);}
 return thick>0?new THREE.ExtrudeGeometry(s,{depth:thick,bevelEnabled:true,bevelThickness:thick*.5,bevelSize:thick*.6,bevelSegments:1,curveSegments:8}):new THREE.ShapeGeometry(s,8);}
const tube=(pts,r,ts=10,rs=6)=>new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(p=>new V3(...p))),ts,r,rs,false);
function taper(len,r0,r1,rs=8){const g=new THREE.CylinderGeometry(r1,r0,len,rs,1); g.translate(0,-len/2,0); return g;}   // hangs down from origin (joint pivot)
// --- materials by plan ---
function mats(c){const glow=c.glow>.05, A=col(c.accent), base=c.color, plan=c.plan;
 const lum=(a,k)=>a.map(v=>Math.min(1,v*k)), belly=plan==='swimmer'?lum(base.map((v,i)=>.5*v+.5*[.85,.88,.9][i]),1.15):lum(base,1.25);
 const P={swimmer:{roughness:.22,metalness:.05,clearcoat:.9,clearcoatRoughness:.15},crawler:{roughness:.88,metalness:0,clearcoat:0},flyer:{roughness:.6,metalness:0,clearcoat:.2,clearcoatRoughness:.5},floater:{roughness:.15,metalness:0,clearcoat:1,clearcoatRoughness:.05}}[plan]||{roughness:.6};
 const em=glow?{emissive:A,emissiveIntensity:.25*c.glow}:{};
 const skin=new THREE.MeshPhysicalMaterial(Object.assign({color:0xffffff,vertexColors:true},P,em));
 const acc=new THREE.MeshPhysicalMaterial(Object.assign({color:A,emissive:A,emissiveIntensity:glow?1.4*c.glow:.08},P,{roughness:Math.min(1,(P.roughness||.6)+.1)}));
 const memb=new THREE.MeshPhysicalMaterial({color:A.clone().lerp(col(base),.35),emissive:A,emissiveIntensity:glow?.6*c.glow:.04,roughness:.5,transparent:true,opacity:plan==='flyer'?.92:.75,side:THREE.DoubleSide,clearcoat:plan==='swimmer'?.8:0});
 const shell=new THREE.MeshPhysicalMaterial({color:col(c.accent.map((v,i)=>.45*v+.25*base[i])),roughness:plan==='crawler'?.55:.3,metalness:.15,clearcoat:.7,clearcoatRoughness:.3});
 const eye=new THREE.MeshPhysicalMaterial({color:0x0a0806,roughness:.05,clearcoat:1,clearcoatRoughness:0,emissive:glow?A:0x000000,emissiveIntensity:glow?.5:0});
 const sclera=new THREE.MeshStandardMaterial({color:0xe8e0cc,roughness:.3});
 const mouth=new THREE.MeshStandardMaterial({color:0x1a0806,roughness:.7});
 const photo=new THREE.MeshBasicMaterial({color:A.clone().multiplyScalar(1.6)});
 if(LOOK==='photo'){const d=MO.skin_detail, sk=detailTex(plan==='crawler'?'scale':'skin'), vn=detailTex('vein'), sc=detailTex('scale'), rep=(t,n)=>{t.repeat.set(n,n); return t;};
  // skin: micro-normal + roughness variation; wet swimmers/floaters get a clearcoat that ignores the pores (smooth mucus layer), dry crawlers/flyers get sheen (fuzz/dust)
  Object.assign(skin,{normalMap:sk.nrm,normalScale:new THREE.Vector2(plan==='crawler'?.25+.4*d:.5+1.1*d,plan==='crawler'?.25+.4*d:.5+1.1*d),roughnessMap:sk.rgh,roughness:Math.min(1,(P.roughness||.6)*1.15)});
  if(plan==='crawler'||plan==='flyer'){skin.sheen=col(base).lerp(new THREE.Color(1,1,1),.5).multiplyScalar(.5);} else {skin.clearcoat=1; skin.clearcoatRoughness=.08;}
  if(glow){skin.emissiveIntensity=.45*c.glow;}   // brighter subsurface-ish glow
  Object.assign(memb,{normalMap:vn.nrm,normalScale:new THREE.Vector2(.6,.6),roughnessMap:vn.rgh,sheen:A.clone().multiplyScalar(.6),emissiveIntensity:glow?.9*c.glow:.12,opacity:plan==='flyer'?.85:.62});
  Object.assign(shell,{normalMap:sc.nrm,normalScale:new THREE.Vector2(.55,.55),roughnessMap:sc.rgh,clearcoat:1,clearcoatRoughness:.12,metalness:.25});
  Object.assign(acc,{normalMap:sk.nrm,normalScale:new THREE.Vector2(.4,.4),clearcoat:.6,clearcoatRoughness:.2});
  [skin,memb,shell,acc].forEach(m=>{m.needsUpdate=true;});
  eye.envMapIntensity=2.5; photo.color.multiplyScalar(1.4);}
 return {skin,acc,memb,shell,eye,sclera,mouth,photo,belly};}
// --- builder: returns {body, limbs, shadow, tris}; caller adds body to its group g (shadow is a child of g, not body: not pickable) ---
function build(c){Q=LOOK==='photo'?1.9:1; MO=Object.assign({},MD,c.morph||{}); const m_=MO; const body=new THREE.Group(), limbs=[], M=mats(c), base=c.color, glow=c.glow>.05, L=Math.max(0,c.limbs|0), S=Math.max(1,c.segments|0);
 const add=(geo,m,par=body,pos=[0,0,0],rot=[0,0,0],sc=[1,1,1])=>{const o=new THREE.Mesh(geo,m); o.position.set(...pos); o.rotation.set(...rot); o.scale.set(...sc); par.add(o); return o;};
 const grp=(par,pos=[0,0,0],rot=[0,0,0])=>{const q=new THREE.Group(); q.position.set(...pos); q.rotation.set(...rot); par.add(q); return q;};
 const eyes=(x,y,z,r,par=body)=>{r*=m_.eye; for(const s of [-1,1]){add(new THREE.SphereGeometry(r,q(12),q(8)),M.sclera,par,[s*x,y,z]); add(new THREE.SphereGeometry(r*.62,q(10),q(8)),M.eye,par,[s*(x+r*.28),y+r*.12,z+r*.5]);}};
 const photophores=(n,len,zc,y,x)=>{if(!glow) return; for(let i=0;i<n;i++) for(const s of [-1,1]) add(new THREE.SphereGeometry(.06,6,4),M.photo,body,[s*x*(1-.4*Math.abs(i/n-.5)),y,zc+len*(i/n-.5)]);};
 if(c.plan==='swimmer'){
  // fusiform body: head/front half rigid, rear split into 3 spine links for a travelling wave
  const E=m_.elong, G=m_.girth, R0=.72*G, LL=.62*E;   // elong stretches the body/spine, girth fattens it
  const front=add(spindle(2.2*E,R0,u=>Math.pow(Math.sin(Math.PI*Math.min(1,.5+u*.5)),.6),q(16),q(16),.82,base,M.belly),M.skin,body,[0,0,.55*E]);
  let par=body, z=-.55*E; const links=[];
  for(let i=0;i<3;i++){const j=grp(par,[0,0,i?-LL:z]); const r0=R0*Math.pow(.72,i), r1=R0*Math.pow(.72,i+1);
   add(spindle(.7*E,1,u=>r1+(r0-r1)*u,q(6),q(16),.82,base,M.belly),M.skin,j,[0,0,-.3*E]); links.push(j); par=j;}
  const fl=grp(par,[0,0,-LL]); fl.scale.setScalar(m_.tail); add(flatShape([[0,0],[.25,-.4],[.9,-.85],[.3,-.55],[0,-.35],[-.3,-.55],[-.9,-.85],[-.25,-.4],[0,0]],.03),M.memb,fl,[0,0,0],[Math.PI/2,0,0]); // forked fluke (vertical-ish)
  fl.rotation.z=Math.PI/2;
  links.forEach((o,i)=>limbs.push({o,k:'spine',i})); limbs.push({o:fl,k:'fluke'});
  add(flatShape([[0,0],[-.2,.5],[-.7,.55],[-.45,.2],[-.8,0]],.02),M.memb,body,[0,.5*G,.1],[0,Math.PI/2,0],[1,m_.fin,m_.fin]);   // dorsal fin (fin morph)
  for(let i=0;i<L;i++){const s=i%2?1:-1, row=Math.floor(i/2), j=grp(body,[s*.58*G,-.22,(.75-.55*row)*E],[0,0,0]);
   add(flatShape([[0,0],[.35,.1],[.85,-.25],[.5,-.3],[0,-.18]],.015),M.memb,j,[0,0,0],[Math.PI/2,0,s<0?Math.PI:0],[(1-.15*row)*m_.fin,m_.fin,1]); limbs.push({o:j,k:'fin',s,i:row});}
  eyes(.42*G,.18,1.25*E,.13); add(new THREE.TorusGeometry(.16,.03,6,12,Math.PI),M.mouth,body,[0,-.12,1.62*E],[0,0,Math.PI],[1,.6,1]);
  for(const s of [-1,1]) for(let g=0;g<3;g++) add(new THREE.TorusGeometry(.32,.012,4,10,Math.PI*.6),M.mouth,body,[s*.55,0,.85-.1*g],[0,s*Math.PI/2,-.9],[1,1,1]);  // gill slits
  photophores(6,1.8,.3,-.32,.5);}
 else if(c.plan==='crawler'){
  const segs=[]; let par=body; const seglen=.95*m_.seg_space, G=m_.girth;
  for(let i=0;i<S;i++){const r=(.72-m_.seg_taper*i)*(i?1:m_.head)*G, j=grp(par,[0,0,i?-seglen:1.0]); add(new THREE.SphereGeometry(r,q(18),q(12)),M.skin,j,[0,.08,0],[0,0,0],[1.05,.7,.95*m_.elong]).geometry=shade(new THREE.SphereGeometry(r,q(18),q(12)),base,M.belly,.6);
   if(i>0) add(new THREE.CylinderGeometry(r*.75,r*.75,.35*m_.seg_space,q(14)),M.skin,j,[0,.06,seglen*.5],[Math.PI/2,0,0],[1.05,.7,1]).geometry=shade(new THREE.CylinderGeometry(r*.75,r*.75,.35*m_.seg_space,q(14)),base,M.belly,.5);  // soft intersegment membrane
   segs.push(j); limbs.push({o:j,k:'seg',i}); par=j;}
  const head=segs[0]; eyes(.28,.38,.52,.12,head);
  for(const s of [-1,1]){add(tube([[s*.15,-.05,.6],[s*.32,-.2,.95],[s*.12,-.28,1.15]],.045,8,5),M.acc,head);             // mandibles
   const an=grp(head,[s*.15,.45,.55]); an.scale.setScalar(m_.fin); add(tube([[0,0,0],[s*.15,.33,.25],[s*.33,.5,.65],[s*.45,.45,1.0]],.014,14,4),M.acc,an); limbs.push({o:an,k:'ant',s});}
  for(let i=0;i<L;i++){const s=i%2?1:-1, pair=Math.floor(i/2), np=Math.max(1,Math.ceil(L/2)); const si=Math.min(S-1,Math.floor(pair*S/np)), seg=segs[si];
   const hip=grp(seg,[s*.62*G,-.12,(np>S?.25-.5*((pair*S/np)%1):0)]), r=.11;
   add(new THREE.SphereGeometry(r*1.25,10,8),M.skin,hip).geometry=shade(new THREE.SphereGeometry(r*1.25,10,8),base,M.belly,.6);
   const fem=grp(hip,[0,0,0],[0,0,s*1.1]); add(taper(.85,r,r*.8),M.skin,fem).geometry=shade(taper(.85,r,r*.8),base,M.belly,.7);
   const knee=grp(fem,[0,-.85,0],[0,0,-s*1.7]); add(new THREE.SphereGeometry(r*.85,8,6),M.acc,knee); add(taper(1.0,r*.8,r*.3),M.skin,knee).geometry=shade(taper(1.0,r*.8,r*.3),base,M.belly,.7);
   limbs.push({o:fem,k:'leg',s,i:pair,knee,kr:-s*1.7});}
  if(glow) for(const sg of segs) for(const s of [-1,1]) add(new THREE.SphereGeometry(.07,6,4),M.photo,sg,[s*.55,.1,0]);}
 else if(c.plan==='flyer'){
  const E=m_.elong, G=m_.girth, WS=m_.wing_span, WA=m_.wing_aspect;   // aspect>1: long narrow (albatross); <1: broad (bat/butterfly)
  add(spindle(2.0*E,.5*G,u=>Math.pow(Math.sin(Math.PI*u),.7)*(u>.7?1:.6+.4*u/.7),q(16),q(14),.9,base,M.belly),M.skin,body);
  add(new THREE.SphereGeometry(.36*m_.head,q(16),q(12)),M.skin,body,[0,.12,1.05*E]).geometry=shade(new THREE.SphereGeometry(.36*m_.head,q(16),q(12)),base,M.belly,.6);
  add(new THREE.ConeGeometry(.09,.45,10),M.acc,body,[0,.05,1.05*E+.45*m_.head],[Math.PI/2,0,0]);   // beak
  eyes(.22*m_.head,.24,1.05*E+.2*m_.head,.09);
  for(const s of [-1,1]){const sh=grp(body,[s*.32*G,.18,.25]); const inner=new THREE.Group(); inner.scale.set(WS*Math.sqrt(WA),1,1/Math.sqrt(WA)); sh.add(inner);
   add(flatShape([[0,.35],[.6,.5],[1.2,.35],[1.25,-.05],[1.2,-.3],[.5,-.45],[0,-.3]],.02),M.memb,inner,[0,0,0],[Math.PI/2,0,0],[s,1,1]);
   const el=grp(inner,[s*1.2,0,0]); add(flatShape([[0,.35],[.7,.45],[1.5,-.1],[.8,-.15],[0,-.3]],.015),M.memb,el,[0,0,0],[Math.PI/2,0,0],[s,1,1]);
   add(tube([[0,0,.3],[s*.6,0,.42],[s*1.2,0,.36]],.028,8,5),M.acc,inner); add(tube([[0,0,.36],[s*.7,0,.4],[s*1.45,0,-.08]],.02,8,5),M.acc,el);   // leading-edge bone
   limbs.push({o:sh,k:'wing',s,el});}
  const tf=grp(body,[0,0,-.95*E]); tf.scale.setScalar(m_.tail); add(flatShape([[0,0],[-.45,-.5],[-.5,-.9],[0,-.7],[.5,-.9],[.45,-.5],[0,0]],.015),M.memb,tf,[0,0,0],[Math.PI/2,0,0]); limbs.push({o:tf,k:'tailfan'});
  for(const s of [-1,1]) add(tube([[s*.15,-.3,.2],[s*.2,-.6,0],[s*.2,-.7,-.2]],.035,6,5),M.acc,body);   // tucked legs
  photophores(4,1.2,0,-.25,.35);}
 else {  // floater: translucent lathe bell with frilled rim, glowing core, jointed trailing tentacles + oral arms
  const prof=[]; for(let i=0;i<=16;i++){const u=i/16, a=u*Math.PI*.55; prof.push(new THREE.Vector2(Math.max(1e-3,Math.sin(a)*(1+.08*Math.sin(u*9))),(.9*Math.cos(a)-.1)/m_.bell_flat));}
  for(let i=0;i<=4;i++){const u=i/4; prof.push(new THREE.Vector2(Math.sin(Math.PI*.55)*(1-.12*u),.9*Math.cos(Math.PI*.55)-.1-.06*u));}
  const bg=new THREE.LatheGeometry(prof,q(28)); shade(bg,base,c.accent,.9);
  const bellM=new THREE.MeshPhysicalMaterial({color:0xffffff,vertexColors:true,emissive:col(c.accent),emissiveIntensity:.18+.6*c.glow,transparent:true,opacity:.55,roughness:.08,clearcoat:1,side:THREE.DoubleSide,depthWrite:false});
  if(LOOK==='photo'){const vt=detailTex('vein'); Object.assign(bellM,{normalMap:vt.nrm,normalScale:new THREE.Vector2(.8,.8),emissiveIntensity:.35+1.0*c.glow,opacity:.42,clearcoatRoughness:.02,sheen:col(c.accent).multiplyScalar(.8)});}
  const bell=add(bg,bellM,body,[0,.6,0]); limbs.push({o:bell,k:'bell'});
  add(new THREE.SphereGeometry(.42,16,12),new THREE.MeshPhysicalMaterial({color:col(c.accent),emissive:col(c.accent),emissiveIntensity:.5+1.6*c.glow,roughness:.3}),bell,[0,.35,0],[0,0,0],[1,.7,1]);   // gonad/core
  const nt=6+S;
  for(let i=0;i<nt;i++){const a=i/nt*TAU; let par=grp(bell,[.92*Math.cos(a),-.2,.92*Math.sin(a)]); const chain=[];
   const TL=.5*m_.tent_len; for(let k=0;k<4;k++){const j=grp(par,k?[0,-TL,0]:[0,0,0]); add(taper(TL+.02,.045*(1-.2*k),.045*(1-.2*(k+1))+.005,5),M.acc,j); chain.push(j); par=j;}
   limbs.push({o:chain[0],k:'tent',a,chain});}
  for(let i=0;i<4;i++){const a=i/4*TAU+.4; add(tube([[0,-.1,0],[.15*Math.cos(a),-.6,.15*Math.sin(a)],[.3*Math.cos(a),-1.1,.25*Math.sin(a)],[.15*Math.cos(a),-1.5,.1*Math.sin(a)]],.07,12,6),M.memb,bell);}}   // oral arms
 if(c.shell){const k=c.shell_k||.5;
  if(c.plan==='crawler'){body.traverse(()=>{}); let par=body; const segs=limbs.filter(l=>l.k==='seg').map(l=>l.o);
   segs.forEach((sg,i)=>{const r=(.74-m_.seg_taper*i)*(i?1:m_.head)*m_.girth; add(new THREE.SphereGeometry(r,q(18),q(8),0,TAU,0,1.15),M.shell,sg,[0,.16,-.05],[-.12,0,0],[1.12,.72*m_.shell_dome,1.05*m_.elong]);
    const nrd=1+Math.round(3*m_.shell_ridges); for(let rr=0;rr<nrd;rr++)
    add(new THREE.TorusGeometry(r*.92,.03,5,q(20),Math.PI),M.shell,sg,[0,.12,-.05+(nrd>1?(rr/(nrd-1)-.5)*.5*r:0)],[0,Math.PI/2,0],[1,.9*m_.shell_dome,1.2]);   // plate ridges (shell_ridges)
    const ns=Math.round(1+2*k); for(let q=0;q<ns;q++){const s=ns>1?(q/(ns-1)-.5)*.9:0; add(new THREE.ConeGeometry(.07+.06*k,(.22+.35*k)*m_.spines,10),M.acc,sg,[s*r,.16+.62*r*Math.cos(s),-.05],[-.35,0,-s*1.1]);}});}
  else if(c.plan==='floater'){const bell=limbs.find(l=>l.k==='bell').o, nr=Math.round(8+8*k); for(let i=0;i<nr;i++){const a=i/nr*TAU; add(new THREE.ConeGeometry(.04+.03*k,.12+.2*k,8),M.shell,bell,[.97*Math.cos(a)*.98,.05,.97*Math.sin(a)*.98],[Math.sin(a)*.5,0,-Math.cos(a)*.5]);}   // stiff rim teeth
   add(new THREE.TorusGeometry(.97,.035,6,40),M.shell,bell,[0,-.12,0],[Math.PI/2,0,0]);}
  else {const fl=c.plan==='flyer', D=m_.shell_dome, E=m_.elong, G=m_.girth; add(new THREE.SphereGeometry(1,q(24),q(10),0,TAU,0,1.25),M.shell,body,[0,fl?.12:.1,fl?.0:.15],[0,0,0],fl?[.42*G,.4*D,.85*E]:[.66*G,.52*D,1.1*E]);   // dorsal carapace
   const nrd=Math.round(1+4*m_.shell_ridges); for(let rr=0;rr<nrd;rr++){const t=(rr+.5)/nrd; add(new THREE.TorusGeometry(1,.022,4,q(16),Math.PI),M.shell,body,[0,fl?.12:.1,((fl?.85:1.1)*(1-2*t)*.8)*E+(fl?0:.15)],[0,Math.PI/2,0],[1,(fl?.4:.52)*D*Math.sqrt(1-Math.pow(1-2*t,2)*.64)*1.02,(fl?.42:.66)*G*Math.sqrt(1-Math.pow(1-2*t,2)*.64)*1.02]);}   // transverse carapace ridges
   const nr=Math.round(3+4*k); for(let i=0;i<nr;i++){const t=i/(nr-1||1); add(new THREE.ConeGeometry(.04+.04*k,(.12+.25*k)*m_.spines,10),M.acc,body,[0,((fl?.48:.55)+.08*Math.sin(Math.PI*t))*D,((fl?.7:.9)-(fl?1.3:1.6)*t)*E+(fl?0:.0)],[-.4,0,0]);}}}
 if(glow){const s=spr(col(c.accent),3.2*c.glow+1.2,.55+.4*c.glow); s.position.y=.3; body.add(s); const s2=spr(col(c.accent),9*c.glow+2,.18*c.glow); s2.position.y=.2; body.add(s2);}
 const shadow=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({map:shadowTex(),transparent:true,depthWrite:false,opacity:.6,polygonOffset:true,polygonOffsetFactor:-2}));
 shadow.rotation.x=-Math.PI/2; shadow.userData.noPick=true; shadow.renderOrder=-1; const sz={swimmer:[1.8,4.2],crawler:[2.2,1.4+S*1.0],flyer:[5,2.4],floater:[2.4,2.4]}[c.plan]||[2,3]; shadow.scale.set(sz[0],sz[1],1);
 let tris=0; body.traverse(o=>{if(o.isMesh){const g=o.geometry; tris+=(g.index?g.index.count:g.attributes.position.count)/3;}});
 return {body,limbs,shadow,tris};}
// --- secondary motion: phase ph (gait clock), t seconds, sp speed ---
function animate(cr,ph,t){const c=cr.c, sw=.35+.4*Math.min(1,c.speed||.5);
 for(const L of cr.limbs){const o=L.o;
  switch(L.k){
   case 'spine': o.rotation.y=sw*.45*(L.i+1)*.55*Math.sin(ph-1.0*(L.i+1)); break;                          // travelling body wave, amplitude grows to tail
   case 'fluke': o.rotation.y=sw*.6*Math.sin(ph-4.0); break;
   case 'fin':   o.rotation.z=L.s*(.15+.35*Math.sin(ph*.5+L.i*.8)); o.rotation.y=L.s*.25*Math.sin(ph*.5+1.2+L.i*.8); break;
   case 'seg':   o.rotation.y=.07*Math.sin(ph-L.i*.9); o.position.y=(L.i?.02*Math.sin(ph*2-L.i*.9):o.position.y); break;  // body sway
   case 'leg':  {const g=ph*1.0+(L.i%2?Math.PI:0)+(L.s>0?Math.PI:0)+L.i*.35, sw2=Math.sin(g), lift=Math.max(0,Math.cos(g));   // alternating tripod-ish gait with stance/swing
                 o.rotation.y=.45*sw2; o.rotation.z=L.s*(1.1-.35*lift); L.knee.rotation.z=L.kr+L.s*.45*lift; break;}
   case 'ant':   o.rotation.x=.12*Math.sin(t*2.3+L.s); o.rotation.y=.1*Math.sin(t*1.7+L.s*2); break;
   case 'wing': {const f=ph*1.4, a=Math.sin(f); o.rotation.z=L.s*(.15+.75*a); L.el.rotation.z=L.s*(.35*Math.sin(f-.9)); o.rotation.x=.12*Math.cos(f); break;}  // downstroke + lagging outer wing + pitch
   case 'tailfan': o.rotation.x=.15*Math.sin(ph*.7); break;
   case 'bell': {const q=ph*.9, c1=Math.pow(.5+.5*Math.sin(q),2.2);           // quick contraction, slow relax
                 o.scale.set(1-.16*c1,1+.12*c1,1-.16*c1); o.position.y=.6+.1*c1;
                 for(const T of cr.limbs) if(T.k==='tent') T.chain.forEach((j,k)=>{j.rotation.x=(.15+.25*(1-c1))*Math.sin(q-k*.8+T.a)*(k?1:.4); j.rotation.z=.18*Math.cos(q*.7-k*.7+T.a*2)*(k?1:.4);}); break;}
  }}}
// --- shadow placement: g is at altitude h on a unit planet; blob sits on the sphere below ---
function placeShadow(cr,h,scale){const s=cr.shadow; if(!s) return; const y=-(h-1)/scale; s.position.y=y+.02; const f=Math.max(0,1-(h-1)/.05); s.material.opacity=.15+.5*f; const k=1+(1-f)*.8; s.scale.z=1; s.userData.k=k;}
// --- lighting pass: tone mapping, PMREM sky env (specular/wetness), warm key + cool rim + hemisphere ---
function lights(scene,renderer,sun){renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.0;
 const key=new THREE.DirectionalLight(0xfff1dc,2.4); key.position.copy(sun); scene.add(key);
 const rim=new THREE.DirectionalLight(0x7fa8ff,.55); rim.position.copy(sun).multiplyScalar(-1).add(new V3(0,30,0)); scene.add(rim);
 scene.add(new THREE.HemisphereLight(0x9cc4ff,0x3a2c1c,.55)); scene.add(new THREE.AmbientLight(0x1a2433,.25));
 try{const env=new THREE.Scene(), gs=new THREE.SphereGeometry(10,32,16), cs=new Float32Array(gs.attributes.position.count*3);
  for(let i=0;i<gs.attributes.position.count;i++){const y=gs.attributes.position.getY(i)/10, a=Math.max(0,y); const cc=y>0?[.25+.35*a,.42+.4*a,.7+.3*a]:[.18,.14,.1]; cs.set(cc,3*i);}
  gs.setAttribute('color',new THREE.BufferAttribute(cs,3)); env.add(new THREE.Mesh(gs,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide})));
  const sd=new THREE.Mesh(new THREE.SphereGeometry(1.2,12,8),new THREE.MeshBasicMaterial({color:new THREE.Color(8,7.5,6.5)})); sd.position.copy(sun).normalize().multiplyScalar(8); env.add(sd);
  const pm=new THREE.PMREMGenerator(renderer); scene.environment=pm.fromScene(env,.04).texture; pm.dispose();}catch(e){console.warn('env',e);}
 return key;}
// --- tech structure: tapered tower + lit window bands + beacon + ground light pool ---
function tower(rnd){const g=new THREE.Group(), h=2.4+3*rnd(); const m=new THREE.MeshPhysicalMaterial({color:0x8a96a6,metalness:.75,roughness:.28,clearcoat:.5});
 const tw=new THREE.Mesh(new THREE.CylinderGeometry(.16,.36,h,10),m); tw.position.y=h/2; g.add(tw);
 const wm=new THREE.MeshBasicMaterial({color:new THREE.Color(1.6,1.15,.6)}); for(let i=1;i<4;i++){const b=new THREE.Mesh(new THREE.CylinderGeometry(.37-.2*i/4*1.0,.37-.2*i/4,.08,10,1,true),wm); b.position.y=h*i/4; b.scale.setScalar(.75+.15*(4-i)/3); g.add(b);}
 const bc=spr(new THREE.Color(1,.82,.5),5.5,1); bc.position.y=h+.3; g.add(bc); const core=spr(new THREE.Color(1,.95,.85),1.6,1); core.position.y=h+.3; g.add(core);
 const pool=spr(new THREE.Color(1,.7,.35),9,.35); pool.position.y=.2; g.add(pool); return g;}
function getLook(){return LOOK;}
function setLook(l){LOOK=l==='photo'?'photo':'stylized'; return LOOK;}
// photo-mode lighting tweak: slightly hotter exposure + brighter env reflections (caller passes renderer/scene)
function applyLookLights(renderer){renderer.toneMappingExposure=LOOK==='photo'?1.12:1.0;}
// shared HUD toggle button (top-right) + key P; onChange(look) is the viewer's rebuild hook
function lookUI(onChange){const b=document.createElement('button'); b.id='lookbtn'; b.style.cssText='position:fixed;top:8px;right:8px;z-index:20;background:#1a2638;color:#cde;border:1px solid #345;border-radius:4px;padding:3px 9px;font:12px monospace;cursor:pointer';
 const lab=()=>{b.textContent='look: '+(LOOK==='photo'?'PHOTO':'stylized')+' [P]';}; lab(); document.body.appendChild(b);
 const flip=()=>{setLook(LOOK==='photo'?'stylized':'photo'); lab(); onChange(LOOK);}; b.onclick=flip;
 addEventListener('keydown',e=>{if(e.key==='p'||e.key==='P') flip();}); return {refresh:lab};}
return {build,animate,placeShadow,lights,tower,spr,glowTex,getLook,setLook,applyLookLights,lookUI};
})();
