/* Streaming browser reduction of sim2.make_galaxies + events.py.
Galaxy units: kpc, 1e8 Msun, G=1, t_unit=47.15 Myr. Stellar masses are
separate physical Msun labels; tracer weights represent stellar populations.
Never stores frame arrays. All streams are keyed by immutable IDs. */
export const TAU=2*Math.PI;
export function hash(seed,key=''){let h=(Number(seed)>>>0)^2166136261;for(const c of String(key)){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}return h>>>0;}
export class RNG{
 constructor(seed,key=''){this.s=hash(seed,key);}
 random(){let t=this.s+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;}
 uniform(a,b){return a+(b-a)*this.random();}
 normal(){return Math.sqrt(-2*Math.log(Math.max(1e-12,this.random())))*Math.cos(TAU*this.random());}
}
const norm=a=>Math.hypot(...a),add=(a,b)=>a.map((v,k)=>v+b[k]);
export const DEFAULTS={spread_mult:1,gal_mass_mult:1,gal_radius_mult:1,vel_disp:.3,disk_frac:.2,thickness:.05,arms_mean:3.5,spiral_prob:.7,imf_slope:2.35,encounter_prob:.3,halo_core:.6,planets_mean:5,spacing_mult:1,moon_dist_mult:1,binary_prob:.3,belt_prob:.6};
export const EDEFAULTS={sn_mass_min:9.5,bh_mass_min:27.5,life_exp:2.5,kick_kms:300,smbh_frac:.01025,agn_prob:.5,tde_r:.26,smbh_merge_r:1.55,grb_prob:.15};
export function lum(m){return m<2?m**4:1.4*m**3.5;}
export function strongField(M,m=1){const rs=1.974e-8*M,rt=.00465*m**.8*(M/m)**(1/3);return {r_s_AU:rs,r_tde_AU:rt,r_isco_AU:3*rs,visible:rt>rs,eta:1-Math.sqrt(8/9)};}
function node(b,ids,c,h,depth=0){
 let m=0,p=[0,0,0],soft=0;for(const i of ids){const a=b[i];m+=a.m;soft+=a.m*(a.soft||0)**2;for(let k=0;k<3;k++)p[k]+=a.m*a.p[k];}
 if(m)for(let k=0;k<3;k++)p[k]/=m;
 const n={c,h,m,p,soft:m?soft/m:0,ids,children:null};
 if(ids.length>1&&depth<24&&h>1e-10){const bins=Array.from({length:8},()=>[]);for(const i of ids){const p=b[i].p;bins[(p[0]>=c[0]?1:0)|(p[1]>=c[1]?2:0)|(p[2]>=c[2]?4:0)].push(i);}n.ids=null;n.children=bins.map((a,j)=>a.length?node(b,a,c.map((v,k)=>v+((j>>k&1)?1:-1)*h/2),h/2,depth+1):null).filter(Boolean);}
 return n;
}
export function gravity(b,theta=.55){
 if(!b.length)return [];
 let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const a of b)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],a.p[k]);hi[k]=Math.max(hi[k],a.p[k]);}
 const c=lo.map((v,k)=>(v+hi[k])/2),h=Math.max(...hi.map((v,k)=>v-lo[k]))/2+1e-6;
 const tree=theta?node(b,b.map((a,i)=>i).filter(i=>b[i].m>0),c,h):null;
 return b.map((a,i)=>{
  const out=[0,0,0];const pull=(p,m,s)=>{const d=p.map((v,k)=>v-a.p[k]),r2=d.reduce((s,v)=>s+v*v,0)+.5*((a.soft||0)**2+s);if(r2===0)return;const f=m/(r2*Math.sqrt(r2));for(let k=0;k<3;k++)out[k]+=f*d[k];};
  if(!theta){for(let j=0;j<b.length;j++)if(j!==i&&b[j].m)pull(b[j].p,b[j].m,(b[j].soft||0)**2);return out;}
  const visit=n=>{if(!n.m)return;if(n.ids){for(const j of n.ids)if(j!==i)pull(b[j].p,b[j].m,(b[j].soft||0)**2);return;}const d=norm(n.p.map((v,k)=>v-a.p[k]));const contains=a.p.every((v,k)=>Math.abs(v-n.c[k])<=n.h);if(!contains&&2*n.h/d<theta)pull(n.p,n.m,n.soft);else for(const ch of n.children)visit(ch);};visit(tree);return out;
 });
}
export function stepGravity(b,dt,theta=.55){let a=gravity(b,theta);for(let i=0;i<b.length;i++)for(let k=0;k<3;k++){b[i].v[k]+=.5*dt*a[i][k];b[i].p[k]+=dt*b[i].v[k];}a=gravity(b,theta);for(let i=0;i<b.length;i++)for(let k=0;k<3;k++)b[i].v[k]+=.5*dt*a[i][k];}
export function energy(b){let e=0;for(let i=0;i<b.length;i++){e+=.5*b[i].m*b[i].v.reduce((s,v)=>s+v*v,0);for(let j=0;j<i;j++){const r2=b[i].p.reduce((s,v,k)=>s+(v-b[j].p[k])**2,0)+.5*((b[i].soft||0)**2+(b[j].soft||0)**2);e-=b[i].m*b[j].m/Math.sqrt(r2);}}return e;}
export class Universe{
 constructor(seed=4,P={},EP={},n=256){
  this.seed=seed;this.P={...DEFAULTS,...P};this.EP={...EDEFAULTS,...EP};P=this.P;EP=this.EP;this.time=0;this.stellarMyr=0;this.events=[];this.bodies=[];this.galaxies=[];this.ejected=0;this.radiated=0;this.stepNo=0;
  for(let g=0;g<3;g++){
   const r=new RNG(seed,`galaxy/${g}`),M=r.uniform(500,2000)*P.gal_mass_mult,R=r.uniform(4,10)*P.gal_radius_mult;
   let c=[(g-1)*R*3.5,r.uniform(-R,R),r.uniform(-R,R)],v=[r.normal(),r.normal(),r.normal()].map(x=>x*P.vel_disp);
   const inc=r.uniform(-.7,.7),az=r.uniform(0,TAU);const rotate=([x,y,z])=>[x*Math.cos(az)-y*Math.sin(az),x*Math.sin(az)*Math.cos(inc)+y*Math.cos(az)*Math.cos(inc)-z*Math.sin(inc),x*Math.sin(az)*Math.sin(inc)+y*Math.cos(az)*Math.sin(inc)+z*Math.cos(inc)];
   const halo={id:`g${g}/halo`,gid:g,kind:0,m:M*(1-P.disk_frac)*(1-EP.smbh_frac),soft:P.halo_core*R,p:[...c],v:[...v],smass:0};this.bodies.push(halo);
   const bh={id:`g${g}/bh`,gid:g,kind:4,m:M*(1-P.disk_frac)*EP.smbh_frac,soft:.05,p:[...c],v:[...v],smass:0,agnUntil:0};this.bodies.push(bh);
   const spiral=r.random()<P.spiral_prob,arms=Math.max(1,Math.min(6,Math.round(P.arms_mean+r.normal()))),slope=1-P.imf_slope;
   const stars=[];for(let j=0;j<n;j++){const q=new RNG(seed,`g${g}/star/${j}`),mass=(.1**slope+q.random()*(30**slope-.1**slope))**(1/slope),rad=Math.min(-Math.log(Math.max(q.random()*q.random(),1e-9))*R/3.5,1.6*R)+.3;let th=q.uniform(0,TAU);if(spiral)th=Math.floor(q.random()*arms)*TAU/arms+2.2*Math.log(rad)+q.normal()*.3;
    const pp=rotate([rad*Math.cos(th),rad*Math.sin(th),q.normal()*P.thickness*R]);
    const vc=Math.sqrt(halo.m*rad**2/(rad**2+.5*(halo.soft**2+.12**2))**1.5+M*P.disk_frac*(1-Math.exp(-rad*3.5/R))/Math.hypot(rad,.12));
    const vv=rotate([-vc*Math.sin(th),vc*Math.cos(th),q.normal()*.03*vc]);
    const lifetime=1e4*mass**-EP.life_exp,age=q.random()*Math.min(lifetime,13000);
    stars.push({id:`g${g}/star/${j}`,gid:g,kind:1,m:mass,smass:mass,soft:.12,p:add(c,pp),v:add(v,vv),age,initialAge:age,lifetime,Z:.0142*10**q.uniform(-.8,.3),color:mass>1.6?[.65,.8,1]:mass>.7?[1,.93,.75]:[1,.45,.22]});
   }
   const total=stars.reduce((s,a)=>s+a.m,0);for(const a of stars){a.m*=M*P.disk_frac/total;this.bodies.push(a);}this.galaxies.push({R,M,c,spiral,arms,id:g});
  }
  this.mass0=this.bodies.reduce((s,a)=>s+a.m,0);
 }
 log(type,b,extra={}){this.events.push({type,id:b.id,g:b.gid,p:[...b.p],t:this.stellarMyr,...extra});if(this.events.length>64)this.events.shift();}
 step(dt=.003,eventMyr=.25,drive=null){
  stepGravity(this.bodies,dt);this.time+=dt;this.stellarMyr+=eventMyr;this.stepNo++;
  const E=this.EP;
  for(const b of this.bodies){
   if(b.kind!==1)continue;b.age+=eventMyr;
   if(b.age<b.lifetime)continue;
   if(b.smass<E.sn_mass_min){b.kind=7;continue;}
   const q=new RNG(this.seed,`${b.id}/death`),z=drive?drive(b.id,this.stellarMyr):[0,0,0,0,0],bh=b.smass>=E.bh_mass_min,rem=bh?.3*b.smass:1.4;
   const before=b.m;b.m*=Math.min(1,rem/b.smass);this.ejected+=before-b.m;b.kind=bh?5:6;
   for(let k=0;k<3;k++)b.v[k]+=q.normal()*E.kick_kms/20.7/Math.sqrt(3)*(bh?.3:1)*(1+.15*z[2]);
   const grb=bh&&q.random()<E.grb_prob*(1+.2*z[3]);this.log(grb?'GRB':'supernova',b,{mass:b.smass,remnant:bh?'black hole':'neutron star',drive:z[2]});
  }
  const holes=this.bodies.filter(b=>b.kind===4&&b.m>0);
  for(const bh of holes){
   for(const b of this.bodies){if(b.kind!==1||!b.m)continue;const d=norm(b.p.map((v,k)=>v-bh.p[k]));if(d>E.tde_r)continue;
    const before=bh.m,f=strongField(bh.m*1e8,b.smass);for(let k=0;k<3;k++)bh.v[k]=(bh.v[k]*before+b.v[k]*b.m)/(before+b.m);
    const radiation=f.visible?b.m*.057:0;bh.m+=b.m-radiation;this.radiated+=radiation;b.m=0;b.kind=9;bh.agnUntil=this.stellarMyr+30;
    this.log(f.visible?'TDE':'capture',bh,{victim:b.id,...f});
   }
   if(this.stepNo%120===0){const q=new RNG(this.seed,`${bh.id}/agn/${this.stepNo}`),z=drive?drive(bh.id,this.stellarMyr):[0,0,0,0,0];if(q.random()<E.agn_prob*(1+.2*z[1])){bh.agnUntil=this.stellarMyr+20;this.log('AGN',bh,{...strongField(bh.m*1e8),source:'phenomenological gas duty cycle'});}}
  }
  for(let i=0;i<holes.length;i++)for(let j=0;j<i;j++){const a=holes[i],b=holes[j];if(!a.m||!b.m||norm(a.p.map((v,k)=>v-b.p[k]))>E.smbh_merge_r)continue;const total=a.m+b.m;for(let k=0;k<3;k++)a.v[k]=(a.m*a.v[k]+b.m*b.v[k])/total;a.m=total*.95;this.radiated+=total*.05;b.m=0;b.kind=9;this.log('BH merger',a,{radiatedMass:.05*total});}
 }
}
