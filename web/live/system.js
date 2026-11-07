import {RNG,hash,TAU,lum,DEFAULTS} from './universe.js';
export const PAL={terran:[[.02,.08,.25],[.05,.3,.5],[.76,.7,.5],[.16,.45,.14],[.4,.33,.25],[.95,.95,.97]],ocean:[[.01,.05,.22],[.04,.25,.5],[.7,.68,.5],[.2,.5,.2],[.35,.35,.3],[.95,.95,.97]],desert:[[.25,.15,.08],[.5,.3,.15],[.85,.65,.4],[.75,.5,.28],[.55,.35,.22],[.9,.8,.65]],ice:[[.15,.25,.4],[.4,.6,.75],[.8,.88,.95],[.85,.9,.97],[.7,.75,.85],[1,1,1]],lava:[[1,.35,.05],[.9,.15,.02],[.12,.08,.07],[.2,.12,.1],[.3,.2,.18],[.45,.4,.38]]};
export function classify(me,a,L,u){if(me>12)return a<2.5*1.67*Math.sqrt(L)?'gas giant':'ice giant';if(a<.55*.95*Math.sqrt(L))return 'lava';if(a<.95*Math.sqrt(L))return 'desert';if(a<=1.67*Math.sqrt(L))return u<.3?'ocean':'terran';return 'ice';}
export function look(seed,id,me,a,L,forced){const r=new RNG(seed,`${id}/look`),type=forced||classify(me,a,L,r.random()),u=()=>r.random(),d={type,mass_earth:me,a,noise_off:[r.uniform(-100,100),r.uniform(-100,100),r.uniform(-100,100)],tilt:r.uniform(-.45,.45),spin:r.uniform(.3,1.8)};
 if(type.includes('giant'))return {...d,bands:6+u()*14,turb:.3+u()*.9,colors:type==='gas giant'?[[.85,.7,.5],[.7,.5,.35],[.95,.9,.8]]:[[.45,.7,.85],[.3,.5,.75],[.7,.85,.95]],storm:[(u()-.5)*1.2,u()*TAU,.08+u()*.12],radius_rel:Math.max(2,Math.min(6,me**.15))};
 return {...d,sea:({terran:.45+.2*u(),ocean:.7+.15*u(),desert:.1*u(),ice:.3+.2*u(),lava:.35+.2*u()})[type],rough:.45+.2*u(),cont_scale:1.2+u()*1.6,mountains:.4+u()*.8,moisture:u()-.5,cap:({ice:.2,lava:1.5,desert:1.2})[type]||.72+.15*u(),clouds:({terran:.6,ocean:.75,desert:.05,ice:.15,lava:.2})[type],colors:PAL[type],atm:({terran:[.35,.6,1],ocean:[.3,.55,1],desert:[.95,.7,.45],ice:[.7,.85,1],lava:[1,.4,.15]})[type],radius_rel:Math.max(.6,Math.min(2.2,me**.27))};
}
export function makeSystem(seed,host,P=DEFAULTS){
 P={...DEFAULTS,...P};const id=`${seed}/${host.id}`,r=new RNG(seed,`${host.id}/system`),mass=host.smass,L=lum(mass),scale=Math.sqrt(L);
 const star={id:host.id,mass,luminosity:L,temperature:5800*Math.max(.08,mass)**.55,age:host.age/1000,Z:host.Z};
 const n=Math.max(3,Math.min(9,Math.round(P.planets_mean+r.normal()))),planets=[];let a=r.uniform(.27,.37)*scale;
 for(let i=0;i<n;i++){
  const pid=`${id}/planet/${i}`,q=new RNG(seed,pid);a*=q.uniform(1.45,1.8)*P.spacing_mult;
  // Explorer preset samples inner rocky worlds, exterior giants; not an observed exoplanet distribution.
  const me=i<3?10**q.uniform(-.45,.8):10**q.uniform(-.2,2.4),e=q.uniform(0,.09),ptype=classify(me,a,L,q.random()),d=look(seed,pid,me,a,L,ptype);
  const p={id:pid,index:i,name:`World ${i+1}`,a,e,m:me*3e-6,phase:q.uniform(0,TAU),inc:q.normal()*.025,look:d,moons:[]};
  if(q.random()<.6){const hill=a*(p.m/(3*mass))**(1/3),nm=1+Math.floor(q.random()*3);for(let j=0;j<nm;j++)p.moons.push({a:hill*q.uniform(.15,.4)*P.moon_dist_mult,m:p.m*.001,phase:q.uniform(0,TAU)});}
  planets.push(p);
 }
 return {id,seed,star,planets,scale};
}
export function position(p,M,t){const mean=p.phase+TAU*t*Math.sqrt(M/p.a**3);let E=mean;for(let i=0;i<6;i++)E-=(E-p.e*Math.sin(E)-mean)/(1-p.e*Math.cos(E));const x=p.a*(Math.cos(E)-p.e),z=p.a*Math.sqrt(1-p.e*p.e)*Math.sin(E);return [x,z*Math.sin(p.inc),z*Math.cos(p.inc)];}
export function lifeSpec(s,p){const d=p.look,L=s.star.luminosity,hz=.95*Math.sqrt(L),ho=1.67*Math.sqrt(L),vol=({terran:.5,ocean:.9,desert:.35,ice:.8,lava:.02,'gas giant':1,'ice giant':1})[d.type];return {id:p.id,seed:hash(s.seed,p.id),star:s.star,plant:p.m<=12*3e-6&&p.a>=.65*hz&&p.a<=1.35*ho,
 habitat:{planet_id:p.index,name:p.name,m:p.m,a:p.a,e:p.e,spin:d.spin,tilt:d.tilt,L_star:L,hz_in:hz,hz_out:ho,in_hz:p.a>=hz&&p.a<=ho,ptype:d.type,volatiles:vol,impact_flux:.02,irradiance:L/p.a**2},
 orbits:{a:s.planets.map(p=>p.a),m:s.planets.map(p=>p.m),idx:p.index},look:d};}
