/* Exact complex statevector gate algebra, not a random-number stand-in.
Qubit 0 is MSB (PennyLane convention). CPU simulation; no quantum hardware. */
import {RNG,hash} from './universe.js';
export class Statevector{
 constructor(n){if(n<1||n>16)throw Error('1..16 wires supported');this.n=n;this.N=1<<n;this.re=new Float64Array(this.N);this.im=new Float64Array(this.N);this.re[0]=1;}
 gate(q,m){const bit=1<<(this.n-1-q),r=this.re,im=this.im;for(let i=0;i<this.N;i++){if(i&bit)continue;const j=i|bit,ar=r[i],ai=im[i],br=r[j],bi=im[j];r[i]=m[0]*ar-m[1]*ai+m[2]*br-m[3]*bi;im[i]=m[0]*ai+m[1]*ar+m[2]*bi+m[3]*br;r[j]=m[4]*ar-m[5]*ai+m[6]*br-m[7]*bi;im[j]=m[4]*ai+m[5]*ar+m[6]*bi+m[7]*br;}}
 h(q){const s=Math.SQRT1_2;this.gate(q,[s,0,s,0,s,0,-s,0]);}
 rx(q,a){const c=Math.cos(a/2),s=Math.sin(a/2);this.gate(q,[c,0,0,-s,0,-s,c,0]);}
 ry(q,a){const c=Math.cos(a/2),s=Math.sin(a/2);this.gate(q,[c,0,-s,0,s,0,c,0]);}
 rz(q,a){const c=Math.cos(a/2),s=Math.sin(a/2);this.gate(q,[c,-s,0,0,0,0,c,s]);}
 cnot(c,t){const cb=1<<(this.n-1-c),tb=1<<(this.n-1-t);for(let i=0;i<this.N;i++)if((i&cb)&&!(i&tb)){const j=i|tb;[this.re[i],this.re[j]]=[this.re[j],this.re[i]];[this.im[i],this.im[j]]=[this.im[j],this.im[i]];}}
 x(q){const bit=1<<(this.n-1-q);let z=0;for(let i=0;i<this.N;i++)if(!(i&bit)){const j=i|bit;z+=2*(this.re[i]*this.re[j]+this.im[i]*this.im[j]);}return z;}
 z(q){const bit=1<<(this.n-1-q);let z=0;for(let i=0;i<this.N;i++)z+=(i&bit?-1:1)*(this.re[i]**2+this.im[i]**2);return z;}
 norm(){let x=0;for(let i=0;i<this.N;i++)x+=this.re[i]**2+this.im[i]**2;return x;}
}
export function qwrk(f,E,W,entangle='layer'){
 if(f.length!==11||E.length!==5)throw Error('5 wires / 11 features required');
 const c=new Statevector(5),x=E.map(row=>Math.tanh(row.reduce((s,e,j)=>s+e*f[j],0))*Math.PI/2);
 for(let q=0;q<5;q++)c.rx(q,x[q]);for(let q=0;q<5;q++)c.ry(q,x[q]);
 const ring=()=>{for(let q=0;q<5;q++)c.cnot(q,(q+1)%5);};
 for(const layer of W){for(let q=0;q<5;q++)for(let g=0;g<3;g++){c[['rx','ry','rz'][g]](q,layer[q][g]);if(entangle==='dense')ring();}if(entangle==='layer')ring();}
 return Array.from({length:5},(_,q)=>c.x(q));
}
export function nimStructure(W){const c=new Statevector(16);for(let q=0;q<16;q++){c.h(q);for(const w of W){c.rx(q,w[0]);c.ry(q,w[1]);c.rz(q,w[2]);}}return Array.from({length:16},(_,q)=>c.z(q));}
const cache=new Map();
export function circuitWeights(seed,id){const key=seed+'/'+id;if(cache.has(key))return cache.get(key);const r=new RNG(seed,id+'/qwrk'),E=Array.from({length:5},()=>Array.from({length:11},()=>r.normal()*.3)),W=Array.from({length:2},()=>Array.from({length:5},()=>Array.from({length:3},()=>r.normal()*.1)));let w={E,W};cache.set(key,w);if(cache.size>64)cache.delete(cache.keys().next().value);return w;}
export function physicsDrive(seed,id,t,enabled=true,features=[]){if(!enabled)return [0,0,0,0,0];const w=circuitWeights(seed,id+'/physics'),r=new RNG(seed,id+'/drive-features');const f=Array.from({length:11},(_,i)=>i<features.length?Math.tanh(features[i]):Math.sin(t*.013*(1+i*.12)+r.uniform(-Math.PI,Math.PI)));f[10]=1;return qwrk(f,w.E,w.W);}
export function structureDrive(seed){const r=new RNG(seed,'NIM-16-wire-structure'),W=Array.from({length:2},()=>Array.from({length:3},()=>r.uniform(0,Math.PI)));return nimStructure(W);}
