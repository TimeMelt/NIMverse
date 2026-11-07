/* Backpressure: callers await each tick, so no simulation backlog accumulates. */
export class LifeWorker{
 constructor(){this.w=new Worker('worker.js',{type:'module'});this.serial=0;this.pending=new Map();this.w.onmessage=({data:d})=>{const p=this.pending.get(d.id);if(!p)return;this.pending.delete(d.id);d.error?p[1](new Error(d.error)):p[0](d.value);};this.w.onerror=e=>{for(const p of this.pending.values())p[1](new Error(e.message));this.pending.clear();};this.ready=this.call('boot');}
 call(op,args={}){return new Promise((a,b)=>{const id=++this.serial;this.pending.set(id,[a,b]);this.w.postMessage({id,op,...args});});}
}
