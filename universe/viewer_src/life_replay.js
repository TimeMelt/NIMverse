// Offline transport: deterministic recorded ticks; no network, storage or backend.
(function(root){
class NIMReplay {
  constructor(data){
    if(!data.ticks || data.ticks.length<2) throw new Error('Replay needs at least two ticks');
    this.info=data.info; this.ticks=data.ticks; this.interval=data.interval||.2;
    this.index=0; this.paused=false; this.speed=1; this.elapsed=0; this.emit=null;
  }
  current(){return Object.assign(structuredClone(this.ticks[this.index]),{
    seq:this.index+1,paused:this.paused,mode:'replay',replay_seq:this.index+1,
    hist_from:1,hist_to:this.ticks.length,live_seq:this.ticks.length,
    realtime_s:this.interval/this.speed});}
  publish(){const tick=this.current(); if(this.emit) this.emit(tick); return tick;}
  start(emit){this.emit=emit; this.publish();}
  advance(seconds){
    if(this.paused) return;
    this.elapsed+=Math.max(0,Math.min(.25,seconds));
    const delay=this.interval/this.speed;
    // At most one sync/render per animation frame, even after a hidden-tab stall.
    if(this.elapsed>=delay){this.elapsed%=delay; this.index=(this.index+1)%this.ticks.length; this.publish();}
  }
  control(body){
    const a=body.action;
    if(a==='pause') this.paused=true;
    else if(a==='resume') this.paused=false;
    else if(a==='toggle') this.paused=!this.paused;
    else if(a==='set_speed'){const s=Number(body.speed); this.speed=Number.isFinite(s)?(s===0?8:Math.max(.25,Math.min(8,s))):1;}
    else if(a==='live'){this.index=0; this.paused=false;}
    else if(['seek','rewind','forward','restore'].includes(a)){
      const target=a==='seek'||a==='restore'?Number(body.seq)-1:this.index+(a==='rewind'?-1:1)*(Number(body.steps)||1);
      if(Number.isFinite(target)) this.index=Math.max(0,Math.min(this.ticks.length-1,Math.round(target)));
      this.paused=true;
    }
    this.elapsed=0; return this.publish();
  }
}
root.NIMReplay=NIMReplay;
})(globalThis);
