/* Offline module worker; every fetch is within the exported package. */
import {loadPyodide} from './pyodide/pyodide.mjs';
import {qwrk,circuitWeights,physicsDrive} from './quantum.js';
let py, specs=new Map(),loop={key:null,paused:false,speed:1,quantum:true},lastStep=0,lastBg=0,streamUnacked=false,latestStream=null;
function tickWorld(key,n=1,enabled=true,time=null){const sp=specs.get(key);py.globals.set('key',key);const t=time??py.runPython('live.runs[key].t');const z=physicsDrive(sp.seed,key,t,enabled,[sp.star.mass,sp.habitat.irradiance,sp.habitat.e,sp.star.age]);py.globals.set('drive_json',JSON.stringify(z));const start=performance.now();let v=JSON.parse(py.runPython(`live.advance(key,${n},__import__('json').loads(drive_json))`));v.tickMs=performance.now()-start;v.memoryBytes=py._module.HEAPU8.buffer.byteLength;return v;}
function circuitEval(s){const d=JSON.parse(s),sp=specs.get(d.key),w=circuitWeights(sp.seed,d.key+'/creature');return JSON.stringify(d.features.map(f=>qwrk(f,w.E,w.W)));}
async function boot(){
 const t=performance.now();py=await loadPyodide({indexURL:new URL('pyodide/',self.location).href});
 await py.loadPackage(['numpy','scipy','rebound']);py.unpackArchive(await(await fetch('leaves.zip')).arrayBuffer(),'zip',{extractDir:'/app'});
 await py.runPythonAsync(`import sys\nsys.path.insert(0,'/app')\nimport live_bridge as live\nimport rebound\nrcheck=rebound.Simulation(); rcheck.add(m=1); rcheck.add(m=1e-6,a=1); rcheck.integrate(.1)\n`);
 py.globals.set('circuit_callback',circuitEval);py.runPython('live.circuit_eval=circuit_callback');
 return {loadMs:performance.now()-t,memoryBytes:py._module.HEAPU8.buffer.byteLength,rebound:py.runPython('rebound.__version__')};
}
const ready=boot();ready.catch(()=>{});
// Physics owns its clock. Slow rendering only drops visual snapshots, never
// queues unbounded messages or makes numerical stepping wait for the UI.
ready.then(()=>setInterval(()=>{
 if(!loop.key||loop.paused||!specs.has(loop.key))return;
 const now=performance.now();
 try{
  if(now-lastStep>=125/Math.max(.1,loop.speed)){lastStep=now;latestStream={stream:true,key:loop.key,value:tickWorld(loop.key,1,loop.quantum)};}
  if(latestStream&&!streamUnacked){postMessage(latestStream);streamUnacked=true;latestStream=null;}
  if(now-lastBg>=5000){lastBg=now;const keys=[...specs.keys()].filter(k=>k!==loop.key);if(keys.length){const key=keys[Math.floor(now/5000)%keys.length];tickWorld(key,1,loop.quantum);}}
 }catch(e){postMessage({stream:true,key:loop.key,error:String(e)});loop.paused=true;}
},15)).catch(()=>{});
onmessage=async({data:d})=>{try{
 if(d.op==='ack'){streamUnacked=false;return;}
 const perf=await ready;let value;
 if(d.op==='boot')value=perf;
 else if(d.op==='create'){specs.delete(d.spec.id);while(specs.size>=4)specs.delete(specs.keys().next().value);specs.set(d.spec.id,d.spec);py.globals.set('spec_json',JSON.stringify(d.spec));value=py.runPython('live.create(__import__("json").loads(spec_json))');}
 else if(d.op==='tick'){value=tickWorld(d.key,Math.max(1,Math.min(100,d.n||1)),d.quantum!==false,d.time);}
 else if(d.op==='loop'){Object.assign(loop,d.state);value=true;}
 else if(d.op==='clear'){py.runPython('live.runs.clear()');specs.clear();loop.key=null;latestStream=null;streamUnacked=false;value=true;}
 else if(d.op==='parity'){py.globals.set('key',d.key);const t=performance.now();value=JSON.parse(py.runPython(`live.advance(key,${Math.max(1,Math.min(100,d.n||1))})`));value.tickMs=performance.now()-t;}
 else if(d.op==='eval')value=py.runPython(d.code);
 postMessage({id:d.id,value});
 }catch(e){postMessage({id:d.id,error:String(e)});}};
