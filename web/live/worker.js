/* Offline module worker; every fetch is within the exported package. */
import {loadPyodide} from './pyodide/pyodide.mjs';
import {qwrk,circuitWeights,physicsDrive} from './quantum.js';
let py, specs=new Map();
function circuitEval(s){const d=JSON.parse(s),sp=specs.get(d.key),w=circuitWeights(sp.seed,d.key+'/creature');return JSON.stringify(d.features.map(f=>qwrk(f,w.E,w.W)));}
async function boot(){
 const t=performance.now();py=await loadPyodide({indexURL:new URL('pyodide/',self.location).href});
 await py.loadPackage(['numpy','scipy','rebound']);py.unpackArchive(await(await fetch('leaves.zip')).arrayBuffer(),'zip',{extractDir:'/app'});
 await py.runPythonAsync(`import sys\nsys.path.insert(0,'/app')\nimport live_bridge as live\nimport rebound\nrcheck=rebound.Simulation(); rcheck.add(m=1); rcheck.add(m=1e-6,a=1); rcheck.integrate(.1)\n`);
 py.globals.set('circuit_callback',circuitEval);py.runPython('live.circuit_eval=circuit_callback');
 return {loadMs:performance.now()-t,memoryBytes:py._module.HEAPU8.buffer.byteLength,rebound:py.runPython('rebound.__version__')};
}
const ready=boot();ready.catch(()=>{});
onmessage=async({data:d})=>{try{
 const perf=await ready;let value;
 if(d.op==='boot')value=perf;
 else if(d.op==='create'){specs.delete(d.spec.id);while(specs.size>=4)specs.delete(specs.keys().next().value);specs.set(d.spec.id,d.spec);py.globals.set('spec_json',JSON.stringify(d.spec));value=py.runPython('live.create(__import__("json").loads(spec_json))');}
 else if(d.op==='tick'){const sp=specs.get(d.key);py.globals.set('key',d.key);const z=physicsDrive(sp.seed,d.key,d.time||0,d.quantum!==false,[sp.star.mass,sp.habitat.irradiance,sp.habitat.e,sp.star.age]);py.globals.set('drive_json',JSON.stringify(z));const t=performance.now();value=JSON.parse(py.runPython(`live.advance(key,${Math.max(1,Math.min(100,d.n||1))},__import__('json').loads(drive_json))`));value.tickMs=performance.now()-t;value.memoryBytes=py._module.HEAPU8.buffer.byteLength;}
 else if(d.op==='clear'){py.runPython('live.runs.clear()');specs.clear();value=true;}
 else if(d.op==='parity'){py.globals.set('key',d.key);const t=performance.now();value=JSON.parse(py.runPython(`live.advance(key,${Math.max(1,Math.min(100,d.n||1))})`));value.tickMs=performance.now()-t;}
 else if(d.op==='eval')value=py.runPython(d.code);
 postMessage({id:d.id,value});
 }catch(e){postMessage({id:d.id,error:String(e)});}};
