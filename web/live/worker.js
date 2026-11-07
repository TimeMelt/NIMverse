/* Offline Python runtime: all URLs are relative to this package. */
import {loadPyodide} from './pyodide/pyodide.mjs';
let py, ready;
async function boot(){
 const t=performance.now();
 py=await loadPyodide({indexURL:new URL('pyodide/',self.location).href});
 await py.loadPackage(['numpy','scipy','rebound']);
 const buf=await (await fetch('leaves.zip')).arrayBuffer();
 py.unpackArchive(buf,'zip',{extractDir:'/app'});
 await py.runPythonAsync(`import sys\nsys.path.insert(0,'/app')\nimport live_bridge as live\nimport rebound\nrcheck=rebound.Simulation(); rcheck.add(m=1); rcheck.add(m=1e-6,a=1); rcheck.integrate(.1)\n`);
 return {loadMs:performance.now()-t,memoryBytes:py._module.HEAPU8.buffer.byteLength,rebound:py.runPython('rebound.__version__')};
}
ready=boot();
onmessage=async({data:d})=>{try{
 const perf=await ready;let value;
 if(d.op==='boot')value=perf;
 else if(d.op==='create'){py.globals.set('spec_json',JSON.stringify(d.spec));value=py.runPython('live.create(__import__("json").loads(spec_json))');}
 else if(d.op==='tick'){py.globals.set('key',d.key);const t=performance.now();value=JSON.parse(py.runPython(`live.advance(key,${Math.max(1,Math.min(100,d.n||1))})`));value.tickMs=performance.now()-t;}
 else if(d.op==='eval')value=py.runPython(d.code);
 postMessage({id:d.id,value});
 }catch(e){postMessage({id:d.id,error:String(e)});}};
