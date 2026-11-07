import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {Statevector,qwrk,nimStructure,physicsDrive,circuitWeights} from './quantum.js';
const fixtures=JSON.parse(fs.readFileSync(new URL('./circuit_fixtures.json',import.meta.url)));
test('quantum gates preserve norm and have correct readout',()=>{const c=new Statevector(3);c.h(0);assert.ok(Math.abs(c.x(0)-1)<1e-14);c.cnot(0,1);assert.ok(Math.abs(c.x(0))<1e-14);assert.ok(Math.abs(c.norm()-1)<1e-14);});
test('QWRK 5-wire XY 2-layer matches PennyLane all entanglers',()=>{let max=0;for(const c of fixtures.cases){const z=qwrk(c.f,c.E,c.W,c.entangle);for(let i=0;i<5;i++)max=Math.max(max,Math.abs(c.z[i]-z[i]));}console.log('QWRK max absolute error',max);assert.ok(max<2e-14);});
test('NIM 16-wire structure matches PennyLane',()=>{const c=fixtures.nim,z=nimStructure(c.W);const max=Math.max(...z.map((a,i)=>Math.abs(a-c.z[i])));console.log('NIM max absolute error',max);assert.ok(max<2e-12);});
test('seeded drive is bounded, live and switchable',()=>{let a=physicsDrive(4,'world/1',0),b=physicsDrive(4,'world/1',100);assert.notDeepEqual(a,b);assert.deepEqual(a,physicsDrive(4,'world/1',0));assert.ok([...a,...b].every(v=>v>=-1&&v<=1));assert.deepEqual(physicsDrive(4,'world/1',0,false),[0,0,0,0,0]);});
