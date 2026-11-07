"""Reference values computed by PennyLane, never needed in browser."""
import json,numpy as np,pennylane as qml
from pathlib import Path
from universe.leaves.mind import CircuitMind
r=np.random.default_rng(412);cases=[]
for ent in ['layer','off','dense']:
    c=CircuitMind(seed=42,entangle=ent)
    for _ in range(4):
        f=r.uniform(-1,1,11);cases.append(dict(f=f.tolist(),E=c.E.tolist(),W=c.W.tolist(),entangle=ent,z=c.z(f).tolist()))
w=r.normal(0,.3,(2,3)).tolist()
@qml.qnode(qml.device('default.qubit',wires=16))
def nim(w):
    for q in range(16):
        qml.Hadamard(q)
        for a in w:qml.RX(a[0],q);qml.RY(a[1],q);qml.RZ(a[2],q)
    return [qml.expval(qml.PauliZ(q)) for q in range(16)]
Path('web/live/circuit_fixtures.json').write_text(json.dumps(dict(source='PennyLane CircuitMind and NIM CircuitClass free-think ansatz',cases=cases,nim=dict(W=w,z=np.array(nim(w)).tolist())),indent=2))
