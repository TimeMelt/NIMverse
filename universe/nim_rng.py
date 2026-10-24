"""Entropy source built on NIM. (1) Runs NIM's real CircuitClass.Circuit in
'free-think' mode (Hadamard + RX/RY/RZ inceptron ansatz) to get expectation values;
(2) samples the same ansatz with shots on a seeded device; shot bits -> uniforms."""
import sys, os, numpy as np, pennylane as qml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Config import settings
from CircuitClass import Circuit

class NIMRandom:
    def __init__(self, seed, n_wires=8, shots=4096):
        settings["data_mode"] = "free-think"
        w = np.random.default_rng(seed).uniform(0, np.pi, (2, 3))
        nim = Circuit(qml.device("default.mixed", wires=n_wires), n_wires, None)
        self.expvals = np.array(nim.run(w.tolist(), None), dtype=float)
        dev = qml.device("default.qubit", wires=n_wires, seed=seed)
        @qml.set_shots(shots)
        @qml.qnode(dev)
        def sampler(weights):
            for i in range(n_wires):
                qml.Hadamard(wires=i)
                for j in range(len(weights)):
                    qml.RX(weights[j][0], wires=i); qml.RY(weights[j][1], wires=i); qml.RZ(weights[j][2], wires=i)
            return qml.sample(wires=range(n_wires))
        self._sampler = sampler
        self._w = np.zeros((1, 3))   # zero rotations => pure Hadamard superposition, unbiased bits
        self._pool = np.empty(0)
    def _refill(self):
        bits = np.asarray(self._sampler(self._w)).astype(np.uint64).reshape(-1)
        bits = bits[: (bits.size // 32) * 32].reshape(-1, 32)
        self._pool = np.concatenate([self._pool, (bits << np.arange(32, dtype=np.uint64)).sum(1) / 2.0**32])
    def random(self, n=None):
        k = 1 if n is None else int(np.prod(n))
        while self._pool.size < k: self._refill()
        out, self._pool = self._pool[:k], self._pool[k:]
        return out[0] if n is None else out.reshape(n)
    def uniform(self, a=0.0, b=1.0, size=None): return a + (b - a) * self.random(size)
    def normal(self, mu=0.0, sd=1.0, size=None):
        n = size if size else 1
        z = np.sqrt(-2*np.log(np.maximum(self.random(n), 1e-12))) * np.cos(2*np.pi*self.random(n))
        z = mu + sd*z
        return z if size else float(z.reshape(-1)[0])
    def integers(self, lo, hi): return int(lo + np.floor(self.random() * (hi - lo)))
