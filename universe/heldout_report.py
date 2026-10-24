import json, numpy as np, sys
sys.path.insert(0, '.')
from universe.learn3 import evaluate
from universe.sim2 import DEFAULTS
C = json.load(open('universe/config3.json')); E = dict(C, **C['eval']); HO = list(range(9000, 9016))
d = np.array([evaluate(DEFAULTS, E, [s]) for s in HO]); dd = evaluate(DEFAULTS, E, HO, full=True)[1]['distinct']
print('defaults held-out %.4f ± %.4f (seed spread), distinct %.3f' % (d.mean(), d.std(), dd))
L = []
for r in range(3):
    J = json.load(open(f'/workspace/nim_universe/out/learned_params_v3_run{r}.json'))
    l = np.array([evaluate(J['params'], E, [s]) for s in HO]); L.append(l)
    print(f'run{r}: best epoch {J["best_epoch"]} val {J["val_best"]:.3f} (def {J["val_default"]:.3f}) held-out {l.mean():.4f}, wins {int((l>d).sum())}/16, distinct {J["heldout_learned"]["distinct"]:.3f}')
L = np.array(L); m = L.mean(1); print('learned held-out mean over 3 runs %.4f ± %.4f (run spread); gain %.4f ± %.4f; paired per-seed gain std %.4f' % (m.mean(), m.std(), (m-d.mean()).mean(), m.std(), (L-d).std()))
