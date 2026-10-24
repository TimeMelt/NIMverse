import json, numpy as np
from scipy import stats
O="/workspace/nim_universe/out"; A=json.load(open(f"{O}/learned_alt2.json")); B=json.load(open(f"{O}/learned_oneway_alt2.json"))
assert A["heldout_seeds"]==B["heldout_seeds"]
a,b=np.array(A["heldout_per_seed"]),np.array(B["heldout_per_seed"]); d=a-b; n=len(d); se=lambda x:x.std(ddof=1)/np.sqrt(len(x))
print("seeds train",A["train_seeds"],"val",A["val_seeds"],"heldout",A["heldout_seeds"])
print(f"alt {a.mean():.4f}±{se(a):.4f}  oneway {b.mean():.4f}±{se(b):.4f}  diff {d.mean():+.4f}±{se(d):.4f} t-test p={stats.ttest_rel(a,b).pvalue:.3f} wilcoxon p={stats.wilcoxon(a,b).pvalue:.3f}")
print("baseline(no-train) heldout", A["heldout"]["one-way v3+v4 (+fixes at mid-range)"]["total"])
for k in ["tde_eff","agn_prob","agn_duty_myr"]:
    print(k,"init",A["rounds"][0]["EP"].get(k) if False else None, "alt",A["event_params"][k],"oneway",B["event_params"][k])
for r in A["rounds"]: print(r["round"],r["val_after_structure"],r["val_after_events"],r["val_gain"],r["dP_struct"],r["dP_events"])
print("oneway val",B["val_best"],B["best_where"],"alt val",A["val_best"],A["best_where"])
