"""Compare LIFE_DEFAULTS vs results/learned_life.json on train + held-out seeds, plus seed-4 long smoke."""
import json, os, sys, time, numpy as np
from .learn_life import loss
from .leaves import LIFE_DEFAULTS
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
C = {**json.load(open(os.path.join(os.path.dirname(__file__), "config3.json"))), "asteroids": 20}
def main():
    L = json.load(open(os.path.join(R, "learned_life.json"))); P = {**LIFE_DEFAULTS, **L["params"]}
    train = L["seeds"]; held = [0, 1, 2, 5, 9, 10, 11, 14, 101, 102]; out = {}
    runs = [("defaults", LIFE_DEFAULTS), ("learned", P)]
    prev = os.path.join(R, "learned_life_prev_impactoff.json")      # 2026-10 run that drove impact_scale -> ~0 (for comparison)
    if os.path.exists(prev): runs.append(("prev_impactoff", {**LIFE_DEFAULTS, **json.load(open(prev))["params"]}))
    for name, PP in runs:
        for tag, ss in (("train", train), ("held", held)):
            f, p, per = loss(PP, ss, C, L["years"], 10., detail=True)
            out[f"{name}/{tag}"] = dict(loss=f, hab=p[0], eco=p[1], rec=p[2], ext=p[3], n_ext=sum(v["ext"] >= 1 for v in per.values()), per=per)
        f, p, per = loss(PP, [4], C, 20000., 20., detail=True)
        out[f"{name}/seed4_20k"] = {**per[4], **dict(loss=f, hab=p[0], eco=p[1], rec=p[2], ext=p[3])}
        f, p, per = loss(PP, [4], C, 4000., 10., culture=True, detail=True)
        out[f"{name}/seed4_culture"] = {**per[4], **dict(loss=f, ext=p[3])}
        print(name, {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items() if kk != "per"} for k, v in out.items() if k.startswith(name)}, flush=True)
    json.dump(out, open(os.path.join(R, "learned_life_validation.json"), "w"), indent=1, default=str)
if __name__ == "__main__": main()
