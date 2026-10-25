"""Life-surface viewer: living planet (same shaders as viewer.py gallery) + stylized creatures from cohort traits.
python -m universe.viewer_life --seed 4   ->  /workspace/nim_universe/out/viewer_life_seed4.html"""
import argparse, json, os
from . import bodies
from .leaves import creature_look as CL
HERE = os.path.dirname(__file__); OUT = "/workspace/nim_universe/out"; RES = os.path.join(HERE, "..", "results")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4); ap.add_argument("--life", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv); J = json.load(open(a.life or f"{RES}/life_demo_seed{a.seed}.json"))
    pl = next(p for p in J["planets"] if p["habitat"]["name"] in J.get("with_life", [])); h = pl["habitat"]
    bid = h["planet_id"]
    for _ in range(60):   # bodies.py look for this planet, forced to the life-stack type (ocean) like viewer.gallery does
        d = bodies.planet_detail(J["seed"], bid, h["m"], h["a"], h["L_star"])
        if d["type"] == h["ptype"]: break
        bid += 100
    cohorts = pl.get("cohorts") or CL.cohorts_from_report(pl, J["LP"], J["seed"])
    cr = CL.creatures(cohorts, J["seed"], pl["biosign"], h["ptype"])
    data = dict(title=f"NIM life · seed {J['seed']} · {h['name']}", planet=d, name=h["name"], ptype=h["ptype"], biomass=pl["biomass"], biosign=pl["biosign"],
                climate=dict(T_K=pl["climate"]["T_K"], water=pl["climate"]["liquid_water_frac"], cls=pl["climate"]["climate_class"]),
                tech=pl.get("tech", 0), culture=pl.get("culture", {}), cohorts=cohorts, creatures=cr, T_eff=J["host"].get("T_eff", 5800))
    R = lambda p: open(os.path.join(HERE, p)).read(); html = R("viewer_src/life_template.html")
    for key, val in [("/*THREE*/", R("vendor/three.min.js")), ("/*ORBIT*/", R("vendor/OrbitControls.js")), ("/*DATA*/", json.dumps(data, separators=(",", ":"))),
                     ("/*SHADERS*/", R("viewer_src/shaders.js")), ("/*APP*/", R("viewer_src/life.js"))]:
        html = html.replace(key, val.replace("</script", "<\\/script"), 1)
    out = a.out or f"{OUT}/viewer_life_seed{a.seed}.html"; open(out, "w").write(html)
    plans = {}; [plans.__setitem__(c["plan"], plans.get(c["plan"], 0)+1) for c in cr]
    print(f"wrote {out}; planet {h['name']} ({d['type']}); {len(cohorts)} cohorts -> {len(cr)} creatures {plans}")

if __name__ == "__main__": main()
