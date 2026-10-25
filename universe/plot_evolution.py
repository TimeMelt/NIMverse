"""Evolution chart from a life report: per-trait mean ± std, population, species, generations over world time.
python -m universe.plot_evolution --life results/life_demo_seed4.json [--planet planet2] -> results/life_evolution_traits_seed4.png"""
import argparse, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .leaves.params import TRAITS
RES = os.path.join(os.path.dirname(__file__), "..", "results")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--life", default=f"{RES}/life_demo_seed4.json"); ap.add_argument("--planet", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv); J = json.load(open(a.life))
    pl = next(p for p in J["planets"] if (p.get("evolution") and (a.planet is None or p["habitat"]["name"] == a.planet)))
    S = [s for s in pl["evolution"]["evolution_series"]]; t = np.array([s["t_w"] for s in S])/1e3
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="#0b0f17")
    for x in ax: x.set_facecolor("#0b0f17"); x.tick_params(colors="#bcd"); [sp.set_color("#345") for sp in x.spines.values()]
    for tr, c in zip(TRAITS, ["#4fc3f7", "#ffb74d", "#81c784", "#e57373"]):
        m = np.array([s["trait_means"][tr] if s["n_alive"] else np.nan for s in S], float); sd = np.array([s["trait_stds"][tr] if s["n_alive"] else np.nan for s in S], float)
        ax[0].plot(t, m, color=c, lw=1.8, label=tr); ax[0].fill_between(t, m-sd, m+sd, color=c, alpha=.18)
    if "env_offset" in S[-1]: ax[0].plot(t, [s.get("env_offset", 0.) + .5 for s in S], color="#9e9e9e", lw=1, ls="--", label="climate drift (+0.5)")
    ax[0].set_title(f"{pl['habitat']['name']}: trait means ± sd (literal individuals)", color="w"); ax[0].set_xlabel("world time [kyr]", color="#bcd"); ax[0].legend(facecolor="#111a26", labelcolor="w", fontsize=8)
    ax[1].plot(t, [s["n_alive"] for s in S], color="#ffd040", label="living individuals"); ax[1].plot(t, [10*s["n_species"] for s in S], color="#ce93d8", label="species ×10")
    ax[1].plot(t, [s["generations_max"]/10 for s in S], color="#80cbc4", label="max generation ÷10")
    ax[1].set_title(f"population · births {pl['evolution']['total_births']} · species ever {pl['evolution']['species_ever']} · alive {pl['evolution'].get('n_species_alive', '?')}", color="w"); ax[1].set_xlabel("world time [kyr]", color="#bcd"); ax[1].legend(facecolor="#111a26", labelcolor="w", fontsize=8)
    out = a.out or f"{RES}/life_evolution_traits_seed{J['seed']}.png"; fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=fig.get_facecolor()); print("wrote", out); return out

if __name__ == "__main__": main()
