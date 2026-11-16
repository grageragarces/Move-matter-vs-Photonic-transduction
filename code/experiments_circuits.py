"""System-level benchmark on the MQT Bench circuits in ../circuits.

For every circuit, parameter regime, layout and architecture, each way of
executing remote gates (workload.MODES) is simulated and the fastest is kept
as "best". Writes one results file and one figure per circuit to
results/circuits/, plus scheduler.json comparing the two scheduling rules.

    python3 experiments_circuits.py
"""
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from core import REGIMES
from qfly import PhotonicQFly, HybridQFly, MatterGrid
from workload import load_qasm, run_circuit, remote_pairs, MODES

HERE = os.path.dirname(os.path.abspath(__file__))
CIRC = os.path.join(HERE, "..", "circuits")
RES = os.path.join(HERE, "..", "results", "circuits")
os.makedirs(RES, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

CIRCUITS = {"ghz": ("ghz_alg_100.qasm", 10), "wstate": ("wstate_alg_100.qasm", 10),
            "qftentangled": ("qftentangled_alg_100.qasm", 3)}        # file, seeds
# (p, g, q): modules per group, groups, data qubits per module. Groups of four
# modules; Q-Fly and the hybrid have the same Benes depth ceil(log2 k) in each.
CONFIGS = [(4, 5, 5), (4, 13, 2), (4, 25, 1)]
MUX = (1, 8)          # communication ions multiplexed on one optical port
SPARE = 1             # free slots per module for moved qubits
PRIORITY = "critical" # scheduling rule for the main results
SHORT = {"Component-demonstrated": "Comp.-demo.", "Near-term": "Near-term", "Optimistic": "Optimistic"}


def half(p):
    return math.ceil(p / 2)


ARCHS = {"qfly": lambda r, g, p, rng: PhotonicQFly(r, g, p, rng),
         "hybrid_half": lambda r, g, p, rng: HybridQFly(r, g, p, half(p), rng),
         "hybrid_full": lambda r, g, p, rng: HybridQFly(r, g, p, p, rng),
         "grid": lambda r, g, p, rng: MatterGrid(r, g, p, rng)}
# control runs: a module may only use its nearest interface module (no pooling)
NOPOOL = {"hybrid_half_nopool": lambda r, g, p, rng: HybridQFly(r, g, p, half(p), rng, pooling=False),
          "hybrid_full_nopool": lambda r, g, p, rng: HybridQFly(r, g, p, p, rng, pooling=False)}
STYLE = {"qfly": (ORANGE, "o", ORANGE, "Q-Fly (all photonic)"),
         "hybrid_half": (AQUA, "D", AQUA, "Hybrid, $c=\\lceil p/2\\rceil$"),
         "hybrid_full": (AQUA, "D", "white", "Hybrid, $c=p$"),
         "grid": (BLUE, "s", BLUE, "All-matter array")}


def pair_split(gates, p, q):
    """Bell pairs a telegate-only run needs within a group and between groups."""
    total = remote_pairs(gates, q)
    inter = remote_pairs(gates, q * p)
    return total - inter, inter


def measure(make, r, g, p, q, n, gates, mono, mode, seeds, priority=PRIORITY):
    v, pairs, shuttles = [], [], []
    for s in range(seeds):
        st = {}
        t = run_circuit(make(r, g, p, np.random.default_rng(300 + s)), gates, n, q, r.t_loc,
                        mode=mode, spare=SPARE, priority=priority, stats=st)[0]
        v.append(t / mono)
        pairs.append(st.get("pairs", 0)); shuttles.append(st.get("shuttles", 0))
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                pairs=float(np.mean(pairs)), shuttles=float(np.mean(shuttles)))


sched = {}
for name, (fname, seeds) in CIRCUITS.items():
    n, gates, counts = load_qasm(os.path.join(CIRC, fname))
    out = {"circuit": dict(file=fname, qubits=n, gates=counts, seeds=seeds),
           "settings": dict(spare_slots=SPARE, priority=PRIORITY, modes=list(MODES)),
           "configs": [dict(p=p, g=g, q=q, N=p * g, k_qfly=p + g - 1, k_hybrid=half(p) + g - 1,
                            pairs_intra_inter=pair_split(gates, p, q)) for p, g, q in CONFIGS],
           "metric": "execution time / monolithic execution time; best = fastest mode",
           "mux": {}}
    fig, axes = plt.subplots(len(MUX), 3, figsize=(7.1, 1.9 * len(MUX)), squeeze=False)
    for row, m in enumerate(MUX):
        out["mux"][str(m)] = {}
        for ax, r0 in zip(axes[row], REGIMES):
            r = r0.but(mux=m)
            mono = run_circuit(None, gates, n, n, r.t_loc)[0]
            res = {"monolithic_ms": mono * 1e3, "modes": {md: {a: [] for a in ARCHS} for md in MODES},
                   "nopool": {md: {a: [] for a in NOPOOL} for md in ("telegate", "burst")},
                   "best": {a: [] for a in ARCHS}}
            for p, g, q in CONFIGS:
                for a, make in ARCHS.items():
                    for md in MODES:
                        res["modes"][md][a].append(measure(make, r, g, p, q, n, gates, mono, md, seeds))
                    b = min(MODES, key=lambda md: res["modes"][md][a][-1]["mean"])
                    res["best"][a].append(dict(mode=b, **res["modes"][b][a][-1]))
                for a, make in NOPOOL.items():
                    for md in res["nopool"]:
                        res["nopool"][md][a].append(measure(make, r, g, p, q, n, gates, mono, md, seeds))
            res["best_speed_vs_qfly"] = {a: [res["best"]["qfly"][i]["mean"] / res["best"][a][i]["mean"]
                                             for i in range(len(CONFIGS))] for a in ARCHS if a != "qfly"}
            out["mux"][str(m)][r0.name] = res
            print(name, f"m={m}", r0.name, "| qfly best/mono",
                  [(round(x["mean"], 1), x["mode"]) for x in res["best"]["qfly"]],
                  "| speed vs qfly", {a: [round(x, 2) for x in v] for a, v in res["best_speed_vs_qfly"].items()},
                  "| modes", {a: [x["mode"] for x in res["best"][a]] for a in ARCHS if a != "qfly"}, flush=True)
            x = np.arange(len(CONFIGS))
            for i, a in enumerate(ARCHS):
                col, mk, face, lab = STYLE[a]
                ax.errorbar(x + (i - 1.5) * 0.13, [b["mean"] for b in res["best"][a]],
                            yerr=[b["std"] for b in res["best"][a]], color=col, marker=mk, ls="none",
                            mfc=face, label=lab, capsize=1.5, elinewidth=0.6)
            ax.set_yscale("log")
            lo_, hi_ = ax.get_ylim()
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter() if hi_ / lo_ < 20 else NullFormatter())
            ax.set_xlim(-0.5, len(CONFIGS) - 0.5); ax.set_xticks(list(x))
            ax.set_xticklabels([f"$N$={p * g}\n$q$={q}" for p, g, q in CONFIGS])
            ax.set_title(f"{SHORT[r0.name]}, {m} ion{'s' if m > 1 else ''}/port", loc="left", fontweight="bold")
        axes[row][0].set_ylabel("Time / monolithic (best mode)")
    fig.tight_layout(w_pad=0.8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, handlelength=1)
    fig.savefig(os.path.join(RES, f"{name}.pdf")); plt.close(fig)
    with open(os.path.join(RES, f"{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # scheduling rule: circuit order against critical path first (one ion per port)
    sched[name] = {}
    for r in REGIMES:
        mono = run_circuit(None, gates, n, n, r.t_loc)[0]
        tab = {}
        for p, g, q in CONFIGS:
            for a, make in ARCHS.items():
                for md in ("telegate", "burst"):
                    t = {pr: measure(make, r, g, p, q, n, gates, mono, md, min(seeds, 3), priority=pr)["mean"]
                         for pr in ("list", "critical")}
                    tab[f"N={p * g},{a},{md}"] = dict(list=t["list"], critical=t["critical"],
                                                      gain=t["list"] / t["critical"])
        sched[name][r.name] = tab
    g_ = [v["gain"] for reg in sched[name].values() for v in reg.values()]
    print(name, "critical-path gain: median %.3f, min %.3f, max %.3f" % (np.median(g_), min(g_), max(g_)), flush=True)
with open(os.path.join(RES, "scheduler.json"), "w") as fh:
    json.dump(sched, fh, indent=1)
print("done")
