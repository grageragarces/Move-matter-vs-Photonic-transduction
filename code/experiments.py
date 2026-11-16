"""Link-level experiments: writes results/results.json and the paper figures.

    python3 experiments.py      # a few seconds
"""
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (REGIMES, DEMO, NEAR, OPTI, F_TARGET, photonic, matter, matter_cycle,
                  dephased, crossover_distance, sample_photonic)
from qfly import path_loss, grid_shape, coords

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
DASH, DOT = (0, (4, 2)), (0, (1, 1.5))
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#dddcd6",
    "grid.linewidth": 0.4, "lines.linewidth": 1.6, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
K_REF = 16          # Q-Fly DPHD with p = 8, g = 9 (Table 2 of the Q-Fly paper)
SEEDS = 8
out = {}


def ms(x):
    return x * 1e3


def top_legend(fig, ax, ncol=3):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=ncol, handlelength=3)


# ------------------------------------------------------------------ E0: Monte Carlo check
rng = np.random.default_rng(2026)
out["mc"] = {}
for r in REGIMES:
    loss = path_loss(r, K_REF, False)
    a = photonic(r, loss)
    s = np.array([sample_photonic(r, loss, rng) for _ in range(20000)])
    out["mc"][r.name] = dict(analytic_ms=ms(a.time), mc_ms=ms(s[:, 0].mean()),
                             mc_sem_ms=ms(s[:, 0].std() / math.sqrt(len(s))),
                             analytic_F=a.fidelity, mc_F=float(s[:, 1].mean()))

# ------------------------------------------------------------------ E1: time vs distance
dist = np.logspace(-0.3, 2.3, 200)   # cm
fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), sharey=True)
out["link"] = {}
for ax, r in zip(axes, REGIMES):
    tm = [ms(matter(r, d / 100 / r.pitch).time) for d in dist]
    li, le = path_loss(r, K_REF, False), path_loss(r, K_REF, True)
    pi, pe = photonic(r, li), photonic(r, le)
    ax.plot(dist, tm, color=BLUE, label="Shuttling")
    ax.axhline(ms(pi.time), color=ORANGE, label="Photonic, intra-group")
    ax.axhline(ms(pe.time), color=ORANGE, ls=DASH, label="Photonic, inter-group")
    di, de = crossover_distance(r, li) * 100, crossover_distance(r, le) * 100
    for d, t, dy in ((di, pi.time, -10), (de, pe.time, 5)):
        if 0 < d < dist[-1]:
            ax.plot([d], [ms(t)], "o", color=INK, ms=3.5, zorder=5)
            ax.annotate(f"{d:.0f} cm" if d >= 10 else f"{d:.1f} cm", (d, ms(t)),
                        textcoords="offset points", xytext=(6 if dy < 0 else -6, dy),
                        ha="left" if dy < 0 else "right", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 3000)
    ax.set_title(r.name, loc="left", fontweight="bold")
    ax.set_xlabel("Transport distance (cm)")
    # dephasing-only raw pairs (best case for distillation) and throughput break-even
    raw = dephased(r.f_phot)
    pd_ = photonic(r, li, raw=raw)
    m1 = matter(r, 1)
    out["link"][r.name] = dict(
        switch_intra_db=2 * (li - 10 * r.lf) / 2, loss_intra_db=li, loss_inter_db=le,
        phot_intra_ms=ms(pi.time), phot_inter_ms=ms(pe.time),
        phot_rounds=pi.rounds, phot_raw_pairs=pi.raw_pairs, phot_F=pi.fidelity,
        raw_rate_bare_hz=r.r_att * 0.5 * r.eta ** 2,
        matter_fixed_ms=ms(r.t_loc + r.t_cool), hop_ms=ms(r.pitch / r.v),
        matter_5cm_ms=ms(matter(r, 5).time), matter_5cm_F=matter(r, 5).fidelity,
        matter_first_distill_cm=next((h for h in range(1, 400) if matter(r, h).rounds > 0), None),
        dstar_intra_cm=di, dstar_inter_cm=de, dstar_bare_cm=crossover_distance(r, 0.0) * 100,
        deph_phot_intra_ms=ms(pd_.time), deph_rounds=pd_.rounds,
        deph_dstar_intra_cm=crossover_distance(r, li, raw=raw) * 100,
        deph_dstar_inter_cm=crossover_distance(r, le, raw=raw) * 100,
        matter_rate_hz=1.0 / (matter_cycle(r) * m1.raw_pairs),
        phot_rate_intra_hz=1.0 / pi.time,
        mux_breakeven_intra=pi.time / (matter_cycle(r) * m1.raw_pairs),
        mux_breakeven_inter=pe.time / (matter_cycle(r) * m1.raw_pairs))
axes[0].set_ylabel("Time per Bell pair at $F\\geq0.99$ (ms)")
top_legend(fig, axes[0])
fig.savefig(os.path.join(FIG, "fig_distance.pdf")); plt.close(fig)

# ------------------------------------------------------------------ E2: target-fidelity table
out["target"] = {}
for r in REGIMES:
    rows = {}
    for ft in (0.98, 0.99, 0.995, 0.999):
        p_, m_ = photonic(r, path_loss(r, K_REF, False), ft), matter(r, 5, ft)
        rows[str(ft)] = dict(phot_ms=ms(p_.time), phot_rounds=p_.rounds,
                             matter_ms=ms(m_.time), matter_rounds=m_.rounds)
    out["target"][r.name] = rows

# ------------------------------------------------------------------ E3: sensitivity of d*
def clean(v):
    return [None if (x is None or not math.isfinite(x) or x <= 0) else float(x) for x in v]


def nan(v):
    return [np.nan if x is None else x for x in v]


fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.85))
out["sens"] = {}
# (a) per-photon efficiency
ax = axes[0]
etas = np.logspace(-2, -0.05, 50)
for r, col, ls, mk in ((DEMO, BLUE, "-", "o"), (NEAR, AQUA, DASH, "s")):
    d = clean([crossover_distance(r.but(eta=e), path_loss(r, K_REF, False)) * 100 for e in etas])
    ax.plot(etas, nan(d), color=col, ls=ls, label=r.name)
    ax.plot([r.eta], [crossover_distance(r, path_loss(r, K_REF, False)) * 100], mk, color=INK, ms=4, zorder=5)
    out["sens"]["eta_" + r.name] = dict(x=list(map(float, etas)), dstar_cm=d)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Per-photon efficiency $\\eta$"); ax.set_ylabel("$d^*$ (cm)", labelpad=1)
ax.set_title("(a) Photon collection", loc="left"); ax.legend(loc="upper right")
# (b) crossing error and coherence time, component-demonstrated
ax = axes[1]
eps = np.logspace(-5, -2.3, 40)
for t2, col, ls in ((50.0, BLUE, "-"), (5.0, AQUA, DASH), (2.0, ORANGE, DOT)):
    d = clean([crossover_distance(DEMO.but(eps_x=e, t2=t2), path_loss(DEMO, K_REF, False)) * 100 for e in eps])
    ax.plot(eps, nan(d), color=col, ls=ls, label=f"$T_2$ = {t2:g} s")
    out["sens"][f"eps_T2={t2:g}"] = dict(x=list(map(float, eps)), dstar_cm=d)
ax.axvline(DEMO.eps_x, color=MUTED, lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_ylim(0, 32)
ax.set_xlabel("Error per boundary crossing $\\epsilon_\\times$")
ax.set_title("(b) Transport noise", loc="left"); ax.legend(loc="upper right")
ax.set_ylabel("$d^*$ (cm)", labelpad=1)
# (c) raw photonic fidelity
ax = axes[2]
fps = np.linspace(0.90, 0.999, 100)
for r, col, ls in ((NEAR, AQUA, DASH), (OPTI, ORANGE, "-")):
    d = [crossover_distance(r.but(f_phot=f), path_loss(r, K_REF, False)) * 100 for f in fps]
    d = [float(x) if math.isfinite(x) else None for x in d]
    ax.plot(fps, nan(d), color=col, ls=ls, label=r.name)
    out["sens"]["fp_" + r.name] = dict(x=list(map(float, fps)), dstar_cm=d)
ax.set_xlabel("Raw photonic fidelity $F_{\\mathrm{p}}$"); ax.set_ylim(0, 14)
ax.set_title("(c) Raw photonic fidelity", loc="left"); ax.legend(loc="upper right")
ax.set_ylabel("$d^*$ (cm)", labelpad=1)
fig.tight_layout(w_pad=0.6)
fig.savefig(os.path.join(FIG, "fig_crossover.pdf")); plt.close(fig)

# communication ions multiplexed on one optical port (m = 1 is the default everywhere else)
out["mux"] = {}
for r in REGIMES:
    li, le = path_loss(r, K_REF, False), path_loss(r, K_REF, True)
    tab = {}
    for m in (1, 2, 4, 8, 16):
        rm = r.but(mux=m)
        pi = photonic(rm, li)
        tab[str(m)] = dict(phot_intra_ms=ms(pi.time), phot_inter_ms=ms(photonic(rm, le).time),
                           dstar_intra_cm=crossover_distance(rm, li) * 100,
                           dstar_inter_cm=crossover_distance(rm, le) * 100,
                           deph_dstar_intra_cm=crossover_distance(rm, li, raw=dephased(r.f_phot)) * 100,
                           # optical ports (each with m ions) needed to match one matter channel
                           ports_to_match_matter=pi.time / (matter_cycle(r) * matter(r, 1).raw_pairs))
    out["mux"][r.name] = tab
    print("mux", r.name, {m: (round(v["phot_intra_ms"], 2), round(v["dstar_intra_cm"], 1),
                              round(v["dstar_inter_cm"], 1), round(v["deph_dstar_intra_cm"], 1))
                          for m, v in tab.items()}, flush=True)

# one-at-a-time sensitivity of the intra-group d* [cm] to a factor of two
out["tornado"] = {}
for r in (DEMO, NEAR):
    loss = path_loss(r, K_REF, False)
    tab = {"base": crossover_distance(r, loss) * 100}
    for key in ("v", "t_cool", "t_loc", "t_gsw", "t_dist", "pitch"):
        tab[key] = [crossover_distance(r.but(**{key: getattr(r, key) * f}), loss) * 100 for f in (0.5, 2.0)]
    tab["gate_infidelity"] = [crossover_distance(r.but(f_gate=1 - (1 - r.f_gate) * f), loss) * 100
                              for f in (0.5, 2.0)]
    out["tornado"][r.name] = tab
    print("tornado", r.name, {k: (round(v, 1) if isinstance(v, float) else [round(x, 1) for x in v])
                              for k, v in tab.items()}, flush=True)


# ------------------------------------------------------------------ E4: group size
def hop_hist_chain(p):
    h = np.arange(1, p)
    return h, (p - h) / (p * (p - 1) / 2)


def hop_hist_grid(p):
    cols, _ = grid_shape(p)
    xy = np.array([coords(i, cols) for i in range(p)])
    d = np.abs(xy[:, None, :] - xy[None, :, :]).sum(-1)[np.triu_indices(p, 1)]
    h, c = np.unique(d, return_counts=True)
    return h, c / c.sum()


def mean_matter(r, hist):
    h, w = hist
    return float(sum(wi * matter(r, int(hi)).time for hi, wi in zip(h, w)))


sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
out["group"] = {}
for r in REGIMES:
    tp = [ms(photonic(r, path_loss(r, 2 * p, False)).time) for p in sizes]
    tc = [ms(mean_matter(r, hop_hist_chain(p))) for p in sizes]
    tg = [ms(mean_matter(r, hop_hist_grid(p))) for p in sizes]
    out["group"][r.name] = dict(p=sizes, phot_ms=tp, chain_ms=tc, grid_ms=tg,
                                chain_wins=[p for p, a, b in zip(sizes, tc, tp) if a < b],
                                grid_wins=[p for p, a, b in zip(sizes, tg, tp) if a < b])

# ------------------------------------------------------------------ summary
# For each regime: the range of distances over which shuttling has the lower
# latency, against physical length scales.
fig, ax = plt.subplots(figsize=(3.45, 2.1))
lo, hi = 0.3, 3000.0                       # cm
rows = ((DEMO, "Demonstrated\ncomponents"), (NEAR, "Near-term"), (OPTI, "Optimistic"))
for y, (r, lab) in zip((2, 1, 0), rows):
    L = out["link"][r.name]
    d_hi, d_lo = L["dstar_intra_cm"], L["deph_dstar_intra_cm"]
    ax.barh(y, hi - lo, left=lo, height=0.5, color=ORANGE, alpha=0.30, lw=0)
    if d_hi > 0:
        ax.barh(y, d_hi - lo, left=lo, height=0.5, color="white", lw=0)
        ax.barh(y, d_hi - lo, left=lo, height=0.5, color=BLUE, alpha=0.30, lw=0)
        ax.barh(y, d_lo - lo, left=lo, height=0.5, color=BLUE, alpha=0.75, lw=0)
        for d, ha, dx in ((d_lo, "right", 3), (d_hi, "left", -3)):
            ax.annotate(f"{d:.0f} cm" if d >= 10 else f"{d:.1f} cm", (d, y), textcoords="offset points",
                        xytext=(dx, 8), ha=ha, fontsize=6.5)
    ax.text(lo * 0.82, y, lab, ha="right", va="center", fontsize=7.5)
for x, lab in ((1, "next\nmodule"), (100, "rack"), (1000, "room")):
    ax.axvline(x, color=MUTED, lw=0.6, ls=":", zorder=0)
    ax.text(x, 2.62, lab, ha="center", va="bottom", fontsize=6.5, color=MUTED)
ax.text(1.6, -0.62, "shuttle the ion", color=INK, fontsize=7, ha="left", va="center")
ax.text(1500, -0.62, "photonic link", color=INK, fontsize=7, ha="right", va="center")
ax.plot([0.42, 1.3], [-0.62, -0.62], color=BLUE, alpha=0.75, lw=5, solid_capstyle="butt")
ax.plot([1700, 2700], [-0.62, -0.62], color=ORANGE, alpha=0.30, lw=5, solid_capstyle="butt")
ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(-0.95, 3.15)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Distance between the two qubits (cm)")
fig.savefig(os.path.join(FIG, "fig_summary.pdf")); plt.close(fig)

with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("done")
