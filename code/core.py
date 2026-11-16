"""Link-level model: photonic heralded entanglement vs. ion shuttling.

Both mechanisms are reduced to the same quantity: the expected time to deliver
one Bell pair between two modules with fidelity >= F_target, including
entanglement distillation where the raw pair is not good enough.

Bell-diagonal states are stored as (A, B, C, D) = (Phi+, Psi-, Psi+, Phi-),
the ordering of Deutsch et al., PRL 77, 2818 (1996).
"""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class Regime:
    name: str
    # --- photonic interconnect ---
    r_att: float    # entanglement attempt rate [Hz]
    eta: float      # per-photon efficiency, ion to detector click, excluding switches/fibre
    f_phot: float   # fidelity of a heralded raw pair
    x2: float       # insertion loss of one 2x2 switch point [dB]
    t_gsw: float    # group-switch reconfiguration time [s]
    lf: float       # fibre loss [dB/m]
    # --- matter link (ion shuttling) ---
    v: float        # mean transport speed, including junctions and module boundaries [m/s]
    pitch: float    # module pitch [m]
    t_loc: float    # local Bell-pair preparation [s]
    t_cool: float   # split/merge + sympathetic recooling at destination [s]
    eps_x: float    # error per module-boundary crossing
    # --- shared ---
    f_gate: float   # local two-qubit gate fidelity
    t2: float       # memory coherence time [s]
    t_dist: float   # duration of one distillation round [s]
    mux: int = 1    # communication ions multiplexed on one optical port

    def but(self, **kw):
        return replace(self, **kw)


# Demonstrated: O'Reilly et al. 2024 (1 MHz attempts, 250 pairs/s, F~0.94) for the
# photonic link; Akhtar et al. 2023 (684 um link at 2424/s -> 1.66 m/s) and
# Pino et al. 2021 (split/combine 2 x 128 us + 2.05 ms cooling) for transport.
# eps_x is the upper bound on coherence infidelity per link reported by Akhtar et al.
# Gate fidelity: Helios (Ransford et al. 2025), 7.9e-4 two-qubit infidelity; the
# optimistic value is the smooth gate of Hughes et al. 2025 (8.4e-5).
DEMO = Regime("Component-demonstrated",
              r_att=1e6, eta=math.sqrt(2 * 2.5e-4), f_phot=0.94, x2=0.4, t_gsw=5e-3, lf=0.03,
              v=684e-6 * 2424, pitch=1e-2, t_loc=1e-3, t_cool=2.3e-3, eps_x=5e-4,
              f_gate=0.9992, t2=50.0, t_dist=1e-3)
NEAR = Regime("Near-term",
              r_att=1e6, eta=0.10, f_phot=0.97, x2=0.4, t_gsw=1e-3, lf=2e-4,
              v=10.0, pitch=1e-2, t_loc=5e-4, t_cool=1e-3, eps_x=1e-5,
              f_gate=0.9995, t2=100.0, t_dist=5e-4)
OPTI = Regime("Optimistic",
              r_att=2.5e5, eta=0.40, f_phot=0.99, x2=0.2, t_gsw=1e-4, lf=2e-4,
              v=50.0, pitch=1e-2, t_loc=2e-4, t_cool=3e-4, eps_x=1e-6,
              f_gate=0.9999, t2=1000.0, t_dist=2e-4)
REGIMES = (DEMO, NEAR, OPTI)

F_TARGET = 0.99
MAX_ROUNDS = 8


# ---------------------------------------------------------------- states
def werner(f):
    e = (1.0 - f) / 3.0
    return np.array([f, e, e, e])


def depolarise(s, e):
    return (1.0 - e) * np.asarray(s) + e / 4.0


def dephase(s, t, t2):
    """Both qubits of the pair idle for time t with coherence time t2."""
    q = 0.5 * (1.0 - math.exp(-2.0 * t / t2))
    a, b, c, d = s
    return np.array([(1 - q) * a + q * d, (1 - q) * b + q * c,
                     (1 - q) * c + q * b, (1 - q) * d + q * a])


def dejmps(s1, s2):
    """One DEJMPS round on two Bell-diagonal pairs. Returns (state, p_success)."""
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    n = (a1 + b1) * (a2 + b2) + (c1 + d1) * (c2 + d2)
    out = np.array([a1 * a2 + b1 * b2, c1 * d2 + d1 * c2,
                    c1 * c2 + d1 * d2, a1 * b2 + b1 * a2]) / n
    return out, n


def gate_noise(s, reg):
    """Two noisy CNOTs per round (one per side): fidelity drops by ~2(1 - f_gate)."""
    return depolarise(s, (8.0 / 3.0) * (1.0 - reg.f_gate))


# ---------------------------------------------------------------- delivery
@dataclass
class Delivery:
    time: float       # expected time to deliver one pair at >= target [s]
    fidelity: float
    rounds: int
    raw_pairs: float  # expected number of raw pairs consumed
    levels: list      # per level: (state, p_success of the round that produced it)


def deliver(cycle, state, reg, offset=0.0, target=F_TARGET, post=None, first=None):
    """Nested distillation with sequential raw-pair generation.

    cycle  : mean time between consecutive raw pairs on this link [s]
    state  : raw Bell-diagonal state
    offset : one-off latency added once per delivered pair [s]
    first  : time to produce the first raw pair, if different from cycle [s]
    post   : optional map applied to the state before comparing with the target
             (noise the pair still has to go through after distillation)
    The pair generated first waits (and dephases) for the mean time needed to
    generate its partner.
    """
    post = post or (lambda x: x)
    t, s, n_raw = cycle, np.asarray(state, float), 1.0
    levels = [(s, 1.0)]
    rounds = 0
    while post(s)[0] < target - 1e-12:
        if rounds == MAX_ROUNDS:
            return Delivery(math.inf, s[0], rounds, math.inf, levels)
        out, p = dejmps(dephase(s, t, reg.t2), s)
        out = gate_noise(out, reg)
        if out[0] <= s[0]:
            return Delivery(math.inf, s[0], rounds, math.inf, levels)
        t = (2.0 * t + reg.t_dist) / p
        n_raw = 2.0 * n_raw / p
        s = out
        rounds += 1
        levels.append((s, p))
    lead = 0.0 if first is None else first - cycle
    return Delivery(offset + lead + t, post(s)[0], rounds, n_raw, levels)


# ---------------------------------------------------------------- photonic
def p_herald(reg, loss_db):
    """Probability that an attempt heralds a pair.

    Two-photon heralding: 1/2 (linear-optics BSA) x both photons arrive. With
    reg.mux ions per port, each attempt slot tries mux ion pairs and succeeds if
    any of them does.
    """
    p = 0.5 * reg.eta ** 2 * 10.0 ** (-loss_db / 10.0)
    return 1.0 - (1.0 - p) ** reg.mux


def dephased(f):
    """Raw pair whose only error is a phase flip (Phi+ <-> Phi-)."""
    return np.array([f, 0.0, 0.0, 1.0 - f])


def photonic(reg, loss_db=0.0, target=F_TARGET, post=None, raw=None):
    cycle = 1.0 / (reg.r_att * p_herald(reg, loss_db))
    raw = werner(reg.f_phot) if raw is None else raw
    return deliver(cycle, raw, reg, offset=reg.t_gsw, target=target, post=post)


# ---------------------------------------------------------------- matter
def matter_state(reg, hops, extra_wait=0.0):
    s = werner(reg.f_gate)
    s = depolarise(s, (4.0 / 3.0) * (1.0 - (1.0 - reg.eps_x) ** hops))
    return dephase(s, hops * reg.pitch / reg.v + reg.t_cool + extra_wait, reg.t2)


def matter(reg, hops, target=F_TARGET):
    """Shuttle one half of a locally prepared pair across `hops` modules.

    Ions are pipelined: after the first, raw pairs arrive every t_loc + t_cool,
    or every pitch / v if a link (one ion at a time) is the bottleneck.
    """
    transit = hops * reg.pitch / reg.v
    return deliver(matter_cycle(reg), matter_state(reg, hops), reg,
                   offset=transit, target=target, first=reg.t_loc + reg.t_cool)


def data_fidelity(reg, hops):
    """Fidelity kept by a data qubit that is itself shuttled across `hops` modules.

    Unlike a Bell pair, a data qubit cannot be distilled afterwards.
    """
    t = hops * reg.pitch / reg.v + reg.t_cool
    return (1.0 - reg.eps_x) ** hops * 0.5 * (1.0 + math.exp(-t / reg.t2))


def max_data_hops(reg, target=F_TARGET):
    """Largest number of boundary crossings a data qubit can make at >= target."""
    h = 0
    while h < 10000 and data_fidelity(reg, h + 1) >= target:
        h += 1
    return h


def matter_cycle(reg):
    return max(reg.t_loc + reg.t_cool, reg.pitch / reg.v)


def crossover_distance(reg, loss_db, target=F_TARGET, raw=None):
    """Transport distance [m] below which shuttling beats the photonic link.

    Returns inf if the photonic link cannot reach the target, 0 if shuttling
    never wins, and nan if neither mechanism can reach the target.
    """
    tp = photonic(reg, loss_db, target, raw=raw).time
    t0 = matter(reg, 0.0, target).time
    if not math.isfinite(tp):
        return math.inf if math.isfinite(t0) else math.nan
    if t0 >= tp:
        return 0.0
    lo, hi = 0.0, 1e4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if matter(reg, mid / reg.pitch, target).time < tp:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- Monte Carlo
def sample_photonic(reg, loss_db, rng, target=F_TARGET, post=None, raw=None):
    """Sample (time, fidelity) of one delivered pair, tracking actual waits."""
    raw = werner(reg.f_phot) if raw is None else raw
    ref = photonic(reg, loss_db, target, post, raw)
    if not math.isfinite(ref.time):
        return math.inf, raw[0]
    p0 = p_herald(reg, loss_db)

    def gen(level):
        if level == 0:
            return rng.geometric(p0) / reg.r_att, raw
        t = 0.0
        while True:
            t1, s1 = gen(level - 1)
            t2, s2 = gen(level - 1)
            out, p = dejmps(dephase(s1, t2, reg.t2), s2)
            t += t1 + t2 + reg.t_dist
            if rng.random() < p:
                return t, gate_noise(out, reg)

    t, s = gen(ref.rounds)
    return reg.t_gsw + t, s[0]
