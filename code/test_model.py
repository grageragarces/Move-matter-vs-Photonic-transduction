"""Consistency checks for the link model, the schedulers and the circuit workload.

    python3 test_model.py        # or: pytest test_model.py
"""
import math
import os
import tempfile
import numpy as np

from core import (REGIMES, DEMO, NEAR, OPTI, photonic, matter, matter_cycle, dephased, werner,
                  dejmps, crossover_distance, sample_photonic, p_herald)
from qfly import path_loss, PhotonicQFly, HybridQFly, MatterGrid
from workload import load_qasm, run_circuit, qft_gates, remote_pairs, MODES

CIRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "circuits")


def rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ link model
def test_states_stay_normalised():
    for r in REGIMES:
        for target in (0.98, 0.99):
            for s, _ in photonic(r, 5.0, target).levels:
                assert abs(sum(s) - 1) < 1e-9


def test_dejmps_removes_phase_flips_in_one_round():
    out, p = dejmps(dephased(0.94), dephased(0.94))
    assert abs(out[0] - 0.94 ** 2 / (0.94 ** 2 + 0.06 ** 2)) < 1e-12
    assert abs(p - (0.94 ** 2 + 0.06 ** 2)) < 1e-12


def test_single_shuttled_pair_latency_is_overhead_plus_transit():
    for r in REGIMES:
        for hops in (1, 3, 5):
            d = matter(r, hops)
            if d.rounds == 0:
                assert abs(d.time - (r.t_loc + r.t_cool + hops * r.pitch / r.v)) < 1e-12


def test_crossover_matches_closed_form_without_distillation():
    r = NEAR
    loss = path_loss(r, 16, False)
    d = crossover_distance(r, loss)
    assert matter(r, d / r.pitch).rounds == 0
    assert abs(d - r.v * (photonic(r, loss).time - r.t_loc - r.t_cool)) < 1e-6


def test_crossover_is_zero_when_photons_beat_the_fixed_overhead():
    assert crossover_distance(OPTI, path_loss(OPTI, 16, False)) == 0.0


def test_monte_carlo_agrees_with_analytic_latency():
    for r in REGIMES:
        loss = path_loss(r, 16, False)
        g = rng(1)
        t = np.mean([sample_photonic(r, loss, g)[0] for _ in range(4000)])
        assert abs(t / photonic(r, loss).time - 1) < 0.03


def test_multiplexing_speeds_up_generation_and_shrinks_crossover():
    for r in (DEMO, NEAR):
        loss = path_loss(r, 16, False)
        assert abs(p_herald(r.but(mux=4), loss) / p_herald(r, loss) - 4) < 0.01   # small-p limit
        times = [photonic(r.but(mux=m), loss).time for m in (1, 2, 4, 8)]
        assert all(a > b for a, b in zip(times, times[1:]))
        d = [crossover_distance(r.but(mux=m), loss) for m in (1, 2, 4, 8)]
        assert all(a > b for a, b in zip(d, d[1:]))
    assert photonic(DEMO.but(mux=1), 3.0).time == photonic(DEMO, 3.0).time


# ------------------------------------------------------------------ schedulers
def test_idle_array_reproduces_analytic_matter_latency():
    for r in REGIMES:
        for u, v in ((0, 1), (0, 37), (5, 60)):
            a = MatterGrid(r, 9, 8, rng())
            t = a.try_start(u, v, 0.0)
            assert abs(t - matter(r, len(a.path(u, v))).time) < 1e-9


def test_idle_qfly_reproduces_analytic_photonic_latency():
    r = NEAR
    for inter, (u, v) in ((False, (0, 1)), (True, (0, 20))):
        t = np.mean([PhotonicQFly(r, 9, 8, rng(s)).try_start(u, v, 0.0) for s in range(2000)])
        assert abs(t / photonic(r, path_loss(r, 16, inter)).time - 1) < 0.04


def test_hybrid_with_all_ports_and_no_pooling_equals_qfly_between_groups():
    for r in REGIMES:
        for u, v in ((0, 20), (3, 71), (9, 40)):
            tq = PhotonicQFly(r, 9, 8, rng(5)).try_start(u, v, 0.0)
            th = HybridQFly(r, 9, 8, 8, rng(5), pooling=False).try_start(u, v, 0.0)
            assert abs(tq - th) < 1e-12


def test_ports_are_never_double_booked():
    h = HybridQFly(NEAR, 5, 4, 2, rng(1))
    log, orig = [], h._photonic

    def wrapped(u, v, gu, gv, mu, mv, now):
        before = [row[:] for row in h.port]
        end = orig(u, v, gu, gv, mu, mv, now)
        if end is not None:
            for grp in (gu, gv):
                for i in range(h.c):
                    if h.port[grp][i] != before[grp][i]:
                        log.append((before[grp][i], now))
        return end

    h._photonic = wrapped
    n, gates, _ = load_qasm(os.path.join(CIRC, "qftentangled_alg_100.qasm"))
    run_circuit(h, gates, n, 5, NEAR.t_loc)
    assert len(log) > 1000 and all(prev <= start + 1e-12 for prev, start in log)


# ------------------------------------------------------------------ circuits
def test_textbook_qft_depth():
    for n in (4, 8, 16):
        assert run_circuit(None, qft_gates(n), n, n, 1.0)[0] == 2 * n - 3


def test_loader_reads_openqasm2_and_inlines_openqasm3_gates():
    q2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncp(pi/2) q[1],q[0];\n' \
         'swap q[0],q[2];\nbarrier q[0],q[1];\nmeasure q[0] -> c[0];\n'
    q3 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\ngate blk a, b {\n  h a;\n  cx a, b;\n}\nbit[3] m;\n' \
         'qubit[3] q;\nblk q[2], q[0];\nry(-1.5) q[1];\nm[0] = measure q[0];\n'
    for text, expect in ((q2, (3, [("h", (0,)), ("cp", (1, 0)), ("swap", (0, 2))])),
                         (q3, (3, [("h", (2,)), ("cx", (2, 0)), ("ry", (1,))]))):
        with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as fh:
            fh.write(text)
        try:
            n, gates, _ = load_qasm(fh.name)
        finally:
            os.unlink(fh.name)
        assert (n, gates) == expect


def test_benchmark_circuits_load_with_expected_gate_counts():
    expect = {"ghz": {"h": 1, "cx": 99}, "wstate": {"x": 1, "ry": 198, "cz": 99, "cx": 99},
              "qftentangled": {"h": 101, "cx": 99, "cp": 4950, "swap": 50}}
    for name, counts in expect.items():
        n, gates, got = load_qasm(os.path.join(CIRC, f"{name}_alg_100.qasm"))
        assert n == 100 and got == counts and len(gates) == sum(counts.values())


def test_circuit_pair_count_matches_gate_list():
    n, gates, _ = load_qasm(os.path.join(CIRC, "qftentangled_alg_100.qasm"))
    _, pairs = run_circuit(MatterGrid(NEAR, 5, 4, rng()), gates, n, 5, NEAR.t_loc)
    assert pairs == remote_pairs(gates, 5) == 4869


def test_burst_shares_pairs_and_all_modes_finish_within_capacity():
    n, gates, _ = load_qasm(os.path.join(CIRC, "qftentangled_alg_100.qasm"))
    mono = run_circuit(None, gates, n, n, NEAR.t_loc)[0]
    pairs = {}
    for mode in MODES:
        for arch in (PhotonicQFly(NEAR, 5, 4, rng()), HybridQFly(NEAR, 5, 4, 2, rng()), MatterGrid(NEAR, 5, 4, rng())):
            stats = {}
            t, p = run_circuit(arch, gates, n, 5, NEAR.t_loc, mode=mode, spare=1, stats=stats)
            assert math.isfinite(t) and t >= mono
            assert stats["max_occupancy"] <= 5 + 1
            pairs[mode, type(arch).__name__] = p
    assert pairs["burst", "PhotonicQFly"] < pairs["telegate", "PhotonicQFly"] / 4
    assert pairs["teledata", "PhotonicQFly"] > 0          # no matter links: teleports only


def test_data_qubits_are_only_shuttled_within_the_fidelity_budget():
    from core import data_fidelity, max_data_hops
    for r in REGIMES:
        h = max_data_hops(r)
        assert data_fidelity(r, h) >= 0.99 > data_fidelity(r, h + 1)
    a = MatterGrid(DEMO, 25, 4, rng())
    assert a.can_move(0, 1) and not a.can_move(0, 99)
    assert not PhotonicQFly(DEMO, 5, 4, rng()).can_move(0, 1)
    h = HybridQFly(DEMO, 5, 4, 2, rng())
    assert h.can_move(0, 3) and not h.can_move(0, 4)       # only inside a group


def test_critical_path_priority_is_a_valid_schedule():
    n, gates, _ = load_qasm(os.path.join(CIRC, "wstate_alg_100.qasm"))
    mono = run_circuit(None, gates, n, n, NEAR.t_loc)[0]
    assert run_circuit(None, gates, n, n, NEAR.t_loc, priority="critical")[0] == mono
    t = run_circuit(PhotonicQFly(NEAR, 5, 4, rng()), gates, n, 5, NEAR.t_loc, priority="critical")[0]
    assert t >= mono


def test_distributed_run_is_never_faster_than_monolithic():
    n, gates, _ = load_qasm(os.path.join(CIRC, "ghz_alg_100.qasm"))
    for r in REGIMES:
        mono = run_circuit(None, gates, n, n, r.t_loc)[0]
        for arch in (PhotonicQFly(r, 5, 4, rng()), HybridQFly(r, 5, 4, 2, rng()), MatterGrid(r, 5, 4, rng())):
            assert run_circuit(arch, gates, n, 5, r.t_loc)[0] >= mono - 1e-12


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as err:
            failed += 1
            print("FAIL", name, err)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
