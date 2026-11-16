"""Compare in-order qubit placement with a KaHyPar partition of the interaction graph.

Needs the kahypar Python package and one of its configuration files, e.g.
config/cut_kKaHyPar_sea20.ini from https://github.com/kahypar/kahypar:

    python3 check_placement.py /path/to/cut_kKaHyPar_sea20.ini
"""
import math
import os
import sys
import kahypar

from workload import load_qasm, remote_pairs

CIRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "circuits")


def kahypar_placement(gates, n, q, ini, seed=1):
    """Module of every qubit: blocks of at most q qubits, minimising cut gate weight."""
    weight = {}
    for name, qs in gates:
        if len(qs) == 2:
            e = tuple(sorted(qs))
            weight[e] = weight.get(e, 0) + (2 if name == "swap" else 1)
    edges, index, flat = list(weight), [0], []
    for a, b in edges:
        flat += [a, b]
        index.append(len(flat))
    k = math.ceil(n / q)
    hg = kahypar.Hypergraph(n, len(edges), index, flat, k, [weight[e] for e in edges], [1] * n)
    ctx = kahypar.Context()
    ctx.loadINIconfiguration(ini)
    ctx.setK(k)
    ctx.setEpsilon((k * q - n) / n)
    ctx.setSeed(seed)
    ctx.suppressOutput(True)
    kahypar.partition(hg, ctx)
    return [hg.blockID(v) for v in range(n)]


if __name__ == "__main__":
    ini = sys.argv[1]
    for name in ("ghz", "wstate", "qftentangled"):
        n, gates, _ = load_qasm(os.path.join(CIRC, f"{name}_alg_100.qasm"))
        for q in (5, 2, 1):
            pl = kahypar_placement(gates, n, q, ini)
            assert max(pl.count(b) for b in set(pl)) <= q
            print(f"{name:13s} q={q}: Bell pairs (telegate) in-order {remote_pairs(gates, q):5d}"
                  f"   KaHyPar {remote_pairs(gates, q, pl):5d}")
