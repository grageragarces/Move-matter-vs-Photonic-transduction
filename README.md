# Move the qubit or send a photon?

Code, benchmark circuits and results for a comparison of two ways to link the
modules of a modular trapped-ion quantum computer:

- a **photonic link**, which heralds entanglement between two modules with photons, and
- a **matter link**, which physically shuttles an ion from one module to the next.

Both are modelled as sources of Bell pairs of a target fidelity (including
distillation) and compared at link level and inside the Q-Fly switched optical
interconnect, an all-matter array, and a hybrid of the two.

## Layout

```
code/      the model, the simulator, the experiment scripts and the tests
circuits/  the three 100-qubit MQT Bench circuits used as workloads (see circuits/README.md)
results/   the numbers behind every figure and table
```

| File | What it does |
|---|---|
| `code/core.py` | Link model: parameter regimes, photonic and matter links, distillation, crossover distance |
| `code/qfly.py` | The three architectures (Q-Fly, hybrid Q-Fly, all-matter array) and their resources |
| `code/workload.py` | OpenQASM loader and the circuit simulator, with four ways of executing remote gates |
| `code/experiments.py` | Link-level experiments and their figures |
| `code/experiments_circuits.py` | System-level benchmark on the three circuits |
| `code/check_placement.py` | Compares in-order qubit placement with a KaHyPar partition |
| `code/test_model.py` | Consistency tests |

## Requirements

Python 3.10 or later with `numpy` and `matplotlib`. `pytest` is optional.
`check_placement.py` additionally needs the `kahypar` package and one of its
configuration files.

## Reproducing the results

Run everything from the `code/` folder.

```
python3 test_model.py              # 19 consistency tests, about a minute
python3 experiments.py             # link-level results and figures, a few seconds
python3 experiments_circuits.py    # benchmark circuits, about half an hour
```

- `experiments.py` writes the link-level numbers to `results/results.json`
  and three figures (latency against distance, sensitivity of the crossover,
  and the summary) to `paper/figures/`, creating that folder if needed.
- `experiments_circuits.py` writes one JSON file and one figure per circuit to
  `results/circuits/`, plus `scheduler.json`, which compares the two
  scheduling rules.

Random seeds are fixed, so repeated runs give the same numbers.

## Where the main parameters live

- The three parameter regimes are the `DEMO`, `NEAR` and `OPTI` objects at the top of `code/core.py`.
- The target fidelity is `F_TARGET` in the same file.
- The number of ions multiplexed on an optical port is the `mux` field of a regime (default 1).
- The module layouts, the number of spare slots per module and the scheduling
  rule are constants at the top of `code/experiments_circuits.py`.

## Reading the results

- `results/results.json`: `link` (latencies and crossover distances), `target`
  (target-fidelity table), `sens` and `tornado` (sensitivity), `mux`
  (multiplexed ports), `group` (group switch against shuttling array), `mc`
  (Monte Carlo check).
- `results/circuits/<circuit>.json`: for each number of ions per port (`mux`),
  regime, way of executing remote gates (`modes`) and architecture, the
  execution time relative to a monolithic machine. `best` holds the fastest
  mode for each case.

## What was removed

The repository contains only what the results depend on. Earlier, exploratory
parts of the study were taken out:

- A synthetic workload of random Bell-pair requests, with its own batch
  scheduler, throughput results and figure. The three benchmark circuits cover
  both local and all-to-all traffic, so it is no longer used.
- A figure of group-switch latency against group size. The numbers behind it
  are still computed and stored under `group` in `results/results.json`.
- A 99-qubit QFT circuit from an older MQT Bench release, replaced by the
  three 100-qubit circuits in `circuits/`.
