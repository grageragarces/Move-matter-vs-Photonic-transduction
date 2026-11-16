# Benchmark circuits

Three 100-qubit circuits from MQT Bench (https://www.cda.cit.tum.de/mqtbench/),
used as workloads for the system-level simulation in `code/experiments_circuits.py`.

| File | Algorithm | Gates after inlining |
|---|---|---|
| `ghz_alg_100.qasm` | GHZ state | 1 H, 99 CX |
| `wstate_alg_100.qasm` | W state | 1 X, 198 RY, 99 CZ, 99 CX |
| `qftentangled_alg_100.qasm` | QFT applied to a GHZ state | 101 H, 99 CX, 4950 CP, 50 SWAP |

## Settings chosen when generating them

- Abstraction level: **algorithm level** (`alg` in the file name), i.e. the
  original circuit, not compiled to a gate set or mapped to a device.
- Output format: **OpenQASM 3**.
- Size: **100 qubits**.
- **Original circuit, no encoding.**
- **Random values** for any circuit parameters.
- Tool versions (from the file headers): MQT Bench 2.0.1, Qiskit 2.1.1.
  GHZ and W state were generated on 2025-10-15, QFT-entangled on 2025-10-17.

Regenerate with the same settings if a circuit is replaced, so that results
stay comparable.

## How the simulator reads them

- Measurements and barriers are dropped; custom gate definitions (the `qft`
  gate in the QFT-entangled file) are inlined.
- Only the qubits each gate acts on and the gate order matter. Rotation angles
  are ignored.
- A two-qubit gate between modules costs one Bell pair; a SWAP between modules
  costs two.

An earlier version of the study used `qft_indep_qiskit_99.qasm` (99 qubits,
target-independent level, OpenQASM 2, MQT Bench 1.1.0). It is no longer in
this folder and its results are not reported.
