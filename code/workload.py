"""Circuit workloads for the system-level simulation.

A circuit is a list of gates (name, qubits). Qubits start on modules in order,
q per module, unless a placement is given. Gate order on every qubit is
respected. How a two-qubit gate between modules is executed depends on `mode`:

  telegate  one Bell pair per remote gate; qubits never move (a SWAP costs two)
  burst     one Bell pair is shared by consecutive remote gates between one
            qubit and one module, as in AutoComm (Wu et al., MICRO 2022)
  teledata  the qubit is moved to the other module whenever that module has a
            spare slot, by shuttling it where a matter link allows and by
            teleporting it (one Bell pair) otherwise; then the gate is local
  adaptive  per gate, whichever of burst and moving serves more upcoming gates

No single mode is best for every circuit and architecture; the experiments run
all four and report the fastest.
"""
import heapq
import math
import re

_DEF = re.compile(r"gate\s+(\w+)\s*(\([^)]*\))?\s*([^{]*)\{([^}]*)\}")
_STMT = re.compile(r"^(\w+)\s*(\([^;]*?\))?\s*(.*)$", re.S)
_INDEX = re.compile(r"^\w+\[(\d+)\]$")
_SKIP = ("openqasm", "include", "bit", "creg", "barrier", "reset")

# gates through which a shared ("cat") copy of a qubit can act
DIAGONAL_2Q = {"cp", "cz", "cu1", "crz", "cphase", "rzz", "cs", "ct"}   # either qubit
CONTROLLED_2Q = {"cx", "cnot", "cy", "ch", "crx", "cry", "csx"}         # first qubit only
DIAGONAL_1Q = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "id"}
MODES = ("telegate", "burst", "teledata", "adaptive")


def load_qasm(path):
    """Read an OpenQASM 2 or 3 file with one flat quantum register.

    Custom gate definitions are inlined. Returns (n_qubits, gates, counts) where
    gates is a list of (name, qubits) and counts is by gate name after inlining.
    Measurements and barriers are dropped.
    """
    with open(path) as fh:
        text = re.sub(r"//[^\n]*", "", fh.read())
    defs = {}

    def grab(m):
        defs[m.group(1)] = ([a.strip() for a in m.group(3).split(",")], m.group(4))
        return ""

    text = _DEF.sub(grab, text)
    n, gates, counts = 0, [], {}

    def run(body, binding):
        nonlocal n
        for stmt in body.split(";"):
            stmt = stmt.strip()
            if not stmt or "measure" in stmt or stmt.lower().startswith(_SKIP):
                continue
            if stmt.startswith(("qubit", "qreg")):
                n = int(re.search(r"\[(\d+)\]", stmt).group(1))
                continue
            m = _STMT.match(stmt)
            name = m.group(1)
            qubits = []
            for arg in m.group(3).split(","):
                arg = arg.strip()
                idx = _INDEX.match(arg)
                qubits.append(int(idx.group(1)) if idx else binding[arg])
            if name in defs:
                formal, inner = defs[name]
                run(inner, dict(zip(formal, qubits)))
                continue
            if len(qubits) > 2:
                raise ValueError(f"gate on more than two qubits: {stmt[:60]}")
            counts[name] = counts.get(name, 0) + 1
            gates.append((name, tuple(qubits)))

    run(text, {})
    return n, gates, counts


def qft_gates(n):
    """Textbook QFT without the final swaps."""
    gates = []
    for i in range(n):
        gates.append(("h", (i,)))
        gates += [("cp", (i, j)) for j in range(i + 1, n)]
    return gates


def remote_pairs(gates, q, placement=None):
    """Bell pairs a telegate-only run needs for a given starting placement."""
    loc = (lambda x: placement[x]) if placement is not None else (lambda x: x // q)
    return sum((2 if name == "swap" else 1) for name, qs in gates
               if len(qs) == 2 and loc(qs[0]) != loc(qs[1]))


def _anchor_ok(name, pos):
    return name in DIAGONAL_2Q or (name in CONTROLLED_2Q and pos == 0)


def run_circuit(arch, gates, n, q, t_gate, t_1q=0.0, mode="telegate", spare=1,
                priority="list", placement=None, stats=None):
    """Execute a circuit; arch=None gives the monolithic baseline.

    spare    : free slots per module that a moved qubit may occupy
    priority : "list" serves ready gates in circuit order, "critical" serves the
               gate with the longest chain of dependent gates first
    placement: list giving the starting module of every qubit
    stats    : optional dict filled with counts of pairs, shuttles, bursts, moves
    Returns (makespan in seconds, number of Bell pairs consumed).
    """
    assert mode in MODES
    loc = list(placement) if placement is not None else [x // q for x in range(n)]
    occ = {}
    for m in loc:
        occ[m] = occ.get(m, 0) + 1
    cap = q + spare
    seq = [[] for _ in range(n)]
    for idx, (_, qs) in enumerate(gates):
        for x in qs:
            seq[x].append(idx)
    pos = [{idx: k for k, idx in enumerate(s)} for s in seq]
    ptr = [0] * n

    rank = None
    if priority == "critical":
        rank = [1] * len(gates)
        for idx in range(len(gates) - 1, -1, -1):
            for x in gates[idx][1]:
                k = pos[x][idx] + 1
                if k < len(seq[x]):
                    rank[idx] = max(rank[idx], 1 + rank[seq[x][k]])

    covered = {}          # gate -> (anchor qubit, module holding its shared copy)
    left = {}             # remote SWAP -> pairs still needed
    counts = {"pairs": 0, "shuttles": 0, "bursts": 0, "moves": 0, "max_occupancy": max(occ.values())}

    def ready(idx):
        return all(seq[x][ptr[x]] == idx for x in gates[idx][1])

    def run_of(x, idx, module, burst):
        """Upcoming gates of x, starting at idx, that a burst (shared copy of x in
        `module`) or a move of x to `module` would serve."""
        served = []
        for j in seq[x][pos[x][idx]:]:
            name, qs = gates[j]
            if len(qs) == 1:
                if burst and name not in DIAGONAL_1Q:
                    break
                continue
            y = qs[1] if qs[0] == x else qs[0]
            if loc[y] != module or (burst and not _anchor_ok(name, qs.index(x))):
                break
            served.append(j)
        return served

    waiting = []
    for x in range(n):
        if seq[x] and ready(seq[x][0]) and seq[x][0] not in waiting:
            waiting.append(seq[x][0])
    done = []             # heap of (time, tiebreak, gate, action)
    tick = 0
    now, makespan = 0.0, 0.0

    def push(t, idx, action):
        nonlocal tick
        tick += 1
        heapq.heappush(done, (t, tick, idx, action))

    def start_remote(idx):
        """Try to start remote gate idx. Returns False if it must wait."""
        name, (a, b) = gates[idx]
        ma, mb = loc[a], loc[b]
        if name == "swap" or mode == "telegate":
            end = arch.try_start(ma, mb, now)
            if end is None:
                return False
            counts["pairs"] += 1
            if name == "swap":
                left[idx] = left.get(idx, 2) - 1
                push(end + (t_gate if left[idx] == 0 else 0.0), idx, "finish" if left[idx] == 0 else "again")
            else:
                push(end + t_gate, idx, "finish")
            return True
        # candidates: (gates served, kind, qubit, destination module)
        options = []
        if mode in ("burst", "adaptive"):
            for x, m in ((a, mb), (b, ma)):
                served = run_of(x, idx, m, burst=True)
                if served:
                    options.append((len(served), 0, "burst", x, m, served))
        if mode in ("teledata", "adaptive"):
            for x, src, m in ((a, ma, mb), (b, mb, ma)):
                if occ.get(m, 0) < cap:
                    served = run_of(x, idx, m, burst=False)
                    # prefer a move over a burst on a tie only if it needs no Bell pair
                    bonus = 1 if arch.can_move(src, m) else -1
                    options.append((len(served), bonus, "move", x, m, served))
        if not options:                                   # plain telegate
            end = arch.try_start(ma, mb, now)
            if end is None:
                return False
            counts["pairs"] += 1
            push(end + t_gate, idx, "finish")
            return True
        _, _, kind, x, m, served = max(options, key=lambda o: (o[0], o[1]))
        src = loc[x]
        if kind == "burst":
            end = arch.try_start(ma, mb, now)
            if end is None:
                return False
            counts["pairs"] += 1
            counts["bursts"] += 1
            for j in served[1:]:
                covered[j] = (x, m)
            push(end + t_gate, idx, "finish")
            return True
        if arch.can_move(src, m):
            end = arch.try_move(src, m, now)
            if end is None:
                return False
            counts["shuttles"] += 1
        else:
            end = arch.try_start(src, m, now)
            if end is None:
                return False
            counts["pairs"] += 1
        counts["moves"] += 1
        occ[m] = occ.get(m, 0) + 1                        # reserve the slot
        counts["max_occupancy"] = max(counts["max_occupancy"], occ[m])
        push(end, idx, ("arrive", x, src, m))
        return True

    while waiting or done:
        if rank is not None:
            waiting.sort(key=lambda i: -rank[i])
        still = []
        for idx in waiting:
            name, qs = gates[idx]
            if len(qs) == 1:
                push(now + t_1q, idx, "finish")
                continue
            a, b = qs
            local = arch is None or loc[a] == loc[b]
            if not local and idx in covered:
                x, m = covered[idx]
                y = b if x == a else a
                local = loc[y] == m and loc[x] != m       # the shared copy is still valid
            if local:
                push(now + t_gate, idx, "finish")
                continue
            ok = start_remote(idx)
            if ok is False:
                still.append(idx)
        waiting = still
        nxt = done[0][0] if done else math.inf
        if arch is not None:
            while arch.events and arch.events[0] <= now:
                heapq.heappop(arch.events)
            if waiting and arch.events:
                nxt = min(nxt, arch.events[0])
        if not math.isfinite(nxt):
            if waiting:
                return math.inf, counts["pairs"]          # a pair cannot reach the target
            break
        now = nxt
        while done and done[0][0] <= now:
            end, _, idx, action = heapq.heappop(done)
            if not math.isfinite(end):
                return math.inf, counts["pairs"]
            makespan = max(makespan, end)
            if action == "again":                         # remote SWAP: second pair
                waiting.append(idx)
                continue
            if action != "finish":                        # a moved qubit arrives
                _, x, src, m = action
                occ[src] -= 1
                loc[x] = m
                waiting.append(idx)
                continue
            for x in gates[idx][1]:
                ptr[x] += 1
            for x in gates[idx][1]:
                if ptr[x] < len(seq[x]):
                    h = seq[x][ptr[x]]
                    if ready(h) and h not in waiting:
                        waiting.append(h)
    if stats is not None:
        stats.update(counts)
    return makespan, counts["pairs"]
