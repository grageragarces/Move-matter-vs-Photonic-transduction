"""Q-Fly topology helpers and the three architectures.

Three architectures over the same N = g * p modules:
  PhotonicQFly : dual-path quasi-half-duplex Q-Fly (Sakuma et al.), every module
                 has one optical port into its group switch.
  HybridQFly   : each group is a shuttling-connected 2D cluster in one vacuum
                 system; only c interface modules per group keep optical ports.
  MatterGrid   : one 2D array of N modules connected by matter links only.
"""
import heapq
import math
import itertools
import numpy as np
from core import (photonic, matter, sample_photonic, werner, depolarise, dephase, data_fidelity,
                  F_TARGET)


def group_loss(k, x2):
    """Benes group switch of radix k: (2 ceil(log2 k) - 1) switch points."""
    return (2 * math.ceil(math.log2(k)) - 1) * x2 if k > 1 else 0.0


def radix_dphd(p, g):
    return p + g - 1


FIBRE_INTRA, FIBRE_INTER = 10.0, 30.0   # total fibre length per path [m]


def path_loss(reg, k, inter):
    """Q-Fly Eq. (1) without the BSA term (kept in p_herald): N_gsw+1 = 2 or 3."""
    n = 3 if inter else 2
    return n * group_loss(k, reg.x2) + (FIBRE_INTER if inter else FIBRE_INTRA) * reg.lf


# ---------------------------------------------------------------- geometry
def grid_shape(p):
    """Most nearly square full rectangle if p factorises, else a ragged grid."""
    cols = math.ceil(math.sqrt(p))
    for c in range(cols, 2 * cols + 1):
        if p % c == 0:
            return c, p // c
    return cols, math.ceil(p / cols)


def coords(i, cols):
    return i % cols, i // cols


def grid_path(a, b, cols, p):
    """Sequence of links (as sorted index pairs) from module a to b on a grid."""
    (xa, ya), (xb, yb) = coords(a, cols), coords(b, cols)
    last = (p - 1) // cols
    steps = []
    x, y = xa, ya

    def go_x():
        nonlocal x
        while x != xb:
            nx = x + (1 if xb > x else -1)
            steps.append(((x, y), (nx, y)))
            x = nx

    def go_y():
        nonlocal y
        while y != yb:
            ny = y + (1 if yb > y else -1)
            steps.append(((x, y), (x, ny)))
            y = ny

    if ya == last and yb != last:
        go_y(); go_x()
    else:
        go_x(); go_y()
    return [tuple(sorted(s)) for s in steps]


def mean_hops_chain(p):
    return (p + 1) / 3.0


def mean_hops_grid(p):
    cols, _ = grid_shape(p)
    pts = [coords(i, cols) for i in range(p)]
    d = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in itertools.combinations(pts, 2)]
    return float(np.mean(d))


# ---------------------------------------------------------------- scheduler
class Pool:
    """A set of identical resources (BSAs, fibres)."""

    def __init__(self, n):
        self.free = [0.0] * n

    def idle(self, now):
        for i, t in enumerate(self.free):
            if t <= now:
                return i
        return None


class _Base:
    """Resources shared by all architectures.

    try_start() returns None if a request cannot start now, else its completion
    time; it pushes onto self.events every time at which a resource it holds is
    released, so that the circuit simulator knows when to try again.
    """

    def __init__(self, reg, g, p, rng, target=F_TARGET):
        self.reg, self.g, self.p, self.rng, self.target = reg, g, p, rng, target
        self.n = g * p
        self.node = [0.0] * self.n
        self.link = {}
        self.events = []
        self._mcache = {}
        self.dt = reg.pitch / reg.v

    def shuttle(self, t, links):
        """Move an ion along links; each link carries one ion at a time."""
        for l in links:
            t = max(t, self.link.get(l, 0.0)) + self.dt
            self.link[l] = t
        return t

    def _matter(self, u, v, path, now):
        """Prepare a pair at u, shuttle one half along path, recool at v."""
        r = self.reg
        if self.node[u] > now:
            return None
        d = self._mcache.get(len(path))
        if d is None:
            d = self._mcache[len(path)] = matter(r, len(path), self.target)
        if not math.isfinite(d.time):
            return math.inf
        extra = d.time - len(path) * self.dt - r.t_loc - r.t_cool   # distillation, if any
        self.node[u] = now + r.t_loc
        arrive = self.shuttle(self.node[u], path)
        end = max(arrive, self.node[v]) + r.t_cool + max(0.0, extra)
        self.node[v] = end
        self._push(self.node[u], end)
        return end

    # ---- moving a data qubit itself (no Bell pair) ----
    def move_path(self, u, v):
        """Matter-link path from module u to v, or None if there is none."""
        return None

    def can_move(self, u, v):
        path = self.move_path(u, v)
        return path is not None and data_fidelity(self.reg, len(path)) >= self.target

    def try_move(self, u, v, now):
        """Shuttle a data qubit from u to v. Returns arrival time, or None to wait."""
        if self.node[u] > now:
            return None
        arrive = self.shuttle(now, self.move_path(u, v))
        end = max(arrive, self.node[v]) + self.reg.t_cool
        self.node[v] = end
        self._push(end)
        return end

    def _push(self, *times):
        for t in times:
            heapq.heappush(self.events, t)


class PhotonicQFly(_Base):
    def __init__(self, reg, g, p, rng, **kw):
        super().__init__(reg, g, p, rng, **kw)
        self.k = radix_dphd(p, g)
        self.bsa = [Pool(max(1, p // 2)) for _ in range(g)]
        self.fibre = {}
        self.loss = {False: path_loss(reg, self.k, False), True: path_loss(reg, self.k, True)}

    def try_start(self, u, v, now):
        if self.node[u] > now or self.node[v] > now:
            return None
        gu, gv = u // self.p, v // self.p
        inter = gu != gv
        for pool in (self.bsa[gu], self.bsa[gv]):
            i = pool.idle(now)
            if i is not None:
                break
        else:
            return None
        if inter:
            fib = self.fibre.setdefault((min(gu, gv), max(gu, gv)), Pool(2))
            j = fib.idle(now)
            if j is None:
                return None
        end = now + sample_photonic(self.reg, self.loss[inter], self.rng, self.target)[0]
        self.node[u] = self.node[v] = pool.free[i] = end
        if inter:
            fib.free[j] = end
        self._push(end)
        return end


class MatterGrid(_Base):
    """All N modules in a single 2D array.

    Each Q-Fly group occupies a contiguous rectangular block, and the blocks are
    tiled row by row, so that "intra-group" requests are spatially local.
    """

    def __init__(self, reg, g, p, rng, **kw):
        super().__init__(reg, g, p, rng, **kw)
        self.cp, self.rp = grid_shape(p)
        assert self.cp * self.rp == p, "group must fill a rectangle"
        self.bc = math.ceil(math.sqrt(g))
        self.last = ((g - 1) // self.bc) * self.rp   # first row of the last block row
        self._paths = {}

    def xy(self, u):
        grp, m = divmod(u, self.p)
        return ((grp % self.bc) * self.cp + m % self.cp,
                (grp // self.bc) * self.rp + m // self.cp)

    def path(self, u, v):
        key = (u, v)
        if key not in self._paths:
            (x, y), (xb, yb) = self.xy(u), self.xy(v)
            steps = []

            def go_x(x, y):
                while x != xb:
                    nx = x + (1 if xb > x else -1)
                    steps.append(tuple(sorted(((x, y), (nx, y)))))
                    x = nx
                return x

            def go_y(x, y):
                while y != yb:
                    ny = y + (1 if yb > y else -1)
                    steps.append(tuple(sorted(((x, y), (x, ny)))))
                    y = ny
                return y

            # the last block row may be incomplete: never travel along it past its end
            if y >= self.last and yb < self.last:
                y = go_y(x, y); go_x(x, y)
            else:
                x = go_x(x, y); go_y(x, y)
            self._paths[key] = steps
        return self._paths[key]

    def try_start(self, u, v, now):
        return self._matter(u, v, self.path(u, v), now)

    def move_path(self, u, v):
        return self.path(u, v)


class HybridQFly(_Base):
    """Groups are shuttling clusters; c interface modules per group hold the optics.

    Photonic pairs are heralded and distilled between interface modules, then
    both halves are shuttled to the end modules. Distillation continues until the
    pair meets the target *after* that final transport. Intra-group pairs use
    whichever of the two mechanisms is faster in expectation.
    """

    def __init__(self, reg, g, p, c, rng, pooling=True, **kw):
        super().__init__(reg, g, p, rng, **kw)
        self.c = c
        self.pooling = pooling        # False: a module may only use its nearest interface
        self.cols, _ = grid_shape(p)
        self.k = radix_dphd(c, g)
        self.iface = [int((i + 0.5) * p / c) for i in range(c)]
        self.port = [[0.0] * c for _ in range(g)]
        self.bsa = [Pool(max(1, c // 2)) for _ in range(g)]
        self.fibre = {}
        self.loss = {False: path_loss(reg, self.k, False), True: path_loss(reg, self.k, True)}
        self.hop = [[len(grid_path(i, m, self.cols, p)) for m in range(p)] for i in self.iface]
        self.wait = {i: photonic(reg, self.loss[i], self.target).time for i in (False, True)}
        self.stats = {"matter": 0, "photonic": 0}
        self._use_photonic = {}

    def _links(self, grp, a, b):
        return [(grp, l) for l in grid_path(a, b, self.cols, self.p)]

    def _post(self, hu, hv):
        """Noise picked up while the two halves are shuttled inward and recooled."""
        r = self.reg
        e = (4.0 / 3.0) * (1.0 - (1.0 - r.eps_x) ** (hu + hv))
        h = max(hu, hv)
        if h == 0:
            return None                     # both ends use their own optical port
        t = h * self.dt + r.t_cool
        return lambda s: dephase(depolarise(s, e), t, r.t2)

    def _idle_port(self, grp, m, now, inter, exclude=None):
        """Nearest idle interface module. A farther one is used only if the extra
        transport costs less than waiting for one photonic delivery."""
        cands = [i for i in range(self.c) if i != exclude]
        hmin = min(self.hop[i][m] for i in cands)
        idle = [i for i in cands if self.port[grp][i] <= now and
                (self.hop[i][m] == hmin or
                 (self.pooling and
                  self.hop[i][m] * self.dt + self.reg.t_cool < self.wait[inter]))]
        return min(idle, key=lambda i: self.hop[i][m]) if idle else None

    def _intra_photonic_faster(self, mu, mv):
        key = (mu, mv)
        if key not in self._use_photonic:
            h = len(grid_path(mu, mv, self.cols, self.p))
            tm = matter(self.reg, h, self.target).time
            tp = math.inf
            if self.c >= 2:
                iu = min(range(self.c), key=lambda i: self.hop[i][mu])
                iv = min((i for i in range(self.c) if i != iu), key=lambda i: self.hop[i][mv])
                hu, hv = self.hop[iu][mu], self.hop[iv][mv]
                post = self._post(hu, hv)
                tp = photonic(self.reg, self.loss[False], self.target, post).time
                if not math.isfinite(tp):
                    tp = photonic(self.reg, self.loss[False], self.target,
                                  raw=post(werner(self.reg.f_phot))).time
                tp += max(hu, hv) * self.dt + self.reg.t_cool if max(hu, hv) else 0.0
            self._use_photonic[key] = tp < tm
        return self._use_photonic[key]

    def _photonic(self, u, v, gu, gv, mu, mv, now):
        r = self.reg
        inter = gu != gv
        iu = self._idle_port(gu, mu, now, inter)
        if iu is None:
            return None
        iv = self._idle_port(gv, mv, now, inter, exclude=None if inter else iu)
        if iv is None:
            return None
        for pool in (self.bsa[gu], self.bsa[gv]):
            ib = pool.idle(now)
            if ib is not None:
                break
        else:
            return None
        if inter:
            fib = self.fibre.setdefault((min(gu, gv), max(gu, gv)), Pool(2))
            jf = fib.idle(now)
            if jf is None:
                return None
        post = self._post(self.hop[iu][mu], self.hop[iv][mv])
        dur = sample_photonic(r, self.loss[inter], self.rng, self.target, post)[0]
        ions = 1
        if not math.isfinite(dur):
            # The target cannot be met after transport: shuttle the raw pairs
            # inward instead and distil at the end modules.
            raw = post(werner(r.f_phot))
            ref = photonic(r, self.loss[inter], self.target, raw=raw)
            if not math.isfinite(ref.time):
                return math.inf
            dur = sample_photonic(r, self.loss[inter], self.rng, self.target, raw=raw)[0]
            ions = max(1, round(ref.raw_pairs))
            self.stats["end_distilled"] = self.stats.get("end_distilled", 0) + 1
        t1 = now + dur
        self.port[gu][iu] = self.port[gv][iv] = pool.free[ib] = t1
        if inter:
            fib.free[jf] = t1
        end = 0.0
        for grp, i, m, node in ((gu, iu, mu, u), (gv, iv, mv, v)):
            if self.hop[i][m] == 0:
                end = max(end, t1)
                continue
            links = self._links(grp, self.iface[i], m)
            for j in range(ions):          # raw ions leave evenly spaced; the last at t1
                arrive = self.shuttle(now + (j + 1) * dur / ions, links)
            done = max(arrive, self.node[node]) + r.t_cool
            self.node[node] = done
            end = max(end, done)
        self._push(t1, end)
        return end

    def move_path(self, u, v):
        if u // self.p != v // self.p:
            return None                      # no matter link between groups
        return self._links(u // self.p, u % self.p, v % self.p)

    def try_start(self, u, v, now):
        gu, gv = u // self.p, v // self.p
        mu, mv = u % self.p, v % self.p
        if gu != gv or self._intra_photonic_faster(mu, mv):
            end = self._photonic(u, v, gu, gv, mu, mv, now)
            if end is not None:
                self.stats["photonic"] += 1
            return end
        end = self._matter(u, v, self._links(gu, mu, mv), now)
        if end is not None:
            self.stats["matter"] += 1
        return end
