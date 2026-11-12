import time
import itertools
from collections import deque
import numpy as np

G = 21

def _parse(obs):
    corner = obs[0:63:3, 0:63:3]
    center = obs[1:63:3, 1:63:3]
    tg = (center == 11) | (corner == 11)
    targets = [(x, y) for y in range(G) for x in range(G) if tg[y, x]]
    is10 = corner == 10
    vm, hm = [], []
    for x in range(G):
        if is10[:, x].sum() >= 7:
            fixed = int((is10[:, x] & (center[:, x] == 10)).sum()) >= 3
            vm.append([x, fixed])
    for y in range(G):
        if is10[y, :].sum() >= 7:
            fixed = int((is10[y, :] & (center[y, :] == 10)).sum()) >= 3
            hm.append([y, fixed])
    p5 = corner == 5
    seen = set()
    pieces = []
    for y in range(G):
        for x in range(G):
            if p5[y, x] and (x, y) not in seen:
                comp = []
                dq = deque([(x, y)])
                seen.add((x, y))
                while dq:
                    cx, cy = dq.popleft()
                    comp.append((cx, cy))
                    for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
                        if 0 <= nx < G and 0 <= ny < G and p5[ny, nx] and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            dq.append((nx, ny))
                mx = min(c[0] for c in comp)
                my = min(c[1] for c in comp)
                w = max(c[0] for c in comp) - mx + 1
                h = max(c[1] for c in comp) - my + 1
                fixed = any(center[cy, cx] == 5 for cx, cy in comp)
                rel = [(cx - mx, cy - my) for cx, cy in comp]
                pieces.append({"x": mx, "y": my, "w": w, "h": h, "cells": rel, "fixed": fixed})
    return targets, vm, hm, pieces


def _solve(targets, vm, hm, pieces, tlimit=5.0):
    t0 = time.time()
    tidx = {}
    for i, c in enumerate(targets):
        tidx[c] = i
    T = (1 << len(targets)) - 1
    cache = {}

    def getmask(pi, fh, fv, ox, oy):
        key = (pi, fh, fv, ox, oy)
        r = cache.get(key)
        if r is not None:
            return r
        p = pieces[pi]
        w, h = p["w"], p["h"]
        m = 0
        if not (ox > G - 1 or ox + w - 1 < 0 or oy > G - 1 or oy + h - 1 < 0):
            for dx, dy in p["cells"]:
                x = ox + (w - 1 - dx if fh else dx)
                y = oy + (h - 1 - dy if fv else dy)
                b = tidx.get((x, y))
                if b is not None:
                    m |= 1 << b
        cache[key] = m
        return m

    movV = [i for i, m in enumerate(vm) if not m[1]]
    movH = [i for i, m in enumerate(hm) if not m[1]]
    nm = len(movV) + len(movH)
    rad = 20
    while (2 * rad + 1) ** nm > 4000 and rad > 1:
        rad -= 1
    ranges = []
    for i in movV:
        ranges.append([v for v in range(G) if abs(v - vm[i][0]) <= rad])
    for i in movH:
        ranges.append([v for v in range(G) if abs(v - hm[i][0]) <= rad])
    configs = []
    for combo in itertools.product(*ranges):
        cost = 0
        k = 0
        for i in movV:
            d = abs(combo[k] - vm[i][0]); cost += d + (1 if d else 0); k += 1
        for i in movH:
            d = abs(combo[k] - hm[i][0]); cost += d + (1 if d else 0); k += 1
        configs.append((cost, combo))
    configs.sort(key=lambda c: c[0])

    best = [10 ** 9, None]
    movP = [i for i, p in enumerate(pieces) if not p["fixed"]]
    fixP = [i for i, p in enumerate(pieces) if p["fixed"]]

    for mcost, combo in configs:
        if mcost >= best[0] or time.time() - t0 > tlimit:
            break
        vpos = [m[0] for m in vm]
        hpos = [m[0] for m in hm]
        k = 0
        for i in movV:
            vpos[i] = combo[k]; k += 1
        for i in movH:
            hpos[i] = combo[k]; k += 1
        trans_cache = {}

        def transforms(w, h):
            key = (w, h)
            if key in trans_cache:
                return trans_cache[key]
            start = (0, 0, 0, 0)
            seenT = {start: 0}
            dq = deque([start])
            while dq:
                st = dq.popleft()
                d = seenT[st]
                if d > 12:
                    continue
                fh, cx, fv, cy = st
                for a in vpos:
                    ns = (1 - fh, 2 * a - (w - 1) - cx, fv, cy)
                    if ns not in seenT:
                        seenT[ns] = d + 1; dq.append(ns)
                for b in hpos:
                    ns = (fh, cx, 1 - fv, 2 * b - (h - 1) - cy)
                    if ns not in seenT:
                        seenT[ns] = d + 1; dq.append(ns)
            res = list(seenT.keys())
            trans_cache[key] = res
            return res

        covered0 = 0
        for pi in fixP:
            p = pieces[pi]
            for fh, cx, fv, cy in transforms(p["w"], p["h"]):
                ox = (-p["x"] if fh else p["x"]) + cx
                oy = (-p["y"] if fv else p["y"]) + cy
                covered0 |= getmask(pi, fh, fv, ox, oy)
        cands = {}
        union = covered0
        for pi in movP:
            p = pieces[pi]
            tr = transforms(p["w"], p["h"])
            lst = []
            for px in range(0, G - p["w"] + 1):
                for py in range(0, G - p["h"] + 1):
                    m = 0
                    for fh, cx, fv, cy in tr:
                        m |= getmask(pi, fh, fv, (-px if fh else px) + cx, (-py if fv else py) + cy)
                    d = abs(px - p["x"]) + abs(py - p["y"])
                    c = d + (1 if d else 0)
                    if m:
                        lst.append((c, px, py, m))
                        union |= m
            lst.sort(key=lambda e: e[0])
            cands[pi] = lst
        if union & T != T:
            continue
        nodes = [0]
        assign = {}

        def dfs(unassigned, covered, cost):
            nodes[0] += 1
            if nodes[0] > 20000:
                return
            if covered & T == T:
                if cost < best[0]:
                    best[0] = cost
                    best[1] = (combo, dict(assign))
                return
            rem = T & ~covered
            t = (rem & -rem).bit_length() - 1
            for pi in list(unassigned):
                for c, px, py, m in cands[pi]:
                    if cost + c >= best[0]:
                        break
                    if (m >> t) & 1:
                        assign[pi] = (px, py)
                        unassigned.remove(pi)
                        dfs(unassigned, covered | m, cost + c)
                        unassigned.add(pi)
                        del assign[pi]

        dfs(set(movP), covered0, mcost)
    if best[1] is None:
        return None
    combo, assign = best[1]
    return combo, assign, movV, movH


def _plan(obs):
    targets, vm, hm, pieces = _parse(obs)
    if not targets:
        return []
    sol = _solve(targets, vm, hm, pieces)
    if sol is None:
        return []
    combo, assign, movV, movH = sol
    acts = []
    cur = {i: (p["x"], p["y"]) for i, p in enumerate(pieces)}

    def piece_cells(excl=None):
        s = set()
        for i, p in enumerate(pieces):
            if i == excl:
                continue
            px, py = cur[i]
            for dx, dy in p["cells"]:
                s.add((px + dx, py + dy))
        return s

    vcur = [m[0] for m in vm]
    hcur = [m[0] for m in hm]
    k = 0
    goals = []
    for i in movV:
        goals.append(("v", i, combo[k])); k += 1
    for i in movH:
        goals.append(("h", i, combo[k])); k += 1
    for kind, i, g in goals:
        pos = vcur[i] if kind == "v" else hcur[i]
        if g == pos:
            continue
        occ = piece_cells()
        cell = None
        for t in range(G):
            c = (pos, t) if kind == "v" else (t, pos)
            if c in occ:
                continue
            if kind == "v" and c[1] in hcur:
                continue
            if kind == "h" and c[0] in vcur:
                continue
            cell = c; break
        if cell is None:
            continue
        acts.append(("ACTION6", cell[1] * 3 + 1, cell[0] * 3 + 1))
        if kind == "v":
            a = "ACTION4" if g > pos else "ACTION3"
            vcur[i] = g
        else:
            a = "ACTION2" if g > pos else "ACTION1"
            hcur[i] = g
        acts += [(a, None, None)] * abs(g - pos)
    for pi, (gx, gy) in assign.items():
        px, py = cur[pi]
        if (gx, gy) == (px, py):
            continue
        occ = piece_cells(excl=pi)
        p = pieces[pi]
        cell = None
        for dx, dy in p["cells"]:
            c = (px + dx, py + dy)
            if c not in occ:
                cell = c; break
        if cell is None:
            dx, dy = p["cells"][0]
            cell = (px + dx, py + dy)
        acts.append(("ACTION6", cell[1] * 3 + 1, cell[0] * 3 + 1))
        acts += [("ACTION4" if gx > px else "ACTION3", None, None)] * abs(gx - px)
        acts += [("ACTION2" if gy > py else "ACTION1", None, None)] * abs(gy - py)
        cur[pi] = (gx, gy)
    return acts


def act(obs, available, memory):
    obs = np.asarray(obs)
    q = memory.get("queue")
    if not q:
        try:
            q = _plan(obs)
        except Exception:
            q = []
        if not q:
            seq = ["ACTION1", "ACTION4", "ACTION2", "ACTION3"]
            n = memory.get("rnd", 0)
            memory["rnd"] = n + 1
            a = seq[n % 4]
            if a not in available and available:
                a = available[0]
            return (a, None, None)
        memory["queue"] = q
    a = q.pop(0)
    memory["queue"] = q
    if a[0] == "ACTION6":
        return ("ACTION6", int(a[1]), int(a[2]))
    return (a[0], None, None)
