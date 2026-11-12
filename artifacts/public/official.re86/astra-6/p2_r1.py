import numpy as np
import itertools
import time
from collections import deque

_COLORS = tuple(range(6, 15))
_DIRS = ((0, -3), (0, 3), (-3, 0), (3, 0))


def _fresh():
    return {
        "phase": "scan",
        "centers": [],
        "clean": {},
        "scan_steps": 0,
        "blind_steps": 0,
        "targets": {},
        "bad_targets": {},
        "geoms": [],
        "gmap": {},
        "turn": 0,
        "idle": 0,
    }


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        steps += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for yy, xx in zip(*np.where(mask)):
        x, y = int(xx), int(yy)
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(x, y)]
        cells = []
        for x, y in q:
            cells.append((x, y))
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((nx, ny))
        result.append(cells)
    return result


def _find_targets(obs):
    a = (obs == 4)
    a[63, :] = False
    count = np.zeros((62, 62), dtype=np.int16)
    for dy in range(3):
        for dx in range(3):
            if dx != 1 or dy != 1:
                count += a[dy:dy + 62, dx:dx + 62]
    candidates = []
    ys, xs = np.where(count >= 4)
    for y, x in zip(ys, xs):
        x, y = int(x + 1), int(y + 1)
        if obs[y, x] != 4:
            candidates.append((int(count[y - 1, x - 1]), x, y))
    candidates.sort(reverse=True)
    out = []
    for score, x, y in candidates:
        if not any(abs(x - xx) <= 2 and abs(y - yy) <= 2
                   for xx, yy in out):
            out.append((x, y))
    return out


def _static_objects(obs):
    groups = _components(obs[:63] == 2)
    groups = [g for g in groups if len(g) >= 2]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            a = groups[i]
            ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
            ay0, ay1 = min(p[1] for p in a), max(p[1] for p in a)
            for j in range(i + 1, len(groups)):
                b = groups[j]
                bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
                by0, by1 = min(p[1] for p in b), max(p[1] for p in b)
                w = max(ax1, bx1) - min(ax0, bx0) + 1
                h = max(ay1, by1) - min(ay0, by0) + 1
                gapx = max(0, ax0 - bx1 - 1, bx0 - ax1 - 1)
                gapy = max(0, ay0 - by1 - 1, by0 - ay1 - 1)
                if w <= 9 and h <= 9 and gapx <= 1 and gapy <= 1:
                    groups[i] += groups.pop(j)
                    changed = True
                    break

    pads = []
    pad_inside = np.zeros((64, 64), dtype=bool)
    for g in groups:
        x0, x1 = min(p[0] for p in g), max(p[0] for p in g)
        y0, y1 = min(p[1] for p in g), max(p[1] for p in g)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if min(w, h) < 4 or max(w, h) > 9 or len(g) < 6:
            continue
        inner = obs[y0 + 1:y1, x0 + 1:x1]
        vals = inner[np.isin(inner, _COLORS)]
        if not len(vals):
            continue
        corners = [int(obs[y0 + 1, x0 + 1]),
                   int(obs[y0 + 1, x1 - 1]),
                   int(obs[y1 - 1, x0 + 1]),
                   int(obs[y1 - 1, x1 - 1])]
        corners = [c for c in corners if c in _COLORS]
        if corners:
            col = int(np.bincount(corners, minlength=16).argmax())
        else:
            col = int(np.bincount(vals, minlength=16).argmax())
        pads.append((x0, y0, w, h, col))
        pad_inside[y0 + 1:y1, x0 + 1:x1] = True

    mould = (255, 231, 195, 129, 129, 195, 231, 255)
    ones = obs[:63] == 1
    obstacles = []
    used = set()
    candidates = set()
    for yy, xx in zip(*np.where(ones)):
        for dy in (0, 1, 6, 7):
            for dx in (0, 1, 6, 7):
                x, y = int(xx) - dx, int(yy) - dy
                if 0 <= x <= 56 and 0 <= y <= 55:
                    candidates.add((x, y))
    fits = []
    for x, y in candidates:
        support = 0
        invalid = 0
        for dy, bits in enumerate(mould):
            for dx in range(8):
                if bits & (1 << dx):
                    v = int(obs[y + dy, x + dx])
                    support += v == 1
                    invalid += v == 5
        if support >= 10 and invalid == 0:
            fits.append((support, x, y))
    for support, x, y in sorted(fits, reverse=True):
        cells = {(x + dx, y + dy)
                 for dy, bits in enumerate(mould)
                 for dx in range(8) if bits & (1 << dx)}
        if len(cells & used) > 8:
            continue
        obstacles.append((x, y, 8, 8, mould))
        used |= cells

    leftovers = ones.copy()
    for x, y in used:
        if y < 63:
            leftovers[y, x] = False
    for cells in _components(leftovers, diagonal=True):
        if len(cells) < 4:
            continue
        x0, x1 = min(p[0] for p in cells), max(p[0] for p in cells)
        y0, y1 = min(p[1] for p in cells), max(p[1] for p in cells)
        rows = [0] * (y1 - y0 + 1)
        for x, y in cells:
            rows[y - y0] |= 1 << (x - x0)
        obstacles.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1,
                          tuple(rows)))
    return pads, obstacles, pad_inside


def _make_geometry(kind, w, h, u=0, v=0, special=False):
    # 0: cross or straight line; 1: box; 2: X; 3: diamond.
    if kind == 0:
        rows = [1 << u] * h
        rows[v] = (1 << w) - 1
    elif kind == 1:
        rows = [1 | (1 << (w - 1))] * h
        rows[0] = rows[-1] = (1 << w) - 1
    elif kind == 2:
        rows = [(1 << y) | (1 << (w - 1 - y)) for y in range(h)]
        if not special:
            rows[h // 2] &= ~(1 << (w // 2))
    else:
        r = w // 2
        rows = []
        for y in range(h):
            d = r - abs(y - h // 2)
            rows.append((1 << (r - d)) | (1 << (r + d)))
    points = []
    for y, bits in enumerate(rows):
        while bits:
            b = bits & -bits
            points.append((b.bit_length() - 1, y))
            bits -= b
    return (kind, w, h, u, v, bool(special), tuple(rows), tuple(points))


def _gid(m, kind, w, h, u=0, v=0, special=False):
    key = (kind, w, h, u, v, bool(special))
    old = m["gmap"].get(key)
    if old is not None:
        return old
    idx = len(m["geoms"])
    m["gmap"][key] = idx
    m["geoms"].append(_make_geometry(*key))
    return idx


def _fit_shape(obs, center, excluded, clean_pixel, m):
    cx, cy = center
    foreground = np.isin(obs, (0,) + _COLORS)
    best = [-1e30, None]

    def evaluate(kind, w, h, u=0, v=0):
        geo = _make_geometry(kind, w, h, u, v, False)
        pts = geo[7]
        xx = np.fromiter((cx - w // 2 + p[0] for p in pts),
                         dtype=np.int16, count=len(pts))
        yy = np.fromiter((cy - h // 2 + p[1] for p in pts),
                         dtype=np.int16, count=len(pts))
        keep = ((xx >= 0) & (xx < 64) & (yy >= 0) & (yy < 63) &
                ((xx != cx) | (yy != cy)))
        xx, yy = xx[keep], yy[keep]
        if not len(xx) or not np.all(foreground[yy, xx]):
            return
        vals = obs[yy, xx]
        take = (~excluded[yy, xx]) & (vals >= 6) & (vals <= 14)
        counts = np.bincount(vals[take], minlength=16)
        col = int(counts.argmax())
        support = int(counts[col])
        score = support - .025 * len(pts) - .002 * (w + h)
        if w % 2 and h % 2:
            score += .005
        if support >= 3 and score > best[0]:
            best[:] = [score, (kind, w, h, u, v, col)]

    line_options = [{c: [] for c in _COLORS},
                    {c: [] for c in _COLORS}]
    for axis in range(2):
        for off in range(-30, 31, 3):
            fixed = (cy if axis == 0 else cx) + off
            if not 0 <= fixed < 63:
                continue
            per_color = {c: (-1e20, 1) for c in _COLORS}
            for length in range(1, 64, 2):
                r = length // 2
                if axis == 0:
                    coords = [(cx + d, fixed) for d in range(-r, r + 1)]
                else:
                    coords = [(fixed, cy + d) for d in range(-r, r + 1)]
                counts = [0] * 16
                valid = True
                for x, y in coords:
                    if not (0 <= x < 64 and 0 <= y < 63):
                        continue
                    if (x, y) == (cx, cy):
                        continue
                    if not foreground[y, x]:
                        valid = False
                        break
                    col = int(obs[y, x])
                    if not excluded[y, x] and col in _COLORS:
                        counts[col] += 1
                if not valid:
                    continue
                for col in _COLORS:
                    score = counts[col] - .027 * length
                    if score > per_color[col][0]:
                        per_color[col] = (score, length)
            for col, (score, length) in per_color.items():
                if score > .5:
                    line_options[axis][col].append((score, length, off))

    for col in _COLORS:
        hs = sorted(line_options[0][col], reverse=True)[:6]
        vs = sorted(line_options[1][col], reverse=True)[:6]
        hs.append((0, 1, 0))
        vs.append((0, 1, 0))
        for _, w, dr in hs:
            for _, h, dc in vs:
                u, v = w // 2 + dc, h // 2 + dr
                if 0 <= u < w and 0 <= v < h and w + h > 4:
                    evaluate(0, w, h, u, v)

    for size in range(3, 64, 2):
        evaluate(2, size, size)
        evaluate(3, size, size)

    invalid = ~foreground
    invalid = invalid.copy()
    if 0 <= cy < 64 and 0 <= cx < 64:
        invalid[cy, cx] = False
    rp = np.pad(np.cumsum(invalid, axis=1), ((0, 0), (1, 0)))
    cp = np.pad(np.cumsum(invalid, axis=0), ((1, 0), (0, 0)))

    def row_bad(y, x0, x1):
        if not 0 <= y < 63:
            return 0
        a, b = max(0, x0), min(64, x1 + 1)
        return int(rp[y, b] - rp[y, a]) if b > a else 0

    def col_bad(x, y0, y1):
        if not 0 <= x < 64:
            return 0
        a, b = max(0, y0), min(63, y1 + 1)
        return int(cp[b, x] - cp[a, x]) if b > a else 0

    for w in range(3, 62):
        x0, x1 = cx - w // 2, cx - w // 2 + w - 1
        for h in range(3, 62):
            y0, y1 = cy - h // 2, cy - h // 2 + h - 1
            if (row_bad(y0, x0, x1) or row_bad(y1, x0, x1) or
                    col_bad(x0, y0, y1) or col_bad(x1, y0, y1)):
                continue
            evaluate(1, w, h)

    if best[1] is None:
        col = 8
        vals = obs[max(0, cy - 4):min(63, cy + 5),
                   max(0, cx - 4):min(64, cx + 5)]
        vals = vals[np.isin(vals, _COLORS)]
        if len(vals):
            col = int(np.bincount(vals, minlength=16).argmax())
        best[1] = (0, 19, 19, 9, 9, col)

    kind, w, h, u, v, col = best[1]
    special = kind == 2 and clean_pixel in _COLORS
    gid = _gid(m, kind, w, h, u, v, special)
    return (cx - w // 2, cy - h // 2, gid, col)


def _hit(geo, x, y, obj, center=False):
    ox, oy, ow, oh = obj[:4]
    w, h, rows = geo[1], geo[2], geo[6]
    if x >= ox + ow or x + w <= ox or y >= oy + oh or y + h <= oy:
        return False
    lo, hi = max(y, oy), min(y + h, oy + oh)
    shift = x - ox
    mask = (1 << ow) - 1
    solid = isinstance(obj[4], (int, np.integer))
    for yy in range(lo, hi):
        bits = rows[yy - y]
        if center and yy - y == h // 2:
            bits |= 1 << (w // 2)
        bits = (bits << shift) if shift >= 0 else (bits >> -shift)
        other = mask if solid else obj[4][yy - oy]
        if bits & other:
            return True
    return False


def _kinematic(m, x, y, gid, action):
    dx, dy = _DIRS[action - 1]
    nx, ny = x + dx, y + dy
    geo = m["geoms"][gid]
    kind, w, h, u, v, special = geo[:6]
    if not (0 <= nx + w // 2 < 64 and 0 <= ny + h // 2 < 64):
        return (x, y, gid, ())
    ng = gid
    for obj in m["obstacles"]:
        geo = m["geoms"][ng]
        if not _hit(geo, nx, ny, obj, True):
            continue
        kind, w, h, u, v, special = geo[:6]
        ox, oy, ow, oh = obj[:4]
        if kind in (1, 3):
            if dx:
                if w <= 6:
                    return (x, y, gid, ())
                nw, nh = w - 3, h + 3
                ny += h // 2 - nh // 2
                ny = int(round(ny / 3.0) * 3)
                if dx < 0:
                    nx -= dx
            else:
                if h <= 6:
                    return (x, y, gid, ())
                nw, nh = w + 3, h - 3
                nx += w // 2 - nw // 2
                nx = int(round(nx / 3.0) * 3)
                if dy < 0:
                    ny -= dy
            ng = _gid(m, 1, nw, nh)
            return (nx, ny, ng, ())

        if kind != 0:
            return (x, y, gid, ())
        col_in = ox <= nx + u < ox + ow
        row_in = oy <= ny + v < oy + oh
        if col_in and row_in:
            nx -= dx
            ny -= dy
        elif dx:
            sign = 1 if dx > 0 else -1
            if col_in:
                can = u > 0 if sign > 0 else u < w - 2
                if can:
                    nu = u - 3 * sign
                    if nu < 0:
                        nu += w
                    if not 0 <= nu < w:
                        return None
                    u = nu
                else:
                    nx -= dx
            elif row_in:
                nx -= dx
                can = u < w - 2 if sign > 0 else u > 0
                if can:
                    nu = u + 3 * sign
                    if nu < 0:
                        nu += w
                    if not 0 <= nu < w:
                        return None
                    u = nu
        elif dy:
            sign = 1 if dy > 0 else -1
            if row_in:
                can = v > 0 if sign > 0 else v < h - 2
                if can:
                    nv = v - 3 * sign
                    if nv < 0:
                        nv += h
                    if not 0 <= nv < h:
                        return None
                    v = nv
                else:
                    ny -= dy
            elif col_in:
                ny -= dy
                can = v < h - 2 if sign > 0 else v > 0
                if can:
                    nv = v + 3 * sign
                    if nv < 0:
                        nv += h
                    if not 0 <= nv < h:
                        return None
                    v = nv
        ng = _gid(m, 0, w, h, u, v, special)

    geo = m["geoms"][ng]
    hits = []
    for pad in m["pads"]:
        if _hit(geo, nx, ny, pad):
            c = pad[4]
            if c not in hits:
                hits.append(c)
    return (nx, ny, ng, tuple(hits))


def _transition(m, state, action, strict=True):
    x, y, gid, color = state
    t = _kinematic(m, x, y, gid, action)
    if t is None:
        return None
    nx, ny, ng, hits = t
    different = [c for c in hits if c != color]
    if strict and len(different) > 1:
        return None
    return (nx, ny, ng, different[0] if different else color)


def _contains(m, state, x, y, selected=False):
    sx, sy, gid, color = state
    geo = m["geoms"][gid]
    dx, dy = x - sx, y - sy
    if not (0 <= dx < geo[1] and 0 <= dy < geo[2]):
        return False
    if selected and dx == geo[1] // 2 and dy == geo[2] // 2:
        return True
    return bool(geo[6][dy] & (1 << dx))


def _render_shapes(m, order=None):
    out = np.full((64, 64), -1, dtype=np.int16)
    if order is None:
        order = m["order"]
    for i in order:
        x, y, gid, color = m["states"][i]
        geo = m["geoms"][gid]
        for dx, dy in geo[7]:
            xx, yy = x + dx, y + dy
            if 0 <= xx < 64 and 0 <= yy < 63:
                out[yy, xx] = color
        if i == m["active"]:
            xx, yy = x + geo[1] // 2, y + geo[2] // 2
            if 0 <= xx < 64 and 0 <= yy < 63:
                out[yy, xx] = 0
    return out


def _choose_order(m, obs):
    n = len(m["states"])
    best = None
    for start in range(n):
        order = [(start + k) % n for k in range(n)]
        pred = _render_shapes(m, order)
        take = pred >= 0
        error = int(np.count_nonzero(pred[take] != obs[take]))
        if best is None or error < best[0]:
            best = (error, order)
    m["order"] = best[1]


def _invalidate(m):
    m["phase"] = "reveal"
    m.pop("commands", None)
    m.pop("search", None)
    m.pop("candidates", None)
    m["plan_index"] = 0


def _issue(m, a):
    m["turn"] = m.get("turn", 0) + 1
    m["idle"] = m.get("idle", 0) + 1 if a == 5 else 0
    if "states" in m:
        if a == 5:
            m["active"] = (m["active"] + 1) % len(m["states"])
        elif 1 <= a <= 4:
            i = m["active"]
            st = _transition(m, m["states"][i], a, False)
            if st is not None:
                m["states"][i] = st
    return ("ACTION" + str(a), None, None)


def _update_targets(m, obs):
    added = False
    for p in _find_targets(obs):
        if p not in m["targets"]:
            if not any(abs(p[0] - q[0]) <= 2 and abs(p[1] - q[1]) <= 2
                       for q in m["targets"]):
                m["targets"][p] = None
                added = True
    if "states" not in m:
        return added

    to_delete = []
    for (x, y), color in list(m["targets"].items()):
        covered = any(_contains(m, st, x, y, i == m["active"])
                      for i, st in enumerate(m["states"]))
        if not covered:
            val = int(obs[y, x])
            if val in _COLORS:
                if color is None:
                    m["targets"][(x, y)] = val
                    added = True
            elif color is None and val in (4, 5):
                key = (x, y)
                m["bad_targets"][key] = m["bad_targets"].get(key, 0) + 1
                if m["bad_targets"][key] >= 3:
                    to_delete.append(key)
    for p in to_delete:
        del m["targets"][p]
        added = True
    return added


def _synchronize(m, obs):
    changed = False
    zeros = list(zip(*np.where(obs[:63] == 0)))
    if len(zeros) == 1:
        yy, xx = map(int, zeros[0])
        i = m["active"]
        x, y, gid, col = m["states"][i]
        geo = m["geoms"][gid]
        ex, ey = x + geo[1] // 2, y + geo[2] // 2
        if (xx, yy) != (ex, ey):
            m["states"][i] = (xx - geo[1] // 2,
                              yy - geo[2] // 2, gid, col)
            changed = True

    owner = np.full((64, 64), -1, dtype=np.int16)
    for i in m["order"]:
        x, y, gid, col = m["states"][i]
        for dx, dy in m["geoms"][gid][7]:
            xx, yy = x + dx, y + dy
            if 0 <= xx < 64 and 0 <= yy < 63:
                owner[yy, xx] = i
    for i, state in enumerate(m["states"]):
        vals = obs[owner == i]
        vals = vals[np.isin(vals, _COLORS)]
        if len(vals) < 4:
            continue
        counts = np.bincount(vals, minlength=16)
        col = int(counts.argmax())
        if col != state[3] and counts[col] >= .70 * len(vals):
            m["states"][i] = state[:3] + (col,)
            changed = True

    pred = _render_shapes(m)
    take = pred >= 0
    errors = int(np.count_nonzero(pred[take] != obs[take]))
    if (errors > max(9, int(np.count_nonzero(take) * .14)) and
            m.get("turn", 0) - m.get("last_refit", -100) >= 10):
        m["last_refit"] = m.get("turn", 0)
        excluded = m["pad_inside"].copy()
        for x, y in m["targets"]:
            excluded[y, x] = True
        states = []
        for i, st in enumerate(m["states"]):
            geo = m["geoms"][st[2]]
            center = (st[0] + geo[1] // 2, st[1] + geo[2] // 2)
            clean = st[3] if geo[5] else 5
            states.append(_fit_shape(obs, center, excluded, clean, m))
        m["states"] = states
        _choose_order(m, obs)
        changed = True
    if changed:
        _invalidate(m)


def _reveal_action(m):
    unknown = [p for p, c in m["targets"].items() if c is None]
    if not unknown:
        return None
    i = m["active"]
    start = m["states"][i]
    covered = [p for p in unknown if _contains(m, start, *p, True)]
    if not covered:
        return 5

    q = deque([(start, ())])
    seen = {start}
    best = None
    while q:
        state, path = q.popleft()
        remaining = sum(_contains(m, state, *p, True) for p in covered)
        if path:
            gain = len(covered) - remaining
            geo_change = state[2] != start[2]
            repaint = state[3] != start[3]
            score = gain * 20 - len(path) - 2 * geo_change - repaint
            if best is None or score > best[0]:
                best = (score, path)
        if len(path) >= 4:
            continue
        for a in range(1, 5):
            nxt = _transition(m, state, a, True)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + (a,)))
    if best and best[0] > 0:
        return best[1][0]
    return 5


def _new_search(m, index):
    start = m["states"][index]
    targets = list(m["targets"].items())
    color_masks = {}
    for j, (p, c) in enumerate(targets):
        color_masks[c] = color_masks.get(c, 0) | (1 << j)
    return {
        "index": index,
        "nodes": [start],
        "parents": [-1],
        "actions": [0],
        "dist": [0],
        "seen": {start: 0},
        "head": 0,
        "kin": {},
        "coverage": {},
        "best": {},
        "targets": targets,
        "color_masks": color_masks,
    }


def _coverage(m, search, key):
    old = search["coverage"].get(key)
    if old is not None:
        return old
    x, y, gid = key
    geo = m["geoms"][gid]
    w, h, rows = geo[1], geo[2], geo[6]
    occ = center = 0
    for j, ((tx, ty), col) in enumerate(search["targets"]):
        dx, dy = tx - x, ty - y
        if 0 <= dx < w and 0 <= dy < h:
            bit = 1 << j
            if rows[dy] & (1 << dx):
                occ |= bit
            if dx == w // 2 and dy == h // 2:
                center |= bit
    answer = (occ, center)
    search["coverage"][key] = answer
    return answer


def _run_search(m, s, deadline):
    nodes, parents, actions = s["nodes"], s["parents"], s["actions"]
    dist, seen, kin = s["dist"], s["seen"], s["kin"]
    head = s["head"]
    limit = 430000
    while head < len(nodes) and head < limit:
        if head % 192 == 0 and time.monotonic() >= deadline:
            s["head"] = head
            return False
        state = nodes[head]
        x, y, gid, color = state
        key = (x, y, gid)
        d = dist[head]
        occ, center = _coverage(m, s, key)
        good = occ & s["color_masks"].get(color, 0)
        bad = occ & ~good
        ckey = (good, bad, center)
        if ckey not in s["best"]:
            s["best"][ckey] = head

        if key not in kin:
            kin[key] = tuple(_kinematic(m, x, y, gid, a)
                             for a in range(1, 5))
        for a, t in enumerate(kin[key], 1):
            if t is None:
                continue
            nx, ny, ng, hits = t
            different = [c for c in hits if c != color]
            if len(different) > 1:
                continue
            nc = different[0] if different else color
            nxt = (nx, ny, ng, nc)
            if nxt in seen:
                continue
            seen[nxt] = len(nodes)
            nodes.append(nxt)
            parents.append(head)
            actions.append(a)
            dist.append(d + 1)
        head += 1
    s["head"] = head
    return True


def _finish_search(s):
    candidates = []
    nodes, parents, actions, dist = (
        s["nodes"], s["parents"], s["actions"], s["dist"])
    for (good, bad, center), idx in s["best"].items():
        path = []
        j = idx
        while parents[j] >= 0:
            path.append(actions[j])
            j = parents[j]
        path.reverse()
        candidates.append((good, bad, center, dist[idx],
                           tuple(path), nodes[idx]))

    grouped = {}
    for c in candidates:
        grouped.setdefault((c[0], c[2]), []).append(c)
    out = []
    for group in grouped.values():
        group.sort(key=lambda c: c[3])
        kept = []
        for c in group:
            if any((p[1] & c[1]) == p[1] and p[3] <= c[3]
                   for p in kept):
                continue
            kept.append(c)
        out.extend(kept)
    out.sort(key=lambda c: (c[3], -c[0].bit_count(), c[1].bit_count()))
    return out


def _visit_order(m, choice, final):
    n, active = len(choice), m["active"]
    others = [i for i in range(n) if i != final and choice[i][4]]
    best = None
    for perm in itertools.permutations(others):
        order = list(perm) + [final]
        cur, switches = active, 0
        for i in order:
            switches += (i - cur) % n
            cur = i
        if best is None or switches < best[0]:
            best = (switches, order)
    return best


def _combine(m):
    n = len(m["states"])
    all_bits = (1 << len(m["targets"])) - 1
    order = list(reversed(m["order"]))
    best_solution = None

    for final in range(n):
        adjusted = {}
        for i in range(n):
            arr = []
            for c in m["candidates"][i]:
                good, bad = c[0], c[1]
                if i == final:
                    good &= ~c[2]
                    bad |= c[2]
                arr.append((good, bad, c))
            adjusted[i] = arr

        unions = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            u = unions[k + 1]
            for good, bad, c in adjusted[order[k]]:
                u |= good
            unions[k] = u
        cache = {}

        def solve(k, need):
            key = (k, need)
            if key in cache:
                return cache[key]
            if k == n:
                ans = (0, ()) if need == 0 else None
                cache[key] = ans
                return ans
            if need & ~unions[k]:
                cache[key] = None
                return None
            ans = None
            for good, bad, c in adjusted[order[k]]:
                if bad & need:
                    continue
                if ans is not None and c[3] >= ans[0]:
                    continue
                nn = need & ~good
                if nn & ~unions[k + 1]:
                    continue
                sub = solve(k + 1, nn)
                if sub is not None:
                    cost = c[3] + sub[0]
                    if ans is None or cost < ans[0]:
                        ans = (cost, (c,) + sub[1])
            cache[key] = ans
            return ans

        answer = solve(0, all_bits)
        if answer is None:
            continue
        choice = [None] * n
        for i, c in zip(order, answer[1]):
            choice[i] = c
        switches, visit = _visit_order(m, choice, final)
        total = answer[0] + switches
        if best_solution is None or total < best_solution[0]:
            best_solution = (total, choice, visit)

    if best_solution is None:
        return None
    _, choice, visit = best_solution
    commands = []
    cur = m["active"]
    for i in visit:
        commands.extend([5] * ((i - cur) % n))
        commands.extend(choice[i][4])
        cur = i
    if not commands:
        commands = [5]
    return commands


def _fallback(m):
    best = None
    n = len(m["states"])
    for i, candidates in enumerate(m.get("candidates", [])):
        for c in candidates:
            if not c[4]:
                continue
            score = (12 * c[0].bit_count() - 15 * c[1].bit_count()
                     - .3 * c[3] - (i - m["active"]) % n)
            if best is None or score > best[0]:
                best = (score, i, c)
    if best:
        _, i, c = best
        return [5] * ((i - m["active"]) % n) + list(c[4])
    return [5, 1, 4]


def _scan(m, obs):
    _update_targets(m, obs)
    zeros = list(zip(*np.where(obs[:63] == 0)))
    center = None
    if zeros:
        yy, xx = zeros[0]
        center = (int(xx), int(yy))

    for p in m["centers"]:
        if p != center:
            m["clean"][p] = int(obs[p[1], p[0]])

    if center is None:
        # A later-drawn shape can hide the active shape's zero pixel.
        # Do not advance the selection cycle until this shape is observed.
        k = m["blind_steps"]
        m["blind_steps"] = k + 1
        directions = (1, 4, 1, 4, 2, 3, 2, 3)
        if k >= 12:
            directions = (2, 4, 2, 4, 1, 3, 1, 3)
        return _issue(m, directions[k % len(directions)])

    m["blind_steps"] = 0
    finished = center in m["centers"] and m["scan_steps"] > 0
    if not finished:
        m["centers"].append(center)
        m["scan_steps"] += 1
        return _issue(m, 5)

    pads, obstacles, inside = _static_objects(obs)
    m["pads"] = pads
    m["obstacles"] = obstacles
    m["pad_inside"] = inside
    excluded = inside.copy()
    for x, y in m["targets"]:
        excluded[y, x] = True

    states = []
    for p in m["centers"]:
        states.append(_fit_shape(obs, p, excluded,
                                 m["clean"].get(p, 5), m))
    m["states"] = states
    m["active"] = m["centers"].index(center)
    m["order"] = list(range(len(states)))
    _choose_order(m, obs)
    m["phase"] = "reveal"
    _update_targets(m, obs)

    a = _reveal_action(m)
    if a is not None:
        return _issue(m, a)

    # Separate visual reconstruction and the expensive search into calls.
    m["phase"] = "plan"
    m["plan_index"] = 0
    m["candidates"] = []
    return _issue(m, 5)


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=np.int16)
    start_time = time.monotonic()

    if "_re86_player" not in memory:
        memory["_re86_player"] = _fresh()
    m = memory["_re86_player"]

    # Recover from an inconsistent model rather than cycling indefinitely.
    if m.get("idle", 0) >= 36 and m["phase"] != "scan":
        old_targets = dict(m["targets"])
        m = _fresh()
        m["targets"] = old_targets
        memory["_re86_player"] = m
        return _issue(m, 1)

    if m["phase"] == "scan":
        return _scan(m, obs)

    _synchronize(m, obs)
    changed_targets = _update_targets(m, obs)
    if changed_targets and m["phase"] in ("plan", "execute", "fallback"):
        _invalidate(m)

    if m["phase"] == "reveal":
        a = _reveal_action(m)
        if a is not None:
            return _issue(m, a)
        m["phase"] = "plan"
        m["plan_index"] = 0
        m["candidates"] = []
        m.pop("search", None)

    if m["phase"] == "plan":
        index = m["plan_index"]
        if "search" not in m:
            m["search"] = _new_search(m, index)
        done = _run_search(m, m["search"], start_time + 7.1)
        if not done:
            return _issue(m, 5)

        m["candidates"].append(_finish_search(m.pop("search")))
        m["plan_index"] += 1
        if m["plan_index"] < len(m["states"]):
            return _issue(m, 5)

        commands = _combine(m)
        if commands is None:
            commands = _fallback(m)
            m["phase"] = "fallback"
        else:
            m["phase"] = "execute"
        m["commands"] = deque(commands)

    if m["phase"] in ("execute", "fallback"):
        commands = m.get("commands")
        if commands:
            return _issue(m, commands.popleft())
        _invalidate(m)
        return _issue(m, 5)

    return _issue(m, 5)
