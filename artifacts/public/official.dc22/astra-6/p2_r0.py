import numpy as np
import heapq
import time

_ALL = (1 << 1024) - 1
_LEFT = sum(1 << (32 * y) for y in range(32))
_RIGHT = _LEFT << 31
_NEIGH = []
for _p in range(1024):
    _x, _y = _p % 32, _p // 32
    _ns = []
    if _y > 0:
        _ns.append((_p - 32, 1))
    if _y < 31:
        _ns.append((_p + 32, 2))
    if _x > 0:
        _ns.append((_p - 1, 3))
    if _x < 31:
        _ns.append((_p + 1, 4))
    _NEIGH.append(tuple(_ns))

_CRUSH_TEMPLATES = []
for _rows in (
    ("...88888", "...8...8", ".8888...", "8888....",
     "8888....", ".8888...", "...8...8", "...88888"),
    ("........", "...88888", ".8888..8", "8888....",
     "8888....", ".8888..8", "...88888", "........"),
):
    _CRUSH_TEMPLATES.append(
        {(x, y) for y, row in enumerate(_rows)
         for x, c in enumerate(row) if c == "8"}
    )

_CONTROLS = {"L", "R", "U", "D", "G"}


def _expand(b):
    return (((b & ~_RIGHT) << 1) | ((b & ~_LEFT) >> 1) |
            (b << 32) | (b >> 32)) & _ALL


def _flood(seed, allowed):
    r = seed & allowed
    while True:
        q = (r | _expand(r)) & allowed
        if q == r:
            return r
        r = q


def _rect(x, y, w, h):
    return {(xx, yy) for yy in range(y, y + h)
            for xx in range(x, x + w)
            if 0 <= xx < 64 and 0 <= yy < 64}


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _components(mask, diagonal=False):
    yy, xx = np.nonzero(mask)
    remaining = set(zip(map(int, xx), map(int, yy)))
    result = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        dirs += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    while remaining:
        first = remaining.pop()
        comp = {first}
        queue = [first]
        for x, y in queue:
            for dx, dy in dirs:
                q = x + dx, y + dy
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    queue.append(q)
        result.append(comp)
    return result


def _project(points, ox, oy, blocking=False):
    ans = 0
    for x, y in points:
        if blocking:
            xx, yy = (x - ox) // 2, (y - oy) // 2
        else:
            if ((x - ox) & 1) or ((y - oy) & 1):
                continue
            xx, yy = (x - ox) // 2, (y - oy) // 2
        if 0 <= xx < 32 and 0 <= yy < 32:
            ans |= 1 << (yy * 32 + xx)
    return ans


def _position(x, y, ox, oy):
    return ((y - oy) // 2) * 32 + (x - ox) // 2


def _find_crane(a, limit, enabled):
    if not enabled:
        return None
    yy, xx = np.nonzero(a[:, :limit] == 8)
    red = set(zip(map(int, xx), map(int, yy)))
    best, best_score = None, -1
    for ti, templ in enumerate(_CRUSH_TEMPLATES):
        first = min(templ, key=lambda p: (p[1], p[0]))
        origins = {(x - first[0], y - first[1]) for x, y in red}
        for x, y in origins:
            if x < 0 or y < 0 or x + 8 > limit or y + 8 > 63:
                continue
            positive = sum((x + dx, y + dy) in red for dx, dy in templ)
            if positive < len(templ) * .94:
                continue
            negative = sum((x + dx, y + dy) not in red
                           for dy in range(8) for dx in range(8)
                           if (dx, dy) not in templ)
            score = positive + .3 * negative
            if score > best_score:
                best_score = score
                best = {
                    "x": x, "y": y, "w": 8, "h": 8,
                    "classic": True, "closed": ti == 1,
                    "pixels": {(x + dx, y + dy) for dx, dy in templ},
                }
    if best:
        return best
    candidates = []
    for comp in _components(a[:, :limit] == 8):
        x0, y0, x1, y1 = _bbox(comp)
        w, h = x1 - x0, y1 - y0
        if w > 4 and h > 4 and len(comp) < .85 * w * h:
            candidates.append((len(comp), comp, (x0, y0, w, h)))
    if candidates:
        _, comp, (x, y, w, h) = max(candidates, key=lambda z: z[0])
        return {
            "x": x, "y": y, "w": w, "h": h,
            "classic": False, "closed": False, "pixels": comp,
        }
    return None


def _scan(a):
    active = [y for y in range(63)
              if np.count_nonzero((a[y] == 4) | (a[y] == 5)) > 20]
    ys = active or list(range(63))
    limit = None
    for x in range(24, 52):
        col = a[ys, x]
        if (np.count_nonzero(col == 0) > len(ys) * .18 and
                np.count_nonzero(col == 4) < len(ys) * .08 and
                np.count_nonzero(a[ys, x + 2] == 5) > len(ys) * .65):
            limit = x
            break
    if limit is None:
        for x in range(24, 52):
            if np.count_nonzero(a[ys, x] == 5) > len(ys) * .78:
                limit = x
                break
    if limit is None:
        limit = 38

    yy, xx = np.nonzero(a[:, :limit] == 14)
    if not len(xx):
        return None
    px, py = int(xx.min()), int(yy.min())
    ox, oy = px & 1, py & 1
    player = _position(px, py, ox, oy)

    goal = None
    small = [c for c in _components(a[:, :limit] == 11) if len(c) <= 9]
    if small:
        gx, gy, _, _ = _bbox(min(small, key=lambda c: abs(len(c) - 4)))
        goal = _position(gx, gy, ox, oy)

    buttons, squares = {}, []
    color_group = {8: "a", 9: "b", 6: "c", 15: "d", 2: "e", 11: "f"}
    for color in range(16):
        if color in (0, 3, 4, 5):
            continue
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, limit:] = a[:, limit:] == color
        for comp in _components(mask):
            x0, y0, x1, y1 = _bbox(comp)
            w, h = x1 - x0, y1 - y0
            if len(comp) < 8:
                continue
            click = min(comp, key=lambda p: (p[1], p[0]))
            center = (x0 + x1) // 2, (y0 + y1) // 2
            if center in comp:
                click = center
            if color in (9, 10) and w == 4 and h == 4:
                squares.append((color, x0, y0, click))
                continue
            if color == 8 and w >= 7 and h <= 3:
                buttons["G"] = click[1], click[0]
                continue
            if ((w >= 7 and h >= 4) or
                    (color in (2, 11) and w >= 4 and h >= 4)):
                g = color_group.get(color, "z" + str(color))
                if g in buttons:
                    g += "_" + str(x0) + "_" + str(y0)
                buttons[g] = click[1], click[0]
    for color, left, right in ((10, "L", "R"), (9, "U", "D")):
        ss = sorted([s for s in squares if s[0] == color],
                    key=lambda s: (s[2], s[1]))
        pair = None
        for s1 in ss:
            for s2 in ss:
                if s1[2] == s2[2] and s1[1] < s2[1]:
                    pair = s1, s2
                    break
            if pair:
                break
        if pair:
            buttons[left] = pair[0][3][1], pair[0][3][0]
            buttons[right] = pair[1][3][1], pair[1][3][0]

    crane = _find_crane(a, limit, bool(_CONTROLS & buttons.keys()))
    yy, xx = np.nonzero(np.isin(
        a[:, :limit], (1, 2, 6, 7, 8, 10, 11, 12, 13, 14)))
    floor = set(zip(map(int, xx), map(int, yy)))
    block = set()
    if crane:
        floor.difference_update(crane["pixels"])
        block.update(crane["pixels"])

    gates, pickups = [], []
    for color in (9, 15):
        for comp in _components(a[:, :limit] == color, True):
            x0, y0, x1, y1 = _bbox(comp)
            w, h = x1 - x0, y1 - y0
            box = _rect(x0, y0, w, h)
            if w >= 3 and h >= 3:
                valid = sum(int(a[y, x]) in (color, 14) for x, y in box)
                solid = valid >= len(box) * .9
                gates.append((color, box, solid))
                if solid:
                    floor.update(box)
                else:
                    block.update(comp)
            elif color == 15 and w <= 5 and h <= 3:
                pickups.append((x0, y0, x1, y1))
                floor.update(comp)

    tele = []
    for comp in _components(np.isin(a[:, :limit], (6, 7))):
        x0, y0, x1, y1 = _bbox(comp)
        if x1 - x0 == 2 and y1 - y0 == 2:
            tele.append((_position(x0, y0, ox, oy), int(a[y0, x0])))

    island_mask = np.isin(a[:, :limit], (2, 6, 7, 11, 14))
    for x0, y0, x1, y1 in pickups:
        island_mask[y0:y1, x0:x1] = True
    island_pixels = set()
    for comp in _components(island_mask):
        x0, y0, x1, y1 = _bbox(comp)
        if (any(a[y, x] == 2 for x, y in comp) and
                x1 - x0 <= 8 and y1 - y0 <= 8 and
                len(comp) >= (x1 - x0) * (y1 - y0) * .55):
            island_pixels.update(_rect(x0, y0, x1 - x0, y1 - y0))

    fb = _project(floor, ox, oy)
    bb = _project(block, ox, oy, True)
    return {
        "limit": limit, "px": px, "py": py, "ox": ox, "oy": oy,
        "p": player, "goal": goal, "buttons": buttons,
        "raw": (fb, fb & ~bb), "gates": gates,
        "pickups": pickups, "tele": tele, "crane": crane,
        "islands": _project(island_pixels, ox, oy),
    }


def _entity(group, variants, center=None, kind="fixed"):
    return {"group": group, "variants": variants,
            "center": center, "kind": kind}


def _cycle_shapes(kind, x, y, h):
    if kind == "e":
        result = []
        for extent in (0, 2, 4, 6, 8, 6, 4, 2):
            result.append(_rect(x, y, extent, h) |
                          _rect(x + 16 - extent, y, extent, h))
        return result
    return [_rect(x + 2 * k, y, 6, h) for k in range(6)]


def _match_phase(a, spec, kind):
    x, y, h, expected = spec["x"], spec["y"], spec["h"], spec["phase"]
    shapes = _cycle_shapes(kind, x, y, h)
    region = _rect(x, y, 16, h)
    scores = []
    for phase, shape in enumerate(shapes):
        error = 0
        for xx, yy in region:
            color = int(a[yy, xx])
            if color == 14:
                continue
            actual = color in (1, 12)
            if actual != ((xx, yy) in shape):
                error += 1
        scores.append((error, (phase - expected) % len(shapes), phase))
    spec["phase"] = min(scores)[2]
    return shapes


def _mechanisms(a, s, memory):
    specs = memory.setdefault("mechanisms", {})
    limit = s["limit"]
    buttons = memory.get("buttons", {})
    if "e" not in specs and "e" in buttons:
        candidates = []
        for comp in _components(a[:, :limit] == 5):
            x0, y0, x1, y1 = _bbox(comp)
            if y1 - y0 <= 3 and 4 <= x1 - x0 <= 16:
                candidates.append(((x0 + x1) // 2 - 8, y0, y1 - y0))
        if not candidates:
            for comp in _components(a[:, :limit] == 12):
                x0, y0, x1, y1 = _bbox(comp)
                if x1 - x0 == 16 and y1 - y0 <= 3:
                    candidates.append((x0, y0, y1 - y0))
        if candidates:
            x, y, h = candidates[0]
            specs["e"] = {"x": x, "y": y, "h": h, "phase": 0}
    if "f" not in specs and "f" in buttons:
        for comp in _components(np.isin(a[:, :limit], (1, 12))):
            x0, y0, x1, y1 = _bbox(comp)
            if x1 - x0 != 6 or y1 - y0 > 3:
                continue
            if "e" in specs and specs["e"]["y"] == y0:
                continue
            base = specs["e"]["x"] if "e" in specs else x0
            if "e" not in specs and a[y0, x0] == 12:
                base = x0 - 10
            phase = max(0, min(5, (x0 - base) // 2))
            specs["f"] = {"x": base, "y": y0, "h": y1 - y0, "phase": phase}
            break
    result = []
    for kind in ("e", "f"):
        if kind not in specs:
            continue
        spec = specs[kind]
        shapes = _match_phase(a, spec, kind)
        phase = spec["phase"]
        variants = [(shapes[(phase + k) % len(shapes)], set())
                    for k in range(len(shapes))]
        result.append(_entity(kind, variants))
    return result


def _build(a, s, memory):
    ox, oy, limit = s["ox"], s["oy"], s["limit"]
    entities = []
    crane = s["crane"]
    crane_pixels = crane["pixels"] if crane else set()

    bridge_mask = np.isin(a[:, :limit], (8, 13, 14))
    for x, y in crane_pixels:
        if x < limit:
            bridge_mask[y, x] = False
    for comp in _components(bridge_mask):
        pivot = {(x, y) for x, y in comp if a[y, x] == 13}
        if not pivot:
            continue
        x0, y0, x1, y1 = _bbox(comp)
        qx0, qy0, qx1, qy1 = _bbox(pivot)
        horizontal = x1 - x0 >= y1 - y0
        cx, cy = (qx0 + qx1) // 2, (qy0 + qy1) // 2
        if qx1 - qx0 < 4:
            if horizontal:
                r = ((qx0 + qx1) / 2 - x0) / max(1, x1 - x0)
                cx = ((x0 + x1) // 2 if .3 < r < .7
                      else x1 - 2 if r >= .7 else x0 + 2)
            else:
                cx = (x0 + x1) // 2
        if qy1 - qy0 < 4:
            if not horizontal:
                r = ((qy0 + qy1) / 2 - y0) / max(1, y1 - y0)
                cy = ((y0 + y1) // 2 if .3 < r < .7
                      else y1 - 2 if r >= .7 else y0 + 2)
            else:
                cy = (y0 + y1) // 2
        shape = {(x, y) for x, y in comp if a[y, x] in (8, 13, 14)}
        shape.update(_rect(cx - 2, cy - 2, 4, 4))
        if horizontal and y1 - y0 <= 4:
            shape.update(_rect(x0, cy - 2, x1 - x0, 4))
        elif not horizontal and x1 - x0 <= 4:
            shape.update(_rect(cx - 2, y0, 4, y1 - y0))
        if horizontal:
            rotated = {(cx - 1 - (y - cy), cy + (x - cx))
                       for x, y in shape}
        else:
            rotated = {(cx + (y - cy), cy - 1 - (x - cx))
                       for x, y in shape}
        centered = (abs((x0 + x1) / 2 - cx) <= 2 and
                    abs((y0 + y1) / 2 - cy) <= 2)
        entities.append(_entity(
            "a", [(shape, set()), (rotated, set())],
            (cx, cy), "bridge" if centered else "fixed"))

    for color, box, solid in s["gates"]:
        variants = [(box, set()), (set(), box)]
        if not solid:
            variants.reverse()
        entities.append(_entity("b" if color == 9 else "d", variants))

    entities.extend(_mechanisms(a, s, memory))

    objects = []
    for comp in _components(a[:, :limit] == 12):
        x0, y0, x1, y1 = _bbox(comp)
        if y1 - y0 >= 4 and x1 - x0 >= 4:
            objects.append(len(entities))
            entities.append(_entity(
                None, [(comp, set())],
                ((x0 + x1) // 2, (y0 + y1) // 2), "object"))

    anchors, crane_edges, crane_blocks = [], {}, []
    ci, attached = -1, -1
    normal = bool(crane and objects)
    if crane:
        if normal:
            anchor = crane["x"] + 9, crane["y"] + 4
        else:
            anchor = (crane["x"] + crane["w"] // 2,
                      crane["y"] + crane["h"] // 2)
        yy, xx = np.nonzero(a[:63, :limit] == 0)
        zeros = set(zip(map(int, xx), map(int, yy)))
        if normal and zeros:
            zx0, zy0, zx1, zy1 = _bbox(zeros)
            bx, by = zx0 + 8, zy1 - 2
            anchors = [(bx + 4 * x, by - 4 * y)
                       for y in range(4) for x in range(4)
                       if x == 0 or y in (0, 3)]
            if anchor not in anchors:
                anchors = [(anchor[0] + 4 * x, anchor[1] - 4 * y)
                           for y in range(4) for x in range(4)
                           if x == 0 or y in (0, 3)]
        else:
            anchors = [anchor]
            for y in range(anchor[1] % 4, 63, 4):
                for x in range(anchor[0] % 4, limit, 4):
                    if ((x, y) in zeros or (x - 1, y) in zeros or
                            (x, y - 1) in zeros):
                        anchors.append((x, y))
            anchors = sorted(set(anchors))
        if anchor not in anchors:
            anchors.append(anchor)
        ci = anchors.index(anchor)
        amap = {p: i for i, p in enumerate(anchors)}
        bad_crane = memory.get("crane_bad", set())
        for i, (x, y) in enumerate(anchors):
            es = []
            for ac, dx, dy in (("L", -4, 0), ("R", 4, 0),
                               ("U", 0, -4), ("D", 0, 4)):
                q = x + dx, y + dy
                if q in amap and (x, y, ac) not in bad_crane:
                    es.append((ac, amap[q]))
            crane_edges[i] = es
            dx, dy = x - anchor[0], y - anchor[1]
            points = {(xx + dx, yy + dy) for xx, yy in crane_pixels}
            crane_blocks.append(_project(points, ox, oy, True))

    force = memory.get("force", set()) - memory.get("mechanisms", {}).keys()
    entities = [e for e in entities if e["group"] not in force]
    if crane and crane["closed"]:
        for i, e in enumerate(entities):
            if e["center"] == anchors[ci] and e["kind"] in ("object", "bridge"):
                attached = i
                break

    groups = {}
    for e in entities:
        g = e["group"]
        if g is not None:
            n = len(e["variants"])
            groups[g] = {
                "n": n, "next": tuple((i + 1) % n for i in range(n)),
                "initial": 0,
            }

    patch_domains = 0
    for g in force:
        records = memory.get("learn", {}).get(g, [])
        domain = 0
        for bf, bw, af, aw in records:
            domain |= (bf ^ af) | (bw ^ aw)
        if not domain:
            continue
        patch_domains |= domain
        values = []
        for rec in records:
            for f, w in ((rec[0], rec[1]), (rec[2], rec[3])):
                v = f & domain, (f & ~w) & domain
                if v not in values:
                    values.append(v)
        current = (s["raw"][0] & domain,
                   (s["raw"][0] & ~s["raw"][1]) & domain)
        if current not in values:
            values.append(current)
        nxt = [(i + 1) % len(values) for i in range(len(values))]
        for bf, bw, af, aw in records:
            v1 = bf & domain, (bf & ~bw) & domain
            v2 = af & domain, (af & ~aw) & domain
            nxt[values.index(v1)] = values.index(v2)
        groups[g] = {
            "n": len(values), "next": tuple(nxt),
            "initial": values.index(current),
        }
        entities.append({
            "group": g, "cell_variants": values,
            "center": None, "kind": "fixed",
        })

    keys = sorted(groups)
    gi = {g: i for i, g in enumerate(keys)}
    static_f = s["raw"][0] & ~patch_domains
    static_b = (s["raw"][0] & ~s["raw"][1]) & ~patch_domains
    if crane:
        static_b &= ~_project(crane_pixels, ox, oy, True)

    for e in entities:
        e["gi"] = gi.get(e["group"], -1)
        e["masks"] = []
        if "cell_variants" in e:
            for f, b in e["cell_variants"]:
                e["masks"].append([(f, b)] * (len(anchors) + 1))
            continue
        for shape, blockers in e["variants"]:
            base = (_project(shape, ox, oy),
                    _project(blockers, ox, oy, True))
            poses = [base]
            for anchor in anchors:
                if e["center"] is not None and e["kind"] in ("bridge", "object"):
                    dx = anchor[0] - e["center"][0]
                    dy = anchor[1] - e["center"][1]
                    f = _project({(x + dx, y + dy) for x, y in shape}, ox, oy)
                    b = _project({(x + dx, y + dy) for x, y in blockers},
                                 ox, oy, True)
                    poses.append((f, b))
                else:
                    poses.append(base)
            e["masks"].append(poses)
        cf, cb = e["masks"][0][0]
        static_f &= ~cf
        static_b &= ~cb

    static_f &= ~(1 << s["p"])
    static_f |= s["islands"]
    if s["goal"] is not None:
        static_f |= 1 << s["goal"]

    tele = list(s["tele"])
    old_tele = memory.get("tele_locations", [])
    known = {p for p, ph in tele}
    if len(tele) == 1:
        for p, phase in old_tele:
            if p == s["p"] and p not in known:
                tele.append((p, 13 - tele[0][1]))
                break
        if len(tele) == 1 and "c" in memory.get("buttons", {}):
            if s["islands"] & (1 << s["p"]):
                tele.append((s["p"], 13 - tele[0][1]))
    if len(tele) >= 2:
        memory["tele_locations"] = list(tele)
    for p, phase in tele:
        static_f |= 1 << p

    pickup_bits = 0
    for x0, y0, x1, y1 in s["pickups"]:
        pickup_bits |= _project(
            _rect(x0 - 1, y0 - 1, x1 - x0 + 1, y1 - y0 + 1), ox, oy)

    allowed = {}
    for g in memory.get("conditional", set()):
        b = 0
        for p in memory.get("visible_at", {}).get(g, set()):
            b |= 1 << p
        allowed[g] = b
    unsafe = {}
    for g, p in memory.get("unsafe", set()):
        unsafe[g] = unsafe.get(g, 0) | (1 << p)

    conf = tuple(groups[g]["initial"] for g in keys)
    unlocked = int("d" in memory.get("buttons", {}) or not s["pickups"])
    M = {
        "ox": ox, "oy": oy, "limit": limit, "goal": s["goal"],
        "static_f": static_f, "static_b": static_b,
        "entities": entities, "keys": keys, "groups": groups,
        "anchors": anchors, "crane_edges": crane_edges,
        "crane_blocks": crane_blocks, "normal": normal,
        "tele": tele, "pickup": pickup_bits,
        "allowed": allowed, "unsafe": unsafe,
        "buttons": set(memory.get("buttons", {})),
        "telelinks": memory.get("telelinks", {}),
        "bad": memory.get("blocked", set()),
        "cache": {}, "current": (conf, ci, attached, unlocked),
        "p": s["p"],
    }
    if not _world(M, M["current"])[0] & (1 << s["p"]):
        M["static_f"] |= 1 << s["p"]
        M["cache"].clear()
    return M


def _world(M, state):
    conf, ci, attached, unlocked = state
    key = conf, ci, attached
    cached = M["cache"].get(key)
    if cached is not None:
        return cached
    f, b = M["static_f"], M["static_b"]
    for i, e in enumerate(M["entities"]):
        v = conf[e["gi"]] if e["gi"] >= 0 else 0
        v %= len(e["masks"])
        pose = ci + 1 if i == attached and ci >= 0 else 0
        ef, eb = e["masks"][v][pose]
        f |= ef
        b |= eb
    if ci >= 0:
        b |= M["crane_blocks"][ci]
    result = [f & _ALL, f & ~b & _ALL, []]
    M["cache"][key] = result
    return result


def _component(world, p):
    bit = 1 << p
    for region in world[2]:
        if region & bit:
            return region
    region = _flood(bit, world[1] | bit)
    world[2].append(region)
    return region


def _bfs(M, world, p):
    allowed = world[1] | (1 << p)
    dist = {p: 0}
    prev = {}
    order = [p]
    region = 1 << p
    bad = M["bad"]
    for u in order:
        for v, action in _NEIGH[u]:
            if v not in dist and allowed & (1 << v):
                if (world[1], u, v) in bad:
                    continue
                dist[v] = dist[u] + 1
                prev[v] = u, action
                order.append(v)
                region |= 1 << v
    return order, dist, prev, region


def _route(M, world, start, target):
    if start == target:
        return []
    _, dist, prev, _ = _bfs(M, world, start)
    if target not in dist:
        return []
    route = []
    p = target
    while p != start:
        q, action = prev[p]
        route.append((action, p))
        p = q
    route.reverse()
    return route


def _relaxed_distances(M):
    allowed = M["static_f"]
    for e in M["entities"]:
        for poses in e["masks"]:
            for f, b in poses:
                allowed |= f
    allowed &= ~M["static_b"]
    goal = M["goal"]
    if goal is None:
        return {}
    allowed |= 1 << goal
    portals = [p for p, phase in M["tele"]]
    dist = {goal: 0}
    queue = [goal]
    for p in queue:
        adjacent = [q for q, ac in _NEIGH[p]]
        if p in portals:
            adjacent += portals
        for q in adjacent:
            if q not in dist and allowed & (1 << q):
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def _valid_crane_move(M, state):
    conf, ci, attached, unlocked = state
    if attached < 0 or M["normal"]:
        return True
    e = M["entities"][attached]
    if e["kind"] != "bridge":
        return True
    v = conf[e["gi"]] if e["gi"] >= 0 else 0
    moving = e["masks"][v % len(e["masks"])][ci + 1][0]
    for i, other in enumerate(M["entities"]):
        if i == attached or other["kind"] != "bridge":
            continue
        q = conf[other["gi"]] if other["gi"] >= 0 else 0
        if moving & other["masks"][q % len(other["masks"])][0][0]:
            return False
    return True


def _reconstruct(M, nodes, index, final_target=None):
    chain = []
    k = index
    while nodes[k][3] >= 0:
        chain.append(k)
        k = nodes[k][3]
    chain.reverse()
    result = []
    for k in chain:
        state, pos, cost, parent, target, action = nodes[k]
        oldstate, oldpos = nodes[parent][0], nodes[parent][1]
        for ac, p in _route(M, _world(M, oldstate), oldpos, target):
            result.append((ac, p, oldstate))
        if action is not None:
            result.append((action, pos, state))
        elif result:
            ac, p, _ = result[-1]
            result[-1] = ac, p, state
    if final_target is not None:
        state, pos = nodes[index][0], nodes[index][1]
        for ac, p in _route(M, _world(M, state), pos, final_target):
            result.append((ac, p, state))
    return result


def _plan(M, seconds=7.0):
    if M["goal"] is None:
        return []
    deadline = time.monotonic() + seconds
    root, start, goal = M["current"], M["p"], M["goal"]
    heuristic = _relaxed_distances(M)

    def h(p):
        return heuristic.get(
            p, 18 + abs(p % 32 - goal % 32) + abs(p // 32 - goal // 32))

    nodes = [(root, start, 0, -1, start, None)]
    reg = _component(_world(M, root), start)
    best = {(root, (reg & -reg).bit_length() - 1): 0}
    heap = [(1.8 * h(start), 0, 0)]
    serial = 0
    progress, progress_h = 0, h(start)

    def add(parent, state, pos, target, action, cost):
        nonlocal serial
        world = _world(M, state)
        region = _component(world, pos)
        key = state, (region & -region).bit_length() - 1
        if cost >= best.get(key, 10**9):
            return
        best[key] = cost
        index = len(nodes)
        nodes.append((state, pos, cost, parent, target, action))
        serial += 1
        heapq.heappush(heap, (cost + 1.8 * h(pos), serial, index))

    expansions = 0
    while heap and expansions < 70000:
        if expansions % 48 == 0 and time.monotonic() > deadline:
            break
        _, _, ni = heapq.heappop(heap)
        state, pos, cost, parent, target, action = nodes[ni]
        world = _world(M, state)
        reg = _component(world, pos)
        key = state, (reg & -reg).bit_length() - 1
        if cost != best.get(key):
            continue
        expansions += 1
        order, dist, prev, reach = _bfs(M, world, pos)
        if goal in dist:
            return _reconstruct(M, nodes, ni, goal)
        closest = min(order, key=h)
        if h(closest) < progress_h:
            progress_h, progress = h(closest), ni

        conf, ci, attached, unlocked = state

        def switch_to(newstate, ac, allowed=_ALL):
            nw = _world(M, newstate)
            candidates = reach & nw[1] & allowed
            if not candidates:
                return
            for q in order:
                if candidates & (1 << q):
                    region = _component(nw, q)
                    candidates &= ~region
                    add(ni, newstate, q, q, ac, cost + dist[q] + 1)
                    if not candidates:
                        break

        # Remote cycles are deliberately NOT restricted to adjacent changes.
        # A platform may need several preparatory clicks before it can be reached.
        for j, g in enumerate(M["keys"]):
            if g not in M["buttons"] and g != "d":
                continue
            if g == "d" and not unlocked:
                continue
            allowed = M["allowed"].get(g, _ALL)
            allowed &= ~M["unsafe"].get(g, 0)
            newconf = list(conf)
            newconf[j] = M["groups"][g]["next"][conf[j]]
            if newconf[j] != conf[j]:
                switch_to((tuple(newconf), ci, attached, unlocked), g, allowed)

        if "c" in M["buttons"] and len(M["tele"]) >= 2:
            allowed = M["allowed"].get("c", _ALL)
            for source, phase in M["tele"]:
                if source not in dist or not allowed & (1 << source):
                    continue
                dest = M["telelinks"].get(source)
                if dest is None:
                    candidates = [p for p, ph in M["tele"]
                                  if p != source and ph != phase]
                    if not candidates:
                        candidates = [p for p, ph in M["tele"] if p != source]
                    dest = candidates[0] if candidates else source
                if world[0] & (1 << dest):
                    add(ni, state, dest, source, "c", cost + dist[source] + 1)

        if not unlocked and M["pickup"] & reach:
            q = next(p for p in order if M["pickup"] & (1 << p))
            add(ni, (conf, ci, attached, 1), q, q, None, cost + dist[q])

        if ci >= 0:
            for ac, nci in M["crane_edges"].get(ci, ()):
                if ac in M["buttons"]:
                    ns = conf, nci, attached, unlocked
                    if _valid_crane_move(M, ns):
                        switch_to(ns, ac)
            if attached < 0 and "G" in M["buttons"]:
                anchor = M["anchors"][ci]
                for i, e in enumerate(M["entities"]):
                    if e["kind"] not in ("bridge", "object"):
                        continue
                    if M["normal"] and e["kind"] != "object":
                        continue
                    if e["center"] == anchor:
                        switch_to((conf, ci, i, unlocked), "G")

    if progress and progress_h + 2 < h(start):
        state, p = nodes[progress][0], nodes[progress][1]
        order, _, _, _ = _bfs(M, _world(M, state), p)
        return _reconstruct(M, nodes, progress, min(order, key=h))
    return []


def _fallback(M, s, memory):
    state, p = M["current"], s["p"]
    world = _world(M, state)
    order, dist, prev, reach = _bfs(M, world, p)
    seen = memory.setdefault("seen", {})
    sig = s["raw"]
    goal = M["goal"] if M["goal"] is not None else p

    def score(q):
        dgoal = abs(q % 32 - goal % 32) + abs(q // 32 - goal // 32)
        return seen.get((sig, q), 0) * 40 + .8 * dist[q] + .15 * dgoal

    choices = [q for q in order if q != p and seen.get((sig, q), 0) == 0]
    if choices:
        target = min(choices, key=score)
        return [(ac, q, state) for ac, q in
                _route(M, world, p, target)], False

    probes = memory.setdefault("probes", {})
    candidates = []
    conf, ci, attached, unlocked = state
    for j, g in enumerate(M["keys"]):
        if g not in s["buttons"] or M["unsafe"].get(g, 0) & (1 << p):
            continue
        nc = list(conf)
        nc[j] = M["groups"][g]["next"][conf[j]]
        ns = tuple(nc), ci, attached, unlocked
        if _world(M, ns)[0] & (1 << p):
            candidates.append((g, ns))
    for g in s["buttons"]:
        if g not in M["keys"] and g not in _CONTROLS:
            candidates.append((g, state))
    if ci >= 0:
        for ac, nci in M["crane_edges"].get(ci, ()):
            if ac in s["buttons"]:
                ns = conf, nci, attached, unlocked
                if _valid_crane_move(M, ns):
                    candidates.append((ac, ns))
        if "G" in s["buttons"] and attached < 0:
            candidates.append(("G", state))
    if candidates:
        ac, ns = min(candidates,
                     key=lambda z: probes.get((sig, p, z[0]), 0))
        probes[(sig, p, ac)] = probes.get((sig, p, ac), 0) + 1
        return [(ac, p, ns)], True
    if len(order) > 1:
        target = min((q for q in order if q != p), key=score)
        return [(ac, q, state) for ac, q in
                _route(M, world, p, target)], False
    return [], False


def _act_impl(obs, available, memory):
    a = np.asarray(obs, dtype=np.int16)
    s = _scan(a)
    if s is None:
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if name in available:
                return name, None, None
        return available[0], None, None

    rebuild = memory.pop("rebuild_next", False)
    buttons = memory.setdefault("buttons", {})
    previous_visible = memory.get("last_visible", set())
    current_visible = set(s["buttons"])
    conditional = memory.setdefault("conditional", set())
    visible_at = memory.setdefault("visible_at", {})
    for g in previous_visible - current_visible:
        if g not in _CONTROLS:
            conditional.add(g)
            rebuild = True
    for g in current_visible:
        visible_at.setdefault(g, set()).add(s["p"])
    for g, xy in s["buttons"].items():
        if g not in buttons and g != "d":
            rebuild = True
        buttons[g] = xy
    memory["last_visible"] = current_visible

    last = memory.pop("last", None)
    M = memory.get("model")
    if last is not None:
        ac = last["action"]
        wrong_position = s["p"] != last["wanted"]
        ignore = (1 << s["p"]) | (1 << last["p"]) | (1 << last["wanted"])
        bf, bw = last["raw"]
        af, aw = s["raw"]
        pf, pw = last["before_world"]
        qf, qw = last["after_world"]
        mismatch = ((((bf ^ af) ^ (pf ^ qf)) |
                     ((bw ^ aw) ^ (pw ^ qw))) & ~ignore).bit_count() > 1
        changed = ((bf ^ af) | (bw ^ aw)) & ~ignore

        if isinstance(ac, str) and ac in memory.get("mechanisms", {}):
            spec = memory["mechanisms"][ac]
            if changed:
                spec["phase"] = (spec["phase"] + 1) % (8 if ac == "e" else 6)

        if wrong_position or mismatch:
            rebuild = True
            memory["plan"] = []
        if isinstance(ac, int) and wrong_position:
            memory.setdefault("blocked", set()).add(
                (pw, last["p"], last["wanted"]))
        elif isinstance(ac, str):
            if ac not in _CONTROLS and ac != "c":
                history = memory.setdefault("learn", {}).setdefault(ac, [])
                history.append((bf, bw, af, aw))
                if len(history) > 24:
                    del history[0]
                recognized = ac in memory.get("mechanisms", {})
                if mismatch and changed and not recognized:
                    memory.setdefault("force", set()).add(ac)
                if mismatch and not changed:
                    memory.setdefault("unsafe", set()).add((ac, last["p"]))
            if ac == "c" and s["p"] != last["p"]:
                memory.setdefault("telelinks", {})[last["p"]] = s["p"]
            if ac in ("L", "R", "U", "D") and mismatch and not changed:
                st = last["before_state"]
                if M and st[1] >= 0:
                    x, y = M["anchors"][st[1]]
                    memory.setdefault("crane_bad", set()).add((x, y, ac))

    if M is not None:
        if (M["goal"] != s["goal"] or M["limit"] != s["limit"] or
                M["ox"] != s["ox"] or M["oy"] != s["oy"]):
            rebuild = True
            memory["plan"] = []

    seen = memory.setdefault("seen", {})
    seen[(s["raw"], s["p"])] = seen.get((s["raw"], s["p"]), 0) + 1

    if M is None or rebuild:
        M = _build(a, s, memory)
        memory["model"] = M
        memory["plan"] = []
    M["p"] = s["p"]
    M["buttons"].update(buttons)
    M["telelinks"] = memory.get("telelinks", {})
    M["bad"] = memory.get("blocked", set())
    for g in conditional:
        mask = 0
        for p in visible_at.get(g, ()):
            mask |= 1 << p
        M["allowed"][g] = mask

    if s["p"] == M["goal"]:
        world = _world(M, M["current"])
        choices = []
        for ac, dx, dy in ((1, 0, -1), (2, 0, 1),
                          (3, -1, 0), (4, 1, 0)):
            x, y = s["p"] % 32 + dx, s["p"] // 32 + dy
            blocked = not (0 <= x < 32 and 0 <= y < 32)
            if not blocked:
                blocked = not bool(world[1] & (1 << (32 * y + x)))
            choices.append((not blocked, ac))
        return "ACTION" + str(min(choices)[1]), None, None

    plan = memory.get("plan", [])
    if plan and isinstance(plan[0][0], str) and plan[0][0] not in s["buttons"]:
        ac = plan[0][0]
        memory["plan"] = []
        plan = []
        if ac not in _CONTROLS:
            conditional.add(ac)
            M["allowed"][ac] = sum(1 << p for p in visible_at.get(ac, ()))

    exploratory = False
    if not plan:
        plan = _plan(M)
        if not plan:
            plan, exploratory = _fallback(M, s, memory)
        memory["plan"] = plan

    if not plan:
        for ac in (1, 2, 3, 4):
            name = "ACTION" + str(ac)
            if name in available:
                return name, None, None
        return available[0], None, None

    ac, destination, after_state = plan.pop(0)
    before_state = M["current"]
    before_world = _world(M, before_state)
    after_world = _world(M, after_state)

    if isinstance(ac, str):
        if ac not in s["buttons"]:
            memory["plan"] = []
            memory["rebuild_next"] = True
            for q, direction in _NEIGH[s["p"]]:
                if before_world[1] & (1 << q):
                    return "ACTION" + str(direction), None, None
            return "ACTION1", None, None
        name = "ACTION6"
        row, col = s["buttons"][ac]
    else:
        name = "ACTION" + str(ac)
        row = col = None

    if name not in available:
        memory["plan"] = []
        for fallback in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if fallback in available:
                return fallback, None, None
        return available[0], None, None

    memory["last"] = {
        "action": ac, "p": s["p"], "wanted": destination,
        "raw": s["raw"],
        "before_world": (before_world[0], before_world[1]),
        "after_world": (after_world[0], after_world[1]),
        "before_state": before_state,
    }
    M["current"] = after_state
    M["p"] = destination
    if exploratory:
        memory["rebuild_next"] = True
    return name, row, col


def act(obs, available, memory):
    try:
        return _act_impl(obs, available, memory)
    except Exception:
        memory.pop("model", None)
        memory.pop("last", None)
        memory["plan"] = []
        a = np.asarray(obs)
        py, px = np.nonzero(a == 14)
        gy, gx = np.nonzero(a == 11)
        directions = [1, 2, 3, 4]
        if len(px) and len(gx):
            dx = int(gx[0]) - int(px[0])
            dy = int(gy[0]) - int(py[0])
            directions = [4 if dx > 0 else 3, 2 if dy > 0 else 1] + directions
        turn = memory.get("emergency_turn", 0)
        memory["emergency_turn"] = turn + 1
        if turn % 3 == 2:
            directions = directions[1:] + directions[:1]
        for d in directions:
            name = "ACTION" + str(d)
            if name in available:
                return name, None, None
        if "RESET" in available:
            return "RESET", None, None
        return available[0], None, None
