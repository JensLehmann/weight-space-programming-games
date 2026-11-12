import numpy as np


def build():
    P, G = {}, []
    size = {"x0": 16, "coords": 3, "mem": 80, "aprev": 8, "avail": 8}
    parameter_cache, operation_caches = {}, {}
    serial = [0]

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(P)
            P[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def node(op, inputs=None, dest=None, name=None, **fields):
        target = G if dest is None else dest
        cache = operation_caches.setdefault(id(target), {})
        if op != "loop":
            key = (op, tuple(inputs or ()), tuple(sorted(fields.items())))
            if name is None and key in cache:
                return cache[key]
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        target.append(nd)
        if op != "loop":
            cache[key] = name
        return name

    def conv(x, w, b=None, dest=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        y = node("conv", [x], dest=dest, **fields)
        size[y] = w.shape[0]
        return y

    def dense(x, w, b=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        y = node("dense", [x], **fields)
        size[y] = w.shape[0]
        return y

    def const(x):
        x = np.asarray(x, np.float32).reshape(-1)
        return dense("aprev", np.zeros((len(x), 8)), x)

    one, zero = const(1), const(0)

    def S(op, a, b=None, dest=None):
        inputs = [a] if b is None else [a, b]
        y = node(op, inputs, dest=dest)
        size[y] = max(size[x] for x in inputs)
        return y

    def V(op, a, b=None):
        inputs = [a] if b is None else [a, b]
        y = node(op, inputs)
        size[y] = max(size[x] for x in inputs)
        return y

    def ss(x, a, b):
        y = node("slice", [x], start=a, stop=b)
        size[y] = b - a
        return y

    def vs(x, a, b):
        y = node("vslice", [x], start=a, stop=b)
        size[y] = b - a
        return y

    def cat(xs):
        y = node("concat", xs)
        size[y] = sum(size[x] for x in xs)
        return y

    def vcat(xs):
        y = node("vconcat", xs)
        size[y] = sum(size[x] for x in xs)
        return y

    def pool(x, op="gmax"):
        y = node(op, [x])
        size[y] = size[x]
        return y

    def vl(x, a=1, b=0):
        n = size[x]
        return dense(x, a * np.eye(n), np.full(n, b))

    def va(a, b):
        return V("vadd", a, b)

    def vm(a, b):
        return V("vmul", a, b)

    def neg(x):
        return vl(x, -1)

    def vn(x):
        return vl(x, -1, 1)

    def vab(x):
        return va(V("vrelu", x), V("vrelu", neg(x)))

    def vc(x):
        return va(V("vrelu", x), neg(V("vrelu", vl(x, 1, -1))))

    def threshold(x, a=1, b=0):
        return vc(vl(x, a, b))

    def choosev(a, b, gate):
        return va(vm(a, gate), vm(b, vn(gate)))

    def sa(x, a=1, b=0, dest=None):
        if size[x] == 1:
            return conv(x, [[[[a]]]], [b], dest=dest)
        y = S("mul", x, const(a), dest=dest)
        if b:
            y = S("add", y, const(b), dest=dest)
        return y

    def sc(x, dest=None):
        return S("min2", S("relu", x, dest=dest), one, dest=dest)

    def sn(x, dest=None):
        return S("sub", one, x, dest=dest)

    def chooses(a, b, gate):
        return S("add", S("mul", a, gate), S("mul", b, vn(gate)))

    def total(x):
        return conv(x, np.ones((1, size[x], 1, 1)))

    def shift(x, dr, dc, dest=None):
        c = size[x]
        k = 1 if dr == dc == 0 else (
            3 if max(abs(dr), abs(dc)) <= 1 else 7)
        w = np.zeros((c, c, k, k), np.float32)
        for j in range(c):
            w[j, j, k // 2 + dr, k // 2 + dc] = 1
        return conv(x, w, dest=dest)

    pix = conv("coords", [
        [[[63]], [[0]], [[0]]],
        [[[0]], [[63]], [[0]]],
        [[[0]], [[0]], [[1]]],
    ])
    row, col, ones = ss(pix, 0, 1), ss(pix, 1, 2), ss(pix, 2, 3)
    xy = ss(pix, 0, 2)
    lex = S("add", col, sa(row, 0.001))
    first, last = sa(lex, -1, 1000), sa(lex, 1, 1000)
    ordering = cat([first, last])

    def peak(mask, score):
        z = S("mul", mask, score)
        p = sc(sa(S("sub", z, pool(z)), 10000, 1))
        return S("mul", mask, p)

    def batch_position(mask, scores):
        p = peak(mask, scores)
        rr, cc = pool(S("mul", row, p)), pool(S("mul", col, p))
        w = np.zeros((4, 4), np.float32)
        for out, src in enumerate([0, 2, 1, 3]):
            w[out, src] = 1
        return dense(vcat([rr, cc]), w)

    wb = np.zeros((1, 16, 1, 1), np.float32)
    wb[0, 10, 0, 0] = wb[0, 1, 0, 0] = 1
    blue = conv("x0", wb)
    wcorner = np.zeros((1, 1, 3, 3), np.float32)
    wcorner[0, 0, 1, 1] = 1
    wcorner[0, 0, 0, 1] = wcorner[0, 0, 1, 0] = -1
    corners = S("relu", conv(blue, wcorner))

    # Infer display scale from the ends of filled player squares.
    evidence = []
    for axis in (0, 1):
        shifted = shift(blue, 3 if axis == 0 else 0,
                        3 if axis == 1 else 0)
        w = np.zeros((7, 2, 7, 7), np.float32)
        for j in range(7):
            source = 0 if j <= 3 else 1
            off = j if j <= 3 else j - 3
            dr, dc = (off, 0) if axis == 0 else (0, off)
            w[j, source, 3 + dr, 3 + dc] = 1
        probes = conv(cat([blue, shifted]), w)
        w = np.zeros((5, 7, 1, 1), np.float32)
        for j, s0 in enumerate(range(2, 7)):
            w[j, s0 - 1, 0, 0] = 1
            w[j, s0, 0, 0] = -1
        evidence.append(pool(S("mul", S("relu", conv(probes, w)),
                               corners)))
    evidence = vc(va(evidence[0], evidence[1]))
    w = np.zeros((3, 5), np.float32)
    for j, s0 in enumerate(range(2, 7)):
        w[:, j] = [s0 - 4, 1 / s0 - 0.25, (s0 - 1) / 2 - 1.5]
    initialized = vs("mem", 0, 1)
    geometry = choosev(vs("mem", 1, 4),
                       dense(evidence, w, [4, 0.25, 1.5]), initialized)
    scale, invscale, half = (
        vs(geometry, 0, 1), vs(geometry, 1, 2), vs(geometry, 2, 3))
    delta = dense(scale, np.ones((5, 1)), -np.arange(2, 7))
    scale_hot = V("vrelu", vl(vab(delta), -1, 1))

    # Track two active pieces and exclude an already joined pair.
    old_done_r, old_done_c, old_done = (
        vs("mem", 12, 13), vs("mem", 13, 14), vs("mem", 14, 15))

    def completed_filter(rp, cp, flag):
        d = S("add", S("abs", S("sub", row, rp)),
              S("abs", S("sub", col, cp)))
        keep = sc(sa(S("sub", d, vm(scale, const(0.4))), 10))
        return chooses(keep, ones, flag)

    active = S("mul", corners,
               completed_filter(old_done_r, old_done_c, old_done))
    old_positions = vs("mem", 4, 8)
    wr = np.zeros((2, 4), np.float32)
    wc = np.zeros((2, 4), np.float32)
    wr[0, 0] = wr[1, 2] = wc[0, 1] = wc[1, 3] = 1
    old_rows, old_cols = dense(old_positions, wr), dense(old_positions, wc)
    tracking_distance = S(
        "add", S("abs", S("sub", row, old_rows)),
        S("abs", S("sub", col, old_cols)))
    tracked_score = S("add", sa(tracking_distance, -1, 1000),
                      sa(ordering, 0.0001))
    raw_positions = batch_position(
        active, chooses(tracked_score, ordering, initialized))
    separation_w = np.asarray([
        [1, 0, -1, 0], [0, 1, 0, -1]], np.float32)
    separation = vm(
        dense(vab(dense(raw_positions, separation_w)), [[1, 1]]),
        invscale)
    same = threshold(separation, -5, 1)
    several = threshold(vl(pool(active, "gmean"), 4096), 1, -1)
    merged = vm(vm(vm(same, several), initialized), vn(old_done))
    done = vc(va(old_done, merged))
    done_r = choosev(vs(raw_positions, 0, 1), old_done_r, merged)
    done_c = choosev(vs(raw_positions, 1, 2), old_done_c, merged)
    remaining = S("mul", corners, completed_filter(done_r, done_c, done))
    fresh_positions = batch_position(remaining, ordering)
    positions = choosev(fresh_positions, raw_positions, merged)
    ar, ac, br, bc = [vs(positions, j, j + 1) for j in range(4)]
    acr, acc, bcr, bcc = (
        va(ar, half), va(ac, half), va(br, half), va(bc, half))

    phase = vs("mem", 15, 16)
    block_mode = threshold(phase)
    old_block_r, old_block_c = vs("mem", 16, 17), vs("mem", 17, 18)
    origin_r, origin_c = vs("mem", 18, 19), vs("mem", 19, 20)

    # Infer horizontal/vertical response signs from observed movement.
    dw = np.zeros((4, 8), np.float32)
    for j, index in enumerate([1, 0, 3, 2]):
        dw[j, index] = 1
        dw[j, 4 + index] = -1
    motion_delta = dense(vcat([positions, old_positions]), dw)
    command_w = np.zeros((4, 8), np.float32)
    command_w[0, 3] = command_w[2, 3] = -1
    command_w[0, 4] = command_w[2, 4] = 1
    command_w[1, 1] = command_w[3, 1] = -1
    command_w[1, 2] = command_w[3, 2] = 1
    response = V("vtanh", vl(
        vm(motion_delta, dense("aprev", command_w)), 20))
    learn = vm(vm(initialized, vn(merged)), vn(block_mode))
    changed = vm(vm(response, response), learn)
    defaults = const([1, 1, -1, 1])
    old_signs = choosev(vs("mem", 8, 12), defaults, initialized)
    signs = choosev(defaults,
                    choosev(response, old_signs, changed), merged)
    ah, av, bh, bv = [vs(signs, j, j + 1) for j in range(4)]

    # Distinguish coloured gates/switches from recoloured walls.
    edge_r = sc(sa(S("abs", S("sub", row, const(1))), -100, 1))
    edge_c = S("max2", sc(sa(col, -100, 1)),
               sc(sa(S("sub", const(63), col), -100, 1)))
    wall_palette = pool(S("mul", "x0", S("mul", edge_r, edge_c)))
    colour_w = np.zeros((3, 16, 1, 1), np.float32)
    select_w = np.zeros((3, 16), np.float32)
    for j, colour in enumerate([14, 12, 15]):
        colour_w[j, colour, 0, 0] = select_w[j, colour] = 1
    colours = S("mul", conv("x0", colour_w),
                vn(dense(wall_palette, select_w)))

    def component_probes(axis):
        shifted = shift(colours, 3 if axis == 0 else 0,
                        3 if axis == 1 else 0)
        w = np.zeros((30, 6, 7, 7), np.float32)
        repeat = np.zeros((30, 5), np.float32)
        reduce_w = np.zeros((6, 30, 1, 1), np.float32)
        for kind in range(2):
            for colour in range(3):
                for j, s0 in enumerate(range(2, 7)):
                    off0 = s0 if kind == 0 else s0 - 1
                    source = colour if off0 <= 3 else colour + 3
                    off = off0 if off0 <= 3 else off0 - 3
                    dr, dc = (off, 0) if axis == 0 else (0, off)
                    out = kind * 15 + colour * 5 + j
                    w[out, source, 3 + dr, 3 + dc] = 1
                    repeat[out, j] = 1
                    reduce_w[kind * 3 + colour, out, 0, 0] = 1
        probes = conv(cat([colours, shifted]), w)
        return conv(S("mul", probes, dense(scale_hot, repeat)), reduce_w)

    px, py = component_probes(1), component_probes(0)
    xe, xi, ye, yi = ss(px, 0, 3), ss(px, 3, 6), ss(py, 0, 3), ss(py, 3, 6)
    w = np.zeros((3, 3, 3, 3), np.float32)
    for j in range(3):
        w[j, j, 1, 1] = 1
        w[j, j, 0, 1] = w[j, j, 1, 0] = -1
    component_corners = S("mul", S("relu", conv(colours, w)),
                          S("min2", xi, yi))
    long_component = S("max2", xe, ye)
    gate_corners = S("mul", component_corners, long_component)
    switch_corners = S("mul", component_corners, sn(long_component))
    objects = cat([gate_corners, switch_corners])
    horizontal_fields = cat([xe, S("mul", colours, zero)])
    gate_indicator = const([1, 1, 1, 0, 0, 0])

    # Remember two descriptors per object type, even while occluded.
    fields = [[] for _ in range(5)]
    for score in (first, last):
        selected = peak(objects, score)
        rr, cc = pool(S("mul", row, selected)), pool(S("mul", col, selected))
        horizontal = pool(S("mul", selected, horizontal_fields))
        width = vm(scale, vl(horizontal, 2, 1))
        height = vm(scale, vl(va(gate_indicator, neg(horizontal)), 2, 1))
        found = pool(objects)
        for collection, value in zip(fields, [rr, cc, width, height, found]):
            collection.append(value)
    new_descriptors = vcat([vcat(collection) for collection in fields])
    old_descriptors = vs("mem", 20, 80)
    old_flags = vs("mem", 68, 80)
    retain = dense(old_flags, np.tile(np.eye(12), (5, 1)))
    rect_memory = choosev(old_descriptors, new_descriptors, retain)
    rect_rows, rect_cols, widths, heights, flags = [
        vs(rect_memory, j * 12, (j + 1) * 12) for j in range(5)]
    rect_r = va(rect_rows, vl(heights, 0.5, -0.5))
    rect_c = va(rect_cols, vl(widths, 0.5, -0.5))
    radius_r, radius_c = vl(heights, 0.5, -0.3), vl(widths, 0.5, -0.3)
    group_w = np.zeros((6, 12, 1, 1), np.float32)
    for j in range(6):
        group_w[j, j, 0, 0] = group_w[j, j + 6, 0, 0] = 1

    def rectangles(rfield, cfield):
        dr = S("abs", S("sub", rfield, rect_r))
        dc = S("abs", S("sub", cfield, rect_c))
        inside = sc(sa(S("min2", S("sub", radius_r, dr),
                          S("sub", radius_c, dc)), 5))
        return sc(conv(S("mul", inside, flags), group_w))

    raster_rects = rectangles(row, col)
    raster_switches = ss(raster_rects, 3, 6)
    switches = sc(total(raster_switches))
    background, yellow, selected_colour = (
        ss("x0", 5, 6), ss("x0", 9, 10), ss("x0", 11, 12))

    # Track the selected movable block's centre.
    bd = S("add", S("abs", S("sub", row, old_block_r)),
           S("abs", S("sub", col, old_block_c)))
    nearby = sc(sa(S("sub", vm(scale, const(2.5)), bd), 10))
    selected_block = S("mul", S("mul", selected_colour, nearby), block_mode)
    selected_found = pool(selected_block)
    penalty = sa(selected_block, 10000, -10000)
    maxima = pool(S("add", xy, penalty))
    minima_negative = pool(S("add", sa(xy, -1), penalty))
    centre = dense(vcat([maxima, minima_negative]), [
        [0.5, 0, -0.5, 0], [0, 0.5, 0, -0.5]])
    centre = choosev(centre, vcat([old_block_r, old_block_c]), selected_found)
    block_r, block_c = vs(centre, 0, 1), vs(centre, 1, 2)

    raw_free = sc(S("add", S("add", background, blue), switches))
    relaxed_free = sc(S("add", S("add", raw_free, yellow), selected_block))
    movable_free = sc(S("add", background, selected_block))

    def erode(x):
        return sc(sa(conv(x, np.ones((1, 1, 3, 3))), 1, -8))

    free, relaxed_free, movable_free = (
        erode(raw_free), erode(relaxed_free), erode(movable_free))

    def seed(rp, cp):
        d = S("add", S("abs", S("sub", row, rp)),
              S("abs", S("sub", col, cp)))
        return sc(sa(d, -1, 2))

    def flood(seeds, traversable, count, name):
        body = []
        rs, ds = name + "_r", name + "_d"
        size[rs] = size[ds] = count
        w = np.zeros((count, count, 3, 3), np.float32)
        for j in range(count):
            for dr, dc in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]:
                w[j, j, dr, dc] = 1
        spread = conv(rs, w, dest=body)
        reached = S("mul", S("min2", spread, one, dest=body),
                    traversable, dest=body)
        distance = S("add", ds, sn(reached, dest=body), dest=body)
        node("loop", name=name, T=256,
             state=[{"id": rs, "init": seeds},
                    {"id": ds, "zeros": [count, 64, 64]}],
             body=body, next={rs: reached, ds: distance})
        r, d = name + "." + rs, name + "." + ds
        size[r] = size[d] = count
        return r, d

    aseed, bseed = seed(acr, acc), seed(bcr, bcc)
    reach, distances = flood(
        cat([aseed, bseed, aseed, bseed,
             S("mul", seed(block_r, block_c), block_mode)]),
        cat([free, free, relaxed_free, relaxed_free, movable_free]),
        5, "navigation")
    ra, rb, rfa, rfb, rblock = [ss(reach, j, j + 1) for j in range(5)]
    da, db, dfa, dfb, dblock = [ss(distances, j, j + 1) for j in range(5)]
    common = S("mul", ra, rb)
    disconnected = vn(pool(common))
    relaxed_connected = pool(S("mul", rfa, rfb))
    reachable_switch = S("mul", S("mul", switches, S("max2", ra, rb)),
                         sn(blue))
    has_switch = pool(reachable_switch)
    need_block = vm(vm(disconnected, pool(yellow)),
                    vc(va(relaxed_connected, vn(has_switch))))
    select_block = vm(need_block, vn(block_mode))
    trigger = vm(vm(disconnected, has_switch), vn(need_block))
    tie = S("add", sa(row, -0.01), sa(col, -0.0001))
    meeting_score = S("add", sa(S("max2", da, db), -1, 2000), tie)
    switch_score = S("add", sa(S("min2", da, db), -1, 2000), tie)
    target_seed = peak(chooses(reachable_switch, common, trigger),
                       chooses(switch_score, meeting_score, trigger))
    target_types = pool(S("mul", raster_switches, target_seed))

    origin_distance = S("add", S("abs", S("sub", row, origin_r)),
                        S("abs", S("sub", col, origin_c)))
    moved_away = sc(sa(S("sub", origin_distance, scale), 10))
    room = conv(movable_free, np.ones((1, 1, 7, 7)))
    parking_mask = S("mul", S("mul", rblock, movable_free), moved_away)
    parking_score = S(
        "add", S("add", S("add", dfa, dfb), sa(dblock, -0.2, 2000)),
        S("add", sa(room, -0.25), tie))
    parking_seed = peak(parking_mask, parking_score)
    _, target_distances = flood(
        cat([target_seed, parking_seed]),
        cat([free, movable_free]), 2, "targets")
    target_distance, parking_distance = (
        ss(target_distances, 0, 1), ss(target_distances, 1, 2))

    # Each matrix axis represents one piece's 7x7 local window.
    N, K = 7, 49
    offsets_r = np.asarray([j // N - 3 for j in range(K)], np.float32)
    offsets_c = np.asarray([j % N - 3 for j in range(K)], np.float32)

    def candidate_centres(rp, cp, rr, cc):
        return (
            dense(vcat([rp, scale]), np.stack([np.ones(len(rr)), rr], 1)),
            dense(vcat([cp, scale]), np.stack([np.ones(len(cc)), cc], 1)))

    def sample_masks(rvec, cvec):
        dr, dc = S("abs", S("sub", row, rvec)), S("abs", S("sub", col, cvec))
        return sc(sa(S("add", dr, dc), -1, 2))

    arv, acv = candidate_centres(acr, acc, offsets_r, offsets_c)
    brv, bcv = candidate_centres(bcr, bcc, offsets_r, offsets_c)
    amask, bmask = sample_masks(arv, acv), sample_masks(brv, bcv)

    def basis(axis):
        z = conv(axis, np.ones((K, 1, 1, 1)), -np.arange(K))
        return S("relu", sa(S("abs", z), -1, 1))

    basis_a, basis_b = basis(row), basis(col)

    def decode(v, b):
        return total(S("mul", b, v))

    def sample(field, masks):
        return pool(S("mul", field, masks))

    afree = sc(sa(decode(sample(free, amask), basis_a), 10, -9))
    bfree = sc(sa(decode(sample(free, bmask), basis_b), 10, -9))
    adist = decode(sample(target_distance, amask), basis_a)
    bdist = decode(sample(target_distance, bmask), basis_b)

    def local_coordinates(axis, rp, cp):
        steps = conv(axis, 2 * np.ones((6, 1, 1, 1)),
                     -2 * (N * np.arange(1, N) - 0.5))
        quotient = total(sc(steps))
        remainder = S("sub", axis, sa(quotient, N))
        rfield = S("add", S("mul", sa(quotient, 1, -3), scale), rp)
        cfield = S("add", S("mul", sa(remainder, 1, -3), scale), cp)
        return rfield, cfield, sc(remainder), sc(sa(remainder, -1, N - 1))

    lar, lac, alow, ahigh = local_coordinates(row, acr, acc)
    lbr, lbc, blow, bhigh = local_coordinates(col, bcr, bcc)
    arects, brects = rectangles(lar, lac), rectangles(lbr, lbc)
    agates, aswitches = ss(arects, 0, 3), ss(arects, 3, 6)
    bgates, bswitches = ss(brects, 0, 3), ss(brects, 3, 6)
    pressed = S("max2", aswitches, bswitches)
    chosen_pressed = sc(total(S("mul", pressed, target_types)))
    afree = sc(S("sub", afree, total(agates)))
    bfree = sc(S("sub", bfree, total(bgates)))

    def logical_shift(x, axis, direction, vertical):
        if not vertical:
            return shift(x, direction if axis == 0 else 0,
                         direction if axis == 1 else 0)
        for step in (3, 3, 1):
            x = shift(x, direction * step if axis == 0 else 0,
                      direction * step if axis == 1 else 0)
        return x

    axis_conditions = []
    for axis, field, gates, low, high in [
        (0, afree, agates, alow, ahigh),
        (1, bfree, bgates, blow, bhigh)
    ]:
        conditions = []
        for vertical in (False, True):
            for direction in (-1, 1):
                ff = logical_shift(field, axis, direction, vertical)
                gg = logical_shift(gates, axis, direction, vertical)
                ff = sc(S("add", ff, total(S("mul", gg, pressed))))
                if not vertical:
                    ff = S("mul", ff, high if direction == 1 else low)
                conditions.append(ff)
        axis_conditions.append(cat(conditions))

    combinations = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    wa, wb = np.zeros((8, 4, 1, 1)), np.zeros((8, 4, 1, 1))
    for index in range(8):
        a, b = combinations[index % 4]
        base = 2 if index >= 4 else 0
        wa[index, base + (a == 1), 0, 0] = 1
        wb[index, base + (b == 1), 0, 0] = 1
    ca, cb = conv(axis_conditions[0], wa), conv(axis_conditions[1], wb)
    coefficients = [
        S("mul", ca, cb), S("mul", ca, sn(cb)),
        S("mul", sn(ca), cb), S("mul", sn(ca), sn(cb))]

    dx, dy = S("sub", lac, lbc), S("sub", lar, lbr)
    separation = S("mul", S("add", S("abs", dx), S("abs", dy)), invscale)
    meet = sc(sa(separation, -100, 1))
    goal = S("max2", sa(meet, 1300, -1000),
             sa(S("mul", chosen_pressed, trigger), 1100, -1000))
    initial = sa(chooses(S("min2", adist, bdist),
                         S("add", adist, bdist), trigger), -1)
    initial = S("max2", initial, goal)

    # Handle the simulator's adjacent horizontal-swap joining rule.
    cross_delta = S("sub", dx, dense(scale, [[1], [-1]]))
    cross = sc(sa(S("mul",
                     S("add", S("abs", cross_delta), S("abs", dy)),
                     invscale), -100, 1))
    cross_select = np.zeros((2, 8, 1, 1))
    cross_select[0, 1, 0, 0] = cross_select[1, 2, 0, 0] = 1
    cross = S("mul", cross, conv(coefficients[0], cross_select))
    cross_w = np.zeros((8, 2, 1, 1))
    cross_w[1, 0, 0, 0] = cross_w[2, 1, 0, 0] = 1300
    crossing_reward = conv(cross, cross_w, -1000 * np.ones(8))

    products = vcat([vm(ah, bh), vm(av, bv)])
    features = vcat([signs, products])
    allow_w = np.zeros((8, 2))
    for index in range(8):
        a, b = combinations[index % 4]
        allow_w[index, index // 4] = 0.5 * a * b
    allowed = dense(products, allow_w, 0.5 * np.ones(8))
    invalid = vl(allowed, 10000, -10000)

    def action_weights(vertical, command):
        w = np.zeros((4, 6))
        ia, ib, ip = (1, 3, 5) if vertical else (0, 2, 4)
        for index, (a, b) in enumerate(combinations):
            w[index, ia] = 0.25 * a * command
            w[index, ib] = 0.25 * b * command
            w[index, ip] = 0.25 * a * b
        return dense(features, w, 0.25 * np.ones(4))

    # Eight transition channels share a single Bellman value field.
    body = []
    value_state, q_state = "joint_value", "joint_q"
    size[value_state], size[q_state] = 1, 8

    def transition_group(kind):
        horizontal = np.zeros((4, 1, 3, 3), np.float32)
        vertical_first = np.zeros((4, 1, 7, 7), np.float32)
        directions = []
        for index, (a, b) in enumerate(combinations):
            dr, dc = (
                (a, b) if kind == 0 else
                ((a, 0) if kind == 1 else (0, b)))
            directions.append((dr, dc))
            horizontal[index, 0, 1 + dr, 1 + dc] = 1
            vertical_first[index, 0, 3 + 3 * dr, 3 + 3 * dc] = 1
        h = conv(value_state, horizontal, dest=body)
        v = conv(value_state, vertical_first, dest=body)
        for step, kernel_size in [(3, 7), (1, 3)]:
            w = np.zeros((4, 4, kernel_size, kernel_size), np.float32)
            for index, (dr, dc) in enumerate(directions):
                w[index, index, kernel_size // 2 + step * dr,
                  kernel_size // 2 + step * dc] = 1
            v = conv(v, w, dest=body)
        y = node("concat", [h, v], dest=body)
        size[y] = 8
        return y

    transitions = [transition_group(kind) for kind in range(3)]
    transitions.append(value_state)
    terms = [S("mul", x, c, dest=body)
             for x, c in zip(transitions, coefficients)]
    q = terms[0]
    for term in terms[1:]:
        q = S("add", q, term, dest=body)

    # A small travel cost preserves geodesic guidance.
    q = S("sub", q, one, dest=body)
    # A jointly blocked action cannot help advance this deterministic game.
    q = S("sub", q, sa(coefficients[3], 40), dest=body)
    q = S("max2", q, crossing_reward, dest=body)
    q = S("add", q, invalid, dest=body)

    # Keep the heuristic as a stopping value at every iteration.
    # Without this anchor, closed local windows flatten after many updates.
    best = initial
    for j in range(8):
        part = node("slice", [q], start=j, stop=j + 1, dest=body)
        size[part] = 1
        best = S("max2", best, part, dest=body)
    node("loop", name="planner", T=32,
         state=[{"id": value_state, "init": initial},
                {"id": q_state, "zeros": [8, 64, 64]}],
         body=body, next={value_state: best, q_state: q})
    final_q = "planner." + q_state
    size[final_q] = 8

    current_distance = S(
        "add", S("abs", S("sub", row, const(24))),
        S("abs", S("sub", col, const(24))))
    current = sc(sa(current_distance, -1, 1))
    qvalues = pool(S("add", final_q, sa(current, 10000, -10000)))
    hq, vq = vs(qvalues, 0, 4), vs(qvalues, 4, 8)

    def weighted(values, weights):
        return dense(vm(values, weights), np.ones((1, 4)))

    player_scores = [
        weighted(vq, action_weights(True, -1)),
        weighted(vq, action_weights(True, 1)),
        weighted(hq, action_weights(False, -1)),
        weighted(hq, action_weights(False, 1))]

    # Independent navigation toward a reachable block-parking cell.
    rr = np.asarray([0, -1, 1, 0, 0], np.float32)
    cc = np.asarray([0, 0, 0, -1, 1], np.float32)
    block_rv, block_cv = candidate_centres(block_r, block_c, rr, cc)
    block_masks = sample_masks(block_rv, block_cv)
    block_cost = sample(parking_distance, block_masks)
    block_valid = vc(vl(sample(movable_free, block_masks), 20, -2))
    block_scores = va(vl(block_cost, -1), vl(block_valid, 1000, -1000))
    block_moves = vs(block_scores, 1, 5)
    near_goal = threshold(va(scale, neg(vs(block_cost, 0, 1))), 1, 0.1)
    release = vm(block_mode, vc(va(near_goal, threshold(phase, 1, -19))))
    deselect = vm(pool(ss("x0", 1, 2)), vn(block_mode))
    click_flag = vc(va(va(select_block, release), deselect))
    moves = [choosev(vs(block_moves, j, j + 1),
                     player_scores[j], block_mode) for j in range(4)]

    click_score = S("add", sa(S("add", dfa, dfb), -1, 2000), tie)
    chosen_click = peak(yellow, click_score)
    click_position = pool(S("mul", xy, chosen_click))
    click_r, click_c = vs(click_position, 0, 1), vs(click_position, 1, 2)
    release_click = S("mul", blue, aseed)
    click_map = S(
        "add", S("mul", chosen_click, select_block),
        S("mul", release_click, vc(va(release, deselect))))
    node("add", [click_map, zero], name="click")
    size["click"] = 1
    logits = vcat([
        const(-10000), moves[0], moves[1], moves[2], moves[3],
        const(-10000), vl(click_flag, 20000, -10000), const(-10000)])
    node("vadd", [logits, const(np.zeros(8))], name="logits")
    size["logits"] = 8

    next_phase = va(vm(vm(vl(phase, 1, 1), block_mode), vn(release)),
                    select_block)
    next_block_r = choosev(
        click_r, choosev(block_r, old_block_r, block_mode), select_block)
    next_block_c = choosev(
        click_c, choosev(block_c, old_block_c, block_mode), select_block)
    memory = vcat([
        one, geometry, positions, signs, done_r, done_c, done, next_phase,
        next_block_r, next_block_c,
        choosev(click_r, origin_r, select_block),
        choosev(click_c, origin_c, select_block), rect_memory])
    node("vadd", [memory, const(np.zeros(80))], name="mem_next")
    size["mem_next"] = 80

    # Remove unused intermediates and their parameters.
    by_id = {nd["id"]: nd for nd in G}
    needed = set()

    def visit(name):
        base = name.split(".")[0]
        if base not in by_id or base in needed:
            return
        needed.add(base)
        nd = by_id[base]
        if nd["op"] != "loop":
            for source in nd["in"]:
                visit(source)
            return
        local = {s["id"] for s in nd["state"]}
        local.update(inner["id"] for inner in nd["body"])
        for state in nd["state"]:
            if "init" in state:
                visit(state["init"])
        for inner in nd["body"]:
            for source in inner.get("in", []):
                if source not in local:
                    visit(source)

    for output in ["logits", "click", "mem_next"]:
        visit(output)
    G = [nd for nd in G if nd["id"] in needed]
    used = set()
    for nd in G:
        for inner in nd["body"] if nd["op"] == "loop" else [nd]:
            for field in ("w", "b"):
                if field in inner:
                    used.add(inner[field])
    P = {name: value for name, value in P.items() if name in used}
    return {"graph": G, "memory_size": 80, "params": P}
