import numpy as np


def build():
    graph, params = [], {}
    sizes = {"x0": 16, "coords": 3, "mem": 95, "aprev": 8, "avail": 8}
    dest = graph
    serial, pserial = [0], [0]

    def parameter(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(a, np.float32)
        return name

    def node(op, inputs=None, name=None, size=None, **fields):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        dest.append(nd)
        if size is None:
            if op in ("concat", "vconcat"):
                size = sum(sizes[z] for z in inputs)
            elif op in ("slice", "vslice"):
                size = fields["stop"] - fields["start"]
            elif op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
                size = max(sizes[z] for z in inputs)
            else:
                size = sizes[inputs[0]]
        sizes[name] = size
        return name

    def cat(xs, vector=False):
        return xs[0] if len(xs) == 1 else node(
            "vconcat" if vector else "concat", xs)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, size=np.shape(w)[0], **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, size=np.shape(w)[0], **fields)

    def ss(x, a=1.0, b=0.0, name=None):
        n = sizes[x]
        return conv(x, np.eye(n, dtype=np.float32)[:, :, None, None] * a,
                    np.full(n, b, np.float32), name)

    def vs(x, a=1.0, b=0.0, name=None):
        n = sizes[x]
        return dense(x, np.eye(n, dtype=np.float32) * a,
                     np.full(n, b, np.float32), name)

    def constant(v):
        a = np.asarray(v, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(a), 8), np.float32), a)

    one, zero = constant([1.0]), constant([0.0])
    negative = constant([-1000.0])

    def clip_s(x):
        return node("min2", [node("relu", [x]), one])

    def smix(xs, coefficients=None, bias=0.0):
        coefficients = coefficients or [1.0] * len(xs)
        w = np.zeros((1, sum(sizes[x] for x in xs), 1, 1), np.float32)
        j = 0
        for x, a in zip(xs, coefficients):
            w[0, j:j + sizes[x], 0, 0] = a
            j += sizes[x]
        return conv(cat(xs), w, [bias])

    def vsum(xs, coefficients=None, bias=0.0, vector_output=False):
        coefficients = coefficients or [1.0] * len(xs)
        if vector_output:
            n = sizes[xs[0]]
            w = np.zeros((n, sum(sizes[x] for x in xs)), np.float32)
            j = 0
            for x, a in zip(xs, coefficients):
                w[:, j:j + sizes[x]] = np.eye(n, dtype=np.float32) * a
                j += sizes[x]
            return dense(cat(xs, True), w, np.full(n, bias, np.float32))
        w = np.concatenate([
            np.full(sizes[x], a, np.float32)
            for x, a in zip(xs, coefficients)
        ])[None, :]
        return dense(cat(xs, True), w, [bias])

    def clip_v(x):
        r = node("vrelu", [x])
        excess = node("vrelu", [vs(r, 1.0, -1.0)])
        return vsum([r, excess], [1.0, -1.0], vector_output=True)

    def sl(x, a, b, vector=False):
        return node("vslice" if vector else "slice", [x], start=a, stop=b)

    def mul(a, b):
        return node("mul", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def not_s(x):
        return ss(x, -1.0, 1.0)

    def not_v(x):
        return vs(x, -1.0, 1.0)

    def query(x, at):
        return node("gmax", [mul(x, at)])

    def shift(x, dx=0, dy=0):
        while dx or dy:
            sx, sy = max(-3, min(3, dx)), max(-3, min(3, dy))
            n = sizes[x]
            w = np.zeros((n, n, 7, 7), np.float32)
            for channel in range(n):
                w[channel, channel, 3 - sy, 3 - sx] = 1.0
            x = conv(x, w)
            dx -= sx
            dy -= sy
        return x

    def dilation(x, radius, axis=None):
        while radius:
            r = min(3, radius)
            n, k = sizes[x], 2 * r + 1
            w = np.zeros((n, n, k, k), np.float32)
            for channel in range(n):
                if axis == "x":
                    w[channel, channel, r, :] = 1.0
                elif axis == "y":
                    w[channel, channel, :, r] = 1.0
                else:
                    w[channel, channel, :, :] = 1.0
            x = clip_s(conv(x, w))
            radius -= r
        return x

    def box_count(x, k=5):
        return conv(x, np.ones((1, 1, k, k), np.float32))

    def tile_count(x):
        w = np.zeros((1, 1, 3, 3), np.float32)
        w[0, 0, 1:3, 1:3] = 1.0
        return conv(x, w)

    def full_tile(x):
        return clip_s(ss(tile_count(x), 1.0, -3.0))

    def rectangle(seed, xlo, xhi, ylo, yhi):
        pieces = []
        for ya in range(ylo, yhi + 1, 7):
            yb = min(ya + 6, yhi)
            cy = (ya + yb) // 2
            for xa in range(xlo, xhi + 1, 7):
                xb = min(xa + 6, xhi)
                cx = (xa + xb) // 2
                w = np.zeros((1, 1, 7, 7), np.float32)
                for yy in range(ya, yb + 1):
                    for xx in range(xa, xb + 1):
                        w[0, 0, 3 - yy + cy, 3 - xx + cx] = 1.0
                pieces.append(conv(shift(seed, cx, cy), w))
        return clip_s(smix(pieces))

    def positive_bit(x, scale=126.0, bias=0.0):
        return clip_v(vs(x, scale, bias))

    def vector_abs(x):
        return vsum([node("vrelu", [x]), node("vrelu", [vs(x, -1.0)])],
                    vector_output=True)

    def winner(x):
        n = sizes[x]
        w = np.zeros((n * n, n), np.float32)
        for i in range(n):
            for j in range(n):
                w[i * n + j, j] += 1.0
                w[i * n + j, i] -= 1.0
        differences = node("vrelu", [dense(x, w)])
        w2 = np.zeros((n, n * n), np.float32)
        for i in range(n):
            w2[i, i * n:(i + 1) * n] = -4000.0
        return node("vrelu", [
            dense(differences, w2, np.ones(n, np.float32))])

    row, col = sl("coords", 0, 1), sl("coords", 1, 2)
    colours = [sl("x0", i, i + 1) for i in range(16)]
    play_columns = node("colmax", [colours[4]])
    not_bottom = clip_s(ss(row, -126.0, 125.0))
    play = mul(play_columns, not_bottom)
    sidebar = mul(not_s(play_columns), not_bottom)
    c = [mul(z, play) for z in colours]
    xy = cat([row, col])

    corner_w = np.zeros((1, 1, 3, 3), np.float32)
    corner_w[0, 0, 1, 1] = 1.0
    corner_w[0, 0, 0, 1] = -1.0
    corner_w[0, 0, 1, 0] = -1.0
    player = node("relu", [conv(c[14], corner_w)])
    goal = node("relu", [conv(c[11], corner_w)])

    def position(at):
        return vs(node("gmean", [mul(xy, at)]), 4096.0)

    # Counts and moments detect actions that leave the board unchanged.
    fp_planes = cat([c[i] for i in
                     [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    row2, col2 = mul(row, row), mul(col, col)
    fingerprint = node("gmean", [cat([
        fp_planes, mul(fp_planes, row), mul(fp_planes, col),
        mul(fp_planes, row2), mul(fp_planes, col2)])])
    difference = vsum([fingerprint, sl("mem", 0, 60, True)],
                      [1.0, -1.0], vector_output=True)
    difference_size = vsum([vector_abs(difference)])
    unchanged = node("vrelu", [vs(difference_size, -1000000.0, 1.0)])
    old_failures = sl("mem", 60, 75, True)
    previous_kind = sl("mem", 75, 86, True)
    executed = cat([
        sl("aprev", 1, 5, True),
        vmul(previous_kind, sl("aprev", 6, 7, True))
    ], True)
    failures = vmul(vsum([old_failures, executed], vector_output=True),
                    unchanged)
    direction_failures = sl(failures, 0, 4, True)
    kind_failures = sl(failures, 4, 15, True)

    # The unlock token has a two-pixel top edge with transparent corners.
    pickup_w = np.zeros((1, 1, 7, 7), np.float32)
    pickup_w[0, 0, 3, 4:6] = 1.0
    pickup_w[0, 0, 3, 3] = -2.0
    pickup_w[0, 0, 3, 6] = -2.0
    visible_pickup = node("relu", [conv(c[15], pickup_w, [-1.0])])

    remembered_pickup_xy = sl("mem", 92, 94, True)
    remembered_pickup_valid = sl("mem", 94, 95, True)
    pickup_delta = node("abs", [
        node("sub", [xy, remembered_pickup_xy])])
    remembered_pickup = clip_s(conv(
        pickup_delta,
        np.full((1, 2, 1, 1), -126.0, np.float32),
        [2.0]))
    remembered_pickup = mul(remembered_pickup, remembered_pickup_valid)
    pickup_seed = node("max2", [visible_pickup, remembered_pickup])
    pickup_position = position(pickup_seed)
    pickup_valid = node("gmax", [pickup_seed])
    pickup_pixels = rectangle(pickup_seed, 0, 3, 0, 1)
    c[15] = node("relu", [node("sub", [c[15], pickup_pixels])])

    # Portal recognition and memory preserve locations under the player.
    portal_w = np.zeros((2, 2, 3, 3), np.float32)
    for out in range(2):
        for yy, xx in [(0, 0), (1, 1)]:
            portal_w[out, 1 - out, 1 + yy, 1 + xx] = 1.0
        for yy, xx in [(0, 1), (1, 0)]:
            portal_w[out, out, 1 + yy, 1 + xx] = 1.0
    portal_patterns = node("relu", [
        conv(cat([c[6], c[7]]), portal_w, [-3.0, -3.0])])
    visible_portals = clip_s(smix([
        sl(portal_patterns, 0, 1), sl(portal_patterns, 1, 2)]))
    portal_memory = sl("mem", 86, 92, True)

    def remembered_point(ri, ci, vi):
        rr = sl(portal_memory, ri, ri + 1, True)
        cc = sl(portal_memory, ci, ci + 1, True)
        valid = sl(portal_memory, vi, vi + 1, True)
        dr = node("abs", [node("sub", [row, rr])])
        dc = node("abs", [node("sub", [col, cc])])
        match = node("relu", [ss(smix([dr, dc]), -63.0, 1.0)])
        return mul(match, valid)

    portals = clip_s(smix([
        visible_portals, remembered_point(0, 1, 4),
        remembered_point(2, 3, 5)]))
    rank = smix([row, col], [4032.0, 63.0])
    maximum_rank = node("gmax", [smix(
        [mul(rank, portals), portals], [1.0, 10000.0], -10000.0)])
    minimum_rank = vs(node("gmax", [smix(
        [mul(rank, portals), portals], [-1.0, 10000.0], -10000.0)]), -1.0)

    def rank_point(value):
        delta = node("abs", [node("sub", [rank, value])])
        return mul(portals, node("relu", [ss(delta, -1.0, 1.0)]))

    portal_min, portal_max = rank_point(minimum_rank), rank_point(maximum_rank)
    next_portals = cat([
        position(portal_min), position(portal_max),
        node("gmax", [portal_min]), node("gmax", [portal_max])], True)

    # Button targets are found by their geometry.
    def large_button(colour):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 1:6, :] = 1.0
        return mul(sidebar, node("relu", [
            conv(colours[colour], w, [-34.0])]))

    def small_button(colour):
        return mul(sidebar, node("relu", [
            conv(colours[colour], np.ones((1, 1, 3, 3), np.float32), [-8.0])]))

    def isolated_square(colour):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 2:6, 2:6] = 1.0
        for dy, dx in [(0, -2), (0, 3), (-2, 0), (3, 0)]:
            w[0, 0, 3 + dy, 3 + dx] = -2.0
        return mul(sidebar, node("relu", [
            conv(colours[colour], w, [-15.0])]))

    def extreme_column(mask, maximum):
        signed_col = col if maximum else ss(col, -1.0)
        value = node("gmax", [smix(
            [mul(signed_col, mask), mask], [1.0, 1000.0], -1000.0)])
        if not maximum:
            value = vs(value, -1.0)
        delta = node("abs", [node("sub", [col, value])])
        return mul(mask, node("relu", [ss(delta, -126.0, 1.0)]))

    button_a, button_b = large_button(8), large_button(9)
    button_c, button_d = large_button(6), large_button(15)
    button_e, button_f = small_button(2), small_button(11)
    square9, square10 = isolated_square(9), isolated_square(10)
    button_up = extreme_column(square9, False)
    button_down = extreme_column(square9, True)
    button_left = extreme_column(square10, False)
    button_right = extreme_column(square10, True)
    grab_w = np.zeros((1, 1, 7, 7), np.float32)
    grab_w[0, 0, 2:5, :] = 1.0
    grab_w[0, 0, 1, 3] = -2.0
    grab_w[0, 0, 5, 3] = -2.0
    button_grab = mul(sidebar, node("relu", [
        conv(colours[8], grab_w, [-20.0])]))
    button_maps = [
        button_a, button_b, button_c, button_d, button_e, button_f,
        button_up, button_down, button_left, button_right, button_grab]
    button_presence = [node("gmax", [z]) for z in button_maps]
    need_pickup = vmul(pickup_valid, not_v(button_presence[3]))
    planning_goal = node("add", [
        mul(goal, not_v(need_pickup)), mul(pickup_seed, need_pickup)])

    # Reconstruct alternative bridge orientations around their pivots.
    c[13] = mul(node("rowmax", [c[13]]), node("colmax", [c[13]]))
    near_hub = dilation(c[13], 10)
    bridge8 = mul(c[8], near_hub)
    crusher8 = mul(c[8], not_s(near_hub))
    hub_corner = node("relu", [conv(c[13], corner_w)])
    horizontal_seed = mul(hub_corner, node("rowmax", [bridge8]))
    vertical_seed = mul(hub_corner, node("colmax", [bridge8]))
    symmetric = clip_s(smix([
        shift(bridge8, -4, 0), shift(bridge8, 0, -4)]))
    sh, sv = mul(horizontal_seed, symmetric), mul(vertical_seed, symmetric)
    hh = node("relu", [node("sub", [horizontal_seed, sh])])
    hv = node("relu", [node("sub", [vertical_seed, sv])])
    other_bridge = clip_s(smix([
        rectangle(hh, 0, 3, -6, 3),
        rectangle(sh, 0, 3, -8, 11),
        rectangle(hv, -6, 3, 0, 3),
        rectangle(sv, -8, 11, 0, 3), c[13]]))
    bridge_current = full_tile(clip_s(smix([bridge8, c[13]])))
    bridge_other = full_tile(other_bridge)

    def gate(colour):
        count = tile_count(c[colour])
        solid = node("relu", [ss(count, 1.0, -3.0)])
        t1 = node("relu", [ss(count, 1.0, -1.0)])
        t2 = node("relu", [ss(count, 1.0, -2.0)])
        t3 = node("relu", [ss(count, 1.0, -3.0)])
        checker = clip_s(smix([t1, t2, t3], [1.0, -2.0, 1.0]))
        wall = node("relu", [node("sub", [clip_s(count), solid])])
        return solid, checker, wall

    bsolid, bchecker, bwall = gate(9)
    dsolid, dchecker, dwall = gate(15)

    tall_w = np.zeros((1, 1, 7, 7), np.float32)
    tall_w[0, 0, :, 3] = 1.0
    tall12 = node("relu", [conv(c[12], tall_w, [-3.0])])
    object12 = mul(c[12], dilation(clip_s(tall12), 2, "x"))
    thin12 = node("relu", [node("sub", [c[12], object12])])
    filler = c[5]
    thin = clip_s(smix([c[1], thin12]))
    filler_rows = node("rowmax", [filler])
    f_rows = mul(node("rowmax", [thin]), not_s(filler_rows))
    f_current = mul(thin, f_rows)
    run_w = np.zeros((1, 1, 7, 7), np.float32)
    run_w[0, 0, 3, :] = 1.0
    long_strip = node("relu", [conv(thin, run_w, [-6.0])])
    master = clip_s(smix([
        filler, mul(thin, node("rowmax", [long_strip]))]))
    master_columns = node("colmax", [master])
    has_master = node("gmax", [master])
    f_columns = clip_s(node("add", [master_columns, not_v(has_master)]))
    e_potential = mul(filler, button_presence[4])
    f_potential = mul(
        mul(dilation(f_current, 10, "x"), f_columns), button_presence[5])

    def coordinate_extreme(mask, coordinate, maximum=True):
        sign = 1.0 if maximum else -1.0
        field = smix(
            [mul(coordinate, mask), mask], [sign, 1000.0], -1000.0)
        result = node("gmax", [field])
        return result if maximum else vs(result, -1.0)

    track = c[0]
    track_box = mul(node("rowmax", [track]), node("colmax", [track]))
    object_present = node("gmax", [object12])
    object_enabled = vmul(object_present, button_presence[10])
    track_min_x = coordinate_extreme(track, col, False)
    left_corridor_center = vs(track_min_x, 1.0, 8.5 / 63.0)
    right_corridor_center = vs(track_min_x, 1.0, 20.5 / 63.0)

    def four_column_band(center):
        delta = node("abs", [node("sub", [col, center])])
        return clip_s(ss(delta, -252.0, 7.5))

    object_columns = clip_s(smix([
        four_column_band(left_corridor_center),
        four_column_band(right_corridor_center)]))
    object_rows = node("rowmax", [dilation(track_box, 4, "y")])
    object_potential = mul(mul(object_columns, object_rows), object_enabled)
    potential_floor = full_tile(clip_s(smix([
        e_potential, f_potential, object_potential])))

    # Classify the terrain hidden under the player.
    ordinary_pad_pixels = clip_s(smix([
        c[2], c[6], c[7], c[11], c[1], thin12, pickup_pixels]))
    pad_near = clip_s(ss(box_count(ordinary_pad_pixels), 1.0, -3.0))
    near_b = clip_s(ss(box_count(c[9]), 1.0, -3.0))
    near_d = clip_s(ss(box_count(c[15]), 1.0, -3.0))
    near_a = clip_s(ss(box_count(bridge8), 1.0, -3.0))
    marker_b = mul(mul(player, near_b), not_s(pad_near))
    marker_d = mul(mul(player, near_d), not_s(pad_near))
    marker_a = mul(mul(player, near_a), clip_s(smix(
        [pad_near, marker_b, marker_d], [-1.0, -1.0, -1.0], 1.0)))
    marker_base = node("relu", [smix(
        [player, marker_a, marker_b, marker_d], [1.0, -1.0, -1.0, -1.0])])
    base_pixels = clip_s(smix([
        c[2], c[6], c[7], c[11], c[1], c[12], pickup_pixels]))
    base_floor = clip_s(smix([
        full_tile(base_pixels), marker_base, portals]))
    bridge_current = clip_s(smix([bridge_current, marker_a]))
    bsolid = clip_s(smix([bsolid, marker_b]))
    dsolid = clip_s(smix([dsolid, marker_d]))
    bother_wall = clip_s(smix([full_tile(c[9]), marker_b]))
    dother_wall = clip_s(smix([full_tile(c[15]), marker_d]))
    crusher_wall = clip_s(tile_count(crusher8))

    floors = []
    for mode in range(8):
        af = bridge_other if mode & 1 else bridge_current
        bf = bchecker if mode & 2 else bsolid
        df = dchecker if mode & 4 else dsolid
        bw = bother_wall if mode & 2 else bwall
        dw = dother_wall if mode & 4 else dwall
        support = clip_s(smix([base_floor, af, bf, df, potential_floor]))
        floors.append(node("relu", [smix(
            [support, bw, dw, crusher_wall], [1.0, -1.0, -1.0, -1.0])]))
    all_floors = cat(floors)
    actual_floor = node("relu", [smix([
        clip_s(smix([base_floor, bridge_current, bsolid, dsolid, player])),
        bwall, dwall, crusher_wall], [1.0, -1.0, -1.0, -1.0])])
    stable_pixels = clip_s(smix([
        c[2], c[6], c[7], c[11], c[13], bridge8, c[9], c[15],
        pickup_pixels, mul(c[1], not_s(f_rows)),
        mul(thin12, not_s(f_rows))]))
    stable_floor = node("relu", [smix(
        [full_tile(stable_pixels), bwall, dwall, crusher_wall],
        [1.0, -1.0, -1.0, -1.0])])
    stable_near = clip_s(ss(box_count(stable_pixels), 1.0, -4.0))
    player_stable = query(stable_near, player)

    # Recurrent wavefront in position/switch-configuration space.
    initial_reach = mul(all_floors, planning_goal)
    body = []
    dest = body
    sizes["reach"], sizes["total"] = 8, 8
    neighbour_w = np.zeros((8, 8, 5, 5), np.float32)
    for m in range(8):
        for dy, dx in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            neighbour_w[m, m, 2 + dy, 2 + dx] = 1.0
    messages = [conv("reach", neighbour_w)]
    for bit, present in [
        (1, button_presence[0]), (2, button_presence[1]),
        (4, button_presence[3])]:
        w = np.zeros((8, 8, 1, 1), np.float32)
        for m in range(8):
            w[m, m ^ bit, 0, 0] = 1.0
        messages.append(mul(conv("reach", w), present))
    portal_reach = node("gmax", [mul("reach", portals)])
    messages.append(mul(mul(portals, portal_reach), button_presence[2]))
    combined = messages[0]
    for z in messages[1:]:
        combined = node("add", [combined, z])
    next_reach = mul(node("min2", [combined, one]), all_floors)
    next_total = node("add", ["total", next_reach])
    dest = graph
    graph.append({
        "id": "wave", "op": "loop", "T": 128,
        "state": [
            {"id": "reach", "init": initial_reach},
            {"id": "total", "zeros": [8, 64, 64]}],
        "body": body,
        "next": {"reach": next_reach, "total": next_total}})
    sizes["wave.reach"], sizes["wave.total"] = 8, 8
    value = "wave.total"
    value0 = sl(value, 0, 1)
    offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def neighbour_queries(fields):
        w = np.zeros((4 * len(fields), len(fields), 5, 5), np.float32)
        for f in range(len(fields)):
            for a, (dy, dx) in enumerate(offsets):
                w[4 * f + a, f, 2 + dy, 2 + dx] = 1.0
        return query(conv(cat(fields), w), player)

    normal_q = neighbour_queries([value0, floors[0]])
    neighbour_value = sl(normal_q, 0, 4, True)
    neighbour_valid = sl(normal_q, 4, 8, True)
    movement_scores = vsum([
        neighbour_value, neighbour_valid, direction_failures
    ], [1.0, 1000.0, -32.0], -1000.0, True)
    movement_scores = node("vadd", [
        movement_scores, constant([-0.03, -0.02, -0.01, 0.0])])
    preferred_move = winner(movement_scores)
    actual_q = neighbour_queries([actual_floor, stable_floor])
    actual_valid = sl(actual_q, 0, 4, True)
    neighbour_stable = sl(actual_q, 4, 8, True)
    destination_parts = []
    for a, (dy, dx) in enumerate(offsets):
        destination_parts.append(mul(
            shift(player, dx, dy), sl(preferred_move, a, a + 1, True)))
    desired_tile = clip_s(smix(destination_parts))
    desired_position = position(desired_tile)
    desired_valid = vsum([vmul(preferred_move, neighbour_valid)])
    desired_actual = vsum([vmul(preferred_move, actual_valid)])
    missing_edge = vmul(desired_valid, not_v(desired_actual))
    needs_e = vmul(missing_edge, query(full_tile(e_potential), desired_tile))
    needs_f = vmul(
        vmul(missing_edge, not_v(needs_e)),
        query(full_tile(f_potential), desired_tile))
    needs_object = vmul(
        vmul(vmul(missing_edge, not_v(needs_e)), not_v(needs_f)),
        query(full_tile(object_potential), desired_tile))

    f_six = query(node("rowmax", [mul(thin12, f_rows)]), player)
    wants_left = sl(preferred_move, 2, 3, True)
    f_unsafe = vmul(
        vmul(needs_f, not_v(player_stable)),
        clip_v(vsum([wants_left, f_six])))
    object_unsafe = vmul(needs_object, not_v(player_stable))
    escape = clip_v(vsum([f_unsafe, object_unsafe]))
    escape_adjustment = vsum(
        [actual_valid, neighbour_stable], [1000.0, 30.0], -800.0, True)
    movement_scores = node("vadd", [
        movement_scores, vmul(escape_adjustment, escape)])
    needs_f = vmul(needs_f, not_v(escape))
    needs_object = vmul(needs_object, not_v(escape))

    # Commands for obtaining and positioning the carried object.
    crusher_corner_w = np.zeros((1, 1, 3, 3), np.float32)
    crusher_corner_w[0, 0, 1:3, 1:3] = 1.0
    for dy, dx in [(-1, 0), (0, -1), (1, -1)]:
        crusher_corner_w[0, 0, 1 + dy, 1 + dx] = -2.0
    crusher_corner = node("relu", [
        conv(crusher8, crusher_corner_w, [-3.0])])
    crusher_pos = node("vadd", [
        position(crusher_corner), constant([1.0 / 63.0, 9.0 / 63.0])])
    unattached = query(shift(crusher8, -3, 3), crusher_corner)
    track_left = vs(track_min_x, 1.0, 8.0 / 63.0)
    track_top = vs(coordinate_extreme(track, row, False), 1.0, 2.0 / 63.0)
    track_bottom = vs(coordinate_extreme(track, row), 1.0, -1.0 / 63.0)
    object_x = vs(coordinate_extreme(object12, col), 1.0, -2.0 / 63.0)
    object_y = vs(coordinate_extreme(object12, row), 1.0, -5.0 / 63.0)
    wanted_y = sl(desired_position, 0, 1, True)
    wanted_x = sl(desired_position, 1, 2, True)
    relative_x = vsum([wanted_x, track_left], [1.0, -1.0])
    x_steps = [
        positive_bit(relative_x, 126.0, -2.0 * (t - 0.5))
        for t in [3.0, 7.0, 11.0]]
    delivery_x = vsum([track_left] + x_steps,
                     [1.0] + [4.0 / 63.0] * 3)
    bottom_choice = positive_bit(
        vsum([wanted_y, track_top], [1.0, -1.0]), 126.0, -11.0)
    delivery_y = vsum([
        vmul(track_top, not_v(bottom_choice)),
        vmul(track_bottom, bottom_choice)])
    target_x = vsum([
        vmul(object_x, unattached),
        vmul(delivery_x, not_v(unattached))])
    target_y = vsum([
        vmul(object_y, unattached),
        vmul(delivery_y, not_v(unattached))])
    crusher_y = sl(crusher_pos, 0, 1, True)
    crusher_x = sl(crusher_pos, 1, 2, True)
    dx = vsum([target_x, crusher_x], [1.0, -1.0])
    dy = vsum([target_y, crusher_y], [1.0, -1.0])
    go_right = positive_bit(dx, 126.0, -1.0)
    go_left = positive_bit(vs(dx, -1.0), 126.0, -1.0)
    go_down = positive_bit(dy, 126.0, -1.0)
    go_up = positive_bit(vs(dy, -1.0), 126.0, -1.0)
    same_y = not_v(clip_v(vsum([go_up, go_down])))
    same_x = not_v(clip_v(vsum([go_left, go_right])))
    at_left = node("vrelu", [vs(
        vector_abs(vsum([crusher_x, track_left], [1.0, -1.0])),
        -126.0, 1.0)])
    cmd_left = clip_v(vsum([
        vmul(not_v(at_left), not_v(same_y)),
        vmul(same_y, go_left)]))
    cmd_right = vmul(same_y, go_right)
    cmd_up, cmd_down = vmul(at_left, go_up), vmul(at_left, go_down)
    cmd_grab = vmul(vmul(unattached, same_x), same_y)

    state_values = query(value, player)
    state_validity = query(all_floors, player)
    portal_value = query(value0, portals)
    on_portal = query(portals, player)

    def switch_score(mode, present):
        val = sl(state_values, mode, mode + 1, True)
        valid = vmul(sl(state_validity, mode, mode + 1, True), present)
        return vsum([val, valid], [1.0, 1000.0], -1000.1)

    basic_scores = [
        switch_score(1, button_presence[0]),
        switch_score(2, button_presence[1]),
        vsum([portal_value, vmul(on_portal, button_presence[2])],
             [1.0, 1000.0], -1000.1),
        switch_score(4, button_presence[3])
    ] + [negative] * 7
    click_scores = vsum(
        [cat(basic_scores, True), kind_failures],
        [1.0, -32.0], vector_output=True)
    force = cat([
        zero, zero, zero, zero, needs_e, needs_f,
        vmul(needs_object, cmd_up), vmul(needs_object, cmd_down),
        vmul(needs_object, cmd_left), vmul(needs_object, cmd_right),
        vmul(needs_object, cmd_grab)], True)
    click_scores = node("vadd", [click_scores, vs(force, 1500.0)])
    click_scores = node("vadd", [
        click_scores, constant([-0.011 * (11 - i) for i in range(11)])])
    selected_kind = winner(click_scores)
    button_stack = cat(button_maps)
    click_fields = node("add", [
        mul(button_stack, click_scores),
        ss(not_s(button_stack), -2000.0)])
    best_click = sl(click_fields, 0, 1)
    for i in range(1, 11):
        best_click = node("max2", [best_click, sl(click_fields, i, i + 1)])
    click = ss(best_click, name="click")
    click_logit = node("gmax", [click])
    node("vconcat", [
        negative, movement_scores, negative, click_logit, negative],
        name="logits")
    node("vconcat", [
        fingerprint, failures, selected_kind, next_portals,
        pickup_position, pickup_valid], name="mem_next")

    return {"graph": graph, "memory_size": 95, "params": params}
