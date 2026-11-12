import numpy as np


def build():
    graph, params = [], {}
    parameter_cache, node_caches = {}, {}
    current = [graph]
    serial = [0]

    NC, NP, R = 5, 5, 15
    N = NC + NP
    GS = N * R
    LENGTHS = list(range(2, 9))
    NL = len(LENGTHS)
    NT = NL + 2
    CAT = NP * 4

    ACTIVE_START = GS
    CODE_START = ACTIVE_START + R
    INDEX_START = CODE_START + NT
    CATALOG_START = INDEX_START + NP
    PHASE_START = CATALOG_START + CAT
    M = PHASE_START + 1

    shapes = {
        "x0": ("s", 16), "coords": ("s", 3),
        "avail": ("v", 8), "aprev": ("v", 8),
        "mem": ("v", M),
    }

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(value):
        value = np.asarray(value, np.float32)
        key = (value.shape, value.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(params))
            parameter_cache[key] = name
            params[name] = value
        return parameter_cache[key]

    def node(op, ins, channels, spatial=True, name=None, **fields):
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        local_cache = node_caches.setdefault(id(current[0]), {})
        global_cache = node_caches.setdefault(id(graph), {})
        if name is None:
            if key in local_cache:
                return local_cache[key]
            if current[0] is not graph and key in global_cache:
                return global_cache[key]
        name = name or fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current[0].append(nd)
        shapes[name] = ("s" if spatial else "v", channels)
        if name not in ("logits", "click", "mem_next"):
            local_cache[key] = name
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        if (
            name is None and w.shape[-2:] == (1, 1)
            and w.shape[0] == w.shape[1]
            and np.array_equal(
                w[:, :, 0, 0], np.eye(w.shape[0], dtype=np.float32)
            )
            and (b is None or not np.any(np.asarray(b)))
        ):
            return x
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], w.shape[0], name=name, **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if (
            name is None and w.shape[0] == w.shape[1]
            and np.array_equal(w, np.eye(w.shape[0], dtype=np.float32))
            and (b is None or not np.any(np.asarray(b)))
        ):
            return x
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], w.shape[0], False,
                    name=name, **fields)

    def cat(xs, name=None):
        return node("concat", xs, sum(shapes[x][1] for x in xs),
                    name=name)

    def vcat(xs, name=None):
        return node("vconcat", xs, sum(shapes[x][1] for x in xs),
                    False, name=name)

    def sl(x, a, b):
        if a == 0 and b == shapes[x][1]:
            return x
        return node("slice", [x], b - a, start=a, stop=b)

    def vs(x, a, b):
        if a == 0 and b == shapes[x][1]:
            return x
        return node("vslice", [x], b - a, False, start=a, stop=b)

    def unary(op, x):
        return node(op, [x], shapes[x][1], shapes[x][0] == "s")

    def binary(op, a, b):
        spatial = shapes[a][0] == "s" or shapes[b][0] == "s"
        return node(op, [a, b], max(shapes[a][1], shapes[b][1]),
                    spatial)

    def add(a, b):
        return binary("add", a, b)

    def sub(a, b):
        return binary("sub", a, b)

    def mul(a, b):
        return binary("mul", a, b)

    def va(a, b):
        return binary("vadd", a, b)

    def vm(a, b):
        return binary("vmul", a, b)

    def pool(x, op="gmax"):
        return node(op, [x], shapes[x][1], False)

    def const(value):
        value = np.atleast_1d(np.asarray(value, np.float32))
        return dense("avail", np.zeros((len(value), 8), np.float32),
                     value)

    def vl(x, scale=1.0, bias=0.0):
        n = shapes[x][1]
        return dense(x, np.eye(n, dtype=np.float32) * scale,
                     np.full(n, bias, np.float32))

    def spl(x, scale=1.0, bias=0.0):
        n = shapes[x][1]
        return conv(x, np.eye(n, dtype=np.float32) * scale,
                    np.full(n, bias, np.float32))

    def vclip(x):
        return vl(unary("vrelu", vl(x, -1, 1)), -1, 1)

    def vstep(x, sharpness=4.0):
        return unary(
            "vrelu",
            vl(unary("vrelu", vl(x, -sharpness)), -1, 1)
        )

    def vabs(x):
        return va(unary("vrelu", x), unary("vrelu", vl(x, -1)))

    one, zero = const([1]), const([0])
    zeros_r = const(np.zeros(R, np.float32))
    zeros_types = const(np.zeros(NT, np.float32))
    zeros_index = const(np.zeros(NP, np.float32))

    def clip(x):
        return binary("min2", unary("relu", x), one)

    def sumsp(x):
        return conv(x, np.ones((1, shapes[x][1]), np.float32))

    def sumv(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def mix(a, b, gate):
        return va(vm(a, gate), vm(b, vl(gate, -1, 1)))

    def smix(a, b, gate):
        return add(mul(a, gate), mul(b, vl(gate, -1, 1)))

    def vmin(a, b):
        positive = unary("vrelu", dense(vcat([a, b]), [[1, -1]]))
        return dense(vcat([a, positive]), [[1, -1]])

    def raw_shift(x, dr, dc):
        k = 2 * max(abs(dr), abs(dc)) + 1
        w = np.zeros((1, 1, k, k), np.float32)
        w[0, 0, k // 2 + dr, k // 2 + dc] = 1
        return conv(x, w)

    yellow = sl("x0", 4, 5)
    purple = sl("x0", 6, 7)
    selected = sl("x0", 9, 10)
    cup_color = sl("x0", 11, 12)
    corner_color = sl("x0", 15, 16)
    flat_color = sl("x0", 8, 9)
    background = sl("x0", 12, 13)

    # Rotation and camera scale.
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    w = np.zeros((4, 16, 3, 3), np.float32)
    for k, (dr, dc) in enumerate(directions):
        w[k, 4, 1, 1] = 1
        w[k, 6, 1 + dr, 1 + dc] = 1
    gravity = pool(unary("relu", conv("x0", w, [-1] * 4)))

    source_tl = unary("relu", conv(
        cat([purple, raw_shift(purple, -1, 0),
             raw_shift(purple, 0, -1)]),
        [[1, -1, -1]]
    ))
    right = {i: raw_shift(purple, 0, i) for i in (1, 2, 3)}
    right[4] = raw_shift(right[1], 0, 3)
    scale_flags = []
    for s in (2, 3, 4):
        z = conv(cat([source_tl, right[s - 1], right[s]]),
                 [[1, 1, -1]], [-1])
        scale_flags.append(pool(unary("relu", z)))
    sf = vcat(scale_flags)
    fallback = unary("vrelu", vl(sumv(sf), -1, 1))
    sf = va(sf, dense(fallback, [[0], [0], [1]]))
    scale = dense(sf, [[2, 3, 4]])
    reciprocal_scale = dense(sf, [[0.5, 1.0 / 3.0, 0.25]])

    uw = np.array([
        [0, 63, 0], [-63, 0, 63],
        [0, -63, 63], [63, 0, 0]
    ], np.float32)
    vw = np.array([
        [63, 0, 0], [0, -63, 63],
        [-63, 0, 63], [0, 63, 0]
    ], np.float32)
    u = sumsp(mul(conv("coords", uw), gravity))
    v = sumsp(mul(conv("coords", vw), gravity))
    uv = cat([u, v])

    gates = vcat([
        vm(gravity, vs(sf, j, j + 1)) for j in range(3)
    ])
    gate23, gate4 = vs(gates, 0, 8), vs(gates, 8, 12)

    def cell_shift(x, du, dv):
        c = shapes[x][1]
        U = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        offsets = [
            (du * U[k][0] + dv * directions[k][0],
             du * U[k][1] + dv * directions[k][1])
            for k in range(4)
        ]
        w23 = np.zeros((8 * c, c, 7, 7), np.float32)
        for j, s in enumerate((2, 3)):
            for k, (dr, dc) in enumerate(offsets):
                for ch in range(c):
                    w23[(4 * j + k) * c + ch, ch,
                        3 + s * dr, 3 + s * dc] = 1
        a = conv(x, w23)
        first_two = sl(a, 0, 4 * c)
        w4 = np.zeros((4 * c, 4 * c, 5, 5), np.float32)
        for k, (dr, dc) in enumerate(offsets):
            for ch in range(c):
                w4[k * c + ch, k * c + ch,
                   2 + 2 * dr, 2 + 2 * dc] = 1
        b = conv(first_two, w4)

        rg23 = np.zeros((8 * c, 8), np.float32)
        rg4 = np.zeros((4 * c, 4), np.float32)
        out23 = np.zeros((c, 8 * c), np.float32)
        out4 = np.zeros((c, 4 * c), np.float32)
        for k in range(8):
            for ch in range(c):
                rg23[k * c + ch, k] = 1
                out23[ch, k * c + ch] = 1
        for k in range(4):
            for ch in range(c):
                rg4[k * c + ch, k] = 1
                out4[ch, k * c + ch] = 1
        return add(
            conv(mul(a, dense(gate23, rg23)), out23),
            conv(mul(b, dense(gate4, rg4)), out4)
        )

    def column_projection(x):
        cp = node("colmax", [x], 1)
        rp = node("rowmax", [x], 1)
        return add(
            mul(cp, dense(gravity, [[1, 0, 1, 0]])),
            mul(rp, dense(gravity, [[0, 1, 0, 1]]))
        )

    source_v = pool(conv(cat([v, yellow]), [[1, 128]], [-128]))
    origin = dense(vcat([source_v, scale]), [[1, -1]], [1])
    max_background_u = pool(conv(
        cat([u, background]), [[1, 128]], [-128]
    ))
    max_first = dense(vcat([max_background_u, scale]),
                      [[1, -1]], [1])

    # Logical-cell lattice.
    normalized_uv = mul(sub(uv, origin), reciprocal_scale)
    distance_init = spl(uv, 0, 64)
    lattice_index, lattice_distance = fresh("st"), fresh("st")
    shapes[lattice_index] = ("v", 1)
    shapes[lattice_distance] = ("s", 2)
    body = []
    current[0] = body
    candidate_distance = unary(
        "abs", sub(normalized_uv, lattice_index)
    )
    next_distance = binary(
        "min2", lattice_distance, candidate_distance
    )
    next_index = vl(lattice_index, 1, 1)
    current[0] = graph
    lattice_loop = fresh("lattice")
    graph.append({
        "id": lattice_loop, "op": "loop", "T": 32,
        "state": [
            {"id": lattice_index, "init": zero},
            {"id": lattice_distance, "init": distance_init},
        ],
        "body": body,
        "next": {
            lattice_index: next_index,
            lattice_distance: next_distance,
        },
    })
    lattice_result = lattice_loop + "." + lattice_distance
    shapes[lattice_result] = ("s", 2)
    lattice_peaks = clip(spl(lattice_result, -100, 2))
    grid = binary("min2", sl(lattice_peaks, 0, 1),
                  sl(lattice_peaks, 1, 2))

    def point(mask, score=None, sharpness=50000.0):
        if score is None:
            score = conv(cat([mask, uv]),
                         [[4, -1 / 128, -1 / 16384]])
        maximum = pool(score)
        p = clip(spl(sub(score, maximum), sharpness, 1))
        return mul(p, mask), maximum

    def point_coords(p):
        return pool(mul(p, uv))

    def cell_box(xy):
        w = np.array([
            [-4, 0], [4, 0], [0, -4], [0, 4]
        ], np.float32)
        b = np.array([
            [4, 0, 0], [-4, 0, -4],
            [0, 4, 0], [0, -4, -4]
        ], np.float32)
        violations = unary("relu", add(
            conv(uv, w),
            dense(vcat([xy, scale]), b,
                  [-0.5, 3.5, -0.5, 3.5])
        ))
        return unary("relu", conv(
            violations, -np.ones((1, 4), np.float32), [1]
        ))

    # Cup openings and cup extraction.
    bu, bm = cell_shift(cup_color, 1, 0), cell_shift(cup_color, -1, 0)
    bv, bt = cell_shift(cup_color, 0, 1), cell_shift(cup_color, 0, -1)
    openings = []
    for a, b, c in [
        (bv, bu, bm), (bu, bv, bt),
        (bt, bu, bm), (bm, bv, bt)
    ]:
        openings.append(unary("relu", conv(
            cat([a, b, c, cup_color]), [[1, 1, 1, -1]], [-2]
        )))
    opening_bank = cat(openings)
    opening_mask = sumsp(opening_bank)

    cup_remaining, cup_records, cup_slot = (
        fresh("st"), fresh("st"), fresh("st")
    )
    shapes[cup_remaining] = ("s", 1)
    shapes[cup_records] = ("v", NC * R)
    shapes[cup_slot] = ("v", NC)
    first_slot = const([1, 0, 0, 0, 0])
    body = []
    current[0] = body
    p, _ = point(cup_remaining)
    xy = point_coords(p)
    validity = pool(cup_remaining)
    direction = pool(mul(opening_bank, p))
    record = vcat([
        xy, zero, direction, vm(cup_slot, validity),
        validity, zero, zero
    ])
    updated_records = vcat([vs(cup_records, R, NC * R), record])
    updated_remaining = unary(
        "relu", sub(cup_remaining, mul(cell_box(xy), validity))
    )
    rotate_slot = np.zeros((NC, NC), np.float32)
    for j in range(NC):
        rotate_slot[(j + 1) % NC, j] = 1
    updated_slot = dense(cup_slot, rotate_slot)
    current[0] = graph
    cup_loop = fresh("cups")
    graph.append({
        "id": cup_loop, "op": "loop", "T": NC,
        "state": [
            {"id": cup_remaining, "init": opening_mask},
            {"id": cup_records, "zeros": [NC * R]},
            {"id": cup_slot, "init": first_slot},
        ],
        "body": body,
        "next": {
            cup_remaining: updated_remaining,
            cup_records: updated_records,
            cup_slot: updated_slot,
        },
    })
    extracted_cups = cup_loop + "." + cup_records
    shapes[extracted_cups] = ("v", NC * R)
    initial_goals = vcat([extracted_cups] + [zeros_r] * NP)

    old_goals = vs("mem", 0, GS)
    old_active = vs("mem", ACTIVE_START, CODE_START)
    old_code = vs("mem", CODE_START, INDEX_START)
    old_index = vs("mem", INDEX_START, CATALOG_START)
    old_catalog = vs("mem", CATALOG_START, PHASE_START)
    old_phase = vs("mem", PHASE_START, M)

    initialized_w = np.zeros((1, GS), np.float32)
    for j in range(NC):
        initialized_w[0, j * R + 12] = 1
    no_memory = unary(
        "vrelu", vl(dense(old_goals, initialized_w), -1, 1)
    )
    initialize = vclip(va(no_memory, vs("aprev", 5, 6)))
    goals = mix(initial_goals, old_goals, initialize)
    phase = vm(old_phase, vl(initialize, -1, 1))

    bit_w = np.zeros((N * NC, GS), np.float32)
    for j in range(N):
        for k in range(NC):
            bit_w[j * NC + k, j * R + 7 + k] = 1
    goal_bits = dense(goals, bit_w)
    priority = list(range(NC, N)) + list(range(NC))
    prior_w = np.zeros((N, N), np.float32)
    for i, gi in enumerate(priority):
        for gj in priority[:i]:
            prior_w[gi, gj] = 1

    # Analytic directional ray tests.
    def ray_base(direction):
        w = np.zeros((4 * N, 2), np.float32)
        b = np.zeros((4 * N, GS), np.float32)
        sw = np.zeros((4 * N, 1), np.float32)
        bias = np.zeros(4 * N, np.float32)
        for j in range(N):
            q, r = j * R, j * 4
            if direction == 0:
                w[r], w[r + 1], w[r + 2] = [-4, 0], [4, 0], [0, 4]
                b[r, q] = 4
                b[r + 1, q] = -4
                b[r + 1, q + 2] = -4
                b[r + 2, q + 1] = -4
                sw[r + 2, 0] = 4
                di = 3
            elif direction == 1:
                w[r], w[r + 1], w[r + 2] = [0, -4], [0, 4], [4, 0]
                b[r, q + 1] = 4
                b[r + 1, q + 1] = -4
                b[r + 1, q + 2] = -4
                b[r + 2, q] = -4
                sw[r + 2, 0] = 4
                di = 4
            else:
                w[r], w[r + 1], w[r + 2] = [0, -4], [0, 4], [-4, 0]
                b[r, q + 1] = 4
                b[r + 1, q + 1] = -4
                b[r + 1, q + 2] = -4
                b[r + 2, q] = 4
                sw[r + 2, 0] = 4
                di = 6
            bias[r:r + 3] = -0.5
            bias[r + 3] = 1
            b[r + 3, q + di] = -1
        return add(
            conv(uv, w),
            va(dense(goals, b, bias), dense(scale, sw))
        )

    ray_bases = [ray_base(k) for k in range(3)]

    def ray(direction, du, dv):
        ow = np.zeros((4 * N, 2), np.float32)
        for j in range(N):
            r = 4 * j
            if direction == 0:
                ow[r:r + 3] = [[-4, 0], [4, 0], [0, 4]]
            elif direction == 1:
                ow[r:r + 3] = [[0, -4], [0, 4], [4, 0]]
            else:
                ow[r:r + 3] = [[0, -4], [0, 4], [-4, 0]]
        z = unary("relu", add(
            ray_bases[direction], dense(vcat([du, dv]), ow)
        ))
        collapse = np.zeros((N, 4 * N), np.float32)
        for j in range(N):
            collapse[j, 4 * j:4 * j + 4] = -1
        hits = unary("relu", conv(z, collapse, np.ones(N)))
        return unary("relu", sub(hits, conv(hits, prior_w)))

    source_hits = ray(0, zero, zero)
    source_good = sumsp(source_hits)
    source_bad = mul(yellow, unary("relu", sub(one, source_good)))
    source_hit_vector = pool(mul(source_hits, yellow))

    repeat_hits = np.zeros((N * NC, N), np.float32)
    bit_sum = np.zeros((NC, N * NC), np.float32)
    weight_sum = np.zeros((N, N * NC), np.float32)
    repeat_missing = np.zeros((N * NC, NC), np.float32)
    for j in range(N):
        for k in range(NC):
            repeat_hits[j * NC + k, j] = 1
            bit_sum[k, j * NC + k] = 1
            weight_sum[j, j * NC + k] = 1
            repeat_missing[j * NC + k, k] = 1

    fed_bits = vclip(dense(
        vm(dense(source_hit_vector, repeat_hits), goal_bits), bit_sum
    ))
    missing_bits = vl(fed_bits, -1, 1)
    goal_weight = dense(
        vm(goal_bits, dense(missing_bits, repeat_missing)), weight_sum
    )
    all_columns = column_projection(yellow)
    bad_columns = column_projection(source_bad)

    # Separate selected-colour detectors prevent merging touching bars.
    snu, spu = cell_shift(selected, -1, 0), cell_shift(selected, 1, 0)
    snv, spv = cell_shift(selected, 0, -1), cell_shift(selected, 0, 1)
    selected_corner = pool(clip(add(
        binary("min2", selected, snv),
        binary("min2", selected, spv)
    )))
    selected_flat = mul(selected, vl(selected_corner, -1, 1))
    corners = add(corner_color, mul(selected, selected_corner))
    first8 = unary("relu", sub(flat_color, cell_shift(flat_color, -1, 0)))
    first9 = unary("relu", sub(selected_flat, cell_shift(selected_flat, -1, 0)))

    shifted = [cat([flat_color, selected_flat, all_columns, bad_columns])]
    for _ in range(8):
        shifted.append(cell_shift(shifted[-1], 1, 0))
    shift_bank = cat(shifted)
    detector_input = cat([first8, first9, shift_bank])
    detector_w = np.zeros((2 * NL, 38), np.float32)
    cover_w = np.zeros((2 * NL, 36), np.float32)
    for j, length in enumerate(LENGTHS):
        for colour in range(2):
            row = colour * NL + j
            detector_w[row, colour] = 1
            detector_w[row, 2 + 4 * (length - 1) + colour] = 1
            detector_w[row, 2 + 4 * length + colour] = -1
        for k in range(length):
            cover_w[j, 4 * k + 2] = 1
            cover_w[NL + j, 4 * k + 3] = 1
    detected_flat = unary("relu", conv(
        detector_input, detector_w, [-1] * (2 * NL)
    ))
    detected8, detected9 = sl(detected_flat, 0, NL), sl(detected_flat, NL, 2 * NL)
    initial_flat = binary("max2", detected8, detected9)
    coverage = clip(conv(shift_bank, cover_w))
    all_cover_bank, bad_cover_bank = sl(coverage, 0, NL), sl(coverage, NL, 2 * NL)

    cm, cp = cell_shift(corners, -1, 0), cell_shift(corners, 1, 0)
    cd = cell_shift(corners, 0, 1)
    cmd, cpd = cell_shift(cm, 0, 1), cell_shift(cp, 0, 1)
    corner_r = unary("relu", conv(
        cat([cm, cd, cmd, corners]), [[1, 1, 1, -1]], [-2]
    ))
    corner_l = unary("relu", conv(
        cat([cp, cd, cpd, corners]), [[1, 1, 1, -1]], [-2]
    ))
    selected_bank = cat([
        detected9, mul(corner_r, snu), mul(corner_l, spu)
    ])
    selected_types = pool(selected_bank)
    selected_point, _ = point(sumsp(selected_bank))
    selected_xy = point_coords(selected_point)
    initial_piece_bank = cat([initial_flat, corner_r, corner_l])

    # Persistent catalogue: initial position, type, and availability.
    object_remaining, object_records = fresh("st"), fresh("st")
    shapes[object_remaining] = ("s", 1)
    shapes[object_records] = ("v", CAT)
    body = []
    current[0] = body
    p, _ = point(object_remaining)
    xy = point_coords(p)
    validity = pool(object_remaining)
    type_vector = pool(mul(initial_piece_bank, p))
    type_number = dense(type_vector, [list(range(1, NT + 1))])
    record = vcat([xy, type_number, validity])
    updated_records = vcat([vs(object_records, 4, CAT), record])
    updated_remaining = unary("relu", sub(
        object_remaining, mul(cell_box(xy), validity)
    ))
    current[0] = graph
    object_loop = fresh("objects")
    graph.append({
        "id": object_loop, "op": "loop", "T": NP,
        "state": [
            {"id": object_remaining, "init": sumsp(initial_piece_bank)},
            {"id": object_records, "zeros": [CAT]},
        ],
        "body": body,
        "next": {
            object_remaining: updated_remaining,
            object_records: updated_records,
        },
    })
    initial_catalog = object_loop + "." + object_records
    shapes[initial_catalog] = ("v", CAT)
    catalog = mix(initial_catalog, old_catalog, initialize)

    def origin_masks(records, width, count, offset=0):
        sw = np.zeros((2 * count, 2), np.float32)
        vw = np.zeros((2 * count, shapes[records][1]), np.float32)
        valid_w = np.zeros((count, shapes[records][1]), np.float32)
        collapse = np.zeros((count, 2 * count), np.float32)
        valid_offset = 3 if width == 4 else 12
        for j in range(count):
            q = offset + j * width
            sw[2 * j, 0] = 1
            sw[2 * j + 1, 1] = 1
            vw[2 * j, q] = -1
            vw[2 * j + 1, q + 1] = -1
            valid_w[j, q + valid_offset] = 1
            collapse[j, 2 * j:2 * j + 2] = -4
        distance = unary("abs", add(conv(uv, sw), dense(records, vw)))
        masks = unary("relu", conv(distance, collapse, np.ones(count)))
        return mul(masks, dense(records, valid_w))

    object_masks = origin_masks(catalog, 4, NP)
    type_w = np.zeros((NP * NT, CAT), np.float32)
    type_b = np.zeros(NP * NT, np.float32)
    valid_w = np.zeros((NP * NT, CAT), np.float32)
    expand_mask = np.zeros((NP * NT, NP), np.float32)
    collapse_type = np.zeros((NT, NP * NT), np.float32)
    for j in range(NP):
        for t in range(NT):
            q = j * NT + t
            type_w[q, j * 4 + 2] = 1
            type_b[q] = -(t + 1)
            valid_w[q, j * 4 + 3] = 1
            expand_mask[q, j] = 1
            collapse_type[t, q] = 1
    type_difference = dense(catalog, type_w, type_b)
    catalog_codes = vm(
        unary("vrelu", vl(vabs(type_difference), -1, 1)),
        dense(catalog, valid_w)
    )
    actual_bank = conv(
        mul(conv(object_masks, expand_mask), catalog_codes),
        collapse_type
    )
    style_counts = dense(catalog_codes, collapse_type)

    installed_masks = origin_masks(goals, R, NP, NC * R)
    selected_frozen = vclip(sumv(pool(mul(installed_masks, selected_point))))
    selected_free = vl(selected_frozen, -1, 1)

    # Cup bounds and safe placement heights.
    cb = np.zeros((4 * NC, GS), np.float32)
    cc = np.zeros((4 * NC, GS), np.float32)
    cv = np.zeros((NC, GS), np.float32)
    cup_u_w = np.zeros((NC, GS), np.float32)
    cup_down_w = np.zeros((NC, GS), np.float32)
    for j in range(NC):
        q = j * R
        cb[4 * j:4 * j + 2, q] = 1
        cb[4 * j + 2:4 * j + 4, q + 1] = 1
        for idx in (3, 5, 6):
            cc[4 * j, q + idx] = -1
        for idx in (3, 4, 5):
            cc[4 * j + 1, q + idx] = 1
        for idx in (4, 5, 6):
            cc[4 * j + 2, q + idx] = -1
        for idx in (3, 4, 6):
            cc[4 * j + 3, q + idx] = 1
        cv[j, q + 12] = 1
        cup_u_w[j, q] = 1
        cup_down_w[j, q + 3] = 1
    cup_bounds = va(dense(goals, cb), vm(dense(goals, cc), scale))
    cup_valid = dense(goals, cv)
    cup_u = dense(goals, cup_u_w)
    missing_down_cups = vm(dense(goals, cup_down_w), missing_bits)

    floor_limit = dense(vcat([origin, scale]), [[-1, -3]], [64])
    limit = floor_limit
    installed_w = np.zeros((NP, GS), np.float32)
    straight_w = np.zeros((1, GS), np.float32)
    for j in range(NP):
        q = (NC + j) * R
        installed_w[j, q + 12] = 1
        straight_w[0, q + 12] = 1
        straight_w[0, q + 13] = -1
        straight_w[0, q + 14] = -1
        wj = np.zeros((1, GS), np.float32)
        wj[0, q + 1] = 1
        wj[0, q + 12] = -1000
        wj[0, q + 13] = 1000
        wj[0, q + 14] = 1000
        limit = vmin(limit, va(dense(goals, wj, [1000]), vl(scale, -2)))
    not_full = vstep(vl(sumv(dense(goals, installed_w)), -1, NP - 0.5))
    first_straight = unary("vrelu", vl(dense(goals, straight_w), -1, 1))

    def placement_valid(lo_u, hi_u, hi_v, height_limit):
        footprint = vcat([lo_u, hi_u, zero, hi_v, scale])
        lw = np.array([[-4, 0], [4, 0], [0, -4], [0, 4]], np.float32)
        inp = vcat([origin, max_first, height_limit, footprint])
        bw = np.zeros((4, 8), np.float32)
        bw[0, [0, 3]] = [4, -4]
        bw[1, [1, 4]] = [-4, 4]
        bw[2, [0, 5, 7]] = [4, -4, 12]
        bw[3, [2, 6]] = [-4, 4]
        loc = unary("relu", add(conv(uv, lw), dense(inp, bw, [-0.5] * 4)))
        loc = unary("relu", conv(loc, -np.ones((1, 4), np.float32), [1]))

        cw = np.zeros((5 * NC, 2), np.float32)
        bw = np.zeros((5 * NC, 4 * NC), np.float32)
        ow = np.zeros((5 * NC, 5), np.float32)
        vw = np.zeros((5 * NC, NC), np.float32)
        bias = np.zeros(5 * NC, np.float32)
        for j in range(NC):
            r, q = 5 * j, 4 * j
            cw[r:r + 4] = [[4, 0], [-4, 0], [0, 4], [0, -4]]
            bw[r, q + 1] = -4
            bw[r + 1, q] = 4
            bw[r + 2, q + 3] = -4
            bw[r + 3, q + 2] = 4
            ow[r, [0, 4]] = [4, -4]
            ow[r + 1, [1, 4]] = [-4, -4]
            ow[r + 2, [2, 4]] = [4, -4]
            ow[r + 3, [3, 4]] = [-4, -4]
            bias[r:r + 4] = -0.5
            bias[r + 4] = 1
            vw[r + 4, j] = -1
        z = unary("relu", add(
            conv(uv, cw),
            va(va(dense(cup_bounds, bw, bias),
                  dense(footprint, ow)), dense(cup_valid, vw))
        ))
        collapse = np.zeros((NC, 5 * NC), np.float32)
        for j in range(NC):
            collapse[j, 5 * j:5 * j + 5] = -1
        forbidden = unary("relu", conv(z, collapse, np.ones(NC)))
        return mul(mul(
            unary("relu", sub(loc, sumsp(forbidden))), grid
        ), not_full)

    # These corner-output rays are invariant throughout the candidate loop.
    horizontal_right = ray(1, scale, zero)
    horizontal_left = ray(2, vl(scale, -1), zero)

    # One candidate loop: seven lengths, right-turn corner, left-turn corner.
    states = {key: fresh("st") for key in
              ("type", "score", "record", "code", "index")}
    for key, width in [
        ("type", 1), ("score", 1), ("record", R),
        ("code", NT), ("index", NP)
    ]:
        shapes[states[key]] = ("v", width)
    type_init, score_init = const([0]), const([-20000])

    body = []
    current[0] = body
    t = states["type"]
    values = np.arange(NT, dtype=np.float32)
    da = dense(t, np.ones((NT, 1), np.float32), -values)
    db = dense(t, -np.ones((NT, 1), np.float32), values)
    code = unary("vrelu", vl(
        va(unary("vrelu", da), unary("vrelu", db)), -1, 1
    ))
    lg = vs(code, 0, NL)
    cr, cl = vs(code, NL, NL + 1), vs(code, NL + 1, NT)
    corner_gate = va(cr, cl)
    flat_gate = vl(corner_gate, -1, 1)
    L = dense(code, [LENGTHS + [0, 0]])
    ls = vm(L, scale)
    span = vm(va(ls, vl(scale, -1)), flat_gate)

    mask = sumsp(mul(actual_bank, code))
    incoming_bad = add(
        sumsp(mul(bad_cover_bank, lg)), mul(bad_columns, corner_gate)
    )
    incoming_all = add(
        sumsp(mul(all_cover_bank, lg)), mul(all_columns, corner_gate)
    )

    left_down = ray(0, vl(scale, -1), vl(scale, -1))
    right_down = ray(0, ls, vl(scale, -1))
    corner_hits = add(mul(horizontal_right, cr), mul(horizontal_left, cl))
    hits_a = add(mul(left_down, flat_gate), corner_hits)
    hits_b = mul(right_down, flat_gate)

    left_drain = clip(add(conv(u, [[-4]], [1]), vl(origin, 4)))
    right_drain = clip(add(
        conv(u, [[4]], [1]),
        dense(vcat([ls, origin]), [[4, 4]], [-256])
    ))
    bar_ends_good = mul(
        clip(add(sumsp(left_down), left_drain)),
        clip(add(sumsp(right_down), right_drain))
    )
    ends_good = add(
        mul(bar_ends_good, flat_gate), clip(sumsp(corner_hits))
    )
    valid = mul(
        placement_valid(
            vm(vl(scale, -1), cr),
            va(span, vm(scale, cl)),
            vm(scale, corner_gate),
            mix(floor_limit, limit, corner_gate)
        ),
        ends_good
    )

    actual_point, _ = point(mask)
    actual_xy = point_coords(actual_point)
    actual_index = pool(mul(object_masks, actual_point))
    valid = mul(valid, pool(mask))
    wa = sumsp(mul(hits_a, goal_weight))
    wb = sumsp(mul(hits_b, goal_weight))
    overlap = sumsp(mul(binary("min2", hits_a, hits_b), goal_weight))
    weight = sub(add(wa, wb), overlap)
    children = add(hits_a, hits_b)
    distance = sumsp(unary("abs", sub(uv, actual_xy)))

    remaining_counts = unary("vrelu", va(
        vs(style_counts, 0, NL), vl(lg, -1)
    ))
    cumulative = np.triu(np.ones((NL, NL), np.float32))
    remaining_presence = vclip(dense(remaining_counts, cumulative))
    remaining_reach = vm(
        dense(remaining_presence, [[3] + [1] * (NL - 1)]), scale
    )
    covered = clip(conv(
        mul(conv(children, repeat_hits), goal_bits), bit_sum
    ))
    left_gap = unary("relu", add(
        conv(u, np.ones((NC, 1), np.float32)), vl(cup_u, -1)
    ))
    right_gap = unary("relu", add(
        conv(u, -np.ones((NC, 1), np.float32)),
        va(cup_u, vl(span, -1))
    ))
    too_far = clip(spl(
        sub(add(left_gap, right_gap), remaining_reach), 4, -0.5
    ))
    penalty = sumsp(mul(
        mul(too_far, sub(one, covered)), missing_down_cups
    ))
    penalty = mul(penalty, vm(first_straight, flat_gate))

    raw = add(
        conv(cat([
            weight, incoming_bad, incoming_all, v, u, distance, penalty
        ]), [[8, 4, 0.1, 0.03, -0.0001, -0.001, -12]], [100]),
        vl(corner_gate, 2)
    )
    score_map = add(mul(raw, valid), spl(sub(one, valid), -10000))
    best_point, best_score = point(valid, score_map, 250000.0)
    xy = point_coords(best_point)
    hit_vector = pool(mul(children, best_point))
    bits = vclip(dense(
        vm(dense(hit_vector, repeat_hits), goal_bits), bit_sum
    ))
    validity = pool(mul(valid, best_point))
    best_record = vcat([
        xy, span, const([1, 0, 0, 0]), bits, validity, cr, cl
    ])
    better = vstep(dense(
        vcat([best_score, states["score"]]), [[1, -1]], [-0.0001]
    ), 10000)
    next_score = mix(best_score, states["score"], better)
    next_record = mix(best_record, states["record"], better)
    next_code = mix(code, states["code"], better)
    next_object_index = mix(actual_index, states["index"], better)
    next_type = vl(t, 1, 1)
    current[0] = graph

    candidate_loop = fresh("candidates")
    graph.append({
        "id": candidate_loop, "op": "loop", "T": NT,
        "state": [
            {"id": states["type"], "init": type_init},
            {"id": states["score"], "init": score_init},
            {"id": states["record"], "init": zeros_r},
            {"id": states["code"], "init": zeros_types},
            {"id": states["index"], "init": zeros_index},
        ],
        "body": body,
        "next": {
            states["type"]: next_type,
            states["score"]: next_score,
            states["record"]: next_record,
            states["code"]: next_code,
            states["index"]: next_object_index,
        },
    })
    outputs = {}
    for key, width in [
        ("record", R), ("code", NT), ("index", NP)
    ]:
        name = candidate_loop + "." + states[key]
        shapes[name] = ("v", width)
        outputs[key] = name

    active = mix(old_active, outputs["record"], phase)
    active_code = mix(old_code, outputs["code"], phase)
    active_index = mix(old_index, outputs["index"], phase)
    active_valid = vs(active, 12, 13)
    target_xy = vs(active, 0, 2)

    catalog_xy_w = np.zeros((2 * NP, CAT), np.float32)
    repeat_index = np.zeros((2 * NP, NP), np.float32)
    reduce_xy = np.zeros((2, 2 * NP), np.float32)
    for j in range(NP):
        catalog_xy_w[2 * j, 4 * j] = 1
        catalog_xy_w[2 * j + 1, 4 * j + 1] = 1
        repeat_index[2 * j:2 * j + 2, j] = 1
        reduce_xy[0, 2 * j] = 1
        reduce_xy[1, 2 * j + 1] = 1
    original_xy = dense(vm(
        dense(catalog, catalog_xy_w), dense(active_index, repeat_index)
    ), reduce_xy)
    origin_error = dense(vcat([selected_xy, original_xy]),
                         [[1, 0, -1, 0], [0, 1, 0, -1]])
    same_origin = vstep(vl(sumv(vabs(origin_error)), -1, 0.25))
    identity_gate = mix(one, same_origin, phase)
    match = vm(
        vm(sumv(vm(selected_types, active_code)), selected_free),
        identity_gate
    )

    error = dense(vcat([target_xy, selected_xy]),
                  [[1, 0, -1, 0], [0, 1, 0, -1]])
    eu, ev = vs(error, 0, 1), vs(error, 1, 2)
    vneg = vstep(vl(ev, -1, -0.5))
    horizontal_allowed = vl(vneg, -1, 1)
    upos = vm(vstep(vl(eu, 1, -0.5)), horizontal_allowed)
    uneg = vm(vstep(vl(eu, -1, -0.5)), horizontal_allowed)
    horizontal = va(upos, uneg)
    vpos = vm(vstep(vl(ev, 1, -0.5)), vl(horizontal, -1, 1))
    moving = va(horizontal, va(vpos, vneg))
    commit = vm(vm(active_valid, match), vl(moving, -1, 1))

    mappings = [
        [4, 1, 3, 2], [3, 2, 4, 1],
        [2, 3, 1, 4], [1, 4, 2, 3],
    ]
    action_terms = []
    for flag, mapping in zip([upos, uneg, vpos, vneg], mappings):
        w = np.zeros((8, 4), np.float32)
        for k, action in enumerate(mapping):
            w[action, k] = 1
        action_terms.append(vm(dense(gravity, w), flag))
    movement_actions = action_terms[0]
    for term in action_terms[1:]:
        movement_actions = va(movement_actions, term)
    movement_actions = vm(movement_actions, vm(active_valid, match))

    click_flag = vm(
        active_valid, unary("vrelu", vl(vm(match, moving), -1, 1))
    )
    test_flag = unary("vrelu", vl(active_valid, -1, 1))
    click_action, test_action = np.zeros((8, 1), np.float32), np.zeros((8, 1), np.float32)
    click_action[6, 0], test_action[5, 0] = 1, 1
    action_vector = va(
        movement_actions,
        va(dense(click_flag, click_action), dense(test_flag, test_action))
    )
    dense(action_vector, np.eye(8, dtype=np.float32) * 10,
          [-20, 0, 0, 0, 0, 0, 0, -20], name="logits")

    object_click_origin = sumsp(mul(object_masks, active_index))
    cr, cl = vs(active_code, NL, NL + 1), vs(active_code, NL + 1, NT)
    object_click = add(
        mul(object_click_origin, vl(va(cr, cl), -1, 1)),
        add(
            mul(cell_shift(object_click_origin, 1, 0), cr),
            mul(cell_shift(object_click_origin, -1, 0), cl)
        )
    )
    click_target = add(
        mul(object_click, vl(commit, -1, 1)), mul(selected, commit)
    )
    conv(cat([click_target, uv]),
         [[1, -0.0000001, -0.00000001]], name="click")

    cups_part, planned_part = vs(goals, 0, NC * R), vs(goals, NC * R, GS)
    shifted_plans = vcat([active, vs(planned_part, 0, (NP - 1) * R)])
    new_goals = vcat([cups_part, mix(shifted_plans, planned_part, commit)])
    remove_w = np.zeros((CAT, NP), np.float32)
    for j in range(NP):
        remove_w[4 * j + 3, j] = -1
    new_catalog = va(catalog, dense(vm(active_index, commit), remove_w))
    next_phase = vm(active_valid, vl(commit, -1, 1))
    vcat([
        new_goals, active, active_code, active_index, new_catalog, next_phase
    ], name="mem_next")

    # Structural compiler: dead-node removal and affine-chain assembly.
    # This does not use observations, game transitions, or fitted weights.
    def prune(nodes, roots):
        needed = set(roots)
        result = []
        for nd in reversed(nodes):
            if nd["op"] != "loop":
                if nd["id"] in needed:
                    result.append(nd)
                    needed.discard(nd["id"])
                    needed.update(nd["in"])
                continue
            final_ids = {
                nd["id"] + "." + s["id"] for s in nd["state"]
            }
            if needed.isdisjoint(final_ids):
                continue
            nd = dict(nd)
            body, external = prune(nd["body"], nd["next"].values())
            nd["body"] = body
            result.append(nd)
            needed.difference_update(final_ids)
            state_ids = {s["id"] for s in nd["state"]}
            needed.update(external - state_ids)
            for state in nd["state"]:
                if "init" in state:
                    needed.add(state["init"])
        return list(reversed(result)), needed

    def assemble(nodes, roots, outer_constants=None):
        constants = dict(outer_constants or {})
        rebuilt = []
        for nd in nodes:
            if nd["op"] == "loop":
                nd = dict(nd)
                nd["body"] = assemble(
                    nd["body"], nd["next"].values(), constants
                )
                rebuilt.append(nd)
                continue
            if nd["op"] == "dense":
                w = params[nd["w"]]
                b = params[nd["b"]] if "b" in nd else np.zeros(w.shape[0], np.float32)
                if not np.any(w):
                    constants[nd["id"]] = b
                elif nd["in"][0] in constants:
                    constants[nd["id"]] = (
                        w @ constants[nd["in"][0]] + b
                    ).astype(np.float32)
            rebuilt.append(nd)
        nodes = rebuilt

        uses = {}
        for root in roots:
            uses[root] = uses.get(root, 0) + 1
        for nd in nodes:
            if nd["op"] == "loop":
                for state in nd["state"]:
                    if "init" in state:
                        x = state["init"]
                        uses[x] = uses.get(x, 0) + 1
                local = {s["id"] for s in nd["state"]}
                local.update(b["id"] for b in nd["body"])
                for b in nd["body"]:
                    for x in b["in"]:
                        if x not in local:
                            uses[x] = uses.get(x, 0) + 1
            else:
                for x in nd["in"]:
                    uses[x] = uses.get(x, 0) + 1

        definitions = {nd["id"]: nd for nd in nodes if nd["op"] != "loop"}
        result = []
        for nd in nodes:
            op = nd["op"]
            eligible = op in ("dense", "vadd")
            if op == "conv":
                eligible = params[nd["w"]].shape[-2:] == (1, 1)
            if not eligible:
                result.append(nd)
                continue

            spatial = shapes[nd["id"]][0] == "s"
            out_n = shapes[nd["id"]][1]
            blocks, expanded = {}, set()
            bias = np.zeros(out_n, np.float32)

            def leaf(x, matrix):
                if x in blocks:
                    blocks[x] = blocks[x] + matrix
                else:
                    blocks[x] = matrix

            def walk(x, matrix, force=False):
                nonlocal bias
                if not spatial and x in constants:
                    bias = bias + matrix @ constants[x]
                    if force or uses.get(x, 0) == 1:
                        expanded.add(x)
                    return
                source = definitions.get(x)
                if source is None or (not force and uses.get(x, 0) != 1):
                    leaf(x, matrix)
                    return
                sop = source["op"]
                valid = False
                if spatial:
                    valid = (
                        sop in ("concat", "slice")
                        or (sop == "conv"
                            and params[source["w"]].shape[-2:] == (1, 1))
                        or (sop in ("add", "sub")
                            and all(shapes[z][0] == "s" for z in source["in"]))
                    )
                else:
                    valid = sop in ("dense", "vconcat", "vslice", "vadd")
                if not valid:
                    leaf(x, matrix)
                    return
                expanded.add(x)
                if sop in ("dense", "conv"):
                    w = params[source["w"]]
                    if spatial:
                        w = w[:, :, 0, 0]
                    if "b" in source:
                        bias = bias + matrix @ params[source["b"]]
                    walk(source["in"][0], matrix @ w)
                elif sop in ("concat", "vconcat"):
                    position = 0
                    for z in source["in"]:
                        width = shapes[z][1]
                        walk(z, matrix[:, position:position + width])
                        position += width
                elif sop in ("slice", "vslice"):
                    z = source["in"][0]
                    w = np.zeros((out_n, shapes[z][1]), np.float32)
                    w[:, source["start"]:source["stop"]] = matrix
                    walk(z, w)
                else:
                    for j, z in enumerate(source["in"]):
                        m = matrix
                        if shapes[z][1] == 1 and matrix.shape[1] != 1:
                            m = matrix.sum(axis=1, keepdims=True)
                        if sop == "sub" and j == 1:
                            m = -m
                        walk(z, m)

            walk(nd["id"], np.eye(out_n, dtype=np.float32), True)
            blocks = {
                x: w.astype(np.float32) for x, w in blocks.items()
                if np.any(w)
            }
            replacement_count = 2 if len(blocks) > 1 else 1
            total_width = sum(shapes[x][1] for x in blocks)
            if (
                len(expanded) <= replacement_count
                or total_width > (512 if spatial else 4096)
                or out_n * max(1, total_width) > 30000
            ):
                result.append(nd)
                continue

            if not blocks:
                if spatial:
                    result.append(nd)
                    continue
                new_nd = {
                    "id": nd["id"], "op": "dense", "in": ["avail"],
                    "w": parameter(np.zeros((out_n, 8), np.float32)),
                    "b": parameter(bias),
                }
                constants[nd["id"]] = bias
            else:
                inputs = list(blocks)
                if len(inputs) == 1:
                    input_id = inputs[0]
                    w = blocks[input_id]
                else:
                    input_id = fresh("assembled")
                    join = {
                        "id": input_id,
                        "op": "concat" if spatial else "vconcat",
                        "in": inputs,
                    }
                    result.append(join)
                    shapes[input_id] = ("s" if spatial else "v", total_width)
                    definitions[input_id] = join
                    w = np.concatenate([blocks[x] for x in inputs], axis=1)
                if spatial:
                    w = w[:, :, None, None]
                new_nd = {
                    "id": nd["id"], "op": "conv" if spatial else "dense",
                    "in": [input_id], "w": parameter(w),
                }
                if np.any(bias):
                    new_nd["b"] = parameter(bias)
            definitions[nd["id"]] = new_nd
            result.append(new_nd)
        return prune(result, roots)[0]

    roots = ["logits", "click", "mem_next"]
    graph = prune(graph, roots)[0]
    for _ in range(5):
        graph = assemble(graph, roots)

    used_parameters = set()

    def collect(nodes):
        for nd in nodes:
            for field in ("w", "b"):
                if field in nd:
                    used_parameters.add(nd[field])
            if nd["op"] == "loop":
                collect(nd["body"])

    collect(graph)
    params = {
        name: value for name, value in params.items()
        if name in used_parameters
    }
    return {"graph": graph, "memory_size": M, "params": params}
