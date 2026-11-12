import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    def pixel(o, colour, dy=0, dx=0, weight=1.0):
        p["W1"][o, colour, dy + 2, dx + 2] += weight

    def colour_value(o, dy=0, dx=0, weight=1.0):
        for colour in range(16):
            pixel(o, colour, dy, dx, weight * colour)

    def c2(o, i, weight=1.0, dy=0, dx=0):
        p["W2"][o, i, dy + 1, dx + 1] += weight

    anchors = [
        (0, 1), (0, 8), (1, 9), (8, 9),
        (9, 8), (9, 1), (8, 0), (1, 0),
    ]

    def covered(j, r, c):
        if j == 0:
            return r < 5
        if j == 1:
            return c >= r
        if j == 2:
            return c >= 5
        if j == 3:
            return c >= 9 - r
        if j == 4:
            return r >= 5
        if j == 5:
            return c <= r
        if j == 6:
            return c < 5
        return c <= 9 - r

    footprint = np.array([
        [covered(j, r, c) for r, c in anchors]
        for j in range(8)
    ], dtype=bool)
    representative = [(j + 1) % 8 for j in range(8)]

    # F1 0:8: corner-adjacent colour samples.
    corner_tests = [
        ((-1, 0), (0, -2), (0, -1)),
        ((-1, 0), (0, 2), (0, 1)),
        ((0, 1), (-2, 0), (-1, 0)),
        ((0, 1), (2, 0), (1, 0)),
        ((1, 0), (0, 2), (0, 1)),
        ((1, 0), (0, -2), (0, -1)),
        ((0, -1), (2, 0), (1, 0)),
        ((0, -1), (-2, 0), (-1, 0)),
    ]
    K = 2048.0
    for o, (a, b, inside) in enumerate(corner_tests):
        colour_value(o)
        pixel(o, 5, *a, K)
        pixel(o, 5, *b, K)
        pixel(o, 5, *inside, -K)
        for colour in (2, 4, 5):
            pixel(o, colour, weight=-K)
        p["b1"][o] = 512.0 - 2.0 * K

    # F1 8:12: reference-square edge samples for narrow tabs.
    edge_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for q, (dy, dx) in enumerate(edge_directions):
        o = 8 + q
        colour_value(o)
        pixel(o, 5, dy, dx, K)
        for colour in (2, 4, 5):
            pixel(o, colour, weight=-K)
        p["b1"][o] = 512.0 - K

    # F1 12:20: positive/negative coordinate hinges.
    centres = [
        (17, 7.5), (16, 7.5),
        (17, 31.5), (16, 38.5),
    ]
    for q, (axis, centre) in enumerate(centres):
        a, b = 12 + 2 * q, 13 + 2 * q
        p["W1"][a, axis, 2, 2] = 63.0
        p["b1"][a] = -centre
        p["W1"][b, axis, 2, 2] = -63.0
        p["b1"][b] = centre

    # Local outline templates; these describe tools, not target images.
    horizontal = np.full((9, 14), 5, dtype=np.int32)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[1:8, 1:13] = 15
    horizontal[8, :] = 2

    diagonal_rows = [
        "......2..........",
        ".....222.........",
        "....22f22........",
        "...22fff22.......",
        "..22fffff22......",
        ".22fffffff22.....",
        "22fffffffff22....",
        "..ffffffffff22...",
        "...ffffffffff22..",
        "....ffffffffff22.",
        ".....ffffffffff22",
        "......ffffffff22.",
        ".......ffffff22..",
        "........ffff22...",
        ".........ff22....",
        "..........22.....",
        "..........2......",
    ]
    diagonal = np.array([
        [5 if s == "." else 2 if s == "2" else 15 for s in row]
        for row in diagonal_rows
    ], dtype=np.int32)

    def rotate_point(r, c, h, w, rotation):
        if rotation == 0:
            return r, c
        if rotation == 90:
            return c, h - 1 - r
        if rotation == 180:
            return h - 1 - r, w - 1 - c
        return w - 1 - c, r

    rotations = [180, 0, 270, 90, 0, 180, 90, 270]
    tool_positions = [
        (25, 24), (33, 21), (38, 32), (33, 40),
        (25, 45), (14, 40), (17, 32), (14, 21),
    ]
    marker_points = [
        (7, 12), (2, 6), (7, 1), (2, 6),
        (7, 12), (2, 6), (7, 12), (2, 6),
    ]
    outward = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]
    marker_locations = []

    # F1 20:28: exact local outline matches, with values at most one.
    for j, rotation in enumerate(rotations):
        base = horizontal if j % 2 == 0 else diagonal
        pr, pc = marker_points[j]
        r, c = rotate_point(
            pr, pc, base.shape[0], base.shape[1], rotation
        )
        x, y = tool_positions[j]
        marker_locations.append((y + r, x + c))

        template = np.rot90(base, (-rotation // 90) % 4)
        required = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                rr, cc = r + dy, c + dx
                colour = (
                    int(template[rr, cc])
                    if 0 <= rr < template.shape[0]
                    and 0 <= cc < template.shape[1]
                    else 5
                )
                if colour in (2, 5):
                    pixel(20 + j, colour, dy, dx, 2.0)
                    required += 1
                else:
                    pixel(20 + j, 2, dy, dx, -2.0)
                    pixel(20 + j, 5, dy, dx, -2.0)
        p["b1"][20 + j] = 1.0 - 2.0 * required

    colour_value(28)

    # F1 29: palette detector, at the upper interior row of a swatch.
    colour_value(29)
    border = [(-1, dx) for dx in range(-2, 3)]
    border += [(dy, dx) for dy in (0, 1, 2) for dx in (-2, 2)]
    for dy, dx in border:
        pixel(29, 4, dy, dx, 128.0)
    pixel(29, 4, weight=-128.0)
    p["b1"][29] = 64.0 - 128.0 * len(border)

    # F1 30,31: a clipped, rotation-invariant cap-black-pair detector.
    # The large centre-colour requirement rejects diagonal basket tips.
    pixel(30, 0, weight=256.0)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            pixel(30, 2, dy, dx, 2.0)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            pixel(30, 2, dy, dx, 16.0)
    p["b1"][30] = -337.0
    p["W1"][31] = p["W1"][30]
    p["b1"][31] = p["b1"][30] - 1.0

    # F2 0:16: separate reference and canvas samples, 128 + colour.
    for j in range(8):
        c2(j, j)
        for i in range(12, 16):
            c2(j, i, -48.0)
        c2(8 + j, j)
        for i in range(16, 20):
            c2(8 + j, i, -48.0)

    # F2 16:24: position-disambiguated basket markers.
    # At the intended match the output is 1000; inward matches vanish.
    for j in range(8):
        o = 16 + j
        mr, mc = marker_locations[j]
        sr, sc = outward[j]
        c2(o, 20 + j, 1000000.0)
        c2(o, 18, 512.0 * sr)
        c2(o, 19, -512.0 * sr)
        c2(o, 16, 512.0 * sc)
        c2(o, 17, -512.0 * sc)
        p["b2"][o] = (
            1000.0 - 1000000.0
            - 512.0 * (
                sr * (mr - 38.5) + sc * (mc - 31.5)
            )
        )

    # F2 24:28: the two central edge pixels of each reference tab.
    for q in range(4):
        c2(24 + q, 8 + q)
        col_weight = 128.0 if q < 2 else 48.0
        row_weight = 48.0 if q < 2 else 128.0
        for i in (12, 13):
            c2(24 + q, i, -col_weight)
        for i in (14, 15):
            c2(24 + q, i, -row_weight)
        p["b2"][24 + q] = -104.0

    # F2 28: selected tool colour, encoded as 64 + colour.
    c2(28, 28)
    for i in range(20, 28):
        c2(28, i, 512.0)
    p["b2"][28] = -448.0
    c2(29, 29)

    # F2 30: cap colour. Subtracting F1 31 clips the marker to [0,1].
    # At the black pair, the 3x3 colour sum is 3*fill_colour + 8.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(30, 28, 1.0 / 3.0, dy, dx)
    c2(30, 30, 512.0)
    c2(30, 31, -512.0)
    p["b2"][30] = 64.0 - 512.0 - 8.0 / 3.0
    c2(31, 30)
    c2(31, 31, -1.0)

    # F3 0:8: pooled packing of reference and canvas samples.
    # max/2 reads the reference; sum-max reads the canvas.
    for j in range(8):
        p["W3"][j, j] = 2.0
        p["W3"][j, 8 + j] = 1.0

        a = representative[j]
        p["W3"][8 + j, 64 + a] = 1.0
        p["W3"][8 + j, 32 + 16 + j] = 1.0
        p["b3"][8 + j] = -1000.0

        p["W3"][16 + j, 64 + 8 + a] = 1.0
        p["W3"][16 + j, 32 + 16 + j] = 1.0
        p["b3"][16 + j] = -1000.0

    cardinals = [0, 2, 4, 6]
    tab_order = [0, 3, 1, 2]
    for q, (j, tab) in enumerate(zip(cardinals, tab_order)):
        p["W3"][24 + q, 64 + 24 + tab] = 1.0
        p["W3"][24 + q, 32 + 16 + j] = 1.0
        p["b3"][24 + q] = -1000.0

    p["W3"][28, 29] = 1.0
    p["W3"][28, 30] = 1.0
    p["W3"][29, 28] = 1.0
    p["W3"][29, 30] = 1.0
    p["W3"][30, 31] = 1.0

    # Three tied broadcast fields, with aligned source rows:
    # local reference colour, local canvas colour, narrow-tab colour.
    field_sources = [
        [(i, 1.0) for i in range(8, 16)],
        [(i, 1.0) for i in range(16, 24)],
        [(i, 0.5) for i in range(24, 28)]
        + [(8 + j, 1.0) for j in (1, 3, 5, 7)],
    ]
    for field, sources in enumerate(field_sources):
        u, left, right = 3 * field, 3 * field + 1, 3 * field + 2
        for i, weight in sources:
            p["Uh"][u, i] = weight
        p["Wh"][u, u, 2, 1] = 1.0
        p["Wh"][left, left, 1, 0] = 1.0
        p["Wh"][right, right, 1, 2] = 1.0
        p["Wh"][left, u, 1, 1] = 1.0
        p["Wh"][right, u, 1, 1] = 1.0

    def field_term(o, field, coefficient):
        u = 3 * field
        p["Wh"][o, u, 1, 1] -= coefficient
        p["Wh"][o, u + 1, 1, 1] += coefficient
        p["Wh"][o, u + 2, 1, 1] += coefficient

    field_term(9, 0, 1.0)
    field_term(9, 1, -1.0)
    field_term(10, 0, -1.0)
    field_term(10, 1, 1.0)
    p["Wh"][11, 9, 1, 1] = -2.0
    p["Wh"][11, 10, 1, 1] = -2.0
    p["bh"][11] = 1.0

    field_term(12, 0, 1.0)
    p["Wh"][12, 11, 1, 1] = -1000.0
    field_term(13, 2, 1.0)
    p["Wh"][13, 11, 1, 1] = 1000.0
    p["bh"][13] = -1000.0

    # Click map: negative absolute colour error, cap preferred on ties.
    for i in (12, 13):
        p["Wh"][14, i, 1, 1] = 1.0
        p["Wh"][15, i, 1, 1] = -1.0
    p["Uh"][14, 28] = -1.0
    p["Uh"][15, 28] = 1.0
    p["bh"][14] = -64.0
    p["bh"][15] = 64.0
    p["Wc"][0, 32 + 14] = -100.0
    p["Wc"][0, 32 + 15] = -100.0
    p["Wc"][0, 30] = 1.0

    # Affine-expression bookkeeping for the pooled controller.
    def expression(terms=(), bias=0.0):
        v = np.zeros(96, dtype=np.float32)
        for i, weight in terms:
            v[i] += weight
        return v, float(bias)

    def difference(a, b):
        return a[0] - b[0], a[1] - b[1]

    next_z = 0

    def positive(e):
        nonlocal next_z
        i = next_z
        next_z += 1
        assert i < 128
        p["Wz"][i] = e[0]
        p["bz"][i] = e[1]
        return i

    def absolute(a, b):
        v, bias = difference(a, b)
        return [positive((v, bias)), positive((-v, -bias))]

    def clipped_absolute(a, b):
        v, bias = difference(a, b)
        return [
            (positive((v, bias)), 1.0),
            (positive((v, bias - 1.0)), -1.0),
            (positive((-v, -bias)), 1.0),
            (positive((-v, -bias - 1.0)), -1.0),
        ]

    target = [
        expression([(32 + j, 0.5)], -128.0)
        for j in range(8)
    ]
    canvas = [
        expression([(64 + j, 4096.0), (32 + j, -1.0)], -128.0)
        for j in range(8)
    ]
    local_target = expression([(1, 1.0)], -128.0)
    local_canvas = expression([(4, 1.0)], -128.0)
    click_goal = expression([(12, 1.0), (13, 1.0)], -128.0)
    selected = expression([(32 + 29, 1.0)], -64.0)

    # Only pairs contained in a common half-plane are required.
    pairs, pair_abs = [], []
    pair_index = {}
    for i in range(8):
        for j in range(i + 1, 8):
            if min(j - i, 8 - (j - i)) < 4:
                pair_index[i, j] = len(pairs)
                pairs.append((i, j))
                pair_abs.append(absolute(target[i], target[j]))

    local_abs = [absolute(t, local_target) for t in target]
    canvas_abs = [absolute(t, b) for t, b in zip(target, canvas)]
    base_selected_abs = absolute(local_target, selected)
    goal_selected_abs = absolute(click_goal, selected)
    tab_unique = clipped_absolute(click_goal, local_target)

    direction_z = []
    for j in range(8):
        e = expression([(32 + 8 + j, 1.0 / 128.0)])
        direction_z.append([
            (positive(e), 1.0),
            (positive((e[0], -1.0)), -1.0),
        ])

    tab_nonuniform = []
    for q in range(4):
        tab_nonuniform.append(positive(expression([
            (32 + 24 + q, 2.0),
            (64 + 24 + q, -4096.0),
        ])))

    anchor_wrong = clipped_absolute(local_target, local_canvas)
    base_wrong = clipped_absolute(local_target, selected)
    assert next_z == 116

    # Memory units:
    # 0:24 pair compatibility after reverse peeling
    # 24:32 latched admissible strokes
    # 32:40 peeled sectors
    # 40:48 repairable sectors of the current colour
    # 48:56 current-position stroke eligibility
    # 56:60 narrow-tab readiness
    # 60 previous click-goal agreement
    # 61 completed-cap-click handshake
    S = 20.0

    def add_units(o, units, coefficient):
        for i in units:
            p["Wmz"][o, i] += coefficient

    def add_terms(o, terms, coefficient):
        for i, weight in terms:
            p["Wmz"][o, i] += coefficient * weight

    for k, ((i, j), units) in enumerate(zip(pairs, pair_abs)):
        add_units(k, units, -S)
        p["Wm"][k, 32 + i] = 30.0 * S
        p["Wm"][k, 32 + j] = 30.0 * S
        p["Wma"][k, :] = 60.0 * S
        p["bm"][k] = 0.5 * S

    # Reverse-layer peeling: retain masks whose remaining footprint
    # is uniform, with an unpeeled representative.
    for j in range(8):
        cells = np.flatnonzero(footprint[j]).tolist()
        for a in range(4):
            for b in range(a + 1, 4):
                i, k = sorted((cells[a], cells[b]))
                p["Wm"][24 + j, pair_index[i, k]] = S
        p["Wm"][24 + j, 24 + j] = 100.0 * S
        p["Wm"][24 + j, 32 + representative[j]] = -100.0 * S
        p["bm"][24 + j] = -5.5 * S

    for i in range(8):
        p["Wm"][32 + i, 32 + i] = S
        for j in range(8):
            if footprint[j, i]:
                p["Wm"][32 + i, 24 + j] = S
        p["Wma"][32 + i, :] = 14.0 * S
        p["bm"][32 + i] = -10.0 * S

    # Feedback repair predicates.
    for i in range(8):
        add_units(40 + i, canvas_abs[i], S)
        add_units(40 + i, local_abs[i], -32.0 * S)
        for j in range(8):
            if footprint[j, i]:
                add_terms(40 + i, direction_z[j], 32.0 * S)
        p["bm"][40 + i] = -32.5 * S

    for j in range(8):
        p["Wm"][48 + j, 24 + j] = S
        add_terms(48 + j, direction_z[j], 2.0 * S)
        p["bm"][48 + j] = -2.0 * S

    # Tab strokes are enabled only after angular agreement.
    for q, j in enumerate(cardinals):
        o = 56 + q
        add_terms(o, direction_z[j], 2.0 * S)
        add_terms(o, tab_unique, 2.0 * S)
        for units in canvas_abs:
            add_units(o, units, -32.0 * S)
        p["Wmz"][o, tab_nonuniform[q]] = -32.0 * S
        p["Wm"][o, 61] = -2.0 * S
        p["bm"][o] = -5.0 * S

    add_units(60, goal_selected_abs, -S)
    p["bm"][60] = 0.5 * S

    # Palette clicks and cap clicks share an action but differ in
    # whether the goal colour was already selected on the prior step.
    p["Wm"][61, 61] = 2.0 * S
    p["Wm"][61, 60] = S
    p["Wm"][61, 56:60] = S
    p["Wma"][61, 6] = 2.0 * S
    for action in (0, 1, 2, 3, 4, 5, 7):
        p["Wma"][61, action] = -10.0 * S
    p["bm"][61] = 2.0 * S

    def memory_output(action, i, weight):
        p["Wo"][action, 128 + i] += weight

    clockwise = [4, 2, 2, 3, 3, 1, 1, 4]
    for j, action in enumerate(clockwise):
        for z, weight in direction_z[j]:
            p["Wo"][action, z] += 10.0 * weight

    # Paint an admissible mask only when it repairs at least one sector.
    p["bo"][5] = 6630.0
    for i in range(48, 56):
        memory_output(5, i, 1000.0)
    for i in range(40, 48):
        memory_output(5, i, 100.0)
    for z in base_selected_abs:
        p["Wo"][5, z] -= 10000.0

    # Palette selection for an incorrect anchor, or a ready tab stroke.
    p["bo"][6] = 1625.0
    for z, weight in anchor_wrong:
        p["Wo"][6, z] += 250.0 * weight
    for z, weight in base_wrong:
        p["Wo"][6, z] += 250.0 * weight
    for i in range(56, 60):
        memory_output(6, i, 1000.0)
    memory_output(6, 61, -2000.0)

    p["bo"][0] = -1000000.0
    p["bo"][7] = -1000000.0
    return p
