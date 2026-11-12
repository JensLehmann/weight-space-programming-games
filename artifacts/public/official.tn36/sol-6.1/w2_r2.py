import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    p["bo"][6] = 1.0

    n1 = 0

    def alloc1():
        nonlocal n1
        i = n1
        n1 += 1
        assert n1 <= 32
        return i

    def template(mask, dy0=-2, dx0=-2):
        i = alloc1()
        positives = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                dy, dx = y + dy0, x + dx0
                assert -2 <= dy <= 2 and -2 <= dx <= 2
                if mask[y, x]:
                    p["W1"][i, 11, dy + 2, dx + 2] = 1.0
                    positives += 1
                else:
                    p["W1"][i, 11, dy + 2, dx + 2] = -1.0
        p["b1"][i] = 1.0 - positives
        return i

    # Scale-one actor, anchored one pixel inside its bounding box.
    body = np.ones((4, 4), dtype=bool)
    body[3, 1:3] = False
    actors = []
    for r in range(4):
        f = template(np.rot90(body, -r), -1, -1)
        actors.append((f, r, 1, 1, 1))

    # Scale-two signatures at the rotated notch corners.
    notch = np.zeros((5, 5), dtype=bool)
    notch[:2, :] = True
    notch[2:4, :2] = True
    anchors = [(2, 6), (1, 2), (5, 1), (6, 5)]
    for r, (ox, oy) in enumerate(anchors):
        f = template(np.rot90(notch, -r))
        actors.append((f, r, 2, ox, oy))

    # Overlapping patches identify the target and its orientation.
    target = np.zeros((6, 6), dtype=bool)
    target[1:5, 0] = True
    target[1:5, 5] = True
    target[4, 2:4] = True
    target[5, :] = True
    goal_parts = []
    for r in range(4):
        mask = np.rot90(target, -r)
        goal_parts.append((
            template(mask[:5, :5]),
            template(mask[1:, 1:])
        ))

    def stroke(colour, vertical):
        i = alloc1()
        points = []
        if vertical:
            points += [(d, 0, colour) for d in (-1, 0, 1)]
            points += [(d, x, 0) for d in (-1, 0, 1)
                       for x in (-1, 1)]
            points += [(-2, 0, 0), (2, 0, 0)]
        else:
            points += [(0, d, colour) for d in (-1, 0, 1)]
            points += [(y, d, 0) for y in (-1, 1)
                       for d in (-1, 0, 1)]
            points += [(0, -2, 0), (0, 2, 0)]
        for dy, dx, c in points:
            p["W1"][i, c, dy + 2, dx + 2] += 1.0
        p["b1"][i] = 1.0 - len(points)
        return i

    hon = stroke(5, False)
    von = stroke(5, True)
    hoff = stroke(1, False)
    voff = stroke(1, True)

    row = alloc1()
    p["W1"][row, 16, 2, 2] = 63.0
    col = alloc1()
    p["W1"][col, 17, 2, 2] = 1.0

    # Filled orange interiors, excluding outlined selector tabs.
    orange = alloc1()
    p["W1"][orange, 9, 1:4, 1:4] = 1.0
    p["b1"][orange] = -8.0

    bit_rows = [33, 36, 39, 42, 45, 48]
    hinges = []
    for y in bit_rows:
        i = alloc1()
        p["W1"][i, 16, 2, 2] = 63.0
        p["b1"][i] = -float(y)
        hinges.append(i)

    n2 = 0

    def alloc2():
        nonlocal n2
        i = n2
        n2 += 1
        assert n2 <= 32
        return i

    def add2(out, inp, weight, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += weight

    actor_presence = alloc2()
    actor_x = alloc2()
    actor_y = alloc2()
    actor_r = alloc2()
    actor_s = alloc2()
    K = 64.0

    for f, r, scale, ox, oy in actors:
        add2(actor_presence, f, 1.0)
        add2(actor_x, f, K + 1.0 - ox / 63.0)
        add2(actor_y, f, K + 1.0 - oy / 63.0)
        add2(actor_r, f, float(r))
        add2(actor_s, f, float(scale))
    add2(actor_x, col, 1.0)
    add2(actor_y, row, 1.0 / 63.0)
    p["b2"][actor_x] = -K
    p["b2"][actor_y] = -K

    goal_presence = []
    for main, other in goal_parts:
        i = alloc2()
        goal_presence.append(i)
        add2(i, main, 1.0)
        add2(i, other, 1.0, 1, 1)
        p["b2"][i] = -1.0

    goal_x = alloc2()
    goal_y = alloc2()
    for main, other in goal_parts:
        for out in (goal_x, goal_y):
            add2(out, main, K)
            add2(out, other, K, 1, 1)
    add2(goal_x, col, 1.0)
    add2(goal_y, row, 1.0 / 63.0)
    p["b2"][goal_x] = -2.0 * K + 1.0 - 1.0 / 63.0
    p["b2"][goal_y] = p["b2"][goal_x]

    def row_peak(out, bit, gain=1.0):
        add2(out, row, gain)
        add2(out, hinges[bit], -2.0 * gain)
        p["b2"][out] += gain * (1.0 - bit_rows[bit])

    bits = []
    for bit in range(6):
        i = alloc2()
        bits.append(i)
        row_peak(i, bit)
        add2(i, hon if bit % 2 == 0 else von, 32.0)
        p["b2"][i] -= 32.0

    ui_column = alloc2()
    row_peak(ui_column, 0)
    add2(ui_column, hon, 32.0)
    add2(ui_column, hoff, 32.0)
    p["b2"][ui_column] -= 32.0

    row_features = []
    for bit in range(6):
        i = alloc2()
        row_features.append(i)
        row_peak(i, bit)

    run_feature = alloc2()
    add2(run_feature, orange, 1.0)
    add2(run_feature, row, 1.0 / 63.0)
    p["b2"][run_feature] = -1.7

    column_feature = alloc2()
    add2(column_feature, col, 1.0)

    introductory_off = alloc2()
    add2(introductory_off, hoff, 1.0)
    add2(introductory_off, voff, 1.0)
    add2(introductory_off, row, 1.0 / 63.0)
    p["b2"][introductory_off] = -1.64

    n3 = 0

    def alloc3():
        nonlocal n3
        i = n3
        n3 += 1
        assert n3 <= 32
        return i

    def local(out, inp, weight):
        p["W3"][out, inp] += weight

    def cm(out, inp, weight):
        p["W3"][out, 64 + inp] += weight

    ax, ay, ar, asc = [alloc3() for _ in range(4)]
    gx, gy, gr, gsc = [alloc3() for _ in range(4)]

    cm(ax, actor_x, 63.0 / 4.0)
    cm(ax, actor_presence, -63.0 / 4.0)
    cm(ay, actor_y, 63.0 / 4.0)
    cm(ay, actor_presence, -63.0 / 4.0)
    cm(ar, actor_r, 1.0)
    cm(asc, actor_s, 1.0)

    cm(gx, goal_x, 63.0 / 4.0)
    cm(gy, goal_y, 63.0 / 4.0)
    for r, gp in enumerate(goal_presence):
        cm(gx, gp, -63.0 / 4.0)
        cm(gy, gp, -63.0 / 4.0)
        cm(gr, gp, float(r))
        cm(gsc, gp, 1.0)

    def code_detector(out, code, bit_row=None):
        cm(out, ui_column, 4.0)
        ones = 0
        for bit in range(6):
            if code & (1 << bit):
                cm(out, bits[bit], 4.0)
                ones += 1
            else:
                cm(out, bits[bit], -4.0)
        p["b3"][out] = -4.0 * ones - 3.0
        if bit_row is not None:
            local(out, row_features[bit_row], 4.0)
            p["b3"][out] -= 4.0

    op = {}
    for code in [1, 2, 3, 33, 5, 6, 8, 9]:
        i = alloc3()
        op[code] = i
        code_detector(i, code)

    # Code 5 contributes one quarter-turn; code 7 contributes two.
    p["W3"][op[5], 64 + bits[1]] = 1.0

    gates = {}
    for key, code, bit_row in [
        ("zero0", 0, 0),
        ("zero1", 0, 1),
        ("zero5", 0, 5),
        ("32_1", 32, 1),
        ("32_0", 32, 0),
        ("32_2", 32, 2),
        ("36_0", 36, 0),
        ("36_1", 36, 1),
        ("zero3", 0, 3),
        ("8_0", 8, 0),
    ]:
        i = alloc3()
        gates[key] = i
        code_detector(i, code, bit_row)

    # Each horizontal gate either adds the desired instruction to an
    # empty slot or removes a completed instruction in the opposite direction.
    local(gates["zero1"], bits[0], 8.0)
    local(gates["zero0"], bits[1], 8.0)

    # The bit-zero scale gate accepts both 8 and 9.
    p["W3"][gates["8_0"], 64 + bits[0]] = 0.0

    finish34 = alloc3()
    code_detector(finish34, 34, 0)

    # Clearing bit five completes 35 -> 3, 37 -> 5, or 38 -> 6.
    finish_rotation = alloc3()
    cm(finish_rotation, ui_column, 1.0)
    local(finish_rotation, row_features[5], 1.0)
    for bit, weight in [
        (5, 1), (2, 1), (0, 1), (1, 1), (3, -1), (4, -1)
    ]:
        cm(finish_rotation, bits[bit], float(weight))
    p["b3"][finish_rotation] = -4.25

    run = alloc3()
    local(run, run_feature, 1.0)

    column = alloc3()
    local(column, column_feature, 1.0)

    intro = alloc3()
    local(intro, introductory_off, 1.0)
    cm(intro, ui_column, -2.0)
    local(intro, bits[4], 3.0)
    p["b3"][intro] = -0.01

    # Clean isolated 4 or 8 codes. A necessary 8 -> 9 completion has
    # higher priority than this cleanup.
    cleanup = alloc3()
    cm(cleanup, ui_column, 4.0)
    for bit in range(6):
        cm(cleanup, bits[bit], -4.0)
    local(cleanup, bits[2], 8.0)
    local(cleanup, bits[3], 8.0)
    p["b3"][cleanup] = -7.0
    assert n3 == 32

    def vector(terms):
        v = np.zeros(32, dtype=np.float32)
        for f, weight in terms:
            v[f] += weight
        return v

    drives = [
        vector([(gx, 1), (ax, -1), (op[1], 1), (op[2], -1)]),
        vector([(gy, 1), (ay, -1), (op[33], 1), (op[3], -1)]),
        vector([(gr, 1), (ar, -1), (op[5], -1), (op[6], 1)]),
        # The detected target is scale one. Count completed shrinking
        # commands; isolated expansion codes are handled by cleanup.
        vector([(gsc, 1), (asc, -1), (op[9], 1)]),
    ]

    # Bidirectional scans aggregate each quantity across all columns.
    K0 = 32.0
    for quantity, drive in enumerate(drives):
        a, b = 2 * quantity, 2 * quantity + 1
        p["Wh"][a, a, 1, 0] = 1.0
        p["Wh"][b, b, 1, 2] = 1.0
        p["Uh"][a] = drive
        p["Uh"][b] = drive
        p["bh"][a] = K0
        p["bh"][b] = K0

    specs = [
        (0, 1, ["zero1"], [], 100.0),
        (0, -1, ["zero0"], [], 100.0),
        (1, 1, ["zero5", "32_1"], ["32_1"], 1.0),
        (1, -1, ["zero5", "32_0"], ["32_0"], 1.0),
        (2, 1, ["zero5", "32_2", "36_0"],
         ["32_2", "36_0"], 1e4),
        (2, -1, ["zero5", "32_2", "36_1"],
         ["32_2", "36_1"], 1e4),
        # Excess shrinking: 9 -> 8, followed by cleanup.
        (3, 1, ["8_0"], ["8_0"], 1e6),
        (3, -1, ["zero3", "8_0"], ["8_0"], 1e6),
    ]

    inhibition = 128.0
    for index, (quantity, sign, allowed, active, weight) in enumerate(specs):
        out = 8 + index
        a, b = 2 * quantity, 2 * quantity + 1
        p["Wh"][out, a, 1, 1] = float(sign)
        p["Wh"][out, b, 1, 1] = float(sign)
        p["Uh"][out] = -float(sign) * drives[quantity]

        for key in allowed:
            p["Uh"][out, gates[key]] += inhibition
        for key in active:
            p["Uh"][out, gates[key]] += 0.2

        p["Uh"][out, column] -= 0.001
        p["bh"][out] = (
            -float(sign) * 65.0 * K0 - inhibition - 0.6
        )
        p["Wc"][0, 32 + out] = weight

    p["Wc"][0, finish34] = 1e9
    p["Wc"][0, finish_rotation] = 1e9
    p["Wc"][0, intro] = 1e8
    p["Wc"][0, cleanup] = 1e5
    p["Wc"][0, run] = 1.0
    p["Wc"][0, column] = -0.0001

    return p
