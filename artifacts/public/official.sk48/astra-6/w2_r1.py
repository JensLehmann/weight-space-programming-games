import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5),
        "b1": (32,),
        "W2": (32, 32, 3, 3),
        "b2": (32,),
        "W3": (32, 96),
        "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32),
        "bh": (16,),
        "Wz": (128, 96),
        "bz": (128,),
        "Wm": (64, 64),
        "Wmz": (64, 128),
        "Wma": (64, 8),
        "bm": (64,),
        "Wo": (8, 192),
        "bo": (8,),
        "Wc": (1, 48),
        "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    colors = (8, 9, 10, 12, 13, 14)

    # ---------- Local visual predicates ----------

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    ring = [
        (r, c)
        for r in range(1, 5)
        for c in range(1, 5)
        if r in (1, 4) or c in (1, 4)
    ]
    center = [(r, c) for r in (2, 3) for c in (2, 3)]

    # Cargo anchor: upper-left pixel of its central 2x2 square.
    # Its outer ring survives the goal-match overlay.
    for j, color in enumerate(colors):
        for r, c in ring:
            w1(j, color, r - 2, c - 2)
        p["b1"][j] = -11

    for r, c in center:
        w1(6, 0, r - 2, c - 2)
    p["b1"][6] = -3

    # Selected horizontal and vertical rails at cargo anchors.
    w1(7, 1, 1, -2)
    w1(8, 1, -2, 0)

    # Rails of either selection state.
    for color in (1, 2, 3):
        w1(9, color, 1, -2)
        w1(10, color, -2, 0)

    # Motor anchor: lower-right central pixel.
    # Detect the ring and notch, without requiring one motor identity.
    motor_colors = tuple(c for c in range(16) if c not in (0, 4))
    for out in (11, 12):
        for r, c in ring:
            w1(out, 0, r - 3, c - 3)
        for r, c in center:
            for color in motor_colors:
                w1(out, color, r - 3, c - 3)
        p["b1"][out] = -16

    for color in (1, 2):
        w1(11, color, 0, 2)
        w1(12, color, 2, 0)

    # Inactive selectable motor.
    for r, c in ring:
        w1(13, 4, r - 3, c - 3)
    for r, c in center:
        for color in motor_colors:
            w1(13, color, r - 3, c - 3)
    p["b1"][13] = -15

    w1(14, 1)
    w1(15, 17)
    w1(16, 16)

    # Board/display separation.
    w1(17, 16, value=63)
    p["b1"][17] = -52
    w1(18, 16, value=-63)
    p["b1"][18] = 54

    w1(19, 18)
    w1(20, 4)

    # Solid background-colour patch. Axial board masks below distinguish
    # an interior obstruction from the exterior background.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w1(21, 5, dy, dx)
    p["b1"][21] = -24

    def f2(out, terms, bias=0.0):
        for inp, value in terms.items():
            p["W2"][out, inp, 1, 1] += value
        p["b2"][out] = bias

    # Free board cargo.
    for c in range(6):
        f2(c, {c: 1, 7: -1, 8: -1, 17: -1})

    # Unmatched icons on the selected goal.
    for c in range(6):
        f2(6 + c, {c: 3, 7: 1, 6: -1, 18: -1}, -3)

    terms = {c: 2 for c in range(6)}
    terms.update({7: 1, 18: -1})
    f2(12, terms, -2)

    terms = {c: 2 for c in range(6)}
    terms.update({7: 1, 8: 1, 17: -3})
    f2(13, terms, -2)

    # Cargo shared with perpendicular rails.
    terms = {c: 3 for c in range(6)}
    terms.update({7: 1, 10: 1, 17: -3})
    f2(14, terms, -4)

    terms = {c: 3 for c in range(6)}
    terms.update({8: 1, 9: 1, 17: -3})
    f2(15, terms, -4)

    f2(16, {11: 1, 17: -1})
    f2(17, {12: 1, 17: -1})
    f2(18, {13: 1, 17: -1})
    f2(19, {14: 1, 17: -1})
    f2(20, {15: 1})
    f2(21, {16: 1})

    terms = {c: 1 for c in range(6)}
    terms.update({7: -1, 6: -1, 18: -1})
    f2(22, terms)

    f2(23, {20: 1, 17: -1})
    f2(24, {21: 1, 17: -1})

    def f3(out, terms, bias=0.0):
        for inp, value in terms.items():
            p["W3"][out, inp] += value
        p["b3"][out] = bias

    for c in range(6):
        f3(c, {6 + c: 1})

    f3(6, {12: 1})
    f3(7, {13: 1})
    f3(8, {14: 1})
    f3(9, {15: 1})
    f3(10, {16: 1})
    f3(11, {17: 1})

    # Coordinate-valued board, rail, and motor masks.
    f3(12, {23: 1, 20: 1}, -1)
    f3(13, {23: 1, 21: 1}, -1)
    f3(14, {19: 1, 20: 1}, -1)
    f3(15, {19: 1, 21: 1}, -1)
    f3(16, {16: 1, 17: 1, 20: 1}, -1)
    f3(17, {16: 1, 17: 1, 21: 1}, -1)

    # A unit motor peak and small unmatched-inactive-goal peaks.
    f3(18, {18: 1, 22: 1.0 / 16.0})

    for c in range(6):
        f3(19 + c, {c: 1, 21: -1})
        f3(25 + c, {c: 1, 20: 1}, -1)

    # Objects obscuring access from the left. The row/column masks
    # restrict background-colour patches to the board's interior.
    blockers = {c: 1 for c in range(6)}
    blockers.update({
        24: 1,
        32 + 23: 1,
        64 + 23: 1,
    })
    f3(31, blockers, -2)

    # ---------- Tied directional scans ----------

    # Prefix count of unmatched goal icons.
    p["Wh"][0, 0, 1, 0] = 1
    p["Uh"][0, :6] = 1

    # Prefix count of all selected goal icons.
    p["Wh"][1, 1, 1, 0] = 1
    p["Uh"][1, 6] = 1

    # Only the first unmatched colour survives.
    for c in range(6):
        p["Uh"][2 + c, c] = 1
        p["Wh"][2 + c, 0, 1, 0] = -1

    # Number of correct entries before the first unmatched entry.
    p["Wh"][8, 1, 1, 0] = 1
    p["Wh"][8, 0, 1, 0] = -100
    p["Uh"][8, :6] = 100
    p["bh"][8] = -100

    # Cargo with no preceding free cargo or interior barrier.
    p["Wh"][9, 9, 1, 0] = 1
    p["Uh"][9, 31] = 1
    for c in range(6):
        p["Uh"][10 + c, 19 + c] = 1
        p["Wh"][10 + c, 9, 1, 0] = -100

    # ---------- Global feature algebra ----------

    def feature(index):
        v = np.zeros(96, dtype=np.float32)
        v[index] = 1
        return v

    def hm(index):
        return feature(index)

    def fm(index):
        return feature(32 + index)

    def favg(index):
        return feature(64 + index)

    z_used = 0

    def z_relu(v, bias=0.0):
        nonlocal z_used
        i = z_used
        z_used += 1
        if z_used > 128:
            raise AssertionError("z capacity exceeded")
        p["Wz"][i] = v
        p["bz"][i] = bias
        result = np.zeros(128, dtype=np.float32)
        result[i] = 1
        return result

    def z_flag(v, bias=0.0, threshold=0.25, width=0.5):
        lo = z_relu(v, bias - threshold)
        hi = z_relu(v, bias - threshold - width)
        return (lo - hi) / width

    def zz():
        return np.zeros(128, dtype=np.float32)

    selected = sum(
        (hm(2 + c) for c in range(6)),
        np.zeros(96, dtype=np.float32),
    )
    axis_h_g, axis_v_g = fm(10), fm(11)

    filtered_up, filtered_down = zz(), zz()
    full_up, full_down = zz(), zz()
    vertical_left, vertical_right = zz(), zz()
    filtered_present, full_present = zz(), zz()

    for c in range(6):
        desired = hm(2 + c)
        horizontal_gate = 200 * (desired + axis_h_g)
        vertical_gate = 200 * (desired + axis_v_g)

        filtered_delta = -63 * (hm(10 + c) + fm(17))
        full_delta = -63 * (fm(19 + c) + fm(17))
        vertical_delta = 63 * (fm(25 + c) - fm(16))

        filtered_up += z_flag(
            -filtered_delta + horizontal_gate, -464, 0.5, 1
        )
        filtered_down += z_flag(
            filtered_delta + horizontal_gate, -336, 0.5, 1
        )
        full_up += z_flag(
            -full_delta + horizontal_gate, -464, 0.5, 1
        )
        full_down += z_flag(
            full_delta + horizontal_gate, -336, 0.5, 1
        )
        vertical_left += z_flag(
            -vertical_delta + vertical_gate, -401, 0.5, 1
        )
        vertical_right += z_flag(
            vertical_delta + vertical_gate, -399, 0.5, 1
        )

        filtered_present += z_flag(
            64 * hm(10 + c) + 200 * desired, -200
        )
        full_present += z_flag(
            64 * fm(19 + c) + 200 * desired, -200
        )

    bad = z_flag(
        4096 * favg(7) - hm(8) + 32 * selected, -32
    )

    horizontal_length = (63.0 / 6.0) * (fm(14) - fm(16))
    vertical_length = (63.0 / 6.0) * (fm(15) - fm(17))

    extra_h = z_flag(
        horizontal_length - hm(8)
        + 32 * (axis_h_g + selected),
        -1.0 / 3.0 - 64,
    )
    extra_v = z_flag(
        vertical_length - hm(8)
        + 32 * (axis_v_g + selected),
        -1.0 / 3.0 - 64,
    )

    # Leave a bypass lane when handing off a completed strand.
    done_extra_h = z_flag(
        63 * (fm(14) - fm(12)) + 128 * axis_h_g, -122
    )
    done_extra_v = z_flag(
        63 * (fm(15) - fm(13)) + 128 * axis_v_g, -122
    )

    axis_h = z_relu(axis_h_g)
    axis_v = z_relu(axis_v_g)
    pin_h = z_relu(fm(8))
    pin_v = z_relu(fm(9))
    done = z_relu(-selected, 1)
    high_h = z_flag(63 * fm(17), -35)
    high_v = z_flag(63 * fm(16), -35)

    inactive_complete_g = 17 * fm(18) - 65536 * favg(18)
    shared_h = z_relu(fm(8) + inactive_complete_g, -1)
    shared_v = z_relu(fm(9) + inactive_complete_g, -1)

    # ---------- Recurrent controller ----------

    m_used = 0

    def memory_unit():
        nonlocal m_used
        i = m_used
        m_used += 1
        if m_used > 64:
            raise AssertionError("memory capacity exceeded")
        return i

    def emit(i, action, priority):
        # Saturated tanh is interpreted as (m + 1) / 2.
        p["Wo"][action, 128 + i] += priority / 2
        p["bo"][action] += priority / 2

    def gate(action, positive, negative=(), priority=10):
        i = memory_unit()
        v = zz()
        for term in positive:
            v += term
        for term in negative:
            v -= term
        p["Wmz"][i] = 16 * v
        p["bm"][i] = -16 * (len(positive) - 0.5)
        emit(i, action, priority)
        return i

    # Remove unwanted suffixes: sidestep once, then retract.
    dump_h, dump_v = memory_unit(), memory_unit()

    for latch, axis, high, backward, low_side, high_side, sides in (
        (dump_h, axis_h, high_h, 3, 2, 1, (1, 2)),
        (dump_v, axis_v, high_v, 1, 4, 3, (3, 4)),
    ):
        p["Wmz"][latch] = 16 * (2 * bad + 2 * axis)
        p["Wm"][latch, latch] = 8
        for action in sides:
            p["Wma"][latch, action] = 16
        p["bm"][latch] = -68
        emit(latch, backward, 80)

        for is_high, action in (
            (False, low_side), (True, high_side)
        ):
            i = memory_unit()
            if is_high:
                p["Wmz"][i] = 16 * (
                    2 * bad + 2 * axis + 4 * high
                )
                p["bm"][i] = -124
            else:
                p["Wmz"][i] = 16 * (
                    2 * bad + 2 * axis - 4 * high
                )
                p["bm"][i] = -60

            p["Wm"][i, latch] = -8
            for previous in sides:
                p["Wma"][i, previous] = -16
            emit(i, action, 80)

    # Horizontal collection.
    gate(1, [axis_h, filtered_present, filtered_up])
    gate(2, [axis_h, filtered_present, filtered_down])
    gate(
        4, [axis_h, filtered_present],
        [filtered_up, filtered_down],
    )

    gate(
        1, [axis_h, full_present, full_up],
        [filtered_present],
    )
    gate(
        2, [axis_h, full_present, full_down],
        [filtered_present],
    )
    gate(
        4, [axis_h, full_present],
        [filtered_present, full_up, full_down],
    )

    # Vertical collection.
    gate(3, [axis_v, full_present, vertical_left])
    gate(4, [axis_v, full_present, vertical_right])
    gate(
        2, [axis_v, full_present],
        [vertical_left, vertical_right],
    )

    # Compact before travelling, unless an intersection is protected.
    gate(
        3, [extra_h, filtered_present, filtered_up + filtered_down],
        [shared_h], priority=40,
    )
    gate(
        3, [extra_h, full_present, full_up + full_down],
        [filtered_present, shared_h], priority=40,
    )
    gate(
        1, [extra_v, full_present, vertical_left + vertical_right],
        [shared_v], priority=40,
    )

    # Release cargo pinned by an unfinished perpendicular strand.
    gate(
        2, [axis_h, pin_h], [shared_h, high_h], priority=300
    )
    gate(
        1, [axis_h, pin_h, high_h], [shared_h], priority=300
    )
    gate(
        4, [axis_v, pin_v], [shared_v, high_v], priority=300
    )
    gate(
        3, [axis_v, pin_v, high_v], [shared_v], priority=300
    )

    gate(3, [done, done_extra_h], priority=200)
    gate(1, [done, done_extra_v], priority=200)
    gate(6, [done], priority=150)

    gate(3, [axis_h], [full_present, done], priority=5)
    gate(1, [axis_v], [full_present, done], priority=5)

    # ---------- Guarded local no-progress recovery ----------
    #
    # Store coordinates at low amplitude so tanh is nearly linear.
    # Successful movements change the coordinate by six pixels; the
    # two-pixel comparison margin dominates approximation error.

    motor_x = z_relu(63 * fm(16))
    motor_y = z_relu(63 * fm(17))

    previous_x = memory_unit()
    previous_y = memory_unit()
    history_valid = memory_unit()

    p["Wmz"][previous_x] = 0.001 * motor_x
    p["Wmz"][previous_y] = 0.001 * motor_y
    p["bm"][history_valid] = 16

    for axis, protected, coord, old_coord, previous_action, sign, retract in (
        (axis_h, shared_h, motor_y, previous_y, 1, -1, 3),
        (axis_h, shared_h, motor_y, previous_y, 2, 1, 3),
        (axis_v, shared_v, motor_x, previous_x, 3, -1, 1),
        (axis_v, shared_v, motor_x, previous_x, 4, 1, 1),
    ):
        i = memory_unit()
        p["Wmz"][i] = (
            64 * axis
            - 64 * protected
            - 64 * done
            - 8 * sign * coord
        )
        p["Wm"][i, old_coord] = 8000 * sign
        p["Wm"][i, history_valid] = 64
        p["Wma"][i, previous_action] = 64
        p["bm"][i] = -190
        emit(i, retract, 600)

    p["bo"][0] = -1000
    p["bo"][5] = -1000
    p["bo"][7] = -1000

    # Inactive board motors dominate the smaller goal-icon peaks.
    p["Wc"][0, 18] = 10

    assert z_used <= 128
    assert m_used <= 64
    return p
