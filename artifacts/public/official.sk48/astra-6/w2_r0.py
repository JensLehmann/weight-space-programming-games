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

    colors = (8, 9, 12, 14)

    def pixel(out, color, dy=0, dx=0, weight=1.0):
        p["W1"][out, color, dy + 2, dx + 2] += weight

    def second(out, inp, weight=1.0, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += weight

    core = ((-1, -1), (-1, 0), (0, -1), (0, 0))
    perimeter = [
        (dy, dx)
        for dy in range(-2, 2)
        for dx in range(-2, 2)
        if (dy, dx) not in core
    ]
    horizontal_stub = (-1, 2)
    vertical_stub = (2, 0)

    # Object anchors lie at the lower-right pixel of the central 2x2.
    # The perimeter test remains valid when a goal marker covers the core.
    for c, color in enumerate(colors):
        for dy, dx in perimeter:
            pixel(c, color, dy, dx)
        p["b1"][c] = -11.0

    pixel(4, 1, *horizontal_stub)
    pixel(4, 1, *vertical_stub)

    for dy, dx in core:
        pixel(5, 0, dy, dx)
    p["b1"][5] = -3.0

    # Selected and unselected movable handles.
    for out, ring_color in ((6, 0), (7, 4)):
        for dy, dx in perimeter:
            pixel(out, ring_color, dy, dx)
        for dy, dx in core:
            pixel(out, 6, dy, dx)
            pixel(out, 15, dy, dx)
        p["b1"][out] = -15.0

    pixel(8, 1)
    pixel(9, 16)
    pixel(10, 17)
    pixel(11, 18)

    for out, stub in ((12, horizontal_stub), (13, vertical_stub)):
        p["W1"][out] = p["W1"][6]
        pixel(out, 1, *stub)
        p["b1"][out] = -16.0

    # Exposed inactive perpendicular-rail edges.
    pixel(14, 3, 2, -1)
    pixel(15, 3, 0, 2)

    for c in range(4):
        p["W1"][16 + c] = p["W1"][c]
        pixel(16 + c, 1, *horizontal_stub)
        p["b1"][16 + c] = -12.0

        p["W1"][20 + c] = p["W1"][c]
        pixel(20 + c, 1, *horizontal_stub, weight=-1.0)
        pixel(20 + c, 1, *vertical_stub, weight=-1.0)
        p["b1"][20 + c] = -11.0

    # Guide probes; the next convolution supplies the remaining offset.
    for out, (dy, dx) in enumerate(
        ((-2, -1), (2, -1), (-1, -2), (-1, 2)), start=24
    ):
        pixel(out, 2, dy, dx)
        pixel(out, 3, dy, dx)

    pixel(30, 16, weight=63.0)
    p["b1"][30] = -52.0
    pixel(31, 16, weight=-63.0)
    p["b1"][31] = 54.0

    def board_gate(out):
        second(out, 30, -100.0)

    def goal_gate(out):
        second(out, 31, -100.0)

    for c in range(4):
        # Unmatched beads of the selected goal.
        second(c, 16 + c)
        second(c, 5, -1.0)
        goal_gate(c)

        # Horizontal acquisition: prefer the topmost candidate row.
        second(4 + c, 20 + c, 4.0)
        second(4 + c, 9, -1.0)
        p["b2"][4 + c] = -2.0
        board_gate(4 + c)

        # Vertical acquisition: prefer the rightmost candidate column.
        second(8 + c, 20 + c, 4.0)
        second(8 + c, 10, 1.0)
        p["b2"][8 + c] = -3.0
        board_gate(8 + c)

        second(12 + c, 20 + c)
        board_gate(12 + c)

    # Carried beads, counted through the spatial mean.
    for c in range(4):
        second(16, c)
    second(16, 4)
    p["b2"][16] = -1.0
    board_gate(16)

    second(17, 12)
    second(18, 13)
    board_gate(17)
    board_gate(18)

    # Handle position and extreme selected-rail position.
    for out, mask, coord in (
        (19, 6, 10), (20, 6, 9),
        (21, 8, 10), (22, 8, 9),
    ):
        second(out, mask, 4.0)
        second(out, coord)
        p["b2"][out] = -4.0
        board_gate(out)

    second(23, 6, 4.0)
    second(23, 10)
    p["b2"][23] = -4.0
    goal_gate(23)

    for c in range(4):
        second(24, 16 + c, 4.0)
    second(24, 5, -4.0)
    second(24, 10)
    p["b2"][24] = -4.0
    goal_gate(24)

    # Carried beads at perpendicular crossings.
    for out, crossing in ((25, 14), (26, 15)):
        for c in range(4):
            second(out, c)
        second(out, 4)
        second(out, crossing)
        p["b2"][out] = -2.0
        board_gate(out)

    second(27, 7)
    board_gate(27)

    second(28, 12, dy=1)
    second(28, 24, dy=-1)
    second(29, 12)
    second(29, 25, dy=1)
    second(30, 13, dx=1)
    second(30, 26, dx=-1)
    second(31, 13)
    second(31, 27, dx=1)
    for out in range(28, 32):
        p["b2"][out] = -1.0
        board_gate(out)

    for c in range(23):
        p["W3"][c, c] = 1.0

    # Goal index relative to the selected example handle.
    p["W3"][23, 24] = 63.0 / 6.0
    p["W3"][23, 32 + 23] = -63.0 / 6.0
    p["W3"][23, 0:4] = 32.0
    p["b3"][23] = -32.0

    p["W3"][24, 25] = 1.0
    p["W3"][24, 32 + 17] = 1.0
    p["b3"][24] = -1.0

    p["W3"][25, 26] = 1.0
    p["W3"][25, 64 + 18] = 1.0
    p["b3"][25] = -1.0

    p["W3"][26, 27] = 1.0
    for out in range(27, 31):
        p["W3"][out, out + 1] = 1.0

    # Tied block: left-to-right goal-prefix sum.
    p["Wh"][0, 0, 1, 0] = 1.0
    p["Uh"][0, 0:4] = 1.0
    for c in range(4):
        p["Uh"][1 + c, c] = 1.0
        p["Wh"][1 + c, 0, 1, 0] = -1.0
    p["Uh"][5, 23] = 1.0
    p["Wh"][5, 0, 1, 0] = -64.0

    # Board scans reject candidates behind another loose bead.
    p["Wh"][6, 6, 1, 0] = 1.0
    p["Wh"][7, 7, 0, 1] = 1.0
    p["Uh"][6, 12:16] = 1.0
    p["Uh"][7, 12:16] = 1.0
    for c in range(4):
        p["Uh"][8 + c, 4 + c] = 1.0
        p["Wh"][8 + c, 6, 1, 0] = -2.0
        p["Uh"][12 + c, 8 + c] = 1.0
        p["Wh"][12 + c, 7, 0, 1] = -2.0

    def basis(i):
        v = np.zeros(96, dtype=np.float32)
        v[i] = 1.0
        return v

    def fm(c):
        return basis(32 + c)

    def fs(c):
        return 4096.0 * basis(64 + c)

    z_used = 0

    def emit(weights, bias=0.0):
        nonlocal z_used
        if z_used >= 128:
            raise RuntimeError("Feature allocation overflow")
        p["Wz"][z_used] = weights
        p["bz"][z_used] = bias
        v = np.zeros(128, dtype=np.float32)
        v[z_used] = 1.0
        z_used += 1
        return v

    def clipped_unit(weights, bias=0.0):
        return emit(weights, bias) - emit(weights, bias - 1.0)

    first = [basis(1 + c) for c in range(4)]
    any_first = sum(first)
    index = basis(5)
    count = fs(16)
    horizontal, vertical = fm(17), fm(18)
    hx, hy = fm(19), fm(20)

    extent_w = (63.0 / 6.0) * (fm(21) + fm(22) - hx - hy)
    extent_b = -2.0 / 6.0

    H = emit(horizontal)
    V = emit(vertical)
    K = emit(index, -1.0)

    compact = clipped_unit(
        2.0 * (count - extent_w), 1.0 - 2.0 * extent_b
    )
    empty = emit(-2.0 * extent_w, 1.0 - 2.0 * extent_b)
    bad = clipped_unit(
        2.0 * (count - index) + 64.0 * any_first, -63.0
    )
    done = emit(-any_first, 1.0)
    other = emit(fm(26))
    pinned = clipped_unit(fm(24) + fm(25))
    has_prefix = clipped_unit(index, -1.0)

    valid_raw = [fm(c) for c in (27, 28, 29, 30)]
    up_ok, down_ok, left_ok, right_ok = [
        emit(w) for w in valid_raw
    ]

    position_w = (63.0 / 6.0) * (hx + hy) + extent_w
    position = emit(position_w, extent_b)

    proposals = {
        a: np.zeros(128, dtype=np.float32)
        for a in (1, 2, 3, 4)
    }

    # Every proposal is computed in the current step. No delayed
    # memory predicates are needed for candidate fallback.
    for c in range(4):
        present = fm(12 + c)

        choices = (
            (horizontal, hy, basis(8 + c), fm(4 + c),
             1.0, -42.0, 1, 2, 4, 0),
            (vertical, hx, basis(12 + c), fm(8 + c),
             -1.0, 21.0, 3, 4, 2, 2),
        )

        for orient, coord, filtered, raw, sign, db, neg_a, pos_a, ext, vi in choices:
            gate = 64.0 * (orient + first[c] + present)
            gb = -192.0
            dw = 21.0 * (coord + sign * filtered)

            negative = emit(
                dw + gate + 64.0 * valid_raw[vi],
                db + gb - 64.0,
            )
            positive = emit(
                -dw + gate + 64.0 * valid_raw[vi + 1],
                -db + gb - 64.0,
            )

            # Cancel the false direction that an absent filtered
            # coordinate would otherwise suggest.
            if sign > 0:
                positive -= emit(
                    -21.0 * coord - 64.0 * filtered
                    + gate + 64.0 * valid_raw[vi + 1],
                    -db + gb - 64.0,
                )
            else:
                negative -= emit(
                    21.0 * coord - 64.0 * filtered
                    + gate + 64.0 * valid_raw[vi],
                    db + gb - 64.0,
                )

            proposals[neg_a] += negative
            proposals[pos_a] += positive

            middle = emit(dw + gate, db + gb)
            aligned = (
                emit(dw + gate, db + gb + 1.0)
                - 2.0 * middle
                + emit(dw + gate, db + gb - 1.0)
            )
            proposals[ext] += aligned

            # When no filtered candidate exists, fall back to a raw one.
            # The filtered coordinate is either zero or at least one.
            fallback_gate = gate - 128.0 * filtered
            raw_dw = 21.0 * (coord + sign * raw)

            fallback_middle = emit(
                raw_dw + fallback_gate, db + gb
            )
            proposals[neg_a] += emit(
                raw_dw + fallback_gate + 64.0 * valid_raw[vi],
                db + gb - 64.0,
            )
            proposals[pos_a] += emit(
                -raw_dw + fallback_gate + 64.0 * valid_raw[vi + 1],
                -db + gb - 64.0,
            )
            proposals[ext] += (
                emit(raw_dw + fallback_gate, db + gb + 1.0)
                - 2.0 * fallback_middle
                + emit(raw_dw + fallback_gate, db + gb - 1.0)
            )

    # A fixed crossing must be cleared before contraction. A crossing
    # belonging to another movable handle is instead kept as an anchor.
    pin_down = emit(
        fm(24) + valid_raw[1] - fm(26), -1.0
    )
    pin_up = emit(
        fm(24) + valid_raw[0] - valid_raw[1] - fm(26), -1.0
    )
    pin_right = emit(
        fm(25) + valid_raw[3] - fm(26), -1.0
    )
    pin_left = emit(
        fm(25) + valid_raw[2] - valid_raw[3] - fm(26), -1.0
    )

    p["bo"][[0, 5]] = -1000.0
    p["bo"][6] = -1.0
    p["bo"][7] = -1.0

    for a in (1, 2, 3, 4):
        p["Wo"][a, :128] += 2.0 * proposals[a]
    p["Wo"][4, :128] += 0.03 * H
    p["Wo"][2, :128] += 0.03 * V
    p["Wo"][6, :128] += 300.0 * done

    for a, feature in (
        (1, pin_up), (2, pin_down), (3, pin_left), (4, pin_right)
    ):
        p["Wo"][a, :128] += 100.0 * feature

    # Small-amplitude geometric registers are nearly linear through tanh.
    eps = 0.001
    p["Wmz"][0] = eps * K
    p["Wmz"][3] = eps * position

    # Preserve the compaction request while releasing a fixed pin.
    p["Wm"][1, 1] = 4.0
    p["Wm"][1, 0] = -12.0 / eps
    p["Wmz"][1] = 12.0 * K - 10.0 * compact
    p["Wma"][1, 6] = 100.0
    p["Wma"][1, 7] = -30.0
    p["bm"][1] = -1.0

    # Dispose of an incorrect load until the rail becomes empty.
    p["Wm"][2, 2] = 4.0
    p["Wmz"][2] = 10.0 * bad - 10.0 * empty
    p["Wma"][2, 6] = -30.0
    p["Wma"][2, 7] = -30.0
    p["bm"][2] = -1.0

    mem_used = 4

    def clause(action, z, old, bias, priority,
               previous=None, slope=4.0):
        nonlocal mem_used
        if mem_used >= 64:
            raise RuntimeError("Memory allocation overflow")
        j = mem_used
        mem_used += 1
        p["Wmz"][j] = slope * z
        for i, weight in old.items():
            p["Wm"][j, i] = slope * weight
        if previous:
            for a, weight in previous.items():
                p["Wma"][j, a] = slope * weight
        p["bm"][j] = slope * bias
        p["Wo"][action, 128 + j] += priority / 2.0
        p["bo"][action] += priority / 2.0
        return j

    for orient, retract in ((H, 3), (V, 1)):
        clause(
            retract,
            12.0 * K + 4.0 * orient
            - 8.0 * compact - 8.0 * bad - 16.0 * pinned
            + 128.0 * has_prefix,
            {1: 4.0, 0: -12.0 / eps},
            -134.0, 20.0,
        )
        clause(
            retract,
            4.0 * orient - 8.0 * empty,
            {2: 4.0},
            -6.0, 50.0,
        )

    # Move an incorrect load aside before retracting it.
    clause(1, 8.0 * bad + 4.0 * up_ok,
           {2: -3.0}, -11.0, 80.0)
    clause(2, 8.0 * bad + 4.0 * down_ok - 4.0 * up_ok,
           {2: -3.0}, -11.0, 80.0)
    clause(3, 8.0 * bad + 4.0 * left_ok,
           {2: -3.0}, -11.0, 80.0)
    clause(4, 8.0 * bad + 4.0 * right_ok - 4.0 * left_ok,
           {2: -3.0}, -11.0, 80.0)

    # Successful arrow actions change the geometric register by +/- one.
    for action, sign in ((1, -1.0), (2, 1.0), (3, -1.0), (4, 1.0)):
        clause(
            6,
            -8.0 * sign * position + 1024.0 * other,
            {3: 8.0 * sign / eps},
            4.0 - 2048.0,
            150.0,
            previous={action: 1024.0},
            slope=1.0,
        )

    # Protect a partial shared sequence against a newly incorrect load.
    clause(
        7, 4.0 * (bad + other + has_prefix),
        {}, -10.0, 700.0,
    )
    clause(
        6, np.zeros(128, dtype=np.float32),
        {}, -2.0, 1000.0,
        previous={7: 4.0},
    )

    # Probe another lane after a stalled forward extension.
    # Prefer the positive direction, away from the usual disposal lane.
    recovery = []
    zero_z = np.zeros(128, dtype=np.float32)
    for orient, forward, action, allowed, forbidden in (
        (H, 4, 2, down_ok, zero_z),
        (H, 4, 1, up_ok, down_ok),
        (V, 2, 4, right_ok, zero_z),
        (V, 2, 3, left_ok, right_ok),
    ):
        recovery.append(clause(
            action,
            -8.0 * position
            + 1024.0 * (orient + allowed - forbidden - other)
            - 1024.0 * bad,
            {3: 8.0 / eps},
            4.0 - 3072.0,
            200.0,
            previous={forward: 1024.0},
            slope=1.0,
        ))

    # The preceding-action gate prevents activation from initial zeros.
    arrow_gate = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}
    clause(
        4, 2.0 * H - 32.0 * bad,
        {recovery[0]: 1.0, recovery[1]: 1.0},
        -3.0, 220.0,
        previous=arrow_gate, slope=8.0,
    )
    clause(
        2, 2.0 * V - 32.0 * bad,
        {recovery[2]: 1.0, recovery[3]: 1.0},
        -3.0, 220.0,
        previous=arrow_gate, slope=8.0,
    )

    # Shorten the rail when a transverse move is obstructed.
    for previous_action, sign, orient, retract in (
        (1, -1.0, H, 3), (2, 1.0, H, 3),
        (3, -1.0, V, 1), (4, 1.0, V, 1),
    ):
        clause(
            retract,
            -8.0 * sign * position
            + 1024.0 * (orient - other - empty),
            {3: 8.0 * sign / eps},
            4.0 - 2048.0,
            200.0,
            previous={previous_action: 1024.0},
            slope=1.0,
        )

    # Hand off after compaction, but not immediately after a click.
    for orient, retract in ((H, 3), (V, 1)):
        clause(
            6,
            4.0 * (orient + compact + other + has_prefix)
            - 16.0 * bad,
            {1: 4.0},
            -22.0, 240.0,
            previous={retract: 4.0},
        )

    # Newly acquired shared beads are useful anchors for the other rail.
    clause(
        6,
        128.0 * (pinned + other) + 4.0 * K - 128.0 * bad,
        {0: -4.0 / eps},
        -258.0, 250.0,
        previous={6: -512.0, 7: -512.0},
    )

    # Click the centre of an unselected movable playfield handle.
    p["Wc"][0, 26] = 1.0

    return p
