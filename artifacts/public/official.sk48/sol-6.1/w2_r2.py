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

    # ---------------------------------------------------------------
    # First convolution: local predicates.
    # Tile/head anchors are sprite position + (3, 3).
    # ---------------------------------------------------------------
    colours = (8, 9, 12, 14)
    TILE = (0, 1, 2, 3)
    SELECTED, UNSELECTED, MATCH = 4, 5, 6
    ACTIVE_H, ACTIVE_V = 7, 8
    INACTIVE_H, INACTIVE_V = 9, 10
    ONE, ROW, COL = 11, 12, 13
    BOTTOM_A, BOTTOM_B = 14, 15
    LEFT_NOTCH, UP_NOTCH = 16, 17
    SHIFTED_V, WHITE = 18, 19

    def k1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def pair(out, entries):
        for channel, dy, dx in entries:
            k1(out, channel, dy, dx)
        p["b1"][out] = -1.0

    corners = ((-2, -2), (-2, 1), (1, -2), (1, 1))
    for out, colour in zip(TILE, colours):
        for dy, dx in corners:
            k1(out, colour, dy, dx)
        p["b1"][out] = -3.0

    middle = ((-1, -1), (-1, 0), (0, -1), (0, 0))
    surround = ((-2, -2), (1, 1), (1, -1), (-1, 1))
    for out, interior in ((SELECTED, 0), (UNSELECTED, 4)):
        for dy, dx in middle:
            k1(out, 6, dy, dx)
            k1(out, 15, dy, dx)
        for dy, dx in surround:
            k1(out, interior, dy, dx)
        p["b1"][out] = -7.0

    pair(MATCH, ((0, -1, -1), (0, 0, 0)))
    pair(ACTIVE_H, ((1, -1, 2), (2, 0, 2)))
    pair(ACTIVE_V, ((2, 2, -1), (1, 2, 0)))
    pair(INACTIVE_H, ((2, -1, 2), (3, 0, 2)))
    pair(INACTIVE_V, ((3, 2, -1), (2, 2, 0)))
    pair(LEFT_NOTCH, ((2, -1, -2), (1, 0, -2)))
    pair(UP_NOTCH, ((1, -2, -1), (2, -2, 0)))
    pair(SHIFTED_V, ((3, 2, -2), (2, 2, -1)))

    k1(ONE, 1)
    k1(ROW, 16)
    k1(COL, 17)
    k1(WHITE, 4)

    k1(BOTTOM_A, 16, value=63.0)
    k1(BOTTOM_B, 16, value=63.0)
    p["b1"][BOTTOM_A] = -52.0
    p["b1"][BOTTOM_B] = -53.0

    # ---------------------------------------------------------------
    # Second convolution: semantic masks.
    # ---------------------------------------------------------------
    def k2(out, channel, value=1.0, dy=0, dx=0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def bottom(out, value):
        k2(out, BOTTOM_A, value)
        k2(out, BOTTOM_B, -value)

    def tiles(out, value=1.0):
        for channel in TILE:
            k2(out, channel, value)

    # Unmatched tiles in the goal display.
    for j in range(4):
        k2(j, TILE[j])
        k2(j, MATCH, -1.0)
        bottom(j, 1.0)
        p["b2"][j] = -1.0

    # Goal heads.
    for out, source in ((4, SELECTED), (5, UNSELECTED)):
        k2(out, source)
        bottom(out, 1.0)
        p["b2"][out] = -1.0

    # Active rod pixels, excluding the goal display.
    k2(6, ONE)
    bottom(6, -1.0)

    # Tiles carried by horizontal and vertical active rods.
    for out, edge in ((7, ACTIVE_H), (8, ACTIVE_V)):
        tiles(out)
        k2(out, edge)
        bottom(out, -1.0)
        p["b2"][out] = -1.0

    # Carried tiles intersecting another rod.
    for out, active, inactive in (
        (9, ACTIVE_H, INACTIVE_V),
        (10, ACTIVE_V, INACTIVE_H),
    ):
        tiles(out)
        k2(out, active)
        k2(out, inactive)
        bottom(out, -1.0)
        p["b2"][out] = -2.0

    # Board-head directions: right, down, left, up.
    for out, edge, dy, dx in (
        (11, ACTIVE_H, 0, 0),
        (12, ACTIVE_V, 0, 0),
        (13, LEFT_NOTCH, 0, -1),
        (14, UP_NOTCH, -1, 0),
    ):
        k2(out, SELECTED)
        k2(out, edge, dy=dy, dx=dx)
        bottom(out, -1.0)
        p["b2"][out] = -1.0

    # Selected board-head coordinates.
    for out, coordinate in ((15, ROW), (16, COL)):
        k2(out, SELECTED)
        k2(out, coordinate)
        bottom(out, -1.0)
        p["b2"][out] = -1.0

    # Available tiles: neither carried nor in the display.
    for j in range(4):
        out = 17 + j
        k2(out, TILE[j])
        k2(out, ACTIVE_H, -1.0)
        k2(out, ACTIVE_V, -1.0)
        bottom(out, -1.0)

    k2(21, UNSELECTED)
    bottom(21, -1.0)
    k2(22, ROW)
    k2(23, COL)

    # Pre-shifted interference sources.
    k2(24, INACTIVE_H, dy=-1)
    bottom(24, -1.0)
    k2(25, SHIFTED_V, dx=-1)
    bottom(25, -1.0)

    # Empty display pixels terminate the goal scan.
    k2(26, WHITE)
    bottom(26, 1.0)
    p["b2"][26] = -1.0

    # ---------------------------------------------------------------
    # Axial features.
    # ---------------------------------------------------------------
    def copy3(out, source):
        p["W3"][out, source] = 1.0

    for j in range(4):
        copy3(j, j)

    copy3(4, 4)
    copy3(5, 5)
    p["W3"][5, 26] = 1.0

    for out, source in (
        (6, 13),
        (7, 6),
        (8, 7),
        (9, 8),
        (10, 9),
        (11, 10),
        (12, 11),
        (13, 12),
        (14, 15),
        (15, 16),
    ):
        copy3(out, source)

    for j in range(4):
        copy3(16 + j, 17 + j)

        # Maximum is 1 minus the first available tile row.
        p["W3"][20 + j, 32 + 17 + j] = 1.0
        p["W3"][20 + j, 22] = -1.0

        # Maximum is 1 minus the first available tile column.
        p["W3"][24 + j, 64 + 17 + j] = 1.0
        p["W3"][24 + j, 23] = -1.0

    copy3(28, 21)
    copy3(29, 24)
    copy3(30, 14)
    copy3(31, 25)

    # ---------------------------------------------------------------
    # Spatial recurrence.
    # ---------------------------------------------------------------
    # Scan the selected goal sequence, stopping at its first mismatch.
    p["Wh"][0, 0, 1, 0] = 1.0
    p["Uh"][0, 4] = 1.0
    p["Uh"][0, 5] = -2.0
    p["Uh"][0, 0:4] = -2.0

    for j in range(4):
        p["Wh"][1 + j, 0, 1, 0] = 1.0
        p["Uh"][1 + j, j] = 1.0
        p["bh"][1 + j] = -1.0

    # Six-pixel look-behind for vertical retraction.
    p["Uh"][5, 29] = 1.0
    for out in range(6, 11):
        p["Wh"][out, out - 1, 0, 1] = 1.0
    p["Uh"][10, 9] = 1.0
    p["bh"][10] = -1.0

    # Six-pixel look-behind for horizontal retraction.
    p["Uh"][11, 31] = 1.0
    for out in range(12, 16):
        p["Wh"][out, out - 1, 1, 0] = 1.0
    p["Uh"][15, 8] = 1.0
    p["bh"][15] = -1.0

    # ---------------------------------------------------------------
    # Global algebra.
    # ---------------------------------------------------------------
    next_z = 0

    def unit(terms=None, bias=0.0):
        nonlocal next_z
        out = next_z
        next_z += 1
        if next_z > 128:
            raise RuntimeError("Unit allocation overflow")
        if terms:
            for index, value in terms.items():
                p["Wz"][out, index] += value
        p["bz"][out] = bias
        return out

    def add(*parts):
        result = {}
        for part in parts:
            for index, value in part.items():
                result[index] = result.get(index, 0.0) + value
        return result

    def mul(terms, factor):
        return {index: value * factor for index, value in terms.items()}

    def fm(channel):
        return 32 + channel

    def fa(channel):
        return 64 + channel

    # Correct-prefix length, with an end-of-sequence correction.
    PREFIX = {16: 4096.0 / 6.0}
    for channel in range(1, 5):
        PREFIX[channel] = -0.5
    prefix_bias = -0.5

    LENGTH = {
        fa(7): 4096.0 / 8.0,
        fa(8): 4096.0 * 6.0 / 8.0,
        fa(9): 4096.0 * 6.0 / 8.0,
    }
    length_bias = 7.0 / 8.0

    NUMBER = {
        fa(8): 4096.0,
        fa(9): 4096.0,
    }

    HORIZONTAL = {fm(12): 1.0, fm(6): 1.0}
    VERTICAL = {fm(13): 1.0, fm(30): 1.0}
    directions = (
        ({fm(12): 1.0}, 4, 3, 1.0, False),
        ({fm(13): 1.0}, 2, 1, 1.0, True),
        ({fm(6): 1.0}, 3, 4, -1.0, False),
        ({fm(30): 1.0}, 1, 2, -1.0, True),
    )

    z_prefix = unit(PREFIX, prefix_bias)
    z_length = unit(LENGTH, length_bias)

    ERROR = add(LENGTH, mul(PREFIX, -1.0))
    error_bias = length_bias - prefix_bias - 1.0

    z_reset_a = unit(mul(ERROR, -1.0), 0.25 - error_bias)
    z_reset_b = unit(mul(ERROR, -1.0), -0.25 - error_bias)

    WRONG = add(NUMBER, mul(PREFIX, -1.0))
    wrong_bias = -prefix_bias - 0.25
    z_wrong = unit(WRONG, wrong_bias)

    z_h = unit(HORIZONTAL)
    z_v = unit(VERTICAL)
    z_direction = [unit(direction[0]) for direction in directions]

    z_wrong_h = unit(
        add(WRONG, mul(HORIZONTAL, 100.0)),
        wrong_bias - 100.0,
    )
    z_wrong_v = unit(
        add(WRONG, mul(VERTICAL, 100.0)),
        wrong_bias - 100.0,
    )

    completed = {channel: -100.0 for channel in range(1, 5)}
    z_done = unit(
        add(completed, mul(ERROR, -100.0)),
        99.5 - 100.0 * error_bias,
    )

    z_anchor_h = unit({fm(10): 1.0, 15: 1.0})
    z_anchor_v = unit({fm(11): 1.0, 10: 1.0})

    navigation = []
    staging = []

    for j in range(4):
        desired = {1 + j: 100.0}
        available = {fm(16 + j): 100.0}

        row_up = {fm(14): 10.5, fm(20 + j): 10.5}
        col_left = {fm(15): 10.5, fm(24 + j): 10.5}

        for terms, bias, orientation, action, vertical in (
            (row_up, -10.5, HORIZONTAL, 1, False),
            (mul(row_up, -1.0), 10.5, HORIZONTAL, 2, False),
            (col_left, -10.5, VERTICAL, 3, True),
            (mul(col_left, -1.0), 10.5, VERTICAL, 4, True),
        ):
            out = unit(
                add(
                    terms,
                    desired,
                    available,
                    mul(orientation, 100.0),
                ),
                bias - 300.0,
            )
            navigation.append((out, action, vertical))

        # Stage before the next tile, then approach axially.
        for orientation, action, _, sign, vertical in directions:
            peak = fm(20 + j) if vertical else fm(24 + j)
            head = fm(14) if vertical else fm(15)
            axial = {
                peak: -10.5 * sign,
                head: -10.5 * sign,
            }
            out = unit(
                add(
                    axial,
                    mul(LENGTH, -1.0),
                    desired,
                    available,
                    mul(orientation, 100.0),
                ),
                10.5 * sign - length_bias - 0.25 - 300.0,
            )
            staging.append((out, action))

    # ---------------------------------------------------------------
    # Memory: compaction and clearing.
    # ---------------------------------------------------------------
    p["Wmz"][0, z_prefix] = 0.01

    for j, z_dir in enumerate(z_direction):
        mi = 1 + j
        p["Wm"][mi, mi] = 12.0
        p["Wm"][mi, 0] = -3000.0
        p["Wmz"][mi, z_prefix] = 30.0
        p["Wmz"][mi, z_wrong] = 48.0
        p["Wmz"][mi, z_reset_a] = -320.0
        p["Wmz"][mi, z_reset_b] = 320.0
        p["Wmz"][mi, z_dir] = 200.0
        p["bm"][mi] = -208.0

    p["Wm"][5, 5] = 12.0
    p["Wmz"][5, z_wrong_h] = 48.0
    p["Wmz"][5, z_h] = 200.0
    p["Wma"][5, 1:3] = -40.0
    p["bm"][5] = -208.0

    p["Wm"][6, 6] = 12.0
    p["Wmz"][6, z_wrong_v] = 48.0
    p["Wmz"][6, z_v] = 200.0
    p["Wma"][6, 3:5] = -40.0
    p["bm"][6] = -208.0

    # Clearing toward the next target is enabled only after compaction.
    for j, (z_nav, action, vertical) in enumerate(navigation):
        mi = 32 + j
        clearing = 6 if vertical else 5
        p["Wm"][mi, clearing] = 200.0
        p["Wmz"][mi, z_nav] = 20.0
        p["Wmz"][mi, z_length] = -2000.0
        p["Wmz"][mi, z_prefix] = 2000.0
        p["bm"][mi] = 1790.0
        if vertical:
            p["Wma"][mi, 3:5] = -1000.0
        else:
            p["Wma"][mi, 1:3] = -1000.0

        p["Wo"][action, 128 + mi] += 15000.0
        p["bo"][action] += 15000.0

    # ---------------------------------------------------------------
    # No-effect detection and temporary rod shifting.
    # ---------------------------------------------------------------
    signature = add(
        {fm(14): 63.0, fm(15): 63.0},
        mul(LENGTH, 1000.0),
    )
    z_signature = unit(signature, 1000.0 * length_bias)
    z_other = unit({fm(28): 1.0})
    p["Wmz"][8, z_signature] = 1.0e-8

    # Opposed tanh units cancel when the movement signature changes.
    # With another selectable rod, a stationary movement produces +2.
    for j, action in enumerate((1, 2, 3, 4)):
        for side, sign in enumerate((1.0, -1.0)):
            mi = 9 + 2 * j + side
            p["Wm"][mi, 8] = -sign * 1.0e9
            p["Wmz"][mi, z_signature] = sign * 10.0
            p["Wmz"][mi, z_other] = 10.0
            p["Wma"][mi, action] = 3.0
            p["bm"][mi] = -10.0
            p["Wo"][6, 128 + mi] += 750000.0

    # After a blocked-move click, shift the newly selected rod once.
    for mi, orientation, shift_action in (
        (20, z_h, 1),
        (21, z_v, 3),
    ):
        p["Wm"][mi, 9:17] = 100.0
        p["Wmz"][mi, orientation] = 1000.0
        p["Wma"][mi, 6] = 1000.0
        p["bm"][mi] = -2100.0

        p["Wo"][shift_action, 128 + mi] += 2000000.0
        p["bo"][shift_action] += 2000000.0

    # Return to the original rod after that single shift.
    for mi, shift_memory, orientation, shift_action in (
        (22, 20, z_h, 1),
        (23, 21, z_v, 3),
    ):
        p["Wm"][mi, shift_memory] = 20.0
        p["Wmz"][mi, orientation] = 100.0
        p["Wma"][mi, shift_action] = 50.0
        p["bm"][mi] = -160.0
        p["Wo"][6, 128 + mi] += 2000000.0
        p["bo"][6] += 2000000.0

    # ---------------------------------------------------------------
    # Final action arbitration.
    # ---------------------------------------------------------------
    for j, (_, grow, retract, _, _) in enumerate(directions):
        p["Wo"][grow, z_direction[j]] += 0.4
        p["Wo"][retract, 128 + 1 + j] += 100000.0
        p["bo"][retract] += 100000.0

    for out, action, _ in navigation:
        p["Wo"][action, out] += 10.0

    for out, action in staging:
        p["Wo"][action, out] += 200.0

    p["Wo"][1, 128 + 5] += 3000.0
    p["bo"][1] += 3000.0
    p["Wo"][3, 128 + 6] += 3000.0
    p["bo"][3] += 3000.0

    p["Wo"][2, z_anchor_h] += 1000000.0
    p["Wo"][4, z_anchor_v] += 1000000.0
    p["Wo"][6, z_done] += 10000.0

    # Click an unselected board-head centre.
    p["Wc"][0, 28] = 1.0

    p["bo"][0] = -1000000.0
    p["bo"][5] = -1000000.0
    p["bo"][7] = -1000000.0

    return p
