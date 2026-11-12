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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    def c1(out, inp, value, dy=0, dx=0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, value, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    # A selected piece has color-0 markers surrounded by color 5.
    # The two channels let W2 clip the neighborhood count to one.
    for out, bias in ((0, -10), (1, -11)):
        c1(out, 5, 10)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                c1(out, 0, 1, dy, dx)
        p["b1"][out] = bias

    c1(2, 11, 1)
    c1(3, 16, 1)
    c1(4, 17, 1)
    c1(5, 18, 1)
    c1(5, 16, -1)
    c1(6, 18, 1)
    c1(6, 17, -1)

    # Masks for the energy column and the bottom padding row.
    c1(7, 17, 63)
    p["b1"][7] = -62
    c1(18, 16, 63)
    p["b1"][18] = -62

    # Left edge of a vertical axis.
    for dy in range(-2, 3):
        c1(8, 10, 2, dy, 0)
        c1(8, 10, -2, dy, -1)
    c1(8, 18, 2, 0, -1)
    p["b1"][8] = -11

    # Top edge of a horizontal axis.
    for dx in range(-2, 3):
        c1(9, 10, 2, 0, dx)
        c1(9, 10, -2, -1, dx)
    c1(9, 18, 2, -1, 0)
    p["b1"][9] = -11

    c1(11, 0, 1)
    c1(12, 0, 1, 2, 0)
    c1(13, 0, 1, 0, 2)

    for dy in (-2, 0, 2):
        c1(14, 10, 1, dy, 0)
    p["b1"][14] = -2

    for dx in (-2, 0, 2):
        c1(15, 10, 1, 0, dx)
    p["b1"][15] = -2
    c1(17, 5, 1)

    # Selected-piece and target masks.
    c2(0, 0, 1)
    c2(0, 1, -1)
    c2(0, 7, -1)
    c2(0, 18, -1)

    c2(1, 2, 1)
    c2(1, 7, -1)

    c2(2, 8, 1)
    c2(3, 9, 1)

    # Selected-axis markers.
    c2(5, 11, 2)
    c2(5, 12, 2, 1, 0)
    c2(5, 14, 2, 0, 1)
    p["b2"][5] = -5

    c2(6, 11, 2)
    c2(6, 13, 2, 0, 1)
    c2(6, 15, 2, 1, 0)
    p["b2"][6] = -5

    # Selected-piece bounding coordinates.
    for out, coord in zip(range(7, 11), (3, 4, 5, 6)):
        c2(out, 0, 1)
        c2(out, 1, -1)
        c2(out, 7, -1)
        c2(out, 18, -1)
        c2(out, coord, 1)
        p["b2"][out] = -1

    # Target bounding coordinates.
    for out, coord in zip(range(11, 15), (3, 4, 5, 6)):
        c2(out, 2, 1)
        c2(out, 7, -1)
        c2(out, coord, 1)
        p["b2"][out] = -1

    # Real-piece pixels for the click map.
    c2(17, 17, 1)
    c2(17, 7, -1)
    c2(17, 18, -1)

    # Axis-edge coordinates.
    for out, mask, coord in (
        (19, 8, 4), (20, 8, 6),
        (21, 9, 3), (22, 9, 5),
    ):
        c2(out, mask, 1)
        c2(out, coord, 1)
        p["b2"][out] = -1

    for c in range(32):
        p["W3"][c, c] = 1

    # Extend detected axes through their entire column or row.
    p["W3"][15, :] = 0
    p["W3"][15, 64 + 2] = 1
    p["W3"][16, :] = 0
    p["W3"][16, 32 + 3] = 1

    # Flood the four sides of the axes.
    p["Wh"][0, 0, 1, 0] = 1
    p["Wh"][1, 1, 1, 2] = 1
    p["Wh"][2, 2, 0, 1] = 1
    p["Wh"][3, 3, 2, 1] = 1
    p["Uh"][0, 15] = 1
    p["Uh"][1, 15] = 1
    p["Uh"][2, 16] = 1
    p["Uh"][3, 16] = 1

    # Target coordinates on each side of each axis.
    for out, flood, coord in (
        (4, 0, 12), (5, 0, 14),
        (6, 1, 12), (7, 1, 14),
        (8, 2, 11), (9, 2, 13),
        (10, 3, 11), (11, 3, 13),
    ):
        p["Wh"][out, flood, 1, 1] = 2
        p["Uh"][out, coord] = 1
        p["bh"][out] = -2

    # Which side contains the selected piece?
    for out, flood in ((12, 0), (13, 1), (14, 2), (15, 3)):
        p["Wh"][out, flood, 1, 1] = 2
        p["Uh"][out, 0] = 1
        p["bh"][out] = -2

    def hm(c):
        return c

    def fm(c):
        return 32 + c

    next_unit = 0

    def unit(terms, bias=0):
        nonlocal next_unit
        i = next_unit
        next_unit += 1
        for idx, value in terms:
            p["Wz"][i, idx] += value
        p["bz"][i] = bias
        return i

    def mul(terms, factor):
        return [(i, factor * v) for i, v in terms]

    selected_piece = unit([(fm(0), 1)])
    selected_v = unit([(fm(5), 1)])
    selected_h = unit([(fm(6), 1)])

    def gated_delta(delta, delta_bias, gate, gate_bias):
        positive = unit(gate + delta, gate_bias + delta_bias)
        negative = unit(gate + mul(delta, -1), gate_bias - delta_bias)
        return positive, negative

    dimension_pairs = {}
    for dim in ("y", "x"):
        if dim == "y":
            target = (fm(11), fm(13))
            piece = (fm(7), fm(9))
            axis = fm(16)
            axis_coords = (fm(21), fm(22))
            sides = [
                ((hm(8), hm(9)), hm(14), (hm(10), hm(11))),
                ((hm(10), hm(11)), hm(15), (hm(8), hm(9))),
            ]
        else:
            target = (fm(12), fm(14))
            piece = (fm(8), fm(10))
            axis = fm(15)
            axis_coords = (fm(19), fm(20))
            sides = [
                ((hm(4), hm(5)), hm(12), (hm(6), hm(7))),
                ((hm(6), hm(7)), hm(13), (hm(4), hm(5))),
            ]

        pairs = []

        # Direct alignment when there is no axis of this orientation.
        direct = [
            (target[0], 1), (target[1], -1),
            (piece[0], -1), (piece[1], 1),
        ]
        pairs.append(gated_delta(direct, 0, [(axis, -100)], 10))

        for own, piece_side, other in sides:
            # Same-side target alignment.
            direct = [
                (own[0], 1), (own[1], -1),
                (piece[0], -1), (piece[1], 1),
            ]
            gate = [
                (piece_side, 100),
                (own[0], 10), (own[1], 10),
            ]
            pairs.append(gated_delta(direct, 0, gate, -105))

            # Opposite-side target alignment via reflection.
            reflected = [
                (other[0], -1), (other[1], 1),
                (piece[0], -1), (piece[1], 1),
                (axis_coords[0], 2), (axis_coords[1], -2),
            ]
            gate = [
                (piece_side, 100),
                (other[0], 10), (other[1], 10),
                (own[0], -100), (own[1], -100),
            ]
            pairs.append(
                gated_delta(reflected, 4 / 63, gate, -105)
            )

        dimension_pairs[dim] = pairs

    # Twice the distance from each selected axis to the target-box center.
    v_delta = [
        (fm(12), 1), (fm(14), -1),
        (fm(19), -1), (fm(20), 1),
    ]
    h_delta = [
        (fm(11), 1), (fm(13), -1),
        (fm(21), -1), (fm(22), 1),
    ]
    axis_bias = -2 / 63
    deadzone = 3 / 63

    axis_units = {}
    for action, delta, selected, sign in (
        (1, h_delta, fm(6), -1),
        (2, h_delta, fm(6), 1),
        (3, v_delta, fm(5), -1),
        (4, v_delta, fm(5), 1),
    ):
        axis_units[action] = unit(
            mul(delta, sign) + [(selected, 4)],
            sign * axis_bias - 4 - deadzone,
        )

    p["bo"][:] = -3
    p["bo"][0] = -100

    for dim, negative_action, positive_action in (
        ("y", 1, 2),
        ("x", 3, 4),
    ):
        for positive, negative in dimension_pairs[dim]:
            p["Wo"][positive_action, positive] += 10
            p["Wo"][positive_action, negative] -= 10
            p["Wo"][negative_action, positive] -= 10
            p["Wo"][negative_action, negative] += 10

    for action in (1, 2, 3, 4):
        p["bo"][action] = 0
        p["Wo"][action, selected_v] -= 20
        p["Wo"][action, selected_h] -= 20
        p["Wo"][action, axis_units[action]] += 1000

    # Cycle objects after an axis is centered, or when piece displacement
    # is approximately zero.
    p["bo"][5] = 0.12
    p["Wo"][5, selected_v] += 15
    p["Wo"][5, selected_h] += 15
    for z in axis_units.values():
        p["Wo"][5, z] -= 100

    # Click a visible piece if no recognizable object is selected.
    p["bo"][6] = 5
    p["Wo"][6, selected_piece] -= 10
    p["Wo"][6, selected_v] -= 20
    p["Wo"][6, selected_h] -= 20
    p["Wc"][0, 17] = 1

    return p
