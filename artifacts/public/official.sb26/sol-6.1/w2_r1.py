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
    colours = (1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    def pixel(out, dy, dx, ids, weight=1.0):
        for colour in ids:
            p["W1"][out, colour, dy + 2, dx + 2] += weight

    def detector(out, terms):
        for dy, dx, ids in terms:
            pixel(out, dy, dx, ids)
        p["b1"][out] = 1.0 - len(terms)

    def scalar_pixel(out, dy, dx):
        for colour in range(16):
            pixel(out, dy, dx, (colour,), float(colour))

    # Target anchor: the upper-left interior pixel of a target ring.
    target_border = [
        (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        (0, -1), (1, -1), (2, -1),
    ]
    target_core = [(0, 0), (0, 1), (1, 0), (1, 1)]
    detector(
        0,
        [(dy, dx, colours) for dy, dx in target_border]
        + [(dy, dx, (5,)) for dy, dx in target_core],
    )
    scalar_pixel(1, -1, -1)

    p["W1"][2, 17, 2, 2] = 63
    p["W1"][3, 16, 2, 2] = 63
    for out, threshold in ((4, 53), (5, 54)):
        p["W1"][out, 16, 2, 2] = 63
        p["b1"][out] = -threshold

    # Solid and hollow movable-tile anchors.
    square = [
        (dy, dx) for dy in range(-1, 3) for dx in range(-1, 3)
    ]
    ring = [
        (dy, dx) for dy, dx in square
        if dy in (-1, 2) or dx in (-1, 2)
    ]
    margin = [(-2, 0, (0, 4)), (0, -2, (0, 4))]
    detector(6, [(dy, dx, colours) for dy, dx in square] + margin)
    detector(7, [(dy, dx, colours) for dy, dx in ring] + margin)
    scalar_pixel(8, -1, -1)

    detector(
        9,
        [(0, 0, (2,)), (0, 1, (2,)),
         (1, 0, (2,)), (1, 1, (2,)),
         (-1, 0, (4,)), (0, -1, (4,))],
    )

    selection_border = (
        [(-2, dx) for dx in range(-2, 3)]
        + [(dy, -2) for dy in range(-1, 3)]
    )
    detector(10, [(dy, dx, (0,)) for dy, dx in selection_border])

    # Vertical stripes encode container borders and target side borders.
    for out, colour in enumerate(colours, 11):
        detector(out, [(dy, 0, (colour,)) for dy in range(-2, 3)])

    for out, threshold in ((22, 52), (23, 53), (24, 54)):
        p["W1"][out, 16, 2, 2] = 63
        p["b1"][out] = -threshold
    p["W1"][25, 3, 2, 2] = 1
    p["W1"][26, 18, 2, 2] = 1
    p["W1"][27, 5, 2, 2] = 1

    # Central-slot tie-breaker for the three-slot structural heuristic.
    for out, threshold in ((29, 30), (30, 31), (31, 32)):
        p["W1"][out, 17, 2, 2] = 63
        p["b1"][out] = -threshold

    def second(out, coefficients, bias=0.0):
        for channel, weight in coefficients.items():
            p["W2"][out, channel, 1, 1] = weight
        p["b2"][out] = bias

    second(0, {0: 1})
    second(1, {1: 1, 0: 32}, -32)
    second(2, {2: 1, 0: 64}, -64)
    second(3, {6: 1, 4: 1, 5: -1}, -1)
    second(4, {7: 1, 4: 1, 5: -1}, -1)
    second(5, {8: 1, 7: 32}, -32)
    second(6, {9: 1, 4: -1, 5: 1})
    second(7, {10: 1, 4: 1, 5: -1}, -1)
    second(8, {
        out: float(16 - colour)
        for out, colour in enumerate(colours, 11)
    })
    second(9, {2: 1})
    second(10, {10: 1})
    second(11, {25: 1, 22: 1, 23: -2, 24: 1}, -1)
    second(12, {3: 1})
    second(13, {7: 1, 6: -1, 4: 1, 5: -1}, -1)

    # Selected supply colour. The half-step distinguishes calls from
    # literals when applying exact container-match bonuses.
    second(
        14,
        {8: 1, 7: 32, 10: 32, 4: 32, 5: -32, 6: 0.5},
        -96.5,
    )
    second(
        15,
        {9: 1, 29: 1, 30: -2, 31: 1, 4: -1, 5: 1},
        -1,
    )

    # Detect same-colour neighbouring target borders across a dark gap.
    # Row maxima remove dependence on which repeated pair supplies it.
    for out, stripe in enumerate(range(11, 22), 17):
        p["W2"][out, stripe, 1, 0] = 1
        p["W2"][out, stripe, 1, 2] = 1
        p["W2"][out, 27, 1, 1] = 1
        p["b2"][out] = -2

    def axial(out, coefficients, bias=0.0):
        for channel, weight in coefficients.items():
            p["W3"][out, channel] = weight
        p["b3"][out] = bias

    RM, CM = 32, 64
    axial(0, {CM + 0: 1})                 # Target columns.
    axial(1, {CM + 1: 1})                 # Target colours.
    axial(2, {CM + 2: 1})
    axial(3, {CM + 3: 1})                 # Remaining-solid columns.
    axial(4, {3: 1, RM + 7: -1})          # Unselected supply solids.
    axial(5, {6: 1})                       # Board-empty anchors.
    axial(6, {5: 1})                       # Anchored movable colour.
    axial(7, {RM + 8: -1, 6: 100}, -84)   # Container colour at empties.
    axial(8, {9: 1})                       # Column coordinate.
    axial(9, {10: 1})                      # Selection marker.
    axial(10, {11: 1})                     # Spent-energy pixels.
    axial(11, {4: 1, RM + 7: -1})
    axial(12, {13: 1, RM + 7: -1})        # Unselected supply calls.
    axial(14, {RM + 14: 1})                # Selected colour along its row.
    axial(15, {15: 1})
    axial(16, {RM + out: 0.5 for out in range(17, 28)})

    def recurrent(out, source, weight=1.0, dx=0, dy=0):
        p["Wh"][out, source, dy + 1, dx + 1] += weight

    def drive(out, feature, weight=1.0):
        p["Uh"][out, feature] += weight

    # Inclusive supply prefix and suffix counts.
    recurrent(0, 0, dx=-1)
    drive(0, 3)
    recurrent(1, 1, dx=1)
    drive(1, 3)

    # Inclusive target suffix count.
    recurrent(2, 2, dx=1)
    drive(2, 0)

    # Broadcast the repeated-target mode downward. A six-pixel target
    # border contributes two half-strength stripe detections.
    recurrent(13, 13, dy=-1)
    drive(13, 16)

    # Shared-block completion signal.
    recurrent(12, 13, 3)
    recurrent(12, 0, -1)
    recurrent(12, 1, -1)
    drive(12, 3)

    # Rank equality triangle. On completing the shared body, correction
    # by three restores suffix-target selection rather than disabling it.
    for out, offset in ((3, 1), (4, 0), (5, -1)):
        recurrent(out, 2)
        recurrent(out, 0, -1)
        recurrent(out, 1, -1)
        recurrent(out, 13, -3)
        recurrent(out, 12, 3)
        drive(out, 3)
        drive(out, 0, 128)
        p["bh"][out] = offset - 128

    recurrent(6, 3, 32)
    recurrent(6, 4, -64)
    recurrent(6, 5, 32)
    drive(6, 1)
    p["bh"][6] = -32

    # Horizontal broadcast of the rank-selected target colour.
    recurrent(7, 7, dx=-1)
    recurrent(7, 6)
    recurrent(8, 8, dx=1)
    recurrent(8, 6)

    recurrent(9, 7)
    recurrent(9, 8)
    recurrent(9, 6, -1)
    drive(9, 6, -1)

    # Actual selected colour, broadcast upward from the supply row.
    recurrent(14, 14, dy=1)
    drive(14, 14)

    recurrent(11, 14)
    recurrent(11, 13)
    drive(11, 7, -1)

    # Calls strictly outrank suffix literals once the shared body is
    # populated, but receive no bonus before that point.
    recurrent(10, 13, 30000)
    recurrent(10, 0, -10000)
    recurrent(10, 1, -10000)
    drive(10, 3, 10000)
    drive(10, 12, 100000)
    p["bh"][10] = -100000

    # Container-match triangle with an endpoint-before-centre preference.
    recurrent(15, 11, -8)
    recurrent(15, 14, 4)
    recurrent(15, 13, 4)
    drive(15, 7, -4)
    drive(15, 5, 10)
    drive(15, 15, -0.75)
    p["bh"][15] = -9

    wc = p["Wc"][0]
    wc[4] = 5000
    wc[5] = 3000
    wc[12] = 4500
    wc[8] = -1

    # Absolute differences use |d| = 2*relu(d) - d.
    wc[32 + 9] = -200
    wc[32 + 7] = 100
    wc[32 + 8] = 100
    wc[32 + 6] = -100
    wc[6] = -100

    wc[32 + 11] = -4
    wc[32 + 14] = 2
    wc[32 + 13] = 2
    wc[7] = -2
    wc[32 + 10] = 1
    wc[32 + 15] = 1000

    # Pooling order is h.max, h.mean, f3.max, f3.mean.
    F3_MAX = 32
    p["Wz"][0, F3_MAX + 5] = 1
    p["Wz"][1, F3_MAX + 10] = 1
    p["Wz"][2, F3_MAX + 9] = 1

    # Episode-start latch allows one reset of an already disturbed start.
    p["bm"][0] = 10
    p["Wm"][1, 0] = 10

    p["bo"][:] = -1000
    p["bo"][6] = 0
    p["bo"][5] = 1
    p["Wo"][5, 0] = -2

    p["bo"][0] = -50
    p["Wo"][0, 1] = 100
    p["Wo"][0, 2] = 100
    p["Wo"][0, 128 + 1] = -300

    return p
