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
    tiles = (1, 2, 9, 10, 11, 12, 15)

    def w1(out, inp, dy=0, dx=0, value=1.0):
        if isinstance(inp, int):
            inp = (inp,)
        for channel in inp:
            p["W1"][out, channel, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # ----------------------------------------------------------
    # First convolution: local visual primitives.
    # ----------------------------------------------------------

    w1(0, tiles)

    for colour, small, large, marker, plane in (
        (11, 1, 3, 5, 7),
        (12, 2, 4, 6, 8),
    ):
        # Anchor of a solid 2x2 component.
        for dy in (0, 1):
            for dx in (0, 1):
                w1(small, colour, dy, dx)
        w1(small, colour, -1, 0, -4.0)
        w1(small, colour, 0, -1, -4.0)
        p["b1"][small] = -3.0

        # Interior anchor of a solid 4x4 component.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w1(large, colour, dy, dx)
        w1(large, colour, -2, 0, -9.0)
        w1(large, colour, 0, -2, -9.0)
        p["b1"][large] = -8.0

        # Separated marker corners and intervening background.
        for dy in (-1, 2):
            for dx in (-1, 2):
                w1(marker, colour, dy, dx)
        w1(marker, 4, -1, 0)
        w1(marker, 4, 0, -1)
        p["b1"][marker] = -5.0

        w1(plane, colour)

    w1(9, 16)
    w1(10, 17)

    samples = {
        11: (0, 2),
        12: (0, -2),
        13: (2, 0),
        14: (-2, 0),
        15: (2, -2),
        16: (2, 2),
        20: (-2, -2),
        21: (-2, 2),
    }
    for channel, (dy, dx) in samples.items():
        w1(channel, tiles, dy, dx)

    for channel, (dy, dx) in zip(
        (17, 18, 19), ((-2, -2), (-2, 2), (2, 2))
    ):
        w1(channel, 11, dy, dx)

    w1(22, 14, 0, 2)

    button_patterns = {
        23: (
            (14, 14, 4),
            (14, 14, 14),
            (14, 14, 14),
            (14, 14, 4),
        ),
        24: (
            (4, 8, 8),
            (8, 8, 8),
            (8, 8, 8),
            (4, 8, 8),
        ),
        25: (
            (4, 14, 14, 4),
            (14, 14, 14, 14),
            (14, 14, 14, 14),
        ),
        26: (
            (8, 8, 8, 8),
            (8, 8, 8, 8),
            (4, 8, 8, 4),
        ),
    }
    for channel, pattern in button_patterns.items():
        for y, row in enumerate(pattern):
            for x, colour in enumerate(row):
                w1(channel, colour, y - 1, x - 1)
        p["b1"][channel] = -11.0

    # Solid button interiors reject the thin HUD.
    for channel, colour in ((27, 14), (28, 8)):
        for dy in (0, 1):
            for dx in (0, 1):
                w1(channel, colour, dy, dx)
        p["b1"][channel] = -3.0

    w1(30, 11, 2, 0)

    # ----------------------------------------------------------
    # Second convolution: graph roles and coordinates.
    # ----------------------------------------------------------

    def neighbour(out, sample, value=1.0):
        dy, dx = samples[sample]
        w2(out, sample, dy // 2, dx // 2, value)

    def small_goal(out, colour=11, value=1.0):
        small, large = (1, 3) if colour == 11 else (2, 4)
        w2(out, small, value=value)
        w2(out, large, 1, 1, -value)

    def square_marker(out):
        w2(out, 5, value=5.0)
        for sample in (12, 13, 15):
            neighbour(out, sample)
        for sample in (11, 14):
            neighbour(out, sample, -1.0)

    # Four-node loop motif and its prepared phase.
    square_marker(0)
    p["b2"][0] = -7.0

    square_marker(1)
    w2(1, 30, 1, 0)
    p["b2"][1] = -8.0

    # Goal at a branch junction.
    small_goal(2, value=4.0)
    for sample in (11, 13):
        neighbour(2, sample)
    for sample in (12, 14, 16):
        neighbour(2, sample, -1.0)
    p["b2"][2] = -5.0

    # Goal at the lower parking endpoint.
    small_goal(3, value=4.0)
    neighbour(3, 14)
    for sample in (11, 12, 13, 15, 16):
        neighbour(3, sample, -1.0)
    p["b2"][3] = -4.0

    # Goal in the middle of a vertical branch.
    small_goal(4, value=4.0)
    for sample in (14, 13):
        neighbour(4, sample)
    for sample in (11, 12):
        neighbour(4, sample, -1.0)
    p["b2"][4] = -5.0

    for out, inp in ((5, 23), (6, 24), (7, 25), (8, 26)):
        w2(out, inp)

    w2(9, 27)
    w2(10, 28)
    w2(11, 0)

    # Occupied matching-marker coordinate at either scale.
    gate = 16.0
    w2(12, 5, value=3.0 * gate)
    w2(12, 17, -1, -1, gate)
    w2(12, 18, -1, 1, gate)
    w2(12, 19, 1, 1, gate)
    w2(12, 7, value=gate)
    w2(12, 10)
    p["b2"][12] = -4.0 * gate

    w2(13, 3, value=gate)
    w2(13, 10, value=-1.0)
    p["b2"][13] = 1.0 - gate

    w2(14, 10)

    # Tile-gated x and 1-x for axial extrema.
    w2(15, 0, value=2.0)
    w2(15, 10)
    p["b2"][15] = -2.0

    w2(16, 0, value=2.0)
    w2(16, 10, value=-1.0)
    p["b2"][16] = -1.0

    small_goal(17, colour=12)
    small_goal(18)
    w2(18, 5, value=-1.0)

    # Clipped vertical connectivity, represented as a difference.
    for out in (20, 21):
        neighbour(out, 13)
        neighbour(out, 14)
    p["b2"][21] = -1.0

    # Adjacent horizontal left/right control pair.
    w2(22, 24)
    w2(22, 22)
    p["b2"][22] = -1.0

    # Shared vertices with diagonal or orthogonal connectivity.
    for out, colour, neighbours in (
        (23, 12, (15, 16, 20, 21)),
        (24, 11, (15, 16, 20, 21)),
        (25, 12, (11, 12, 13, 14)),
        (26, 11, (11, 12, 13, 14)),
    ):
        small_goal(out, colour=colour, value=5.0)
        for sample in neighbours:
            neighbour(out, sample)
        p["b2"][out] = -8.0

    # ----------------------------------------------------------
    # Axial features and routing requests.
    # ----------------------------------------------------------

    for channel in range(5):
        p["W3"][channel, 32 + channel] = 1.0

    for channel in range(5, 9):
        p["W3"][channel, channel] = 1.0

    p["W3"][9, 10] = 1.0
    p["W3"][10, 9] = 1.0

    p["W3"][11, 9] = 1.0
    p["W3"][11, 64 + 11] = 1.0
    p["b3"][11] = -1.0

    # Endpoint-buffer request.
    p["W3"][12, 9] = 10.0
    p["W3"][12, 32 + 12] = 1.0
    p["W3"][12, 14] = -1.0
    p["b3"][12] = -10.0 + 12.0 / 63.0

    # Triangular spacing penalty on a short occupied row.
    for channel, threshold in zip((13, 14, 15), (9.0, 16.0, 23.0)):
        p["W3"][channel, 9] = 10.0
        p["W3"][channel, 32 + 12] = 1.0
        p["W3"][channel, 32 + 13] = 1.0
        p["b3"][channel] = -11.0 - threshold / 63.0

    slope = 128.0
    logical_gate = 512.0
    xmax = 32 + 15
    inv_xmin = 32 + 16

    def left_lobe_features(first, goal_channel):
        # Nonvertical arc on the left half of the occupied row.
        q = first
        p["W3"][q, goal_channel] = logical_gate
        p["W3"][q, 20] = -logical_gate
        p["W3"][q, 21] = logical_gate
        p["W3"][q, xmax] = 0.5 * slope
        p["W3"][q, inv_xmin] = -0.5 * slope
        p["W3"][q, 14] = -slope
        p["b3"][q] = (
            -logical_gate + slope * (0.5 - 0.5 / 63.0)
        )

        # Outer-left arc, including vertical portions.
        q = first + 1
        p["W3"][q, goal_channel] = logical_gate
        p["W3"][q, inv_xmin] = -slope
        p["W3"][q, 14] = -slope
        p["b3"][q] = (
            -logical_gate + slope * (1.0 + 1.5 / 63.0)
        )

        # A triangular window for the inner-right vertical arc.
        thresholds = (
            (0.5, -0.5, 0.5),
            (0.75, -0.25, 0.25 - 0.75 / 63.0),
            (1.0, 0.0, -1.5 / 63.0),
        )
        for offset, (hi, inv_lo, constant) in enumerate(thresholds):
            q = first + 2 + offset
            p["W3"][q, goal_channel] = logical_gate
            p["W3"][q, 20] = logical_gate
            p["W3"][q, 21] = -logical_gate
            p["W3"][q, 14] = slope
            p["W3"][q, xmax] = -slope * hi
            p["W3"][q, inv_xmin] = -slope * inv_lo
            p["b3"][q] = (
                -2.0 * logical_gate - slope * constant
            )

    left_lobe_features(16, 17)
    left_lobe_features(21, 18)

    p["W3"][26, 24] = 1.0
    p["W3"][26, 26] = 1.0
    p["W3"][27, 23] = 1.0
    p["W3"][27, 25] = 1.0
    p["W3"][28, 17] = 1.0

    p["W3"][29, 22] = 1.0

    p["W3"][30, 22] = 2.0
    p["W3"][30, 14] = 1.0
    p["b3"][30] = -2.0

    p["W3"][31, 22] = 2.0
    p["W3"][31, 14] = -1.0
    p["b3"][31] = -1.0

    # ----------------------------------------------------------
    # Recurrent fields.
    #
    # h0: branch motif field
    # h1: prepared-phase field, with exactly the same propagation
    # h2-h3: junction occupancy
    # h4-h5: actual lower parking occupancy
    # h6-h7: middle-branch occupancy / orange-goal presence
    # h8-h9: loop-routing request, kept separate from parking
    # ----------------------------------------------------------

    # Linear matched fields: when the phase is prepared, h1 == h0.
    # Their difference therefore represents the unprepared phase.
    for channel in (0, 1):
        for dy in (-1, 0, 1):
            wh(channel, channel, dy, 0, 1.0 / 3.0)
        p["Uh"][channel, channel] = 1.0

    JUNCTION = 2
    PARKED = 4
    MIDDLE = 6
    ROUTE = 8

    # Bounded Boolean broadcast:
    # the difference of each pair is clip(v, 0, 1).
    for even in (JUNCTION, PARKED, MIDDLE, ROUTE):
        odd = even + 1
        for out in (even, odd):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    wh(out, even, dy, dx, 2.0)
                    wh(out, odd, dy, dx, -2.0)
        p["bh"][even] = -0.25
        p["bh"][odd] = -1.25

    def source(even, channel, weight=1.0):
        p["Uh"][even, channel] += weight
        p["Uh"][even + 1, channel] += weight

    source(JUNCTION, 2)
    source(JUNCTION, 27)

    # No loop-routing sources are permitted in this parking field.
    source(PARKED, 3)

    source(MIDDLE, 4)
    source(MIDDLE, 28)

    for first in (16, 21):
        for offset, weight in enumerate((1.0, 1.0, 1.0, -2.0, 1.0)):
            source(ROUTE, first + offset, weight)
    source(ROUTE, 26)

    def flag(out, even, value):
        wh(out, even, value=value)
        wh(out, even + 1, value=-value)

    # ----------------------------------------------------------
    # Six spatial control outputs.
    # ----------------------------------------------------------

    # Intersecting loops: prefer the left paired control when a
    # routing request exists, unless orange occupies a shared node.
    flag(10, MIDDLE, 4.0)
    flag(10, JUNCTION, -4.0)
    flag(10, ROUTE, 1.0)
    wh(10, 0, value=-64.0)
    p["Uh"][10, 29] = 4.0
    p["Uh"][10, 30] = -0.1
    p["bh"][10] = -8.0

    # Otherwise use the rightmost paired left control.
    flag(14, MIDDLE, 1.0)
    wh(14, 0, value=-64.0)
    p["Uh"][14, 29] = 1.0
    p["Uh"][14, 31] = -0.1
    p["bh"][14] = -1.0

    control_gate = 128.0

    # Main left: finish from a junction in the prepared phase.
    wh(11, 1)
    flag(11, JUNCTION, control_gate)
    p["Uh"][11, 6] = control_gate
    p["bh"][11] = -2.0 * control_gate

    # Main left: adjust phase while the goal is genuinely parked.
    wh(12, 0)
    wh(12, 1, value=-1.0)
    flag(12, PARKED, control_gate)
    p["Uh"][12, 6] = control_gate
    p["bh"][12] = -2.0 * control_gate

    # Side right: park an unprepared junction goal, or advance a
    # goal from the middle of the branch back to its junction.
    wh(13, 0)
    wh(13, 1, value=-1.0)
    flag(13, JUNCTION, control_gate)
    flag(13, MIDDLE, 2.0 * control_gate)
    p["Uh"][13, 7] = 4.0 * control_gate
    p["bh"][13] = -5.0 * control_gate

    # Side left: retrieve the parked goal once phase is prepared.
    wh(15, 1)
    flag(15, PARKED, control_gate)
    p["Uh"][15, 8] = control_gate
    p["bh"][15] = -2.0 * control_gate

    # ----------------------------------------------------------
    # Click readout and action selection.
    # ----------------------------------------------------------

    p["Wc"][0, 5] = 3.0
    p["Wc"][0, 9] = 1.0
    p["Wc"][0, 10] = 0.1
    p["Wc"][0, 11] = 2.0
    p["Wc"][0, 12] = 100.0

    p["Wc"][0, 13] = -500.0
    p["Wc"][0, 14] = 1000.0
    p["Wc"][0, 15] = -500.0

    p["Wc"][0, 32 + 10] = 100.0
    for channel in (11, 12, 13, 14, 15):
        p["Wc"][0, 32 + channel] = 20.0

    p["bo"].fill(-1.0)
    p["bo"][0] = -2.0
    p["bo"][6] = 1.0

    return p
