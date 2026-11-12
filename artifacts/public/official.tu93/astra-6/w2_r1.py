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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    cross = [(0, 0)] + directions

    def add_pixel(out, colors, dy, dx, weight=1.0):
        for color in colors:
            p["W1"][out, color, 2 + dy, 2 + dx] += weight

    def center_detector(out, colors):
        for dy, dx in corners:
            add_pixel(out, colors, dy, dx)
        p["b1"][out] = -3.0

    # F1: centres, traversable pixels, and bridge pixels.
    center_detector(0, [0, 8, 9, 12, 13, 14])
    center_detector(1, [2])
    add_pixel(2, [0, 2, 4, 8, 9, 11, 12, 13, 14, 15], 0, 0)
    center_detector(3, [14])
    center_detector(4, [8])
    center_detector(5, [12])
    center_detector(6, [13])
    add_pixel(7, [2], 0, 0)

    for d, (dy, dx) in enumerate(directions):
        # Front markers of red enemies and patrols.
        c = 8 + d
        add_pixel(c, [15, 11], 0, 0)
        add_pixel(c, [8, 12], -dy, -dx)
        add_pixel(c, [8, 12], dy, dx, -1.0)
        p["b1"][c] = -1.0

        # Front markers of activated purple enemies.
        c = 12 + d
        add_pixel(c, [11], 0, 0)
        add_pixel(c, [13], -dy, -dx)
        add_pixel(c, [13], dy, dx, -1.0)
        p["b1"][c] = -1.0

        # First translation for legal-action probes.
        c = 16 + d
        for sy, sx in corners:
            add_pixel(c, [9], sy - dy, sx - dx)
        p["b1"][c] = -3.0

    center_detector(20, [9])

    # Player-masked row and column coordinates.
    for out, coordinate in [(21, 16), (22, 17)]:
        for dy, dx in corners:
            add_pixel(out, [9], dy, dx, 2.0)
        p["W1"][out, coordinate, 2, 2] = 1.0
        p["b1"][out] = -8.0

    p["W1"][23, 16, 2, 2] = 1.0
    p["W1"][24, 17, 2, 2] = 1.0

    # Incident-bridge count at node centres.
    for dy, dx in directions:
        add_pixel(25, [2], 2 * dy, 2 * dx)

    # Horizontal/vertical patrol markers, independent of direction.
    add_pixel(26, [15, 11], 0, 0)
    add_pixel(26, [12], -1, 0)
    add_pixel(26, [12], 1, 0)
    p["b1"][26] = -2.0

    add_pixel(27, [15, 11], 0, 0)
    add_pixel(27, [12], 0, -1)
    add_pixel(27, [12], 0, 1)
    p["b1"][27] = -2.0

    # F2: expand centres into a thin navigation graph.
    for dy, dx in cross:
        p["W2"][0, 0, 1 + dy, 1 + dx] = 1.0
        p["W2"][0, 1, 1 + dy, 1 + dx] = 1.0

    for dst, src in {
        1: 0, 2: 3, 3: 4, 4: 5,
        5: 6, 6: 2, 7: 7,
    }.items():
        p["W2"][dst, src, 1, 1] = 1.0

    for d, (dy, dx) in enumerate(directions):
        p["W2"][8 + d, 8 + d, 1, 1] = 1.0
        p["W2"][8 + d, 12 + d, 1, 1] = 1.0
        p["W2"][12 + d, 16 + d, 1 - dy, 1 - dx] = 1.0

    for c in range(20, 28):
        p["W2"][c, c, 1, 1] = 1.0

    # Coordinates masked by the patrol markers.
    p["W2"][28, 26, 1, 1] = 2.0
    p["W2"][28, 23, 1, 1] = 1.0
    p["b2"][28] = -2.0
    p["W2"][29, 27, 1, 1] = 2.0
    p["W2"][29, 24, 1, 1] = 1.0
    p["b2"][29] = -2.0

    # F3: graph, nodes, objects, directional markers, legal probes.
    p["W3"][0, 0] = 1.0
    p["W3"][0, 6] = 1.0
    p["b3"][0] = -1.0

    for c in range(1, 6):
        p["W3"][c, c] = 1.0

    for d in range(4):
        p["W3"][6 + d, 8 + d] = 1.0
        p["W3"][10 + d, 12 + d] = 1.0
        p["W3"][10 + d, 7] = 1.0
        p["b3"][10 + d] = -1.0

    # Absolute coordinate errors for each six-pixel landing offset.
    tolerance = 0.002
    for d, (dy, dx) in enumerate(directions):
        if dy:
            coordinate = 23
            player_coordinate = 64 + 21
            displacement = 6.0 * dy
        else:
            coordinate = 24
            player_coordinate = 32 + 22
            displacement = 6.0 * dx

        positive = 14 + 2 * d
        negative = positive + 1

        p["W3"][positive, coordinate] = 63.0
        p["W3"][positive, player_coordinate] = -63.0
        p["b3"][positive] = -displacement - tolerance
        p["W3"][negative, coordinate] = -63.0
        p["W3"][negative, player_coordinate] = 63.0
        p["b3"][negative] = displacement - tolerance

    p["W3"][22, 32 + 20] = 1.0
    p["W3"][23, 64 + 20] = 1.0

    # Penalize non-objective leaf nodes.
    p["W3"][24, 1] = 2.0
    p["W3"][24, 25] = -1.0
    for c in (2, 3, 4, 5):
        p["W3"][24, c] = -2.0

    for dst, src in {
        25: 26, 26: 27, 27: 28,
        28: 29, 29: 21, 30: 22,
    }.items():
        p["W3"][dst, src] = 1.0

    # H0..3: directional threat projections.
    for d, (dy, dx) in enumerate(directions):
        p["Wh"][d, d, 1 - dy, 1 - dx] = 1.0
        p["Uh"][d, 6 + d] = 7.0
        p["bh"][d] = -1.0

    # H4-H5: clipped threatened-node indicator.
    graph_gate = 512.0
    for c in (4, 5):
        for d in range(4):
            p["Wh"][c, d, 1, 1] = 1.0
        p["Uh"][c, 1] = graph_gate
        p["bh"][c] = -graph_gate
    p["bh"][5] -= 1.0

    # H6: clock. H7: delayed sources.
    p["Wh"][6, 6, 1, 1] = 1.0
    p["bh"][6] = 1.0

    p["Wh"][7, 6, 1, 1] = 1.0
    p["bh"][7] = -256.0
    p["Uh"][7, 2] = 248.0
    p["Uh"][7, 3] = 192.0
    p["Uh"][7, 5] = 184.0
    p["Uh"][7, 4] = 176.0

    # H8-H9: clipped flood occupancy.
    for c in (8, 9):
        for dy, dx in directions:
            p["Wh"][c, 8, 1 + dy, 1 + dx] = 1.0
            p["Wh"][c, 9, 1 + dy, 1 + dx] = -1.0
        p["Wh"][c, 7, 1, 1] = 1.0
        p["Wh"][c, 4, 1, 1] = -graph_gate
        p["Wh"][c, 5, 1, 1] = graph_gate
        p["Uh"][c, 0] = graph_gate
        p["bh"][c] = -graph_gate
    p["bh"][9] -= 1.0

    # H10: arrival-time accumulator.
    p["Wh"][10, 10, 1, 1] = 1.0
    p["Wh"][10, 8, 1, 1] = 1.0
    p["Wh"][10, 9, 1, 1] = -1.0

    # H11: local danger diffusion.
    for dy, dx in directions:
        p["Wh"][11, 11, 1 + dy, 1 + dx] = 0.24
    p["Wh"][11, 4, 1, 1] = 1.0
    p["Wh"][11, 5, 1, 1] = -1.0
    p["Uh"][11, 0] = 64.0
    p["bh"][11] = -64.0

    # H12..15: scores sampled at the actual landing centres.
    spatial_gate = 4096.0
    for d, (dy, dx) in enumerate(directions):
        c = 12 + d
        p["Wh"][c, 10, 1, 1] = 1.0
        p["Wh"][c, 11, 1, 1] = -5000.0
        p["Uh"][c, 24] = -32.0
        p["Uh"][c, 14 + 2 * d] = -spatial_gate
        p["Uh"][c, 15 + 2 * d] = -spatial_gate
        p["Uh"][c, 23 if dy else 22] = spatial_gate
        p["bh"][c] = 2048.0 - spatial_gate

    # Z0..3 scores; Z4..7 legality.
    for d in range(4):
        p["Wz"][d, 12 + d] = 1.0
        p["Wz"][4 + d, 32 + 10 + d] = 1.0

    # Z8/Z9: straight vertical/horizontal corridors at the player.
    for d in range(4):
        p["Wz"][8, 42 + d] = 1.0 if d < 2 else -1.0
        p["Wz"][9, 42 + d] = -1.0 if d < 2 else 1.0
    p["bz"][8:10] = -1.0

    # Patrol visibility/coordinates and current player coordinates.
    for z, feature in {
        12: 25, 13: 26, 14: 27,
        15: 28, 16: 29, 17: 30,
    }.items():
        p["Wz"][z, 32 + feature] = 1.0

    s = float(np.tanh(1.0))

    # M0..3 previous actions; M4..7 one further action delay.
    for d in range(4):
        p["Wma"][d, 1 + d] = 1.0
        p["Wm"][4 + d, d] = 1.0 / s

    # M16..23: two generations of nearly linear score memory.
    score_scale = 1.0e6
    for d in range(4):
        p["Wmz"][16 + d, d] = 1.0 / score_scale
        p["Wm"][20 + d, 16 + d] = 1.0

    # M8..15: paired thresholds. Their difference activates only when
    # (a) the last three moves alternate,
    # (b) the player is in a straight corridor, and
    # (c) the forward score repeats its value from two decisions ago.
    steepness = 8.0
    gate = 8192.0
    for d in range(4):
        forward = opposite[d]
        corridor = 8 if d < 2 else 9

        for j, edge in enumerate((1.0, -1.0)):
            m = 8 + 2 * d + j
            p["Wmz"][m, forward] = steepness
            p["Wm"][m, 20 + forward] = -steepness * score_scale

            p["Wm"][m, d] = -steepness * gate / s
            p["Wm"][m, 4 + forward] = -steepness * gate / s
            p["Wma"][m, 1 + forward] = -steepness * gate
            p["Wmz"][m, corridor] = -steepness * gate
            p["bm"][m] = steepness * (4.0 * gate + edge)

    # M24..27 retain patrol visibility and its movement-axis coordinate.
    for axis in range(2):
        p["Wmz"][24 + axis, 12 + axis] = 1.0
        p["Wmz"][26 + axis, 14 + axis] = 0.001

    # M28..31: disappeared-patrol warning, restricted to its previous
    # row (horizontal patrol) or column (vertical patrol).
    for axis in range(2):
        for j, edge in enumerate((1.0, -1.0)):
            m = 28 + 2 * axis + j
            p["Wmz"][m, 16 + axis] = steepness * 63.0
            p["Wm"][m, 26 + axis] = -steepness * 63000.0
            p["Wm"][m, 24 + axis] = -steepness * gate / s
            p["Wmz"][m, 12 + axis] = steepness * gate
            p["bm"][m] = steepness * (gate + edge)

    p["bo"][:] = -1.0e6
    p["bo"][0] = 0.1

    for d in range(4):
        action = 1 + d
        p["bo"][action] = -100000.0
        p["Wo"][action, d] = 1.0
        p["Wo"][action, 4 + d] = 100000.0

        p["Wo"][action, 128 + d] = 0.05 / s
        p["Wo"][action, 128 + opposite[d]] = -10.0 / s

        # Six additional points only for detected persistent oscillation.
        p["Wo"][action, 128 + 8 + 2 * d] = -3.0
        p["Wo"][action, 128 + 9 + 2 * d] = 3.0

        # Prefer leaving the movement axis of a newly occluded patrol.
        axis = 1 if d < 2 else 0
        p["Wo"][action, 128 + 28 + 2 * axis] = -24.0
        p["Wo"][action, 128 + 29 + 2 * axis] = 24.0

    return p
