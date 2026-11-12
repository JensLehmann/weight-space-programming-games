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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz = p["Wz"]
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo = p["Wo"]

    # Candidate recolored-wall colors.
    colors = [4, 6, 7, 8, 11, 12, 13, 14, 15]
    n = len(colors)

    for i, color in enumerate(colors):
        W1[i, color, 2, 2] = 1

    # Candidate pixels immediately outside each player's four edges.
    edge1 = 9
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for j, (dy, dx) in enumerate(offsets):
        for color in (1, 10):
            W1[edge1 + j, color, 2 + dy, 2 + dx] += 1
            W1[edge1 + j, color, 2, 2] -= 1

    # A clipped ramp gives a left-half indicator.
    ramp_a, ramp_b = 13, 14
    W1[ramp_a, 17, 2, 2] = -63
    W1[ramp_b, 17, 2, 2] = -63
    b1[ramp_a] = 32
    b1[ramp_b] = 31

    # Display edges, used for identifying wall colors and timer rows.
    vertical_border1 = 15
    W1[vertical_border1, 18, 2, 2] = 2
    W1[vertical_border1, 18, 2, 1] = -1
    W1[vertical_border1, 18, 2, 3] = -1

    horizontal_border1 = 16
    W1[horizontal_border1, 18, 2, 2] = 2
    W1[horizontal_border1, 18, 1, 2] = -1
    W1[horizontal_border1, 18, 3, 2] = -1

    player1, selected1 = 17, 18
    W1[player1, 1, 2, 2] = 1
    W1[player1, 10, 2, 2] = 1
    W1[selected1, 1, 2, 2] = 1

    # Player-masked x, 1-x, y, and 1-y.
    coord1 = 19
    for j, axis in enumerate((17, 17, 16, 16)):
        out = coord1 + j
        W1[out, 1, 2, 2] = 1
        W1[out, 10, 2, 2] = 1
        if j % 2 == 0:
            W1[out, axis, 2, 2] = 1
            b1[out] = -1
        else:
            W1[out, axis, 2, 2] = -1

    # Expand the inset color-9 block marking by one display pixel.
    block1 = 23
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            W1[block1, 9, 2 + dy, 2 + dx] = 1

    # Second layer: retain colors and identify colors at row edges.
    for i in range(n):
        W2[i, i, 1, 1] = 1
        W2[n + i, i, 1, 1] = 1
        W2[n + i, vertical_border1, 1, 1] = 1
        b2[n + i] = -1

    edge2 = 18
    for j in range(4):
        W2[edge2 + j, edge1 + j, 1, 1] = 1

    side2 = 22
    player2 = 23
    selected2 = 24
    coord2 = 25
    horizontal_border2 = 29
    block2 = 30

    W2[side2, ramp_a, 1, 1] = 1
    W2[side2, ramp_b, 1, 1] = -1
    W2[player2, player1, 1, 1] = 1
    W2[selected2, selected1, 1, 1] = 1

    for j in range(4):
        W2[coord2 + j, coord1 + j, 1, 1] = 1

    W2[horizontal_border2, horizontal_border1, 1, 1] = 1
    W2[block2, block1, 1, 1] = 1

    # Wall channels compare each pixel with the edge colors on its row.
    for i in range(n):
        W3[i, i] = 1
        W3[i, 32 + n + i] = 1
        b3[i] = -1

    # Split each player-edge indicator into left and right halves.
    gates3 = 9
    for direction in range(4):
        left = gates3 + 2 * direction
        right = left + 1
        W3[left, edge2 + direction] = 1
        W3[left, side2] = 1
        b3[left] = -1
        W3[right, edge2 + direction] = 1
        W3[right, side2] = -1

    source_left = 17
    source_right = 18
    selected3 = 19
    coord3 = 20
    horizontal_border3 = 24
    block3 = 25

    W3[source_left, player2] = 1
    W3[source_left, side2] = 1
    b3[source_left] = -1
    W3[source_right, player2] = 1
    W3[source_right, side2] = -1
    W3[selected3, selected2] = 1

    for j in range(4):
        W3[coord3 + j, coord2 + j] = 1

    W3[horizontal_border3, horizontal_border2] = 1
    W3[block3, block2] = 1

    # Two flood fields. For each field, h[base] - h[base+1]
    # is a capped binary reachability signal, while h[base+2]
    # accumulates the time for which a pixel has been reached.
    for base, source in ((0, source_left), (3, source_right)):
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            for out in (base, base + 1):
                Wh[out, base, 1 + dy, 1 + dx] = 1
                Wh[out, base + 1, 1 + dy, 1 + dx] = -1

        bh[base + 1] = -1
        for out in (base, base + 1):
            Uh[out, source] = 1
            for wall in range(n):
                Uh[out, wall] = -7
            Uh[out, horizontal_border3] = -7
            Uh[out, block3] = -7

        Wh[base + 2, base + 2, 1, 1] = 1
        Wh[base + 2, base, 1, 1] = 1
        Wh[base + 2, base + 1, 1, 1] = -1

    # Read the opposite flood field at each candidate edge. Including
    # the current player cell gives a no-movement fallback when an
    # edge is blocked.
    for direction in range(4):
        for side in range(2):
            j = 2 * direction + side
            out = 6 + j
            opposite_time = 5 if side == 0 else 2
            own_source = source_left if side == 0 else source_right

            Wh[out, opposite_time, 1, 1] = 1
            Uh[out, gates3 + j] = 200
            Uh[out, own_source] = 200
            for wall in range(n):
                Uh[out, wall] = -200
            Uh[out, horizontal_border3] = -200
            Uh[out, block3] = -200
            bh[out] = -200

    for j in range(8):
        Wz[j, 6 + j] = 1

    for j in range(4):
        Wz[8 + j, 32 + coord3 + j] = 1

    Wz[12, 32 + selected3] = 1

    # Store the preceding player-coordinate extrema.
    coordinate_scale = 0.1
    for j in range(4):
        Wmz[j, 8 + j] = coordinate_scale

    # Estimate whether the previous action changed the extrema.
    directions = [
        {2: -1, 3: 1},   # up
        {2: 1, 3: -1},   # down
        {0: 1, 1: 1},    # outward
        {0: -1, 1: -1},  # inward
    ]
    for a, terms in enumerate(directions):
        unit = 4 + a
        for j, sign in terms.items():
            Wmz[unit, 8 + j] = 200 * coordinate_scale * sign
            Wm[unit, j] = -200 * sign
        Wma[unit, 1 + a] = 100
        bm[unit] = -101

    # Record the previous direction.
    for a in range(4):
        Wma[8 + a, 1 + a] = 5

    # Directional scores, with horizontal controls mirrored.
    Wo[1, 0] = 1
    Wo[1, 1] = 1
    Wo[2, 2] = 1
    Wo[2, 3] = 1
    Wo[3, 4] = 1
    Wo[3, 7] = 1
    Wo[4, 6] = 1
    Wo[4, 5] = 1

    # Penalize retrying a failed action and immediate reversals.
    reverse = [1, 0, 3, 2]
    for a in range(4):
        action = 1 + a
        Wo[action, 128 + 4 + a] += 30
        Wo[action, 128 + 8 + a] -= 30
        Wo[action, 128 + 8 + reverse[a]] -= 8

    # Deselect a selected movable block by clicking a player.
    Wo[6, 12] = 500
    p["Wc"][0, selected3] = 1

    return p
