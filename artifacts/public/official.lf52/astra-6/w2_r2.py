import math
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

    def sample(ch, dy, dx, colours, weight=1.0):
        if isinstance(colours, int):
            colours = (colours,)
        for colour in colours:
            p["W1"][ch, colour, dy + 2, dx + 2] += weight

    def detector(ch, positive, negative=()):
        for dy, dx, colours in positive:
            sample(ch, dy, dx, colours)
        for dy, dx, colours in negative:
            sample(ch, dy, dx, colours, -1.0)
        p["b1"][ch] = 1.0 - len(positive)

    # Tile-local anchor: row 2, column 2.
    circle = [
        (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
    core = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Green, red, blue, and failure-grey pieces.
    for ch, colour in ((0, 14), (1, 8), (2, 9), (8, 2)):
        detector(
            ch,
            [(dy, dx, colour) for dy, dx in circle],
            [(dy, dx, colour) for dy, dx in corners],
        )

    # Destination marker.
    ring = [point for point in circle if point not in core]
    detector(
        3,
        [(dy, dx, 2) for dy, dx in ring],
        [(dy, dx, 2) for dy, dx in core],
    )

    # Selected-piece surround.
    outline = [
        (-2, 0), (-2, 1), (0, -2), (1, -2),
        (-1, -1), (-1, 2), (2, -1), (2, 2),
    ]
    detector(
        4,
        [(dy, dx, 3) for dy, dx in outline]
        + [(0, 0, (14, 8, 9, 2))],
    )

    # Empty ordinary tile and empty orange platform.
    for ch, inside, border in ((5, 1, 0), (6, 12, 11)):
        positive = [
            (dy, dx, inside)
            for dy in range(-1, 3)
            for dx in range(-1, 3)
        ]
        positive += [(-2, 0, border), (0, -2, border)]
        detector(ch, positive)

    # Permanent purple obstacle.
    box = []
    for dy in range(-2, 2):
        for dx in range(-1, 3):
            if dy in (-2, 1) or dx in (-1, 2):
                colour = 15
            elif (dy, dx) == (-1, 0):
                colour = 0
            else:
                colour = 7
            box.append((dy, dx, colour))
    detector(7, box)

    # Piece carried by an orange platform.
    detector(
        9,
        [(dy, dx, (14, 8, 9, 2)) for dy, dx in circle]
        + [(dy, dx, 12) for dy, dx in corners],
    )

    # Action order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d, (dy, dx) in enumerate(directions):
        exit_pattern = [
            (0, 0, 11),
            (dy, dx, 5),
            (2 * dy, 2 * dx, 5),
        ]
        detector(
            10 + d,
            exit_pattern,
            [(-dy, -dx, 11)],
        )
        detector(
            14 + d,
            exit_pattern + [(-dy, -dx, (14, 8, 9, 2))],
        )

    # Upper exit obscured by a carried purple obstacle.
    detector(
        18,
        [(0, -2, 11), (0, 0, 15), (-1, 0, 5), (-2, 0, 5)],
    )

    p["W1"][19, 16, 2, 2] = 1.0
    p["W1"][20, 17, 2, 2] = 1.0
    p["W1"][21, 1, 2, 2] = 1.0

    for ch in range(22):
        p["W2"][ch, ch, 1, 1] = 1.0
    p["W2"][10, 18, 1, 1] = 1.0

    def feature(out_ch, in_ch, weight=1.0):
        p["W3"][out_ch, in_ch] += weight

    # f3:
    # 0 green/grey, 1 red, 2 blue, 3 marker, 4 selection,
    # 5 empty anchor, 6 box, 7 row, 8 column, 9 failure-grey.
    feature(0, 0)
    feature(0, 8)
    feature(1, 1)
    feature(2, 2)
    feature(3, 3)
    feature(4, 4)
    for ch in (5, 6, 3):
        feature(5, ch)
    feature(6, 7)
    feature(7, 19)
    feature(8, 20)
    feature(9, 8)

    for ch in range(10, 18):
        feature(ch, ch)

    # Small coordinate tie-breaker, strictly gated to empty anchors.
    for ch in (5, 6, 3):
        feature(18, ch, 8.0)
    feature(18, 19, 0.010)
    feature(18, 20, 0.020)
    p["b3"][18] = -8.0

    feature(19, 9)       # carried piece
    feature(20, 6)       # empty platform
    feature(21, 21)      # ordinary floor pixels

    # h15: bounded diffusion of floor pixels.
    # The recurrent kernel has total mass 0.92.
    for dy, dx in directions:
        p["Wh"][15, 15, 1 + dy, 1 + dx] = 0.23
    p["Uh"][15, 21] = 0.10

    # h0:4: landing-site rays.
    # h4:8: relays restricted to intervening pieces.
    # h8:12: rays travelling from the relays towards origins.
    relay_gate = 256.0

    # Ray direction is opposite to the contemplated jump direction.
    box_threshold = [0.85, 1.50, 0.85, 1.50]
    blue_threshold = [0.92, 0.75, 0.85, 1.10]

    for d, (dy, dx) in enumerate(directions):
        landing_ray = d
        midpoint = 4 + d
        jump_ray = 8 + d
        ky, kx = 1 - dy, 1 - dx

        p["Wh"][landing_ray, landing_ray, ky, kx] = 0.5
        p["Wh"][landing_ray, 12, 1, 1] = 0.07
        p["Uh"][landing_ray, 5] = 1.0
        p["Uh"][landing_ray, 18] = 1.0
        p["Uh"][landing_ray, 20] = 0.20

        # Pieces stop the first ray. Its maximum intended
        # amplitude is far below this blocking margin.
        for ch in (0, 1, 2, 6):
            p["Uh"][landing_ray, ch] = -16.0

        # Six pixels between anchors: compensate five incoming
        # attenuation steps, then apply the midpoint threshold.
        p["Wh"][midpoint, landing_ray, ky, kx] = 32.0
        p["bh"][midpoint] = -relay_gate - 0.9
        p["Uh"][midpoint, 0] = relay_gate + 0.4
        p["Uh"][midpoint, 1] = relay_gate + 0.4
        p["Uh"][midpoint, 2] = (
            relay_gate + 0.9 - blue_threshold[d]
        )
        p["Uh"][midpoint, 6] = (
            relay_gate + 0.9 - box_threshold[d]
        )

        p["Wh"][jump_ray, jump_ray, ky, kx] = 0.5
        p["Wh"][jump_ray, midpoint, 1, 1] = 2.0

        # Incoming signals only: no self-generated relay score.
        p["Wh"][12, jump_ray, ky, kx] = 32.0
        p["Wh"][13, jump_ray, ky, kx] = 32.0

    # Large logical gates prevent short-distance signals at
    # non-anchor pixels from entering the continuation feedback.
    score_gate = 2048.0

    # h12: future utility at empty landing anchors.
    p["bh"][12] = -score_gate - 0.06
    p["Uh"][12, 5] = score_gate
    p["Wh"][12, 15, 1, 1] = 1.5

    # h13: origin utility. Blue pieces may move but are secondary.
    p["bh"][13] = -score_gate - 0.06
    p["Uh"][13, 0] = score_gate
    p["Uh"][13, 1] = score_gate
    p["Uh"][13, 2] = score_gate - 0.15

    # h14: future utility restricted to displayed destinations.
    # The selected origin is not artificially treated as empty.
    p["bh"][14] = -score_gate
    p["Uh"][14, 3] = score_gate
    p["Wh"][14, 12, 1, 1] = 1.0

    # Click selection.
    p["Wc"][0, 32 + 13] = 100.0
    p["Wc"][0, 32 + 14] = 100.0
    p["Wc"][0, 3] = 10000.0
    p["Wc"][0, 0] = 0.02
    p["Wc"][0, 1] = 0.01
    p["Wc"][0, 2] = -0.02
    p["Wc"][0, 7] = 0.001
    p["Wc"][0, 8] = 0.002

    # Global action features.
    p["Wz"][0, 13] = 1.0
    p["Wz"][1, 32 + 3] = 1.0
    for d in range(4):
        p["Wz"][2 + d, 32 + 10 + d] = 1.0
        p["Wz"][6 + d, 32 + 14 + d] = 1.0
    p["Wz"][10, 32 + 9] = 1.0

    p["bo"][:] = -1000.0
    p["bo"][6] = -50.0
    p["Wo"][6, 0] = 1000.0
    p["Wo"][6, 1] = 10000.0
    p["bo"][7] = -12.0

    for d in range(4):
        action = d + 1
        p["bo"][action] = -30.0
        p["Wo"][action, 2 + d] = 30.0
        p["Wo"][action, 6 + d] = 70.0
    p["bo"][3] += 0.25

    # Four persistent heading units.
    for d in range(4):
        p["Wm"][d, d] = 3.0
        p["bm"][d] = 0.02 if d == 2 else -0.02
        for previous_d in range(4):
            p["Wma"][d, previous_d + 1] = (
                12.0 if previous_d == d else -12.0
            )

    clockwise = {0: 3, 3: 1, 1: 2, 2: 0}
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    for candidate in range(4):
        for previous in range(4):
            if candidate == clockwise[previous]:
                preference = 7.0
            elif candidate == previous:
                preference = 3.0
            elif candidate == opposite[previous]:
                preference = -5.0
            else:
                preference = 0.0
            p["Wo"][candidate + 1, 128 + previous] = (
                0.5 * preference
            )

    # Small changing junction tie-breaker.
    angle = 0.19
    gain = 1.10
    cosine = gain * math.cos(angle)
    sine = gain * math.sin(angle)
    p["Wm"][4, 4] = cosine
    p["Wm"][4, 5] = -sine
    p["Wm"][5, 4] = sine
    p["Wm"][5, 5] = cosine
    p["bm"][4] = 0.01
    p["Wo"][1, 128 + 5] = -1.0
    p["Wo"][2, 128 + 5] = 1.0
    p["Wo"][3, 128 + 4] = -1.0
    p["Wo"][4, 128 + 4] = 1.0

    # Grey feedback requests two undos.
    p["Wmz"][8, 10] = 6.0
    p["Wm"][9, 8] = 6.0
    p["Wo"][7, 128 + 8] = 100000.0
    p["Wo"][7, 128 + 9] = 100000.0

    # A decaying failure trace briefly favors changing the
    # transport state instead of immediately replaying the failure.
    p["Wm"][10, 10] = 0.85
    p["Wmz"][10, 10] = 6.0
    p["Wm"][11, 10] = 40.0
    p["bm"][11] = -10.0
    p["bo"][6] -= 10000.0
    p["Wo"][6, 128 + 11] = -10000.0

    return p
