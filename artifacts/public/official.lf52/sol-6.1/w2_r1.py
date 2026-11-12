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

    def detector(channel, positive, negative=()):
        for dy, dx, colour in positive:
            p["W1"][channel, colour, dy + 2, dx + 2] += 2.0
        for dy, dx, colour in negative:
            p["W1"][channel, colour, dy + 2, dx + 2] -= 2.0
        p["b1"][channel] = 1.0 - 2.0 * len(positive)

    # All central tile detectors use sprite pixel (2, 2).
    for channel, colour in enumerate((14, 8, 9)):
        detector(channel, [
            (0, 0, colour), (0, -1, colour),
            (-1, 0, colour), (-1, 1, colour),
            (1, -1, colour), (1, 1, colour),
        ], [(-1, -1, colour)])

    # Permanent pink obstacle.
    detector(3, [
        (0, 0, 7), (0, 1, 7),
        (-1, 0, 0), (-1, 1, 7),
        (0, -1, 15), (1, 0, 15),
    ])

    # Completely empty white and orange pads.
    for channel, interior, border in ((4, 1, 0), (5, 12, 11)):
        positive = [
            (dy, dx, interior)
            for dy in range(-1, 3)
            for dx in range(-1, 3)
        ]
        positive += [(-2, 0, border), (0, -2, border)]
        detector(channel, positive)

    # Displayed jump targets.
    for channel, interior in ((6, 1), (7, 12)):
        detector(channel, [
            (0, 0, interior), (1, 1, interior),
            (0, -1, 2), (-1, 0, 2),
            (-1, -1, interior),
        ])

    # Currently selected piece.
    detector(8, [
        (-2, -1, 3), (-1, -2, 3),
        (-1, -1, 3), (-2, 0, 3), (0, -2, 3),
    ])

    # Orange pad, regardless of its passenger.
    detector(9, [
        (-2, -2, 11), (-2, 0, 11), (-2, 2, 11),
        (0, -2, 11), (2, -2, 11),
    ], [(-1, 0, 11), (0, -1, 11)])

    # Up, down, left, and right rail entrances.
    detector(10, [
        (0, -1, 11), (0, 0, 11), (0, 1, 11),
        (-2, 0, 5), (-2, 1, 5),
    ], [(-2, -1, 5)])
    detector(11, [
        (0, -1, 11), (0, 0, 11), (0, 1, 11),
        (2, 0, 5), (2, 1, 5),
    ], [(2, -1, 5)])
    detector(12, [
        (-1, 0, 11), (0, 0, 11), (1, 0, 11),
        (0, -2, 5), (1, -2, 5),
    ], [(-1, -2, 5)])
    detector(13, [
        (-1, 0, 11), (0, 0, 11), (1, 0, 11),
        (0, 2, 5), (1, 2, 5),
    ], [(-1, 2, 5)])

    p["W1"][14, 16, 2, 2] = 1.0
    p["W1"][15, 17, 2, 2] = 1.0
    p["W1"][16, 18, 2, 2] = 1.0

    # Grey round pieces indicate the game's failed-puzzle state.
    detector(17, [
        (0, 0, 2), (0, -1, 2),
        (-1, 0, 2), (-1, 1, 2),
        (1, -1, 2), (1, 1, 2),
    ], [(-1, -1, 2)])

    for channel in range(17):
        p["W2"][channel, channel, 1, 1] = 1.0

    # Coordinate-valued stationary green/red goal detectors.
    coordinate_specs = (
        (17, 15, False), (18, 15, True),
        (19, 14, False), (20, 14, True),
    )
    for out, coordinate, reverse in coordinate_specs:
        p["W2"][out, 0, 1, 1] = 1.0
        p["W2"][out, 1, 1, 1] = 1.0
        p["W2"][out, 9, 1, 1] = -1.0
        if reverse:
            p["W2"][out, coordinate, 1, 1] = -1.0
        else:
            p["W2"][out, coordinate, 1, 1] = 1.0
            p["b2"][out] = -1.0

    # Corresponding carrier coordinate detectors.
    for out, coordinate, reverse in (
        (21, 15, False), (22, 15, True),
        (23, 14, False), (24, 14, True),
    ):
        p["W2"][out, 9, 1, 1] = 1.0
        if reverse:
            p["W2"][out, coordinate, 1, 1] = -1.0
        else:
            p["W2"][out, coordinate, 1, 1] = 1.0
            p["b2"][out] = -1.0

    p["W2"][25, 17, 1, 1] = 1.0

    for channel in range(21):
        p["W3"][channel, channel] = 1.0

    # Axial attraction, gated by an available rail entrance.
    for out, edge, axial, coordinate, sign, constant in (
        (21, 10, 64 + 20, 14, 1.0, -1.0),
        (22, 11, 64 + 19, 14, -1.0, 0.0),
        (23, 12, 32 + 18, 15, 1.0, -1.0),
        (24, 13, 32 + 17, 15, -1.0, 0.0),
    ):
        p["W3"][out, axial] = 10.0
        p["W3"][out, coordinate] = 10.0 * sign
        p["W3"][out, edge] = 10.0
        p["b3"][out] = 10.0 * constant - 10.0

    for out, inp in zip(range(25, 29), range(21, 25)):
        p["W3"][out, inp] = 1.0

    p["W3"][29, 25] = 1.0
    p["W3"][30, 64] = 1.0
    p["W3"][31, 9] = 1.0
    p["W3"][31, 0] = 1.0
    p["W3"][31, 1] = 1.0
    p["b3"][31] = -1.0

    # H13: stable discounted spatial attraction.
    p["Wh"][13, 13, 1, 1] = 0.04
    for ky, kx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        p["Wh"][13, 13, ky, kx] = 0.23
    p["Uh"][13, 0] = 1.0
    p["Uh"][13, 1] = 1.0
    p["Uh"][13, 2] = 2.0
    p["Uh"][13, 3] = 0.25

    field_scale = 100.0

    # H14: attraction at empty pads and displayed destinations.
    p["Wh"][14, 13, 1, 1] = field_scale
    p["bh"][14] = -512.0
    for channel in (4, 5, 6, 7):
        p["Uh"][14, channel] = 512.0

    # H15: attraction at permanent obstacles.
    p["Wh"][15, 13, 1, 1] = field_scale
    p["bh"][15] = -512.0
    p["Uh"][15, 3] = 512.0

    # Fixed linear-filter impulse response, used solely to compensate
    # for a jumping piece's and midpoint obstacle's own field.
    q = np.zeros((129, 129), dtype=np.float64)
    response = {}
    for t in range(1, 120):
        nq = 0.04 * q
        nq[1:, :] += 0.23 * q[:-1, :]
        nq[:-1, :] += 0.23 * q[1:, :]
        nq[:, 1:] += 0.23 * q[:, :-1]
        nq[:, :-1] += 0.23 * q[:, 1:]
        nq[64, 64] += 1.0
        q = nq
        if t in (107, 113, 119):
            response[t] = q.copy()

    rho = 0.3
    gain = rho ** -5
    gradient_weight = 0.4

    self_landing = response[113][64, 76]
    self_midpoint = response[119][64, 70]
    obstacle_landing = response[113][64, 70]
    obstacle_midpoint = response[119][64, 64]
    tail = response[107][64, 82]

    pink_baseline = (
        1.0
        + field_scale * (
            self_landing - gradient_weight * self_midpoint
            + 0.25 * (
                obstacle_landing
                - gradient_weight * obstacle_midpoint
            )
        )
        + rho ** 6 * (1.0 + field_scale * tail)
        + 0.005
    )

    # Four directional circuits:
    # H0..3 rays from landings;
    # H4..7 occupied midpoint gates;
    # H8..11 return rays;
    # H12 source scores.
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    centres = (0, 1, 2, 3, 4, 5, 6, 7)

    for d, (dy, dx) in enumerate(directions):
        ky, kx = 1 + dy, 1 + dx
        ray, midpoint, return_ray = d, 4 + d, 8 + d

        p["Wh"][ray, ray, ky, kx] = rho
        p["Wh"][ray, 14, 1, 1] = 1.0
        for channel in (4, 6):
            p["Uh"][ray, channel] = 1.0
        for channel in (5, 7):
            p["Uh"][ray, channel] = 1.2
        for channel in (0, 1, 2, 3):
            p["Uh"][ray, channel] = -512.0
        p["Uh"][ray, 8] = 514.0

        p["Wh"][midpoint, ray, ky, kx] = gain
        p["Wh"][midpoint, 15, 1, 1] = -gradient_weight
        p["bh"][midpoint] = -32768.0
        for channel in (0, 1, 2):
            p["Uh"][midpoint, channel] = 32767.5
        p["Uh"][midpoint, 3] = (
            32768.0 + 0.38 - pink_baseline
        )

        p["Wh"][return_ray, return_ray, ky, kx] = rho
        p["Wh"][return_ray, midpoint, 1, 1] = 6.0
        for channel in centres:
            p["Uh"][return_ray, channel] = -2.0

        p["Wh"][12, return_ray, ky, kx] = gain

    # The barrier exceeds unmasked intermediate relay amplitudes.
    # The remaining positive threshold rejects distant ray tails.
    p["bh"][12] = -262144.0
    for channel in (0, 1, 2, 6, 7):
        p["Uh"][12, channel] = 262143.5

    # Global jump and target-presence features.
    p["Wz"][0, 12] = 1.0
    p["bz"][0] = -0.001
    p["Wz"][1, 32 + 6] = 1.0
    p["Wz"][1, 32 + 7] = 1.0

    for i in range(4):
        p["Wz"][2 + i, 32 + 10 + i] = 1.0
        p["Wz"][6 + i, 32 + 21 + i] = 1.0

    # Bounding-box-centre attraction: carrier centre toward goal centre.
    up_terms = ((27, 1), (28, -1), (19, -1), (20, 1))
    left_terms = ((25, 1), (26, -1), (17, -1), (18, 1))
    for unit, terms, sign in (
        (10, up_terms, 1),
        (11, up_terms, -1),
        (12, left_terms, 1),
        (13, left_terms, -1),
    ):
        for feature, coefficient in terms:
            p["Wz"][unit, 32 + feature] = (
                4.0 * sign * coefficient
            )

    p["Wz"][14, 32 + 29] = 1.0

    p["Wo"][6, 0] = 100.0
    p["Wo"][6, 1] = 10000.0
    p["bo"][6] = -10.0

    for i, action in enumerate((1, 2, 3, 4)):
        p["Wo"][action, 2 + i] = 50.0
        p["Wo"][action, 6 + i] = 0.2
        p["Wo"][action, 10 + i] = 2.0
        p["bo"][action] = -40.0

    p["Wo"][0, 14] = 10000.0
    p["bo"][0] = -6.0
    p["bo"][5] = -1000.0
    p["bo"][7] = -1000.0

    # Persistent movement direction. New movements overwrite it;
    # intervening clicks preserve it.
    clockwise = (1, 4, 2, 3)
    rewards = (5.0, 8.0, -12.0, 3.0)
    for d in range(4):
        p["Wm"][d, d] = 1.5
        p["Wma"][d, 0] = -3.0
        for qd, action in enumerate(clockwise):
            p["Wma"][d, action] = 3.0 if qd == d else -3.0
            p["Wo"][action, 128 + d] = rewards[(qd - d) % 4]

    # Slowly varying exploration preference.
    p["Wm"][4, 4] = 1.08
    p["Wm"][4, 5] = -0.16
    p["Wm"][5, 4] = 0.16
    p["Wm"][5, 5] = 1.08
    p["bm"][4] = 0.04

    p["Wo"][1, 128 + 4] = 4.0
    p["Wo"][2, 128 + 4] = -4.0
    p["Wo"][3, 128 + 5] = 4.0
    p["Wo"][4, 128 + 5] = -4.0

    # Mild penalty, rather than suppressing valid consecutive clicks.
    p["Wma"][6, 6] = 2.0
    p["Wo"][6, 128 + 6] = -3.0

    p["bo"][2] += 0.02
    p["bo"][3] += 0.01
    p["bo"][4] += 0.04

    # Spatial click scoring. Empty pads and pink obstacles cannot win
    # source selection; displayed targets dominate all source scores.
    p["Wc"][0, 32 + 12] = 1000.0
    p["Wc"][0, 32 + 14] = 10000.0
    p["Wc"][0, 6] = 100000000.0
    p["Wc"][0, 7] = 100000050.0

    p["Wc"][0, 4] = -10000000.0
    p["Wc"][0, 5] = -10000000.0
    p["Wc"][0, 3] = -10000000.0

    p["Wc"][0, 0] = 0.02
    p["Wc"][0, 1] = 0.01
    p["Wc"][0, 2] = -0.02
    p["Wc"][0, 30] = 0.002
    p["Wc"][0, 14] = -0.002
    p["Wc"][0, 15] = -0.001

    return p
