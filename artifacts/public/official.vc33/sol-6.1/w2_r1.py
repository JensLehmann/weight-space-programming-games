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
    p = {k: np.zeros(s, dtype=np.float32)
         for k, s in shapes.items()}
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def c1(o, i, dr=0, dc=0, v=1.0):
        p["W1"][o, i, 2 + dr, 2 + dc] += v

    def c2(o, i, dr=0, dc=0, v=1.0):
        p["W2"][o, i, 1 + dr, 1 + dc] += v

    def ch(o, i, dr=0, dc=0, v=1.0):
        p["Wh"][o, i, 1 + dr, 1 + dc] += v

    def f3(o, i, v=1.0, axial=0):
        p["W3"][o, axial + i] += v

    # Local colours and directional foot/bridge-end evidence.
    c1(0, 11)
    c1(0, 14)
    c1(1, 5)
    c1(2, 9)
    c1(3, 12)
    c1(4, 1)
    c1(4, 12)

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            c1(5, 4, dr, dc)

    for d, (dr, dc) in enumerate(dirs):
        c1(6 + d, 11)
        c1(6 + d, 14)
        c1(6 + d, 4, -dr, -dc, .5)
        c1(6 + d, 4, -2 * dr, -2 * dc, .5)
        p["b1"][6 + d] = -1.0

        c1(10 + d, 1)
        c1(10 + d, 12)
        c1(10 + d, 5, dr, dc, .5)
        c1(10 + d, 5, 2 * dr, 2 * dc, .5)
        p["b1"][10 + d] = -1.0

        a = 14 + 2 * d
        for o in (a, a + 1):
            c1(o, 9, dr, dc)
            c1(o, 9, 2 * dr, 2 * dc)
        p["b1"][a + 1] = -1.0

        for colour in (3, 5, 9):
            c1(22 + d, colour)
        c1(22 + d, 0, dr, dc)
        p["b1"][22 + d] = -1.0

    # Row/column coordinates confined to player foot stripes.
    for o, coord, sign in (
        (26, 16, 1), (27, 16, -1),
        (28, 17, 1), (29, 17, -1),
    ):
        c1(o, 11)
        c1(o, 14)
        for dr, dc in dirs:
            c1(o, 4, 2 * dr, 2 * dc)
        c1(o, coord, v=sign)
        p["b1"][o] = -2.0 if sign == 1 else -1.0
    c1(30, 16)
    c1(31, 17)

    # Cap directional evidence using two ReLU channels.
    c2(0, 0)
    c2(0, 5, v=-10.0)
    c2(1, 1)
    c2(2, 2)
    c2(3, 3)

    for d in range(4):
        c2(4 + d, 6 + d, v=2.0)
        c2(16 + d, 6 + d, v=2.0)
        p["b2"][16 + d] = -1.0

        c2(8 + d, 10 + d, v=2.0)
        c2(20 + d, 10 + d, v=2.0)
        p["b2"][20 + d] = -1.0

    surface = [24, 25, 30, 31]
    for d in range(4):
        c2(surface[d], 22 + d)

    def near(o, d, dr, dc):
        a = 14 + 2 * d
        c2(o, a, dr, dc)
        c2(o, a + 1, dr, dc, -1.0)

    # Paired-button divider seeds.
    c2(12, 1)
    near(12, 1, 0, -1)
    near(12, 0, 0, 1)
    p["b2"][12] = -2.0

    c2(13, 1)
    near(13, 3, -1, 0)
    near(13, 2, 1, 0)
    p["b2"][13] = -2.0

    for dst, src in (
        (14, 26), (15, 27),
        (26, 28), (27, 29),
        (28, 30), (29, 31),
    ):
        c2(dst, src)

    def foot(o, d, axial=0, v=1.0):
        f3(o, 4 + d, v, axial)
        f3(o, 16 + d, -v, axial)

    def end(o, d, axial=0, v=1.0):
        f3(o, 8 + d, v, axial)
        f3(o, 20 + d, -v, axial)

    def divider(o):
        f3(o, 12, axial=64)
        f3(o, 13, axial=32)
        for d in (0, 1):
            end(o, d, axial=32)
        for d in (2, 3):
            end(o, d, axial=64)
        f3(o, 1)

    # Shared potential domain, with bridge endpoints open.
    divider(0)
    for d in range(4):
        end(0, d, v=-8.0)
    f3(0, 0, -8.0)

    # Request transport does not cross dividers.
    divider(4)

    # Bridge masks: active endpoint, matching foot height, and player
    # on the appropriate transverse side.
    # (F3 channel, horizontal gravity?, positive transfer?)
    bridge_masks = [
        (1, True, True),
        (2, True, False),
        (3, False, True),
        (5, False, False),
    ]
    for out, horizontal, positive in bridge_masks:
        p["b3"][out] = 12.0
        f3(out, 3, -4.0)

        ds = (0, 1) if horizontal else (2, 3)
        axial = 64 if horizontal else 32
        for d in ds:
            end(out, d, v=-4.0)
            foot(out, d, axial=axial, v=-4.0)

        foot_coordinate = 14 if horizontal else 26
        local_coordinate = 28 if horizontal else 29
        sign = 1.0 if positive else -1.0
        f3(out, foot_coordinate, sign, axial)
        f3(out, local_coordinate, -sign)

    # Cross-wall relay and click-domain masks.
    p["b3"][6] = 1.0
    f3(6, 1, -1.0)
    f3(6, 2, -1.0)

    p["b3"][7] = 1.0
    f3(7, 2, -1.0)
    f3(7, 3, -1.0)

    for d in range(4):
        p["b3"][8 + d] = 1.0
        foot(8 + d, d, v=-1.0)
        f3(8 + d, surface[d], -1.0)
        f3(20 + d, surface[d])

    f3(12, 0)

    # Replenishment only when the corresponding room has no exposed
    # support boundary. Projection follows the support, not its wall.
    p["b3"][13] = -1.0
    f3(13, 2)
    f3(13, 12, axial=32)
    f3(13, surface[2], -1.0, 64)
    f3(13, surface[3], -1.0, 64)

    p["b3"][14] = -1.0
    f3(14, 2)
    f3(14, 13, axial=64)
    f3(14, surface[0], -1.0, 32)
    f3(14, surface[1], -1.0, 32)

    # Weak sources at all inactive endpoints.
    for d in range(4):
        end(15, d)
    f3(15, 3, -4.0)

    # Stronger sources when the player's longitudinal position matches
    # a nearby inactive endpoint.
    for d in range(4):
        axial = 64 if d < 2 else 32
        out = 16 + d
        p["b3"][out] = -1.0
        end(out, d)
        foot(out, d, axial=axial)
        f3(out, 3, -2.0)

        local_coord = 28 if d < 2 else 29
        foot_coord = 14 if d < 2 else 26
        foot_inverse = 15 if d < 2 else 27
        distance = 15.0 / 63.0

        plus = 24 + 2 * d
        minus = plus + 1
        for o in (plus, minus):
            end(o, d)
            foot(o, d, axial=axial)
            f3(o, 3, -4.0)
            p["b3"][o] = -2.0 - distance

        f3(plus, local_coord)
        f3(plus, foot_coord, -1.0, axial)

        f3(minus, local_coord, -1.0)
        f3(minus, foot_inverse, -1.0, axial)
        p["b3"][minus] += 1.0

    # Recurrent channels:
    # 0 potential; 1,2,3,15 directional bridge interest;
    # 4..7 growth requests; 8..11 shrink requests;
    # 12,13 room transport; 14 cross-wall relay.
    mask = float(2 ** 100)
    p["Uh"][0, 0] = -mask
    p["Uh"][0, 12] = float(2 ** 20)
    p["Uh"][0, 15] = float(2 ** 16)
    for d in range(4):
        p["Uh"][0, 16 + d] = float(2 ** 40)
        p["Uh"][0, 24 + 2 * d] = -mask
        p["Uh"][0, 25 + 2 * d] = -mask

    ch(0, 0, v=.02)
    for dr, dc in dirs:
        ch(0, 0, dr, dc, .245)

    for h, mask_channel, dr, dc in (
        (1, 1, 1, 0),
        (2, 2, -1, 0),
        (3, 3, 0, 1),
        (15, 5, 0, -1),
    ):
        p["Uh"][h, mask_channel] = -mask
        ch(h, 0, dr, dc, 1.0)
        ch(h, 0, -dr, -dc, -1.0)

    for d, (dr, dc) in enumerate(dirs):
        grow, shrink = 4 + d, 8 + d
        for out in (grow, shrink):
            p["Uh"][out, 8 + d] = -mask
            p["Uh"][out, 20 + d] = -float(2 ** 26)

        ch(grow, 0, -dr, -dc, 1.0)
        ch(grow, 0, dr, dc, -1.0)
        ch(shrink, 0, dr, dc, 1.0)
        ch(shrink, 0, -dr, -dc, -1.0)

    for out, first in ((12, 4), (13, 8)):
        p["Uh"][out, 4] = -mask
        ch(out, out, v=.05)
        for dr, dc in dirs:
            ch(out, out, dr, dc, .2375)
        for d in range(4):
            ch(out, first + d)

    p["Uh"][14, 6] = -mask
    ch(14, 14, v=.05)
    for dr, dc in dirs:
        ch(14, 14, dr, dc, .2375)
        ch(14, 13, dr, dc, .15)
        ch(14, 12, dr, dc, -.15)

    # Deterministic spatial action ranking.
    p["Wc"][0, 7] = -float(2 ** 120)
    p["Wc"][0, 13] = float(2 ** 110)
    p["Wc"][0, 14] = float(2 ** 110)
    for h in (1, 2, 3, 15):
        p["Wc"][0, 32 + h] = float(2 ** 60)
    p["Wc"][0, 32 + 12] = 1.0
    p["Wc"][0, 32 + 13] = -1.0
    p["Wc"][0, 32 + 14] = 1.0

    p["bo"][6] = 1.0
    return p
