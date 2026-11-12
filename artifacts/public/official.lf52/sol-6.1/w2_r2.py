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
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    def c1(o, c, dy=0, dx=0, w=1.0):
        p["W1"][o, c, dy + 2, dx + 2] += w

    def c2(o, c, dy=0, dx=0, w=1.0):
        p["W2"][o, c, dy + 1, dx + 1] += w

    def c3(o, c, w=1.0, axial=0):
        p["W3"][o, c + 32 * axial] += w

    def hc(o, c, dy=0, dx=0, w=1.0):
        p["Wh"][o, c, dy + 1, dx + 1] += w

    def template(o, requirements):
        for dy, dx, colours in requirements:
            for colour in colours:
                c1(o, colour, dy, dx)
        p["b1"][o] = 1.0 - len(requirements)

    def circle(o, colour, oy=0, ox=0):
        for dy, dx in ((0, 0), (-1, 0), (0, -1),
                       (1, 0), (0, 1)):
            c1(o, colour, oy + dy, ox + dx)
        c1(o, colour, oy - 1, ox - 1, -5.0)
        p["b1"][o] = -4.0

    def interior(o, inside, border):
        template(o, [
            (0, 0, (inside,)), (0, 1, (inside,)),
            (1, 0, (inside,)), (-1, 0, (inside,)),
            (0, -1, (inside,)), (-1, -1, (inside,)),
            (-2, 0, (border,)), (0, -2, (border,)),
            (-2, -2, (border,)),
        ])

    def portal(o, oy=0, ox=0):
        template(o, [
            (oy, ox, (7,)),
            (oy - 1, ox, (0,)),
            (oy, ox - 1, (15,)),
            (oy - 1, ox - 1, (15,)),
            (oy + 1, ox, (7,)),
            (oy, ox + 1, (7,)),
        ])

    # Pixel templates, anchored at the upper-left central pixel.
    for o, colour in enumerate((14, 8, 9, 2)):
        circle(o, colour)
    interior(4, 1, 0)

    ring = [
        (-1, 0), (-1, 1), (0, -1), (1, -1),
        (0, 2), (1, 2), (2, 0), (2, 1),
    ]
    template(5, [(dy, dx, (2,)) for dy, dx in ring]
             + [(0, 0, (1, 12))])

    inside_platform = (2, 7, 8, 9, 12, 14, 15)
    template(6, [
        (-2, 0, (11,)), (0, -2, (11,)),
        (-2, -2, (11,)), (0, 0, inside_platform),
    ])

    c1(7, 5, 1, -2)
    c1(11, 5, -2, 1)
    c1(12, 5, -2, 2)
    c1(13, 5, 2, -2)
    c1(8, 16)
    c1(9, 17)
    c1(10, 18)

    interior(14, 12, 11)
    template(15, [
        (-2, 0, (0,)), (0, -2, (0,)),
        (-2, -2, (0,)), (0, 0, (1, 2, 7, 8, 9, 14)),
    ])

    for o, colour in zip(range(16, 20), (14, 8, 9, 2)):
        circle(o, colour, -1, -1)

    template(20, [
        (0, 1, (11,)), (1, 0, (11,)),
        (1, 1, (11,)), (-2, -2, inside_platform),
    ])

    # Absence of nearby rail/outline pixels; later inverted.
    p["b1"][21] = 1.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            c1(21, 5, dy, dx, -1.0)
            c1(21, 11, dy, dx, -1.0)

    portal(22)
    portal(23, -1, -1)

    # Semantic second layer.
    for ch in range(4):
        c2(0, ch)
        c2(1, ch)                 # All peg colours are selectable.
    for ch in (0, 3):
        c2(2, ch)
    c2(3, 1)
    c2(4, 2)

    for ch in (4, 5, 14):
        c2(5, ch)
    c2(6, 5)
    c2(7, 6)

    # Platform occupants at the two rail-inspection anchors.
    c2(8, 6, 1, 1)
    for ch in (0, 1, 2, 3, 22):
        c2(8, ch, 1, 1)
    p["b2"][8] = -1.0

    c2(9, 20)
    c2(10, 20)
    for ch in (16, 17, 18, 19, 23):
        c2(10, ch, -1, -1)
    p["b2"][10] = -1.0

    # Actual rail beyond the platform, rather than its own outline.
    c2(11, 6, 1, 1)
    c2(11, 7, 0, -1)
    p["b2"][11] = -1.0

    c2(12, 6, 1, 1)
    c2(12, 11, -1, 0)
    p["b2"][12] = -1.0

    c2(13, 20)
    c2(13, 12, 0, 1)
    p["b2"][13] = -1.0

    c2(14, 20)
    c2(14, 13, 1, 0)
    p["b2"][14] = -1.0

    def peg_coordinate(o, channels, coordinate, inverse=False,
                       dy=0, dx=0, adjustment=0.0):
        for ch in channels:
            c2(o, ch, dy, dx)
        c2(o, coordinate, dy, dx, -1.0 if inverse else 1.0)
        p["b2"][o] = (-adjustment if inverse
                     else -1.0 + adjustment)

    peg_coordinate(15, (0, 1, 2, 3), 8)
    peg_coordinate(31, (0, 1, 2, 3), 9)

    for o, coord, inverse in (
        (16, 9, False), (17, 9, True),
        (18, 8, False), (19, 8, True),
    ):
        peg_coordinate(o, (0, 1, 2, 3), coord, inverse, 1, 1)

    for o, coord, inverse in (
        (20, 9, False), (21, 9, True),
        (22, 8, False), (23, 8, True),
    ):
        peg_coordinate(o, (16, 17, 18, 19), coord, inverse,
                       -1, -1, -1.0 / 63.0)

    c2(24, 8)
    p["b2"][24] = 1.0 / 63.0
    c2(25, 9)
    p["b2"][25] = 1.0 / 63.0
    c2(26, 8)
    p["b2"][26] = -2.0 / 63.0
    c2(27, 9)
    p["b2"][27] = -2.0 / 63.0

    c2(28, 3)
    c2(29, 21)
    c2(30, 22)

    # Local semantics and axial transport hints.
    for ch in range(11):
        c3(ch, ch)
    c3(11, 30)                   # Portal centre.

    c3(12, 12, 2.0)
    c3(12, 19, axial=2)
    c3(12, 24)
    p["b3"][12] = -3.0

    c3(13, 14, 2.0)
    c3(13, 22, axial=2)
    c3(13, 26, -1.0)
    p["b3"][13] = -2.0

    c3(14, 11, 2.0)
    c3(14, 17, axial=1)
    c3(14, 25)
    p["b3"][14] = -3.0

    c3(15, 13, 2.0)
    c3(15, 20, axial=1)
    c3(15, 27, -1.0)
    p["b3"][15] = -2.0

    rails = (12, 14, 11, 13)      # Up, down, left, right.
    occupants = (8, 10, 8, 10)
    for d in range(4):
        c3(16 + d, rails[d])
        c3(20 + d, rails[d])
        c3(20 + d, occupants[d])
        p["b3"][20 + d] = -1.0

    c3(24, 28)
    c3(25, 31)
    c3(26, 15)
    c3(27, 15)
    c3(27, 7)

    c3(28, 29, -1.0)
    p["b3"][28] = 1.0

    # Stationary portals are rail-routing goals.
    c3(30, 30)
    c3(30, 7, -1.0)

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Two symmetric one-dimensional Green-function rays.
    # For neighbour coefficient .4, attenuation per pixel is .5
    # and the infinite-line self response is 5/3.
    for ray, offsets in (
        (0, ((0, -1), (0, 1))),
        (1, ((-1, 0), (1, 0))),
    ):
        for dy, dx in offsets:
            hc(ray, ray, dy, dx, 0.4)
        p["Uh"][ray, 5] = 1.0
        hc(ray, 10, w=2.0)

    # Peg/portal midpoints with a nearby empty destination.
    for middle, empty_ray in ((2, 0), (3, 1)):
        hc(middle, empty_ray)
        p["Uh"][middle, 0] = 5.0
        p["Uh"][middle, 11] = 5.0 - 0.034
        p["bh"][middle] = -5.010

    # Return midpoint evidence to possible source pegs.
    for ray, middle, offsets in (
        (4, 2, ((0, -1), (0, 1))),
        (5, 3, ((-1, 0), (1, 0))),
    ):
        for dy, dx in offsets:
            hc(ray, ray, dy, dx, 0.4)
        hc(ray, middle)

    hc(6, 4)
    hc(6, 5)
    hc(6, 2, w=-5.0 / 3.0)
    hc(6, 3, w=-5.0 / 3.0)
    p["Uh"][6, 1] = 10.0
    p["bh"][6] = -10.00008

    # Clipped confidence is used only for action selection.
    hc(7, 6)
    p["bh"][7] = -0.0004

    # Peg proximity and floor-support potentials.
    for dy, dx in directions:
        hc(8, 8, dy, dx, 0.24)
        hc(9, 9, dy, dx, 0.245)
    p["Uh"][8, 0] = 0.04
    p["Uh"][9, 27] = 0.025

    # Destination quality, gated to empty tiles.
    hc(10, 8, w=8.0)
    hc(10, 9, w=3.0)
    p["Uh"][10, 5] = 4.0
    p["bh"][10] = -4.0

    # Potential restricted to a thickened rail/outline network.
    for dy, dx in directions:
        hc(11, 11, dy, dx, 0.249)
    hc(11, 8, w=0.03)
    p["Uh"][11, 30] = 0.05
    p["Uh"][11, 28] = 100.0
    p["bh"][11] = -100.0

    # Positive directional potential gradients at legal platforms.
    for d, (dy, dx) in enumerate(directions):
        hc(12 + d, 11, dy, dx)
        hc(12 + d, 11, w=-1.0)
        p["Uh"][12 + d, 16 + d] = 10.0
        p["bh"][12 + d] = -10.0

    # Source clicks favour useful destinations, not screen position.
    p["Wc"][0, 1] = 3.0
    p["Wc"][0, 6] = 5000.0
    p["Wc"][0, 25] = -0.3
    p["Wc"][0, 26] = -0.1
    p["Wc"][0, 32 + 6] = 100000.0
    p["Wc"][0, 32 + 8] = 80.0
    p["Wc"][0, 32 + 9] = 25.0

    # Global action features.
    p["Wz"][0, 6] = 10000.0
    p["Wz"][0, 7] = -10000.0
    p["Wz"][1, 32 + 6] = 1.0
    p["Wz"][2, 32 + 24] = 1.0

    for d in range(4):
        p["Wz"][10 + d, 12 + d] = 2000.0
        p["Wz"][14 + d, 32 + 16 + d] = 1.0
        p["Wz"][18 + d, 32 + 20 + d] = 1.0
        p["Wz"][22 + d, 32 + 12 + d] = 1.0

    p["bo"][:] = -100.0
    p["bo"][7] = 0.0
    p["bo"][0] = -20.0
    p["Wo"][0, 2] = 60.0

    p["bo"][6] = -1.0
    p["Wo"][6, 0] = 12.0
    p["Wo"][6, 1] = 100.0

    for d, action in enumerate((1, 2, 3, 4)):
        p["bo"][action] = -1.0
        p["Wo"][action, 10 + d] = 1.0
        p["Wo"][action, 14 + d] = 1.2
        p["Wo"][action, 18 + d] = 0.4
        p["Wo"][action, 22 + d] = 0.7

        # Mild continuation bias, with stronger reversal inhibition.
        p["Wm"][d, d] = 1.0
        for a in (1, 2, 3, 4):
            p["Wma"][d, a] = 2.0 if a == action else -2.0
        p["Wo"][action, 128 + d] = 0.15
        opposite = (1, 0, 3, 2)[d]
        p["Wo"][action, 128 + opposite] = -0.35

    p["bo"][4] += 0.003
    p["bo"][2] += 0.001

    return p
