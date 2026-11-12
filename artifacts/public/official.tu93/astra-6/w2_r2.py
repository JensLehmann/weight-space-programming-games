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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def w1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # ------------------------------------------------------------
    # First visual layer.
    # ------------------------------------------------------------

    # Walkable artwork, including occupied nodes.
    for color in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        w1(0, color)

    w1(1, 14)  # Exit.
    w1(2, 2)   # Connector.
    for color in (9, 4):
        w1(3, color)

    # Exact player-center points translated one pixel.
    for a, (dy, dx) in enumerate(directions):
        out = 4 + a
        for yy in (-1, 0, 1):
            for xx in (-1, 0, 1):
                for color in (9, 4):
                    w1(out, color, yy - dy, xx - dx)
        p["b1"][out] = -8.0

    # Enemy-head detectors, positioned two pixels ahead of center.
    # Inactive purple enemies do not generate prediction rays.
    for a, (dy, dx) in enumerate(directions):
        out = 8 + a
        w1(out, 15, -dy, -dx, 1.0)
        w1(out, 11, -dy, -dx, 2.0)

        for color in (8, 12):
            w1(out, color, -2 * dy, -2 * dx, 2.0)
        w1(out, 13, -2 * dy, -2 * dx, 1.0)

        for color in (8, 12, 13):
            w1(out, color, 0, 0, -4.0)
        p["b1"][out] = -2.0

    for color in (8, 12, 13):
        w1(12, color)
    w1(13, 12)

    # Degree at node centers.
    for dy, dx in directions:
        w1(14, 2, 2 * dy, 2 * dx)

    # Sprite-gated coordinates.
    for out, coord, colors in (
        (15, 16, (14,)),
        (16, 17, (14,)),
        (17, 16, (9, 4)),
        (18, 17, (9, 4)),
    ):
        w1(out, coord)
        for color in colors:
            w1(out, color, value=2.0)
        p["b1"][out] = -1.0

    for color in (11, 15):
        w1(19, color)

    # ------------------------------------------------------------
    # Second visual layer.
    # ------------------------------------------------------------
    for channel in range(4):
        w2(channel, channel)

    # Legal directional seeds at player_center + 2 * direction.
    for a, (dy, dx) in enumerate(directions):
        out = 4 + a
        w2(out, 4 + a, -dy, -dx)
        w2(out, 2)
        p["b2"][out] = -1.0

    # Paired channels clamp enemy detections to one.
    for a in range(4):
        aa, bb = 8 + 2 * a, 9 + 2 * a
        w2(aa, 8 + a)
        w2(bb, 8 + a)
        p["b2"][bb] = -1.0

    w2(16, 12)
    w2(17, 13)

    w2(18, 14, value=-1.0)
    p["b2"][18] = 2.0

    for out, inp in (
        (19, 15), (20, 16), (21, 17), (22, 18),
        (23, 19),
    ):
        w2(out, inp)

    # ------------------------------------------------------------
    # Pointwise representation; axial inputs are unused.
    #
    # 0 walkable; 1 exit; 2 player body
    # 3..6 legal directional seeds
    # 7..10 directional enemy seeds
    # 11 enemy body; 12 patrol body; 13 dead-end cost
    # 14..17 coordinates; 18 obstacle; 19 enemy marker
    # ------------------------------------------------------------
    for out, inp in ((0, 0), (1, 1), (2, 3)):
        p["W3"][out, inp] = 1.0

    for a in range(4):
        p["W3"][3 + a, 4 + a] = 1.0
        p["W3"][7 + a, 8 + 2 * a] = 1.0
        p["W3"][7 + a, 9 + 2 * a] = -1.0

    for out, inp in (
        (11, 16), (12, 17), (13, 18),
        (14, 19), (15, 20), (16, 21), (17, 22),
        (19, 23),
    ):
        p["W3"][out, inp] = 1.0

    p["W3"][18, 0] = -1.0
    p["b3"][18] = 1.0

    # ------------------------------------------------------------
    # Recurrent spatial computation.
    #
    # 0,1: paired clipped reachability
    # 2: integrated reachability / arrival-time preference
    # 3..6: forward enemy rays
    # 7..10: legal player rays
    # 11..14: destination-node readouts
    # 15: bounded patrol-proximity field
    # ------------------------------------------------------------

    # Tiny positive sources grow through recurrent neighborhood gain.
    # The stronger exit source starts its wave earlier than enemies.
    # Explicit obstacle features avoid cancellation of tiny sources
    # against a large constant on traversable pixels.
    for out in (0, 1):
        for yy in (-1, 0, 1):
            for xx in (-1, 0, 1):
                wh(out, 0, yy, xx, 1.0)
                wh(out, 1, yy, xx, -1.0)

        p["Uh"][out, 18] = -64.0
        p["Uh"][out, 1] = 1.0e-10
        p["Uh"][out, 11] = 1.0e-30
        p["Uh"][out, 12] = -64.0
        p["bh"][out] = -float(out == 1)

        # Thicken the center-line predictions to cover the artwork's
        # full three-pixel width during route propagation.
        for ray in range(3, 7):
            for yy in (-1, 0, 1):
                for xx in (-1, 0, 1):
                    wh(out, ray, yy, xx, -64.0)

    wh(2, 2)
    wh(2, 0)
    wh(2, 1, value=-1.0)

    for a, (dy, dx) in enumerate(directions):
        enemy_ray = 3 + a
        wh(enemy_ray, enemy_ray, -dy, -dx)
        p["Uh"][enemy_ray, 7 + a] = 11.0
        p["Uh"][enemy_ray, 19] = -64.0
        p["bh"][enemy_ray] = -1.0

        # Finite ramp from the legal connector toward the destination.
        player_ray = 7 + a
        wh(player_ray, player_ray, -dy, -dx)
        p["Uh"][player_ray, 3 + a] = 5.0
        p["bh"][player_ray] = -1.0

    # Leaky diffusion has total recurrent gain 0.96.
    for dy, dx in directions:
        wh(15, 15, dy, dx, 0.24)
    p["Uh"][15, 12] = 1.0

    selection_gate = 100000.0
    for a, (dy, dx) in enumerate(directions):
        out = 11 + a
        player_ray = 7 + a

        # The ramp's second difference selects its far endpoint:
        # precisely the node six pixels from the player.
        wh(out, player_ray, -dy, -dx, selection_gate)
        wh(out, player_ray, 0, 0, -2.0 * selection_gate)
        wh(out, player_ray, dy, dx, selection_gate)

        wh(out, 2)

        # Actual next-node danger dominates route preferences.
        for enemy_ray in range(3, 7):
            wh(out, enemy_ray, value=-1000.0)

        # Modest caution near patrols, without strong global repulsion.
        wh(out, 15, value=-32.0)

        p["bh"][out] = 1000.0 - selection_gate

        # Remove the near-source endpoint of the second difference.
        p["Uh"][out, 2] = -5.0 * selection_gate

        # Landing on an enemy removes it before its movement phase.
        p["Uh"][out, 11] = 20.0
        p["Uh"][out, 12] = 1000.0

        p["Uh"][out, 13] = -40.0
        p["Uh"][out, 1] = 40.0

    # ------------------------------------------------------------
    # Pooled decoder.
    # ------------------------------------------------------------
    for a in range(4):
        p["Wz"][a, 11 + a] = 1.0
        p["Wz"][4 + a, 32 + 3 + a] = 1.0

    for z, positive, negative in (
        (8, 14, 16),
        (9, 16, 14),
        (10, 15, 17),
        (11, 17, 15),
    ):
        p["Wz"][z, 32 + positive] = 1.0
        p["Wz"][z, 32 + negative] = -1.0

    p["bo"][:] = -1.0e6
    for a in range(4):
        action = a + 1
        p["bo"][action] = (a + 1) * 0.0002
        p["Wo"][action, a] = 1.0
        p["Wo"][action, 4 + a] = 2.0

    for action, positive, negative in (
        (1, 9, 8),
        (2, 8, 9),
        (3, 11, 10),
        (4, 10, 11),
    ):
        p["Wo"][action, positive] = 0.05
        p["Wo"][action, negative] = -0.05

    # Small action memory discourages immediate reversals, but safety
    # and the legal-action gates take precedence.
    opposite = (1, 0, 3, 2)
    for previous in range(4):
        p["Wma"][previous, previous + 1] = math.atanh(0.5)
        p["Wo"][opposite[previous] + 1, 128 + previous] = -36.0
        p["Wo"][previous + 1, 128 + previous] = 0.2

    return p
