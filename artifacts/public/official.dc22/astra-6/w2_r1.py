import numpy as np
import math


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
    p = {name: np.zeros(shape, np.float32)
         for name, shape in shapes.items()}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cross = [(0, 0)] + directions

    def c1(out, channel, dy=0, dx=0, value=1.0):
        p["W1"][out, channel, dy + 2, dx + 2] += value

    def c2(out, channel, dy=0, dx=0, value=1.0):
        p["W2"][out, channel, dy + 1, dx + 1] += value

    def rec(out, channel, dy=0, dx=0, value=1.0):
        p["Wh"][out, channel, dy + 1, dx + 1] += value

    surface_colors = [1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]

    # Pointwise visible support.
    for color in surface_colors:
        c1(0, color)

    # Exact 2x2 checkerboard templates.
    for color, first in [(9, 1), (15, 3)]:
        for phase in range(2):
            out = first + phase
            for dy in range(2):
                for dx in range(2):
                    expected = color if (dy + dx + phase) % 2 else 4
                    c1(out, expected, dy, dx)
            p["b1"][out] = -3

    # Unique upper-left player pixel.
    c1(5, 14)
    c1(5, 14, -1, 0, -1)
    c1(5, 14, 0, -1, -1)

    # Both teleporter phases.
    for phase in range(2):
        out = 7 + phase
        for dy in range(2):
            for dx in range(2):
                expected = 6 + ((dy + dx + phase) % 2)
                c1(out, expected, dy, dx)
        p["b1"][out] = -3

    # Collectible cap and base.
    collectible = [
        (0, 0, 2), (0, 1, 15), (0, 2, 15),
        (1, 0, 15), (1, 1, 15), (1, 2, 15),
    ]
    for dy, dx, color in collectible:
        c1(9, color, dy, dx)
    p["b1"][9] = 1 - len(collectible)

    # Ordinary control-button interiors.
    for out, color in zip(range(10, 14), [8, 9, 6, 15]):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                c1(out, color, dy, dx)
        p["b1"][out] = -24

    # Complete visible 2x2 support at a candidate destination.
    for dy in range(2):
        for dx in range(2):
            for color in surface_colors:
                c1(14, color, dy, dx)
    p["b1"][14] = -3

    # Pixel coordinates. The offset makes float32 rounding occur before
    # subtraction, yielding integer-valued coordinates.
    for out, coordinate in [(15, 16), (16, 17)]:
        c1(out, coordinate, value=63)
        c1(out, 18, value=128)
        p["b1"][out] = -128

    # Occlusion-tolerant pivot detection.
    #
    # A 4x3 pivot stripe has two adjacent horizontal detections; a 3x4
    # stripe has two adjacent vertical detections. Player-colored pixels
    # may replace pivot pixels, but a lone 2x2 player cannot match.
    for out, vertical, inverse in [
        (17, True, False),
        (18, True, True),
        (19, False, False),
        (20, False, True),
    ]:
        ys = range(-1, 3) if vertical else range(-1, 2)
        xs = range(-1, 2) if vertical else range(-1, 3)
        for dy in ys:
            for dx in xs:
                c1(out, 13, dy, dx, 128)
                c1(out, 14, dy, dx, 128)

        coordinate = 16 if vertical else 17
        c1(out, coordinate, value=-63 if inverse else 63)
        p["b1"][out] = -12 * 128 + (64 if inverse else -1)

    # Isolated 2x2 goal, excluding larger same-colored controls.
    for dy in range(2):
        for dx in range(2):
            c1(21, 11, dy, dx)
    for dy, dx in [(-1, 0), (2, 0), (0, -1), (0, 2)]:
        c1(21, 11, dy, dx, -1)
    p["b1"][21] = -3

    # One-pixel player probes.
    for i, (dy, dx) in enumerate(directions):
        out = 22 + i
        c1(out, 14, -dy, -dx)
        c1(out, 14, -dy - 1, -dx, -1)
        c1(out, 14, -dy, -dx - 1, -1)

    # Fill interior checkerboard holes, not general empty space.
    for out, bias in [(30, -5), (31, -6)]:
        c1(out, 4, value=4)
        for dy, dx in directions:
            c1(out, 9, dy, dx)
            c1(out, 15, dy, dx)
        p["b1"][out] = bias

    # Surface union.
    c2(0, 0)
    c2(0, 30)
    c2(0, 31, value=-1)

    # Purple checkerboard coverage, represented by a clipped difference.
    for out in [1, 2]:
        for dy in [-1, 0]:
            for dx in [-1, 0]:
                c2(out, 3, dy, dx)
                c2(out, 4, dy, dx)
    p["b2"][2] = -1
    c2(3, 5)

    # Legal destinations and teleporter-entry flags.
    for i, (dy, dx) in enumerate(directions):
        c2(4 + i, 22 + i, -dy, -dx)
        c2(4 + i, 14)
        p["b2"][4 + i] = -1

        c2(8 + i, 22 + i, -dy, -dx)
        c2(8 + i, 7)
        c2(8 + i, 8)
        p["b2"][8 + i] = -1

        c2(28, 22 + i, -dy, -dx)

    # Checkerboard destinations that require a switch.
    for out, checker_channels in [(12, [1, 2]), (13, [3, 4])]:
        for i, (dy, dx) in enumerate(directions):
            c2(out, 22 + i, -dy, -dx, 10)
        for channel in checker_channels:
            c2(out, channel)
        p["b2"][out] = -10

    c2(14, 21)
    c2(14, 9)
    c2(15, 7)
    c2(15, 8)

    for i in range(4):
        c2(16 + i, 10 + i)

    # Spread paired stripe detections into exactly four lanes.
    for out, source in [(20, 17), (21, 18)]:
        c2(out, source, 0, -1)
        c2(out, source, 0, 1)
    for out, source in [(22, 19), (23, 20)]:
        c2(out, source, -1, 0)
        c2(out, source, 1, 0)

    c2(29, 15)
    c2(30, 16)

    # Planning wall mask. Purple paths start closed.
    p["W3"][0, 0] = -1
    p["W3"][0, 1] = 1
    p["W3"][0, 2] = -1
    p["b3"][0] = 1

    p["W3"][1, 1] = 1
    p["W3"][1, 2] = -1

    # Immediate goal/collectible seeds and delayed teleporter seeds.
    p["W3"][2, 14] = 1
    p["W3"][2, 15] = 1e-32

    epsilon = 1.0 / 256.0
    for i in range(4):
        p["W3"][3 + i, 4 + i] = 1
        p["W3"][3 + i, 8 + i] = epsilon

    p["W3"][7, 12] = 1
    p["W3"][8, 13] = 1
    p["W3"][9, 3] = 1

    for i in range(4):
        p["W3"][10 + i, 16 + i] = 1

    signs = np.array([1, -1, -1, 1], dtype=np.float32)

    # Integer ReLU hinges describe alternate bridge arms. Their signed
    # differences are exact binary bands rather than cancellation noise.
    def alternate_arm(first, vertical, probe_gated):
        axial = 64 if vertical else 32
        coordinate = 29 if vertical else 30
        edge = 20 if vertical else 22

        for j, shift in enumerate([9, 8, -11, -12]):
            out = first + j
            p["W3"][out, coordinate] = 1
            p["W3"][out, axial + edge] = 3
            p["W3"][out, axial + edge + 1] = 4
            p["W3"][out, 0] = -256
            p["b3"][out] = shift - 252

            if probe_gated:
                p["W3"][out, 28] = 256
                p["b3"][out] -= 256

    alternate_arm(15, True, False)
    alternate_arm(19, False, False)
    alternate_arm(23, True, True)
    alternate_arm(27, False, True)

    # Main bounded flood: channels 0-1 encode a value in [0,1].
    for out in [0, 1]:
        for dy, dx in cross:
            rec(out, 0, dy, dx)
            rec(out, 1, dy, dx, -1)
        rec(out, 9, value=6)
        p["Uh"][out, 0] = -6
        p["Uh"][out, 2] = 1
        p["Uh"][out, 15:19] = 6 * signs
        p["Uh"][out, 19:23] = 6 * signs
    p["bh"][1] = -1

    # Arrival-time integrals at each legal movement destination.
    gate = 256.0
    for i in range(4):
        out = 2 + i
        rec(out, out)
        rec(out, 0)
        rec(out, 1, value=-1)
        p["Uh"][out, 3 + i] = gate
        p["bh"][out] = -gate

    # Arrival-time integral at switch-requiring destinations.
    rec(6, 6)
    rec(6, 0)
    rec(6, 1, value=-1)
    p["Uh"][6, 7:9] = gate
    p["Uh"][6, 23:27] = gate * signs
    p["Uh"][6, 27:31] = gate * signs
    p["bh"][6] = -gate

    # Broadcast purple-button availability.
    for out in [7, 8]:
        for dy, dx in cross:
            rec(out, 7, dy, dx)
            rec(out, 8, dy, dx, -1)
        p["Uh"][out, 13] = 1
    p["bh"][8] = -1

    # Availability opens only purple checkerboard locations.
    rec(9, 7)
    rec(9, 8, value=-1)
    p["Uh"][9, 1] = 2
    p["bh"][9] = -2

    # Three request broadcasts select the matching control color.
    for high, low, own_button, source in [
        (10, 11, 11, 7),
        (12, 13, 13, 8),
        (14, 15, 10, None),
    ]:
        for out in [high, low]:
            for dy, dx in cross:
                rec(out, high, dy, dx)
                rec(out, low, dy, dx, -1)

            for button in range(10, 14):
                if button != own_button:
                    p["Uh"][out, button] = -8

            if source is None:
                p["Uh"][out, 23:27] = signs
                p["Uh"][out, 27:31] = signs
            else:
                p["Uh"][out, source] = 1

        p["bh"][low] = -1

    # Spatial click decoding.
    p["Wc"][0, 10:14] = [100, 100.02, 100.2, 100.01]
    for high, low in [(10, 11), (12, 13), (14, 15)]:
        p["Wc"][0, 32 + high] = 10
        p["Wc"][0, 32 + low] = -10

    # Global action readout.
    for i in range(4):
        p["Wz"][i, 2 + i] = 1
        p["Wz"][5 + i, 32 + 3 + i] = 1

        # Extract the packed teleporter-entry flag.
        p["Wz"][9 + i, 32 + 3 + i] = 256
        p["bz"][9 + i] = -256

    p["Wz"][4, 6] = 1

    # Low-priority switch exploration when no route signal is present.
    for unit, request in [(13, 7), (14, 8)]:
        p["Wz"][unit, 32 + request] = 2
        p["Wz"][unit, 2:7] = -1
        p["bz"][unit] = -1

    # Remember whether the last movement entered a teleporter.
    for i in range(4):
        p["Wmz"][i, 9 + i] = 6
        p["bm"][i] = -3

        p["Wm"][4 + i, i] = 6
        p["Wma"][4 + i, 1 + i] = 12
        p["bm"][4 + i] = -12

        p["Wma"][8 + i, 1 + i] = 6
        p["bm"][8 + i] = -3

    # Small bounded tie-breaking oscillator.
    theta = 0.7
    cosine, sine = math.cos(theta), math.sin(theta)
    p["Wm"][12:14, 12:14] = 1.1 * np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float32
    )
    p["bm"][12] = 0.07

    p["bo"][:] = -10000
    opposite = [1, 0, 3, 2]
    preferences = [0.12, 0.10, 0.14, 0.08]

    for i in range(4):
        action = 1 + i
        p["bo"][action] = 0
        p["Wo"][action, i] = 1
        p["Wo"][action, 5 + i] = preferences[i]

        # Cancel the integrator offset introduced by the packed flag.
        p["Wo"][action, 9 + i] = -128
        p["Wo"][action, 128 + 8 + opposite[i]] = -0.01

    p["Wo"][1, 128 + 12] = -0.005
    p["Wo"][2, 128 + 12] = 0.005
    p["Wo"][3, 128 + 13] = -0.005
    p["Wo"][4, 128 + 13] = 0.005

    p["Wo"][6, 4] = 1
    p["Wo"][6, 13:15] = 2
    p["Wo"][6, 128 + 4:128 + 8] = 200
    p["bo"][6] = 800 - 0.5

    return p
