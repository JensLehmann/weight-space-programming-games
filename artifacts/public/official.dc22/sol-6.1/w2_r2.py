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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    def c1(out, src, dy=0, dx=0, w=1.0):
        p["W1"][out, src, dy + 2, dx + 2] += w

    def c2(out, src, dy=0, dx=0, w=1.0):
        p["W2"][out, src, dy + 1, dx + 1] += w

    def ax(out, src, w=1.0, mode=0):
        p["W3"][out, src + 32 * mode] += w

    def rec(out, src, dy=0, dx=0, w=1.0):
        p["Wh"][out, src, dy + 1, dx + 1] += w

    square = ((0, 0), (0, 1), (1, 0), (1, 1))
    outside = ((-1, 0), (0, -1), (2, 0), (0, 2))
    perimeter = (
        (-1, 0), (-1, 1), (0, -1), (1, -1),
        (2, 0), (2, 1), (0, 2), (1, 2),
    )

    # Player planes shifted to prospective two-pixel destinations.
    for out, (dy, dx) in enumerate(
        ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2))
    ):
        c1(out, 14, dy, dx)

    # A fully visible 2x2 footprint is usable without a toggle.
    for dy, dx in square:
        for color in (2, 6, 7, 8, 9, 11, 12, 13, 14, 15):
            c1(5, color, dy, dx)
    p["b1"][5] = -3.0

    # Optimistic terrain counts. Single pixels are retained because
    # the internal wavefront propagates at stride one.
    for out, color in ((6, 9), (7, 15), (8, 1)):
        for dy, dx in square:
            c1(out, color, dy, dx)

    c1(9, 13)

    def isolated_square(out, colors):
        for dy, dx in square:
            for color in colors:
                c1(out, color, dy, dx)
        for dy, dx in outside:
            for color in colors:
                c1(out, color, dy, dx, -4.0)
        p["b1"][out] = -3.0

    isolated_square(10, (11,))
    isolated_square(11, (6, 7))

    # Small pink pickup on a red landing.
    for color, dy, dx in (
        (2, 0, 0), (15, 0, 1), (15, 1, 0),
        (15, 1, 1), (15, 1, 2),
    ):
        c1(12, color, dy, dx)
    p["b1"][12] = -4.0

    # Button edges and an additional lower-body test.
    for i, color in enumerate((9, 15, 8, 6)):
        out = 13 + i
        c1(out, color)
        c1(out, 5, -1, 0)
        c1(out, 0, -1, 0)
        p["b1"][out] = -1.0
        c1(23 + i, color, 2, 0)

    # Tall black panel-column evidence.
    for dy in (-2, -1, 0, 1, 2):
        c1(17, 5, dy, 0)
    p["b1"][17] = -4.0

    # Unsafe-to-toggle terrain under the player.
    for out, color in ((19, 9), (20, 15), (21, 8)):
        c1(out, 14, w=8.0)
        for dy, dx in perimeter:
            c1(out, color, dy, dx)
        p["b1"][out] = -11.0

    # Controls occupy the right-side panel in the canonical renderer.
    # This rejects similarly colored board objects, without encoding
    # particular button rows or particular level layouts.
    c1(22, 17, w=-100.0)
    p["b1"][22] = 50.0

    # Extract top-left player corners.
    for out in range(5):
        c2(out, out)
        c2(out, out, -1, 0, -1.0)
        c2(out, out, 0, -1, -1.0)

    c2(5, 5)

    # raw - excess implements min(count, 1).
    for raw, excess, src in ((6, 7, 6), (8, 9, 7), (10, 11, 8)):
        c2(raw, src)
        c2(excess, src)
        p["b2"][excess] = -1.0

    for out, src in (
        (12, 9), (13, 10), (14, 11), (15, 12),
        (20, 17), (24, 19), (25, 20), (26, 21),
    ):
        c2(out, src)

    # Filtered controls: a top edge with a colored body three rows
    # below it, situated on the panel side of the image.
    for i in range(4):
        out = 16 + i
        c2(out, 13 + i)
        c2(out, 23 + i, 1, 0)
        c2(out, 22, w=-10.0)
        p["b2"][out] = -1.0

    def orange(out, w):
        ax(out, 6, w)
        ax(out, 7, -w)

    def pink(out, w):
        ax(out, 8, w)
        ax(out, 9, -w)

    def pale(out, w):
        ax(out, 10, w)
        ax(out, 11, -w)

    def destinations(out, w):
        for src in (1, 2, 3, 4):
            ax(out, src, w)

    def pivot_axes(out, w):
        ax(out, 12, w, 1)
        ax(out, 12, w, 2)

    def unsafe_axes(out, src, w):
        ax(out, src, w, 1)
        ax(out, src, w, 2)

    # Optimistic navigation mask. Pink checkerboards are not included:
    # they can require an unlock pickup before their control exists.
    p["b3"][0] = 1.0
    ax(0, 5, -1.0)
    orange(0, -1.0)
    pale(0, -1.0)
    pivot_axes(0, -1.0)
    ax(0, 20, 100.0, 2)

    ax(1, 13)  # Goal.
    ax(2, 15)  # Uncollected pickup.

    # Currently usable destinations: up, down, left, right.
    for out, dst in zip(range(3, 7), range(1, 5)):
        ax(out, dst)
        ax(out, 5)
        p["b3"][out] = -1.0

    # Relay gates open at an appropriate blocked player destination
    # or a matching, filtered control.
    p["b3"][7] = 2.0
    destinations(7, -1.0)
    orange(7, -1.0)
    ax(7, 5)
    ax(7, 16, -4.0)
    unsafe_axes(7, 24, 4.0)

    p["b3"][8] = 2.0
    destinations(8, -1.0)
    pink(8, -1.0)
    ax(8, 5)
    ax(8, 17, -4.0)
    unsafe_axes(8, 25, 4.0)

    p["b3"][9] = 5.0
    destinations(9, -4.0)
    pivot_axes(9, -1.0)
    ax(9, 5, 4.0)
    orange(9, 4.0)
    pink(9, 4.0)
    ax(9, 18, -8.0)
    unsafe_axes(9, 26, 4.0)

    ax(10, 14)

    # Teleporter virtual wires.
    p["b3"][11] = 1.0
    ax(11, 14, -1.0, 1)
    ax(11, 14, -1.0, 2)
    ax(11, 20, 100.0, 2)

    # Legal button pixels.
    for out, src in zip(range(12, 16), range(16, 20)):
        ax(out, src)
        ax(out, 20, 1.0, 2)
        p["b3"][out] = -1.0

    # Prospective teleporter entries.
    for out, dst in zip(range(16, 20), range(1, 5)):
        ax(out, dst)
        ax(out, 14)
        p["b3"][out] = -1.0

    # Non-button click penalty.
    p["b3"][20] = 2.0
    ax(20, 20, -1.0, 2)
    for src in range(16, 20):
        ax(20, src, -1.0)

    # Player-local blocked-frontier seeds.
    p["b3"][24] = -1.0
    destinations(24, 1.0)
    orange(24, 1.0)
    ax(24, 5, -1.0)
    unsafe_axes(24, 24, -4.0)

    p["b3"][25] = -1.0
    destinations(25, 1.0)
    pink(25, 1.0)
    ax(25, 5, -1.0)
    unsafe_axes(25, 25, -4.0)

    p["b3"][26] = -4.0
    destinations(26, 4.0)
    pivot_axes(26, 1.0)
    ax(26, 5, -4.0)
    orange(26, -4.0)
    pink(26, -4.0)
    unsafe_axes(26, 26, -4.0)

    BIG = 1000.0
    cross = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    # H0-H1: clipped wavefront. A very weak goal source gives an
    # existing pickup a substantial head start.
    for out in (0, 1):
        for dy, dx in cross:
            rec(out, 0, dy, dx)
            rec(out, 1, dy, dx, -1.0)
        rec(out, 15)
        p["Uh"][out, 0] = -BIG
        p["Uh"][out, 1] = np.float32(1e-36)
        p["Uh"][out, 2] = 1.0
    p["bh"][1] = -1.0

    # H2: accumulated wavefront arrival value.
    rec(2, 2)
    rec(2, 0)
    rec(2, 1, w=-1.0)
    p["Uh"][2, 0] = -BIG

    # H3..H6: arrival values at currently usable destinations.
    for out, mask in zip(range(3, 7), range(3, 7)):
        rec(out, 2)
        p["Uh"][out, mask] = BIG
        p["bh"][out] = -BIG

    # H7/9/11 contain local requests and gated button readouts.
    # H8/10/12 diffuse each request independently.
    relay_specs = (
        (7, 8, 7, 24),
        (9, 10, 8, 25),
        (11, 12, 9, 26),
    )
    for local, spread, block, seed in relay_specs:
        rec(local, 2)
        rec(local, spread)
        p["Uh"][local, block] = -BIG
        p["Uh"][local, seed] = 1.0

        rec(spread, spread, w=0.10)
        for dy, dx in cross[1:]:
            rec(spread, spread, dy, dx, 0.22)
        rec(spread, local, w=0.02)

    # H13-H14: clipped teleporter-wire wavefront.
    for out in (13, 14):
        for dy, dx in cross:
            rec(out, 13, dy, dx)
            rec(out, 14, dy, dx, -1.0)
        rec(out, 15)
        p["Uh"][out, 11] = -BIG
    p["bh"][14] = -1.0

    # Couple the two wavefronts at actual terminal patterns only.
    rec(15, 0)
    rec(15, 1, w=-1.0)
    rec(15, 13)
    rec(15, 14, w=-1.0)
    p["Uh"][15, 10] = BIG
    p["bh"][15] = -BIG

    # Movement scores.
    for i in range(4):
        p["Wz"][i, 3 + i] = 1.0
        p["Wo"][1 + i, i] = 1.0

        p["Wz"][8 + i, 32 + 3 + i] = 1.0
        p["Wo"][1 + i, 8 + i] = 0.01

    # Switch scores subtract relay feedback and seed contributions.
    # Button visibility is a hard gate at this scale.
    for i, (local, spread, _, seed) in enumerate(relay_specs):
        unit = 4 + i
        button = 12 + i
        p["Wz"][unit, local] = 1.0
        p["Wz"][unit, spread] = -1.0
        p["Wz"][unit, 32 + seed] = -1.0
        p["Wz"][unit, 32 + button] = 300.0
        p["bz"][unit] = -300.5
        p["Wo"][6, unit] = 1.0

        # Low-priority fallback when a blocked frontier exists but the
        # optimistic wavefront has not reached the current component.
        fallback = 24 + i
        p["Wz"][fallback, 32 + seed] = 1.0
        p["Wz"][fallback, 32 + button] = 2.0
        p["bz"][fallback] = -2.0
        p["Wo"][6, fallback] = 0.1

    # A visible pink control can now be used to make a nearby
    # checkerboard solid even though that checkerboard was excluded
    # from the initial planning graph.
    p["Wo"][6, 25] = 150.0

    p["bo"][0] = -10000.0
    p["bo"][5] = -10000.0
    p["bo"][7] = -10000.0
    p["bo"][6] = -1e-5

    # Prospective teleporter-entry memory.
    for i in range(4):
        unit = 16 + i
        p["Wz"][unit, 32 + 16 + i] = 1.0
        p["Wmz"][i, unit] = 5.0

        yes, no, action = 4 + i, 8 + i, 1 + i
        for mem_unit, bias in ((yes, -30.0), (no, -50.0)):
            p["Wm"][mem_unit, i] = 20.0
            p["Wma"][mem_unit, action] = 20.0
            p["bm"][mem_unit] = bias
        p["Wo"][6, 128 + yes] = 200.0
        p["Wo"][6, 128 + no] = -200.0

        # Continue rather than reverse when spatial scores tie.
        mem_unit = 16 + i
        p["Wma"][mem_unit, action] = 20.0
        p["Wo"][action, 128 + mem_unit] = 0.001
        opposite = (2, 1, 4, 3)[i]
        p["Wo"][opposite, 128 + mem_unit] = -0.001

    # One-step cooldown for non-teleport clicks.
    p["Wma"][20, 6] = 5.0
    p["Wo"][6, 128 + 20] = -1.2

    # Matching request relay readouts select matching legal controls.
    for channel in (7, 9, 11):
        p["Wc"][0, 32 + channel] = 1.0
    p["Wc"][0, 20] = -1000.0

    # With no other request, a memory-triggered teleport click selects
    # the green control.
    p["Wc"][0, 15] = np.float32(1e-30)

    return p
