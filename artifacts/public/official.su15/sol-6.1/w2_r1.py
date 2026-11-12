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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def ch(out, inp, dy=0, dx=0, value=1.0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    def f3(out, inp, value=1.0, axial=0):
        p["W3"][out, inp + 32 * axial] += value

    # Numbered-piece landmarks.
    c1(0, colors[0])
    for rank in range(1, 9):
        color = colors[rank]
        br = 8 + rank

        if rank < 6:
            for dy, dx in [(-1, -1), (-1, 0), (0, -1)]:
                c1(rank, color, dy, dx)
                c1(br, color, -dy, -dx)
            for dy, dx in [(-2, -1), (-1, -2),
                           (-2, 0), (0, -2)]:
                c1(rank, color, dy, dx, -3.0)
                c1(br, color, -dy, -dx, -3.0)
            p["b1"][rank] = -2.0
            p["b1"][br] = -2.0
        else:
            # Full square corners reject rounded goals and enemy patterns.
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    c1(rank, color, dy, dx)
                    c1(br, color, dy, dx)
            for t in range(-1, 2):
                c1(rank, color, -2, t, -10.0)
                c1(rank, color, t, -2, -10.0)
                c1(br, color, 2, t, -10.0)
                c1(br, color, t, 2, -10.0)
            p["b1"][rank] = -8.0
            p["b1"][br] = -8.0

    # Expanded support of the unambiguous numbered-piece colors.
    for color in colors[:6]:
        p["W1"][17, color, :, :] = 1.0

    # Visible target surface remains useful under partial occlusion.
    c1(18, 9)

    # Coordinate gates.
    c1(20, 16, value=-63.0)
    p["b1"][20] = 9.5
    c1(21, 16, value=63.0)
    p["b1"][21] = -9.5
    c1(22, 17, value=-63.0)
    p["b1"][22] = 23.5

    for out, (dy, dx) in enumerate(
        [(0, -2), (0, 2), (-2, 0), (2, 0)], start=23
    ):
        c1(out, 9, dy, dx)

    c1(27, 2)
    c1(29, 16, value=63.0)

    # Patterned-enemy landmarks.
    first_pattern = [
        (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2),
        (1, -1), (1, 0), (1, 1),
    ]
    for dy, dx in first_pattern:
        c1(31, 7, dy, dx)
        c1(30, 14, dy, dx)
    for out, color in [(31, 7), (30, 14)]:
        c1(out, color, 0, -1, -4.0)
        c1(out, color, 0, 1, -4.0)
        p["b1"][out] = -7.0

    third_pattern = [
        (-2, -2), (-2, 0), (-2, 2),
        (-1, -1), (-1, 1),
        (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1),
    ]
    for dy, dx in third_pattern:
        c1(28, 13, dy, dx)
    c1(28, 13, -1, 0, -5.0)
    p["b1"][28] = -12.0

    # Approximate piece-center sources.
    for rank, width in enumerate(sizes):
        if rank == 0:
            c2(rank, 0)
        elif width <= 5:
            tl = width // 2 - 1
            br = width // 2 - (width - 2)
            c2(rank, rank, -tl, -tl, 0.5)
            c2(rank, 8 + rank, -br, -br, 0.5)
        else:
            c2(rank, rank, -1, -1)
            c2(rank, 8 + rank, 1, 1)
        c2(rank, 20, value=-100.0)

    # Row priorities: larger values represent smaller ranks.
    for out, dy in [(9, 0), (10, -1), (11, 1)]:
        for rank in range(9):
            c2(out, rank, dy, 0, 9.0 - rank)
        c2(out, 20, dy, 0, -100.0)

    # Header requirement, encoded as requested rank + 1.
    for rank in range(9):
        c2(12, rank, value=rank + 1.0)
    c2(12, 21, value=-100.0)
    c2(12, 22, value=-100.0)

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            c2(13, 17, dy, dx)
            for rank in range(6, 9):
                c2(14, rank, dy, dx)
                c2(14, 8 + rank, dy, dx)
    c2(13, 20, value=-10000.0)
    c2(14, 20, value=-10000.0)

    c2(15, 18)
    c2(15, 20, value=-100.0)

    # Opposite target pixels approximately recognize delivered pieces.
    c2(16, 23, 0, -1)
    c2(16, 24, 0, 1)
    p["b2"][16] = -1.0
    c2(17, 25, -1, 0)
    c2(17, 26, 1, 0)
    p["b2"][17] = -1.0

    c2(18, 27)
    for source in [28, 30, 31]:
        c2(19, source)
    c2(19, 20, value=-100.0)
    c2(20, 29)

    for rank in range(9):
        f3(rank, rank)
    f3(9, 9, axial=1)
    f3(10, 10, axial=1)
    f3(11, 11, axial=1)
    f3(12, 12, axial=1)

    # Paired ReLUs encode clipped support masks.
    f3(13, 13)
    f3(14, 13)
    p["b3"][14] = -1.0
    f3(15, 14)
    f3(16, 14)
    p["b3"][16] = -1.0

    f3(17, 15)
    f3(18, 16)
    f3(19, 17)
    f3(20, 18)
    f3(22, 20)
    f3(24, 19)

    f3(26, 20, -1.0)
    p["b3"][26] = 9.5
    f3(27, 20)
    p["b3"][27] = -62.5

    # H0: downward broadcast of the header requirement.
    ch(0, 0, -1, 0)
    p["Uh"][0, 12] = 1.0

    # H1/H2: directional prefix maxima, stored relative to row priority.
    ch(1, 1, -1, 0)
    p["Uh"][1, 10] = 1.0
    p["Uh"][1, 9] = -1.0
    ch(2, 2, 1, 0)
    p["Uh"][2, 11] = 1.0
    p["Uh"][2, 9] = -1.0

    # Global priority is H3 + H2 + F3[9].
    ch(3, 1)
    ch(3, 2, value=-1.0)

    # H4/H9: iteration counter and initialization exclusion gate.
    ch(4, 4)
    p["bh"][4] = 1.0
    ch(9, 4, value=-1.0)
    p["bh"][9] = 76.0

    # H10: positive when merging is needed.
    ch(10, 0)
    ch(10, 3)
    ch(10, 2)
    p["Uh"][10, 9] = 1.0
    p["bh"][10] = -10.0

    # H11: positive when the smallest remaining rank is oversized.
    ch(11, 0, value=-1.0)
    ch(11, 3, value=-1.0)
    ch(11, 2, value=-1.0)
    p["Uh"][11, 9] = -1.0
    p["bh"][11] = 10.0

    # H7: minimum-rank sources below the requirement.
    gate = 20.0
    ch(7, 0)
    ch(7, 3, value=-20.0)
    ch(7, 2, value=-20.0)
    ch(7, 9, value=-1000.0)
    p["Uh"][7, 9] = -20.0
    for rank in range(9):
        p["Uh"][7, rank] = (
            gate - rank - 1.5 + 20.0 * (9.0 - rank)
        )
    p["bh"][7] = -gate

    # H12: delivery sources, enabled only in exact-rank mode.
    for rank in range(9):
        p["Uh"][12, rank] = 1.0
    p["Uh"][12, 24] = 0.50
    ch(12, 10, value=-2.0)
    ch(12, 11, value=-2.0)
    ch(12, 9, value=-1000.0)
    p["Uh"][12, 18] = -3.0
    p["Uh"][12, 19] = -3.0

    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def diffusion(out, center, neighbor):
        ch(out, out, value=center)
        for dy, dx in cardinal:
            ch(out, out, dy, dx, neighbor)

    # H8: merge attraction, without a persistent momentum buffer.
    diffusion(8, 0.218, 0.20)
    ch(8, 7)

    # H13: delivery attraction.
    diffusion(13, 0.198, 0.20)
    ch(13, 12)

    # H5: shorter-range repulsion.
    diffusion(5, 0.18, 0.20)
    ch(5, 7)
    ch(5, 12)
    p["Uh"][5, 24] = 0.35

    # H6: reject distant clicks, except while waiting for downgrading.
    ch(6, 5, value=-1.0)
    ch(6, 10, value=0.012)
    ch(6, 11, value=-0.10)
    p["bh"][6] = 0.015

    # H14/H15: broad target guidance.
    diffusion(14, 1.05, 0.20)
    ch(14, 15, value=-0.85)
    ch(14, 10, value=-2.0)
    p["Uh"][14, 17] = 1.0
    ch(15, 14)

    wc = p["Wc"][0]
    wc[32 + 8] = 500.0
    wc[32 + 13] = 500.0
    wc[32 + 5] = -525.0
    wc[32 + 14] = 10.0
    wc[32 + 6] = -100000000.0

    wc[13] = -1000000.0
    wc[14] = 1000000.0
    wc[15] = -1000000.0
    wc[16] = 1000000.0
    wc[26] = -1000000000.0
    wc[27] = -1000000000.0

    # Action head: click normally, undo a red impossibility indication.
    p["Wz"][0, 32 + 20] = 1.0
    p["bo"][:] = -100.0
    p["bo"][6] = 1.0
    p["bo"][7] = -1.0
    p["Wo"][7, 0] = 10.0

    return p
