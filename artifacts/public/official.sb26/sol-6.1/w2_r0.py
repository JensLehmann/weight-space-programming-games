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
    colours = (6, 8, 9, 11, 12, 14, 15)
    index = {colour: i for i, colour in enumerate(colours)}

    def c1(out, inp, dy=0, dx=0, value=1.0):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def c2(out, inp, dy=0, dx=0, value=1.0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def ax(out, inp, which="local", value=1.0):
        offset = {"local": 0, "row": 32, "col": 64}[which]
        p["W3"][out, offset + inp] += value

    # W1: seven solid templates, seven target-border templates,
    # socket and selection templates, seven reference templates,
    # seven frame-border templates, and two coordinates.
    for i, colour in enumerate(colours):
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                c1(i, colour, dy, dx)
        p["b1"][i] = -15.0

        c1(7 + i, colour)
        c1(7 + i, 5, -1, 0)
        p["b1"][7 + i] = -1.0

        for dy in range(-1, 3):
            for dx in range(-1, 3):
                border = dy in (-1, 2) or dx in (-1, 2)
                c1(16 + i, colour, dy, dx,
                   1.0 if border else -1.0)
        p["b1"][16 + i] = -11.0

        for dy in range(-2, 3):
            c1(23 + i, colour, dy, 0)
        p["b1"][23 + i] = -4.0

    for dy in (0, 1):
        for dx in (0, 1):
            c1(14, 2, dy, dx)
    p["b1"][14] = -3.0

    for dy, dx in ((-1, -1), (-1, 0), (0, -1)):
        c1(15, 0, dy, dx)
    p["b1"][15] = -2.0

    c1(30, 16)
    c1(31, 17)

    # W2 tile identities use amplitude 1 for solids, 1/2 for refs.
    # Selected-colour detectors retain unit amplitude for either.
    for i in range(7):
        c2(i, i)
        c2(i, 16 + i, value=0.5)

        for dx in (-1, 0, 1):
            c2(7 + i, 7 + i, 0, dx)

        c2(16 + i, i)
        c2(16 + i, 16 + i)
        c2(16 + i, 15, -1, -1)
        p["b2"][16 + i] = -1.0

        c2(23 + i, 23 + i)

    c2(14, 14)

    # Two ramps whose difference is the inventory indicator.
    c2(15, 30, value=63.0)
    p["b2"][15] = -53.0
    c2(31, 30, value=63.0)
    p["b2"][31] = -54.0

    # Selected reference detector.
    for i in range(7):
        c2(30, 16 + i)
    c2(30, 15, -1, -1)
    p["b2"][30] = -1.0

    # Mix weak coordinates into two target channels. Away from
    # target borders these channels are simply coordinates.
    c2(7 + index[6], 31, value=0.01)
    c2(7 + index[12], 30, value=0.01)

    col_channel = 7 + index[6]
    row_channel = 7 + index[12]

    def add_board_socket(out, value=1.0):
        ax(out, 14, value=value)
        ax(out, 15, value=-value)
        ax(out, 31, value=value)

    def add_inventory_indicator(out, value=1.0):
        ax(out, 15, value=value)
        ax(out, 31, value=-value)

    # Palette-specific terminal priorities and geometric biases.
    priorities = {
        14: 0.90,
        15: 0.80,
        6: 0.70,
        9: 0.60,
        8: 0.50,
        11: 0.40,
        12: 0.30,
    }
    depth_bias = {
        6: 0.03,
        8: 2.00,
        9: 0.03,
        11: 0.03,
        12: 0.03,
        14: 1.50,
        15: 0.03,
    }
    horizontal_bias = {
        6: -0.75,
        8: -0.10,
        9: 0.00,
        11: 0.75,
        12: 0.00,
        14: -0.75,
        15: 0.75,
    }

    # W3 0..6: socket capacities for each selected colour.
    # W3 7..13: unselected inventory identities.
    # W3 14..20: selected-colour axial crosses.
    for i, colour in enumerate(colours):
        add_board_socket(i, 4.0)

        # Root colour 8 is not treated as a named subroutine
        # when deciding where to put solid colour-8 terminals.
        if colour != 8:
            ax(i, 23 + i, "row", 0.75)

        if colour != 14:
            ax(i, 7 + i, "col", 0.35 / 3.0)

        ax(i, row_channel, value=100.0 * depth_bias[colour])
        ax(i, col_channel,
           value=100.0 * horizontal_bias[colour])
        p["b3"][i] = -3.0

        ax(7 + i, i)
        add_inventory_indicator(7 + i)
        for j in range(7):
            ax(7 + i, 16 + j, "row", -1.0)
        p["b3"][7 + i] = -1.0

        ax(14 + i, 16 + i, "row")
        ax(14 + i, 16 + i, "col")

    # Board sockets only: inventory vacancies must not delay RUN.
    add_board_socket(21)

    # Unselected inventory support.
    for i in range(7):
        ax(22, i)
        ax(22, 16 + i, "row", -1.0)
    add_inventory_indicator(22)
    p["b3"][22] = -1.0

    for i in range(7):
        ax(23, 16 + i, "row")
        ax(24, 16 + i, "row")
        ax(24, 16 + i, "col")

    ax(25, row_channel, value=100.0)
    ax(26, col_channel, value=100.0)

    # Candidate support: unit sockets/solids, half-unit refs.
    add_board_socket(27, 2.0)
    for i in range(7):
        ax(27, i)
    add_inventory_indicator(27)
    p["b3"][27] = -1.0

    # Reference-selection source.
    ax(28, 30, "row")
    ax(28, 30, "col")

    # Selected board tile, used to recover perturbed starts.
    for i in range(7):
        ax(29, 16 + i)
    add_inventory_indicator(29, -1.0)

    # Gate used by the reference flood. At a socket this is
    # 1-row, giving the flood a row-dependent saturation cap.
    add_board_socket(30, 2.0)
    for i in range(7):
        ax(30, i)
    add_inventory_indicator(30)
    ax(30, row_channel, value=-100.0)
    p["b3"][30] = -1.0

    # Secondary alignment for colour 14. Alignment is retained
    # in the root and matching-colour frame, but suppressed in
    # other named frames, where repeated-substring alignment
    # is ambiguous.
    add_board_socket(31)
    ax(31, 7 + index[14], "col", 1.0 / 3.0)
    for i, colour in enumerate(colours):
        if colour not in (8, 14):
            ax(31, 23 + i, "row", -1.0)
    p["b3"][31] = -1.0

    # Seven paired routing fields. Their represented value is
    # A-B. The update clips incoming signals to local capacity.
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for i in range(7):
        a, b = 2 * i, 2 * i + 1
        for out in (a, b):
            for dy, dx in neighbours:
                p["Wh"][out, a, dy + 1, dx + 1] += 1.0
                p["Wh"][out, b, dy + 1, dx + 1] -= 1.0
            p["Uh"][out, 14 + i] = 2.0

        p["bh"][b] = -1.0
        p["Uh"][b, 27] += 1.0
        p["Uh"][b, i] -= 1.0
        p["Uh"][b, 7 + i] -= 1.0

    p["Uh"][2 * index[14] + 1, 31] -= 0.35

    # Reference-only flood, also represented as A-B.
    # Its capacity is 1 in background and approximately row
    # at sockets, so subtracting it prefers upper frames.
    ra, rb = 14, 15
    for out in (ra, rb):
        for dy, dx in ((0, 0),) + neighbours:
            p["Wh"][out, ra, dy + 1, dx + 1] += 1.0
            p["Wh"][out, rb, dy + 1, dx + 1] -= 1.0
        p["Uh"][out, 28] = 2.0
    p["bh"][rb] = -1.0
    p["Uh"][rb, 30] = 1.0

    # Action head; memory tensors remain zero.
    p["Wz"][0, 32 + 21] = 1.0
    p["Wz"][1, 32 + 29] = 1.0

    p["bo"].fill(-10.0)
    p["bo"][5] = 0.0
    p["bo"][6] = -1.0
    p["Wo"][6, 0] = 4.0

    p["bo"][0] = -4.0
    p["Wo"][0, 1] = 8.0

    # Click head.
    for i, colour in enumerate(colours):
        p["Wc"][0, 32 + 2 * i] = 1.0
        p["Wc"][0, 32 + 2 * i + 1] = -1.0
        p["Wc"][0, 7 + i] = priorities[colour]

    p["Wc"][0, 32 + ra] = -8.0
    p["Wc"][0, 32 + rb] = 8.0

    p["Wc"][0, 21] = 0.20
    p["Wc"][0, 22] = 16.0
    p["Wc"][0, 25] = 0.002
    p["Wc"][0, 26] = -0.002
    p["Wc"][0, 27] = 16.0
    p["bc"][0] = -16.0

    return p
