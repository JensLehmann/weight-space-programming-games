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

    def numeric_kernel(out, dr, dc, weight=1.0):
        for colour in range(16):
            W1[out, colour, 2 + dr, 2 + dc] += (
                weight * (colour + 1)
            )

    # Shifted numerical colour planes. Two planes cover the central
    # octants; four more supply the small-stamp samples.
    numeric_offsets = [
        (0, 0), (1, 1), (-1, 0),
        (0, 2), (2, 0), (0, -1),
    ]
    for ch, (dr, dc) in enumerate(numeric_offsets):
        numeric_kernel(ch, dr, dc)

    red_ch, row_ch, col_ch = 6, 7, 8
    W1[red_ch, 2, 2, 2] = 1
    W1[row_ch, 16, 2, 2] = 63
    W1[col_ch, 17, 2, 2] = 63

    row_hinges = {}
    col_hinges = {}
    for ch, row in enumerate([24, 38, 53], 9):
        row_hinges[row] = ch
        W1[ch, 16, 2, 2] = 63
        b1[ch] = -row

    for ch, col in enumerate([17, 21, 31, 42, 46], 12):
        col_hinges[col] = ch
        W1[ch, 17, 2, 2] = 63
        b1[ch] = -col

    # Isolate the two reading centres on the integer pixel lattice.
    slope = 24.0 / 31.0
    eps = 0.001

    W1[17, 17, 2, 2] = 63
    W1[17, 16, 2, 2] = -63 * slope
    b1[17] = -7 + 7 * slope - eps

    W1[18, 17, 2, 2] = -63
    W1[18, 16, 2, 2] = 63 * slope
    b1[18] = 7 - 7 * slope - eps

    W1[19, 16, 2, 2] = -63
    b1[19] = 7 - eps
    W1[20, 16, 2, 2] = 63
    b1[20] = -38 - eps

    # Palette centres.
    for dr, dc in [
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2), (0, 2),
        (2, -2), (2, 0), (2, 2),
    ]:
        W1[21, 4, 2 + dr, 2 + dc] = 1
    W1[21, 4, 2, 2] = -8
    b1[21] = -7

    # Selected swatch underline, with one further shift in W2.
    W1[22, 0, 4, 2] = 1

    # Small-stamp eye and red surround.
    for dr, dc in [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]:
        W1[23, 2, 2 + dr, 2 + dc] = 1
    W1[23, 0, 2, 2] = 4
    b1[23] = -6

    # Signed differences across the inner edge of each small stamp.
    # Paired ReLUs give an absolute colour difference.
    difference_stencils = [
        ((-2, 0), (-1, 0)),
        ((0, 2), (0, 1)),
        ((2, 0), (1, 0)),
        ((0, -2), (0, -1)),
    ]
    for k, (outside, inside) in enumerate(difference_stencils):
        for s, sign in enumerate((1.0, -1.0)):
            ch = 24 + 2 * k + s
            numeric_kernel(ch, *outside, weight=sign)
            numeric_kernel(ch, *inside, weight=-sign)

    def second(out, inp, weight=1.0, dr=0, dc=0):
        W2[out, inp, 1 + dr, 1 + dc] += weight

    def reading_gate(out):
        second(out, 17, -1024)
        second(out, 18, -1024)
        second(out, 19, -32)
        second(out, 20, -32)

    # Eight interior sectors in clockwise order.
    core_sources = [
        (0, 0, -1),
        (0, -1, 0),
        (0, -1, 1),
        (1, -1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, -1),
        (0, 1, -1),
    ]
    for out, (source, dr, dc) in enumerate(core_sources):
        second(out, source, dr=dr, dc=dc)
        reading_gate(out)

    stamp_sources = [
        (2, -1, 0),
        (3, 0, 1),
        (4, 1, 0),
        (5, 0, -1),
    ]
    for k, (source, dr, dc) in enumerate(stamp_sources):
        second(8 + k, source, dr=dr, dc=dc)
        reading_gate(8 + k)

    # Each marker is unique to one resting basket orientation.
    markers = [
        (24, 31), (24, 42), (38, 46), (53, 42),
        (53, 31), (53, 21), (38, 17), (24, 21),
    ]
    location_penalty = 32.0
    for i, (row, col) in enumerate(markers):
        out = 12 + i
        second(out, red_ch)
        second(out, row_hinges[row], -2 * location_penalty)
        second(out, col_hinges[col], -2 * location_penalty)
        second(out, row_ch, location_penalty)
        second(out, col_ch, location_penalty)
        b2[out] = -location_penalty * (row + col) - 0.01

    second(20, 0)
    second(20, 21, 32)
    b2[20] = -32
    second(21, 21)

    second(22, 0)
    second(22, 21, 32)
    second(22, 22, 32, dr=1)
    b2[22] = -64

    second(23, 21)
    second(23, 22, dr=1)
    b2[23] = -1

    second(24, 23)
    second(25, 23)
    b2[25] = -1

    second(26, col_ch)
    b2[26] = -7.01
    second(27, col_ch, -1)
    b2[27] = 30.99

    difference_shifts = [(0, 0), (0, 1), (1, 0), (0, 0)]
    for k, (dr, dc) in enumerate(difference_shifts):
        second(28 + k, 24 + 2 * k, dr=dr, dc=dc)
        second(28 + k, 25 + 2 * k, dr=dr, dc=dc)
        reading_gate(28 + k)

    def local(out, inp, weight=1.0):
        W3[out, inp] += weight

    def rowmax(out, inp, weight=1.0):
        W3[out, 32 + inp] += weight

    def colmax(out, inp, weight=1.0):
        W3[out, 64 + inp] += weight

    orientation_scale = 32.0
    colour_offset = 0.32

    def active_read(out, reading, orientation, working=False):
        colmax(out, reading)
        rowmax(out, 12 + orientation, orientation_scale)
        local(out, 27 if working else 26, -32)
        b3[out] = -orientation_scale

    for i in range(8):
        active_read(i, i, i)
        active_read(8 + i, i, i, working=True)

    for k in range(4):
        active_read(16 + k, 8 + k, 2 * k)
        active_read(20 + k, 8 + k, 2 * k, working=True)

    # Palette values are offset consistently with the active readings.
    # Adding 30 to the selected swatch discourages clicking it again.
    local(24, 20)
    local(24, 21, -colour_offset)
    local(24, 23, 30)
    local(25, 21)

    local(26, 22)
    local(26, 23, -colour_offset)

    local(28, 24)
    local(28, 25, -1)

    necessity_channels = [27, 29, 30, 31]
    for k, out in enumerate(necessity_channels):
        active_read(out, 28 + k, 2 * k)

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def recurrent(out, inp, weight=1.0, dr=0, dc=0):
        Wh[out, inp, 1 + dr, 1 + dc] += weight

    # Sparse measurement broadcasts. Prefix streams settle within T=128.
    recurrent(0, 0, dr=1)
    Uh[0, 0:8] = 1
    recurrent(1, 1, dc=-1)
    recurrent(1, 0)

    recurrent(2, 2, dr=1)
    Uh[2, 8:16] = 1
    recurrent(3, 3, dc=-1)
    recurrent(3, 2)
    recurrent(4, 4, dc=1)
    recurrent(4, 2)

    recurrent(5, 5, dr=1)
    Uh[5, 16:20] = 1
    recurrent(6, 6, dc=-1)
    recurrent(6, 5)

    gate = 128.0

    def palette_gate(channel, bias=0.0):
        Uh[channel, 25] += gate
        bh[channel] = bias - gate

    # Compare target and working interior readings.
    recurrent(7, 1)
    recurrent(7, 3, -1)
    recurrent(7, 4, -1)
    recurrent(7, 2)
    palette_gate(7, -0.25)

    recurrent(8, 1, -1)
    recurrent(8, 3)
    recurrent(8, 4)
    recurrent(8, 2, -1)
    palette_gate(8, -0.25)

    recurrent(9, 7, -2)
    recurrent(9, 8, -2)
    palette_gate(9, 1)

    # Choose the broad-stroke colour on an interior mismatch;
    # otherwise expose the small-stamp colour.
    recurrent(10, 1)
    recurrent(10, 9, -64)
    palette_gate(10)

    recurrent(11, 6)
    recurrent(11, 7, -64)
    recurrent(11, 8, -64)
    palette_gate(11)

    # Palette distance from the required colour.
    Uh[12, 24] = 1
    recurrent(12, 10, -1)
    recurrent(12, 11, -1)
    palette_gate(12)

    Uh[13, 24] = -1
    recurrent(13, 10)
    recurrent(13, 11)
    palette_gate(13)

    p["Wc"][0, 25] = 20
    p["Wc"][0, 28] = 19.5
    p["Wc"][0, 32 + 12] = -1
    p["Wc"][0, 32 + 13] = -1

    next_unit = 0

    def unit(coefficients, bias=0.0):
        nonlocal next_unit
        j = next_unit
        next_unit += 1
        for index, value in coefficients.items():
            p["Wz"][j, index] = value
        p["bz"][j] = bias
        return j

    def add_action(action, units, multiplier=1.0):
        for j, coefficient in units:
            p["Wo"][action, j] += multiplier * coefficient

    equality = unit({9: 1})

    # Boolean selected-colour mismatch.
    selected_difference = {10: 1, 11: 1, 32 + 26: -1}
    selected_wrong = []
    for sign in (1, -1):
        coefficients = {
            k: sign * v for k, v in selected_difference.items()
        }
        lo = unit(coefficients, -0.25)
        hi = unit(coefficients, -0.75)
        selected_wrong.extend([(lo, 2), (hi, -2)])

    # Small-stamp mismatch, conditional on a correct interior and
    # the presence of a clickable small stamp.
    tab_difference = {}
    for k in range(4):
        tab_difference[32 + 16 + k] = 1
        tab_difference[32 + 20 + k] = -1

    tab_needed = []
    for sign in (1, -1):
        coefficients = {
            k: sign * v for k, v in tab_difference.items()
        }
        coefficients[9] = 32
        coefficients[32 + 28] = 32
        lo = unit(coefficients, -64.25)
        hi = unit(coefficients, -64.75)
        tab_needed.extend([(lo, 2), (hi, -2)])

    # Crucial guard: an outer reading identical to its same-sector
    # interior is not evidence that a small stamp should be applied.
    no_stamp_coefficients = {9: 1}
    for ch in necessity_channels:
        no_stamp_coefficients[32 + ch] = -2
    no_stamp_needed = unit(no_stamp_coefficients)

    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    for orientation, action in enumerate(clockwise_action):
        coefficients = {32 + orientation: 1}
        lo = unit(coefficients)
        hi = unit(coefficients, -0.5)
        add_action(action, [(lo, 2), (hi, -2)])

    p["bo"][0] = -100
    p["bo"][7] = -100

    p["bo"][5] = 5
    p["Wo"][5, equality] = -10
    add_action(5, selected_wrong, -10)

    p["bo"][6] = -5
    p["Wo"][6, equality] = -10
    add_action(6, selected_wrong, 10)
    add_action(6, tab_needed, 20)
    p["Wo"][6, no_stamp_needed] = -20

    return p
