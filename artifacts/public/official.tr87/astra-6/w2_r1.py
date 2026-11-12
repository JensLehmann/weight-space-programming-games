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

    text = [
        ("00100", "00100", "01110", "00100", "11111"),
        ("00001", "00101", "11111", "10100", "10000"),
        ("11111", "10100", "10000", "10100", "11111"),
        ("00100", "11111", "10101", "10101", "00100"),
        ("10001", "11111", "00100", "11111", "10001"),
        ("11111", "00100", "11100", "10000", "11111"),
        ("10001", "11111", "10001", "10001", "11011"),
        ("10000", "11110", "10010", "10010", "11111"),
        ("11111", "10001", "11101", "10101", "11111"),
        ("00111", "00101", "11111", "10100", "11100"),
        ("01110", "01010", "11111", "10001", "11111"),
        ("11111", "10001", "10001", "11111", "00100"),
        ("11110", "10011", "10001", "11001", "01111"),
        ("00100", "11111", "10101", "11111", "00100"),
        ("10101", "10000", "10101", "10000", "11111"),
        ("11101", "10000", "00000", "00001", "10111"),
        ("10111", "00100", "00100", "00100", "11101"),
        ("00100", "10101", "00100", "10101", "00100"),
        ("11011", "10001", "10101", "10001", "11011"),
        ("10111", "10101", "11101", "00000", "10101"),
        ("10001", "00000", "10001", "11011", "01010"),
    ]
    masks = np.array(
        [[[int(c) for c in row] for row in glyph] for glyph in text],
        dtype=np.int32,
    )

    # Pixel-count signatures invariant under quarter turns.
    orbit_set = set()
    for y in range(-2, 3):
        for x in range(-2, 3):
            a, b = y, x
            orbit = []
            for _ in range(4):
                orbit.append((a, b))
                a, b = b, -a
            orbit_set.add(tuple(sorted(set(orbit))))
    orbits = sorted(orbit_set, key=lambda q: (len(q), q))
    counts = np.array([
        [sum(int(g[y + 2, x + 2]) for y, x in orbit)
         for orbit in orbits]
        for g in masks
    ], dtype=np.int32)

    raw = []
    hinges = {}
    ch = 0
    for j, orbit in enumerate(orbits):
        r = ch
        ch += 1
        raw.append(r)
        for y, x in orbit:
            p["W1"][r, 5, y + 2, x + 2] = 1.0
        for v in sorted({
            int(v) for v in counts[:, j] if 0 < v < len(orbit)
        }):
            h = ch
            ch += 1
            hinges[j, v] = h
            p["W1"][h] = p["W1"][r]
            p["b1"][h] = -float(v)
    assert ch == 21

    family_channels = []
    for colour in (10, 7, 11):
        family_channels.append(ch)
        p["W1"][ch, colour, 0, 2] = 1.0
        ch += 1

    borders = [
        ([(-2, x) for x in range(-2, 3)], (-1, 0)),
        ([(2, x) for x in range(-2, 3)], (1, 0)),
        ([(y, -2) for y in range(-2, 3)], (0, -1)),
        ([(y, 2) for y in range(-2, 3)], (0, 1)),
    ]
    border_channels = []
    for line, shift in borders:
        border_channels.append((ch, shift))
        for y, x in line:
            p["W1"][ch, 18, y + 2, x + 2] += 1.0
            for colour in (10, 7, 11):
                p["W1"][ch, colour, y + 2, x + 2] -= 1.0
        ch += 1

    pointer1, row1, line1, col1 = 28, 29, 30, 31

    # A constant-height attention strip at the bracket's left end.
    # It covers the first glyph centre and the three columns to its left.
    # The right end of a wide bracket does not activate this detector.
    p["W1"][pointer1, 0, 2, 3:5] = 10.0
    p["W1"][pointer1, 0, 3, 0:4] = 1.0
    p["b1"][pointer1] = -20.0

    p["W1"][row1, 16, 2, 2] = 63.0
    p["W1"][col1, 17, 2, 2] = 63.0
    p["W1"][line1, 3, 2, 1:4] = 1.0
    p["b1"][line1] = -2.0

    # Exact detectors from L1 distances between signatures.
    for i in range(20):
        bias = 1.0
        for j, orbit in enumerate(orbits):
            v = int(counts[i, j])
            if v == 0:
                p["W2"][i, raw[j], 1, 1] -= 1.0
            elif v == len(orbit):
                p["W2"][i, raw[j], 1, 1] += 1.0
                bias -= v
            else:
                p["W2"][i, raw[j], 1, 1] += 1.0
                p["W2"][i, hinges[j, v], 1, 1] -= 2.0
                bias -= v
        p["W2"][i, family_channels[i // 7], 0, 1] += 1.0
        bias -= 1.0
        for c, (dy, dx) in border_channels:
            p["W2"][i, c, dy + 1, dx + 1] -= 1.0
        p["b2"][i] = bias

    connector2, pointer2 = 20, 21
    lower_guard, source_end_guard, upper_guard = 22, 23, 24
    p["W2"][connector2, line1, 1, 1] = 1.0
    p["W2"][pointer2, pointer1, 1, 1] = 1.0

    p["W2"][lower_guard, row1, 1, 1] = -1.0
    p["b2"][lower_guard] = 38.0

    # Exclude the lower destination band from source queries.
    p["W2"][source_end_guard, row1, 1, 1] = 1.0
    p["b2"][source_end_guard] = -49.0

    # Column maximum is 25 + x/128, an exact coordinate carrier.
    p["W2"][upper_guard, row1, 1, 1] = 1.0
    p["W2"][upper_guard, col1, 1, 1] = 1.0 / 128.0
    p["b2"][upper_guard] = -38.0

    centres = [10, 17, 24, 31, 38, 45, 52]
    for j, x in enumerate(centres):
        p["W2"][25 + j, col1, 1, 1] = 1.0
        p["b2"][25 + j] = -float(x)

    gate = 32.0
    for i in range(20):
        index = i % 7 + 1
        family = i // 7
        for bit in range(3):
            if (index >> bit) & 1:
                p["W3"][bit, i] = 1.0

        # These two features cover both upper and lower glyphs.
        p["W3"][4, i] = 1.0
        p["W3"][15, i] = index + 8 * family

        # Upper-only weighted count for the rule-editing raster scan.
        p["W3"][31, i] = 1.0 if family == 0 else 16.0

    for c in (0, 1, 2, 31):
        p["W3"][c, upper_guard] = -gate

    p["W3"][3, connector2] = 1.0
    p["W3"][3, upper_guard] = -gate

    # Source identities selected by bracket-column attention.
    for k in range(7):
        for i in range(20):
            if i % 7 == k and i // 7 != 1:
                p["W3"][5 + k, i] = 1.0
        p["W3"][5 + k, 64 + pointer2] = gate
        p["W3"][5 + k, lower_guard] = -gate
        p["W3"][5 + k, source_end_guard] = -gate
        p["b3"][5 + k] = -gate

    p["W3"][13, pointer2] = 1.0
    p["W3"][14, pointer2] = 1.0
    p["W3"][14, upper_guard] = -gate

    p["W3"][23, 64 + upper_guard] = -128.0
    p["b3"][23] = 3263.0

    # Lower-string columns, including both centre parities.
    large = 131072.0
    for family_class, start in ((0, 16), (1, 24)):
        for j, centre in enumerate(centres):
            out = start + j
            for i in range(20):
                belongs = (
                    i // 7 == 0 if family_class == 0 else i // 7 != 0
                )
                if belongs:
                    p["W3"][out, i] = large + i % 7 + 1
            p["W3"][out, lower_guard] = -large
            p["W3"][out, 25 + j] = -128.0
            p["W3"][out, 64 + upper_guard] = 8192.0
            p["b3"][out] = (
                -large + 208.0 - 64.0 * centre - 204800.0
            )

    # Visual substitution lookup via horizontal transport.
    for bit in range(3):
        p["Wh"][bit, bit, 1, 0] = 0.5
        p["Uh"][bit, bit] = 512.0
    p["Wh"][3, 3, 1, 0] = 0.5
    p["Uh"][3, 3] = 16.0

    spatial = 65536.0
    for k in range(1, 7):
        out = 3 + k
        for bit in range(3):
            p["Uh"][out, bit] = float(1 << bit)
            if (k >> bit) & 1:
                p["Wh"][out, bit, 1, 0] = 8.0
                p["bh"][out] -= 8.0
            else:
                p["Wh"][out, bit, 1, 0] = -8.0
        p["Wh"][out, 3, 1, 0] = 128.0
        p["Uh"][out, 4] = spatial
        p["bh"][out] -= spatial + 128.0

    # Upper-text raster prefix counts.
    p["Wh"][10, 10, 1, 0] = 1.0
    p["Uh"][10, 31] = 1.0
    p["Wh"][11, 11, 0, 1] = 1.0
    p["Wh"][11, 10, 1, 1] = 1.0
    p["Uh"][11, 23] = -4096.0
    p["Wh"][12, 12, 1, 2] = 1.0
    p["Wh"][12, 11, 0, 1] = 1.0

    # Direct readout beneath the upper bracket, in either game mode.
    p["Wh"][13, 13, 0, 1] = 0.5
    p["Uh"][13, 13] = 1.0
    attention = 1024.0
    for out in (14, 15):
        p["Wh"][out, 13, 1, 1] = 64.0 * attention
        p["Uh"][out, 4] = large
        p["bh"][out] = -large - attention
    p["Uh"][14, 15] = 1.0
    p["Wh"][15, 10, 1, 1] = 1.0
    p["Wh"][15, 12, 1, 1] = 1.0

    nz = 0

    def zunit(coefficients, bias=0.0):
        nonlocal nz
        i = nz
        nz += 1
        for g, value in coefficients.items():
            p["Wz"][i, g] = value
        p["bz"][i] = bias
        v = np.zeros(128, np.float32)
        v[i] = 1.0
        return v

    normal_target = np.zeros(128, np.float32)
    normal_present = np.zeros(128, np.float32)
    for k in range(6):
        terms = {4 + k: 1.0, 32 + 5 + k: 8.0}
        a = zunit(terms, -8.0)
        b = zunit(terms, -8.5)
        normal_target += a
        normal_present += 2.0 * (a - b)

    mode = zunit({32 + 14: 1.0})

    code_hinges = {}
    for threshold in (0, 1, 7, 8, 9, 15, 16, 17, 23, 24):
        code_hinges[threshold] = zunit(
            {14: 1.0}, -float(threshold)
        )

    family_forms = []
    for start in (0, 8, 16):
        family_forms.append(
            code_hinges[start] - code_hinges[start + 1]
            - code_hinges[start + 7] + code_hinges[start + 8]
        )
    current_index = (
        code_hinges[0] - 8.0 * family_forms[1]
        - 16.0 * family_forms[2]
    )

    rank = zunit({15: 1.0})
    quotient = np.zeros(128, np.float32)
    for j in range(1, 9):
        quotient += (
            zunit({15: 1.0}, -(16.0 * j - 1.0))
            - zunit({15: 1.0}, -16.0 * j)
        )

    values = [[], []]
    occupied = [[], []]
    for family_class, start in ((0, 16), (1, 24)):
        for j in range(7):
            g = 32 + start + j
            v0 = zunit({g: 1.0})
            v1 = zunit({g: 1.0}, -1.0)
            v100 = zunit({g: 1.0}, -100.0)
            v101 = zunit({g: 1.0}, -101.0)
            values[family_class].append(
                -15.0 * v0 + 16.0 * v1
                - 192.0 * v100 + 192.0 * v101
            )
            occupied[family_class].append(v0 - v1)

    assert nz <= 128

    bin_forms = [
        rank - 16.0 * quotient,
        quotient.copy(),
    ]
    for family_class in range(2):
        for j in range(3):
            bin_forms[family_class] -= occupied[family_class][j]

    # Thermometer encoding of the lower column associated with a rule.
    for family_class in range(2):
        for j in range(8):
            m = 8 * family_class + j
            p["Wmz"][m] = 16.0 * bin_forms[family_class]
            p["bm"][m] = 16.0 * (2.5 - j)

    select_gate = 256.0
    rule_families = [
        family_forms[0],
        family_forms[1] + family_forms[2],
    ]
    for family_class in range(2):
        for j in range(7):
            branch = 7 * family_class + j
            delta = values[family_class][j] - current_index
            for t, cutoff in enumerate((-3.5, 0.0, 3.5)):
                m = 16 + 3 * branch + t
                p["Wmz"][m] = (
                    16.0 * delta
                    + select_gate * (mode + rule_families[family_class])
                )
                p["Wm"][m, 8 * family_class + j] = select_gate / 2.0
                p["Wm"][m, 8 * family_class + j + 1] = -select_gate / 2.0
                p["bm"][m] = -16.0 * cutoff - 3.0 * select_gate

    # Ordinary translation now compares against the directly attended
    # destination, not an independently band-gated column maximum.
    normal_delta = normal_target - current_index
    for t, (cutoff, sharpness) in enumerate(
        ((-3.5, 16.0), (0.0, 2.0), (3.5, 16.0))
    ):
        m = 58 + t
        p["Wmz"][m] = (
            sharpness * normal_delta
            + select_gate * (normal_present - mode)
        )
        p["bm"][m] = -sharpness * cutoff - select_gate

    p["Wmz"][61] = 16.0 * (normal_present - mode)
    p["bm"][61] = -8.0

    # Synchronization for the one-action-delayed rule-position register.
    p["Wmz"][62] = 16.0 * mode
    p["Wma"][62, 4] = 16.0
    p["Wm"][62, 63] = -16.0
    p["bm"][62] = -8.0
    p["bm"][63] = 16.0

    p["bo"][:] = -100.0
    p["bo"][1] = 29.0
    p["bo"][2] = -29.0
    p["bo"][3] = -2.0
    p["bo"][4] = 1.0

    # Cyclic shortest-direction decisions.
    for branch in range(14):
        for t, coefficient in enumerate((-2.0, 2.0, -2.0)):
            m = 16 + 3 * branch + t
            p["Wo"][2, 128 + m] += coefficient
            p["Wo"][1, 128 + m] -= coefficient
    p["Wo"][2, :128] += 2.0 * mode
    p["Wo"][1, :128] -= 2.0 * mode

    for t, coefficient in enumerate((-2.0, 2.0, -2.0)):
        p["Wo"][2, 128 + 58 + t] += coefficient
        p["Wo"][1, 128 + 58 + t] -= coefficient
    p["Wo"][2, 128 + 61] += 1.0
    p["Wo"][1, 128 + 61] -= 1.0

    p["Wo"][1, 128 + 62] += 100.0
    p["bo"][1] += 100.0

    return p
