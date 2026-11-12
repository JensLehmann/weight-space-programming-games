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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    colours = [0, 15, 12, 11, 14, 8, 9]
    labels = {c: i + 1 for i, c in enumerate(colours)}

    # Outer sectors clockwise; then paired samples of each inset.
    outer = [
        (0, 2), (0, 7), (2, 9), (7, 9),
        (9, 7), (9, 2), (7, 0), (2, 0),
    ]
    inner = [
        (1, 3), (1, 6),
        (3, 8), (6, 8),
        (8, 3), (8, 6),
        (3, 1), (6, 1),
    ]
    samples = outer + inner

    # This coordinate is invariant under the displacement between
    # the target and work panels.
    lines = [24 * (r + 3) - 31 * (c + 3) for r, c in samples]

    for i, v in enumerate(lines):
        p["W1"][i, 16, 2, 2] = 24 * 63
        p["W1"][i, 17, 2, 2] = -31 * 63
        p["b1"][i] = -v

    p["W1"][16, 16, 2, 2] = 24 * 63
    p["W1"][16, 17, 2, 2] = -31 * 63
    p["b1"][16] = 2000

    for c, v in labels.items():
        p["W1"][17, c, 2, 2] = v

    # Five-red-pixel edge detectors with affine coordinate preferences.
    # "d" follows slope +1; "a" follows slope -1.
    pose_specs = [
        ("h", -1, 0, 35, 11),
        ("d", -1, 1, -10, 8),
        ("v", 0, 1, -32, 14),
        ("a", 1, 1, -85, 10),
        ("h", 1, 0, -35, 18),
        ("d", 1, -1, -25, 7),
        ("v", 0, -1, 32, 15),
        ("a", -1, -1, 50, 5),
    ]
    for i, (kind, ry, cx, offset, scale) in enumerate(pose_specs):
        q = 18 + i
        for t in range(-2, 3):
            if kind == "h":
                dr, dc = 0, t
            elif kind == "v":
                dr, dc = t, 0
            elif kind == "d":
                dr, dc = t, t
            else:
                dr, dc = t, -t
            p["W1"][q, 2, 2 + dr, 2 + dc] += 100 / scale
        p["W1"][q, 16, 2, 2] = ry * 63 / scale
        p["W1"][q, 17, 2, 2] = cx * 63 / scale
        p["b1"][q] = (-500 + offset) / scale

    # Paint colour immediately above the black selection underline.
    for c, v in labels.items():
        p["W1"][26, c, 2, 2] = v
    p["W1"][26, 0, 4, 2] += 20
    p["b1"][26] = -20

    # Right endpoint of that underline.
    p["W1"][27, 0, 4, 1] = 10
    p["W1"][27, 3, 4, 2] = 10
    p["b1"][27] = -19

    # The small tool contains a black pair bracketed by red pixels.
    for q, vertical in [(28, False), (29, True)]:
        if not vertical:
            black = [(0, 0), (0, 1)]
            red = [(0, -1), (0, 2)]
            optional = [(-1, 0), (1, 0)]
        else:
            black = [(0, 0), (1, 0)]
            red = [(-1, 0), (2, 0)]
            optional = [(0, -1), (0, 1)]

        for dr, dc in black:
            p["W1"][q, 0, 2 + dr, 2 + dc] += 10
        for dr, dc in red:
            p["W1"][q, 2, 2 + dr, 2 + dc] += 10
        for dr, dc in optional:
            p["W1"][q, 2, 2 + dr, 2 + dc] += 1
        p["b1"][q] = -40.5

    p["W1"][30, 16, 2, 2] = 63
    p["W1"][31, 17, 2, 2] = 63

    # Sample extraction by distance from the invariant lines.
    for i, v in enumerate(lines):
        p["W2"][i, 17, 1, 1] = 1
        p["W2"][i, i, 1, 1] = -40
        p["W2"][i, 16, 1, 1] = 20
        p["b2"][i] = -20 * (2000 + v)

    # Coordinate hinges.
    p["W2"][16, 31, 1, 1] = 1
    p["b2"][16] = -15
    p["W2"][17, 31, 1, 1] = -1
    p["b2"][17] = 20
    p["W2"][18, 30, 1, 1] = 1
    p["b2"][18] = -5

    # Reject the short side-edge responses of other basket poses.
    # Correct responses are one; incorrect cardinal responses are
    # at most 5/9.
    for i in range(8):
        p["W2"][19 + i, 18 + i, 1, 1] = 4
        p["b2"][19 + i] = -3

    p["W2"][27, 26, 1, 1] = 1
    p["W2"][28, 27, 2, 1] = 1
    p["W2"][29, 17, 1, 1] = 1
    p["W2"][30, 17, 1, 1] = 1
    p["b2"][30] = -1
    p["W2"][31, 28, 1, 1] = 1
    p["W2"][31, 29, 1, 1] = 1

    # Target labels and work-panel outer labels.
    for i in range(16):
        p["W3"][i, i] = 1
        p["W3"][i, 16] = -20
    for i in range(8):
        p["W3"][16 + i, i] = 1
        p["W3"][16 + i, 17] = -20

    def palette_gate(out):
        p["W3"][out, 17] = -20
        p["W3"][out, 18] = -20

    p["W3"][24, 27] = 1
    palette_gate(24)

    # Three bits of the clean basket-pose signal.
    for bit in range(3):
        for i in range(8):
            if (i >> bit) & 1:
                p["W3"][25 + bit, 19 + i] = 1

    p["W3"][28, 28] = 1
    palette_gate(28)

    # Unit indicator for palette interiors.
    p["W3"][29, 29] = 1
    p["W3"][29, 30] = -1
    palette_gate(29)

    p["W3"][30, 31] = 1

    # Recurrent spatial routing to the next palette entry.
    # At the final entry, the small baseline score wraps to the first.
    p["Wh"][0, 0, 1, 0] = 1
    p["Uh"][0, 28] = 1
    p["Wh"][1, 0, 1, 1] = 1
    p["Uh"][1, 29] = 1
    p["bh"][1] = -1

    p["Wc"][0, 29] = 0.01
    p["Wc"][0, 30] = 100
    p["Wc"][0, 33] = 10

    def z_feature(z, coefficients, bias=0):
        for f, v in coefficients.items():
            p["Wz"][z, 32 + f] += v
        p["bz"][z] = bias

    for i in range(8):
        coefficients = {}
        ones = 0
        for b in range(3):
            bit = (i >> b) & 1
            coefficients[25 + b] = 1 if bit else -1
            ones += bit
        z_feature(i, coefficients, 1 - ones)

    def absolute_pair(start, a, b):
        z_feature(start, {a: 1, b: -1})
        z_feature(start + 1, {a: -1, b: 1})
        return start, start + 1

    selected_diff = [
        absolute_pair(8 + 2 * i, 24, i) for i in range(16)
    ]
    edge_diff = [
        absolute_pair(40 + 2 * i, i, (i + 1) % 8)
        for i in range(8)
    ]
    for i in range(8):
        absolute_pair(56 + 2 * i, i, 16 + i)

    inset_pair = [
        absolute_pair(72 + 2 * s, 8 + 2 * s, 9 + 2 * s)
        for s in range(4)
    ]
    associated_outer = [(0, 1), (2, 3), (5, 4), (7, 6)]
    inset_outer = []
    for s in range(4):
        pair = []
        for j in range(2):
            pair.append(absolute_pair(
                80 + 4 * s + 2 * j,
                8 + 2 * s + j,
                associated_outer[s][j],
            ))
        inset_outer.append(pair)

    z_feature(96, {30: 2})
    p["bz"][97] = 1

    # Signed Boolean controller memory.
    C = list(range(0, 8))
    K = list(range(8, 16))
    P = list(range(16, 24))
    SC = list(range(24, 28))
    SK = list(range(28, 32))
    SP = list(range(32, 36))
    SS = list(range(36, 40))
    F = list(range(40, 48))
    I = list(range(48, 56))
    Q, B = 56, 57
    counts = list(range(58, 63))
    done, after = counts[4], 63

    gain = 20.0
    M = 1000.0

    def memory_row(out, mi=None, zz=None, bias=0, boot=True):
        mi = dict(mi or {})
        zz = dict(zz or {})
        if boot:
            mi[Q] = mi.get(Q, 0) + M
            bias -= M
        for j, v in mi.items():
            p["Wm"][out, j] += gain * v / 2
            p["bm"][out] += gain * v / 2
        for j, v in zz.items():
            p["Wmz"][out, j] += gain * v
        p["bm"][out] += gain * bias

    def add_absolute(d, indices, value):
        for j in indices:
            d[j] = d.get(j, 0) + value

    p["bm"][Q] = 20
    memory_row(B, {Q: 2}, bias=-1.5, boot=False)

    # Find a monochromatic four-sector window. A five-stroke sweep
    # ending at that window reconstructs the outer pattern.
    for r in range(8):
        zz = {}
        for e in [(r - 1) % 8, r, (r + 1) % 8]:
            add_absolute(zz, edge_diff[e], -1)
        memory_row(F[r], zz=zz, bias=0.5, boot=False)

        mi = {Q: 1, B: -1, F[r]: 1}
        for j in range(r):
            mi[F[j]] = -1
        memory_row(I[r], mi, bias=-1.5, boot=False)

    for r in range(8):
        target = (r - 1) % 8
        delta = {}
        add_absolute(delta, selected_diff[target], 1)

        zz = dict(delta)
        zz[r] = zz.get(r, 0) - 1
        memory_row(
            C[r],
            {
                C[r]: M,
                P[(r - 1) % 8]: M + 20,
                I[(r - 4) % 8]: M + 20,
                done: -M,
            },
            zz, bias=-M + 0.5,
        )
        memory_row(
            K[r], {C[r]: M, done: -M},
            delta, bias=-M - 0.5,
        )

        zz = {}
        add_absolute(zz, selected_diff[target], -10)
        zz[r] = 10
        memory_row(
            P[r], {C[r]: 1, done: -M},
            zz, bias=-10.5,
        )

    # Prefix counter for the five completed large strokes.
    for n, out in enumerate(counts):
        mi = {out: 2}
        for j in P:
            mi[j] = mi.get(j, 0) + 1
        bias = -1.5
        if n == 0:
            bias += 1
        else:
            mi[counts[n - 1]] = mi.get(counts[n - 1], 0) + 1
        memory_row(out, mi, bias=bias)
    memory_row(after, {done: 1}, bias=-0.5)

    # Process the four possible small stamps. A stamp is needed when
    # its samples agree with each other but differ from the outer paint.
    for s in range(4):
        pose = 2 * s
        target = 8 + 2 * s
        need = {}
        add_absolute(need, inset_pair[s], -20)
        for pair in inset_outer[s]:
            add_absolute(need, pair, 1)

        mi = {
            SC[s]: M,
            SP[s]: -(M + 20),
            SS[s]: -(M + 20),
        }
        if s == 0:
            mi[done] = M + 20
            mi[after] = -(M + 20)
        else:
            mi[SP[s - 1]] = M + 20
            mi[SS[s - 1]] = M + 20

        zz = {}
        add_absolute(zz, selected_diff[target], 1)
        zz[pose] = -1
        memory_row(SC[s], mi, zz, bias=-M + 0.5)

        mi = {SC[s]: M, SP[s]: -M, SS[s]: -M}
        zz = {}
        add_absolute(zz, selected_diff[target], 1)
        memory_row(SK[s], mi, zz, bias=-M - 0.5)

        zz = dict(need)
        add_absolute(zz, selected_diff[target], -20)
        zz[pose] = zz.get(pose, 0) + 20
        memory_row(SP[s], mi, zz, bias=-M - 20.5)

        zz = {j: -v for j, v in need.items()}
        memory_row(SS[s], mi, zz, bias=-M + 0.5)

    grid = [
        (0, 1), (0, 2), (1, 2), (2, 2),
        (2, 1), (2, 0), (1, 0), (0, 0),
    ]

    p["bo"][:] = -1000
    p["bo"][1:5] = [-1, 1, -1, 1]
    p["bo"][5] = -10
    p["bo"][6] = -10

    invalid = [
        (1, 2), (1, 4), (3, 4), (2, 4),
        (1, 2), (2, 3), (3, 4), (1, 3),
    ]
    for i, (y, x) in enumerate(grid):
        p["Wo"][1, i] += y
        p["Wo"][2, i] -= y
        p["Wo"][3, i] += x
        p["Wo"][4, i] -= x
        for a in invalid[i]:
            p["Wo"][a, i] -= 10

    def out_memory(action, unit, value):
        p["Wo"][action, 128 + unit] += value / 2
        p["bo"][action] += value / 2

    def goal_contribution(unit, dy, dx):
        out_memory(1, unit, -dy)
        out_memory(2, unit, dy)
        out_memory(3, unit, -dx)
        out_memory(4, unit, dx)

    for r, (y, x) in enumerate(grid):
        goal_contribution(C[r], y - 1, x - 1)

        # A diagonal pose removes the small tool, making palette
        # clicks safe. Cardinal goals detour through a neighbour.
        diagonal = r if r % 2 else (r - 1) % 8
        py, px = grid[diagonal]
        goal_contribution(K[r], py - y, px - x)

        out_memory(5, P[r], 100)
        out_memory(6, K[r], 30)

    for s in range(4):
        r = 2 * s
        y, x = grid[r]
        py, px = grid[(r - 1) % 8]
        goal_contribution(SC[s], y - 1, x - 1)
        goal_contribution(SK[s], py - y, px - x)
        out_memory(6, SK[s], 30)
        out_memory(6, SP[s], 100)
        out_memory(1, SS[s], 100)

    # Stamp-performing states override this palette-click penalty.
    p["Wo"][6, 96] = -60

    return p
