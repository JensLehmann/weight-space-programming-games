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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    # Opposite polarities for each canonical two-colour palette:
    # (9, 8), (8, 12), and (14, 15).
    colour_a = (9, 12, 15)
    colour_b = (8, 14)
    palette = colour_a + colour_b

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    def w1(out, colours, dy=0, dx=0, value=1.0):
        for colour in colours:
            p["W1"][out, colour, dy + 2, dx + 2] += value

    def w2(out, inp, value=1.0, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    # First-layer shared feature indices.
    CA, CB = 16, 17
    TILE = 18
    BG_UP, BG_LEFT = 19, 20
    SIX_COUNT = 21
    MARK_BASE = 22

    w1(CA, colour_a)
    w1(CB, colour_b)

    # At a tile centre, sample the nine logical sprite pixels.
    # This accepts both solid tiles and tiles decorated with colour 6.
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            w1(TILE, palette + (6,), dy, dx)
            w1(SIX_COUNT, (6,), dy, dx)
    p["b1"][TILE] = -8.0

    # Background predicates. Zero padding is accepted as an outer
    # boundary; black tutorial backgrounds are deliberately rejected.
    non_background = tuple(c for c in range(16) if c != 4)
    w1(BG_UP, non_background, -2, 0, -1.0)
    w1(BG_LEFT, non_background, 0, -2, -1.0)
    p["b1"][BG_UP] = 1.0
    p["b1"][BG_LEFT] = 1.0

    for d, (dy, dx) in enumerate(directions):
        first, second = 2 * d, 2 * d + 1
        sy, sx = 2 * dy, 2 * dx

        # Their sum is:
        #   1 + [this mark requests polarity A]
        # at a valid coloured clue centre.
        w1(first, colour_a, value=2.0)
        w1(first, (0,), sy, sx)
        p["b1"][first] = -1.0

        w1(second, colour_b, value=2.0)
        w1(second, (2, 3), sy, sx)
        p["b1"][second] = -1.0

        # Validate the mark independently for each direction.
        mark = MARK_BASE + d
        w1(mark, palette)
        w1(mark, (0, 2, 3), sy, sx)
        p["b1"][mark] = -1.0

    def add_anchor_gate(out):
        # At the upper-left pixel of a central 2x2 block, these
        # inspect the background immediately above and left of
        # the complete 6x6 sprite.
        w2(out, BG_UP, dy=-1)
        w2(out, BG_LEFT, dx=-1)

    for d in range(8):
        out_a, out_b = 2 * d, 2 * d + 1
        first, second = 2 * d, 2 * d + 1

        # A request.
        w2(out_a, first)
        w2(out_a, second)
        w2(out_a, CA, -1.0)
        w2(out_a, CB, -1.0)
        add_anchor_gate(out_a)
        p["b2"][out_a] = -2.0

        # B request: the complement, restricted to valid marks.
        w2(out_b, MARK_BASE + d, 2.0)
        w2(out_b, first, -1.0)
        w2(out_b, second, -1.0)
        add_anchor_gate(out_b)
        p["b2"][out_b] = -2.0

    # Second-layer candidate masks.
    TILE_A, TILE_B, ORDINARY = 16, 17, 18

    for out, state in ((TILE_A, CA), (TILE_B, CB)):
        w2(out, TILE)
        w2(out, state)
        add_anchor_gate(out)
        p["b2"][out] = -3.0

    w2(ORDINARY, TILE)
    w2(ORDINARY, SIX_COUNT, -1.0)
    add_anchor_gate(ORDINARY)
    p["b2"][ORDINARY] = -2.0

    # Axial pooling is unnecessary; retain only local features.
    for c in range(19):
        p["W3"][c, c] = 1.0

    # Directed, damped relays.
    #
    # An isolated seed has amplitude 766. After n pixels:
    #       h(n) = max(768 * 2**(-n) - 2, 0).
    # Thus the signal is 1 at the neighbouring centre, eight
    # pixels away, and disappears beyond it.
    #
    # Matching candidates suppress the signal. Ordinary candidates
    # retain about 0.05 per violation; special candidates retain 1.
    # Damping also bounds interference from consecutive clue seeds.
    for d, (dy, dx) in enumerate(directions):
        for bit in range(2):
            channel = 2 * d + bit
            p["Wh"][channel, channel, 1 - dy, 1 - dx] = 0.5
            p["Uh"][channel, channel] = 767.0
            p["Uh"][channel, TILE_A if bit == 0 else TILE_B] = -4.0
            p["Uh"][channel, ORDINARY] = -0.95
            p["bh"][channel] = -1.0

    # The large candidate bonus excludes clue centres and relay paths.
    # Remaining differences rank unsatisfied playable tiles.
    p["Wc"][0, TILE_A] = 16384.0
    p["Wc"][0, TILE_B] = 16384.0
    p["Wc"][0, 32:48] = 1.0

    p["bo"][6] = 1.0
    return p
