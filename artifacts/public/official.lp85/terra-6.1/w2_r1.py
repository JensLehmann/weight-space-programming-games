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
    p = {k: np.zeros(shape, dtype=np.float32)
         for k, shape in shapes.items()}

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Colour samples.
    W1[0, 8, 2, 2] = 1.0
    W1[1, 11, 2, 2] = 1.0
    W1[2, 12, 2, 2] = 1.0
    W1[13, 14, 2, 2] = 1.0

    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for j, (dy, dx) in enumerate(corners):
        W1[3 + j, 11, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        W1[7 + j, 12, 2 + 2 * dy, 2 + 2 * dx] = 1.0

    # Unfilled scale-1 markers.
    for out, colour in [(11, 11), (12, 12)]:
        for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
            W1[out, colour, 2 + dy, 2 + dx] = 1.0
        W1[out, colour, 2, 2] = -4.0
        b1[out] = -3.5

    # Samples two pixels below; f2 shifts these one further pixel.
    W1[14, 11, 4, 2] = 1.0
    W1[15, 12, 4, 2] = 1.0

    W2[0, 0, 1, 1] = 1.0
    W2[1, 11, 1, 1] = 1.0
    W2[2, 12, 1, 1] = 1.0

    # Unfilled scale-2 markers.
    for j, (dy, dx) in enumerate(corners):
        W2[3, 3 + j, 1 + dy, 1 + dx] = 1.0
        W2[4, 7 + j, 1 + dy, 1 + dx] = 1.0
    W2[3, 1, 1, 1] = -4.0
    W2[4, 2, 1, 1] = -4.0
    b2[3:5] = -3.5

    # A matching goal one scale-1 tile below an unfilled marker.
    W2[5, 11, 1, 1] = 2.0
    W2[5, 14, 2, 1] = 1.0
    b2[5] = -1.5
    W2[6, 12, 1, 1] = 2.0
    W2[6, 15, 2, 1] = 1.0
    b2[6] = -1.5
    W2[7, 13, 1, 1] = 1.0

    # f3: unfilled markers, blue controls, local red hint, red pixels.
    W3[0, 1:5] = 1.0
    W3[1, 0] = 1.0
    W3[2, 5] = 1.0
    W3[2, 6] = 1.0
    W3[3, 7] = 1.0

    # Diffuse marker and red-hint fields.
    for channel in (0, 2):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            Wh[channel, channel, 1 + dy, 1 + dx] = 0.249

    Uh[0, 0] = 1.0
    Uh[2, 2] = 1.0

    # Gate the two fields at their corresponding control colours.
    Wh[1, 0, 1, 1] = 1.0
    Uh[1, 1] = 16.0
    bh[1] = -16.0

    Wh[3, 2, 1, 1] = 1.0
    Uh[3, 3] = 16.0
    bh[3] = -16.0

    p["bo"][6] = 1.0
    p["Wc"][0, 32 + 1] = 1.0
    p["Wc"][0, 32 + 3] = 4.0

    return p
