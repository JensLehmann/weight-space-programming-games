import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def node(name, op, inputs, **fields):
        entry = {"id": name, "op": op, "in": list(inputs)}
        entry.update(fields)
        graph.append(entry)
        return name

    def conv(name, inp, weights, bias=None):
        key = "p%d" % counter[0]
        counter[0] += 1
        params[key] = np.asarray(weights, dtype=np.float32)
        fields = {"w": key}
        if bias is not None:
            bkey = "p%d" % counter[0]
            counter[0] += 1
            params[bkey] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bkey
        return node(name, "conv", [inp], **fields)

    def dense(name, inp, weights, bias=None):
        key = "p%d" % counter[0]
        counter[0] += 1
        params[key] = np.asarray(weights, dtype=np.float32)
        fields = {"w": key}
        if bias is not None:
            bkey = "p%d" % counter[0]
            counter[0] += 1
            params[bkey] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bkey
        return node(name, "dense", [inp], **fields)

    def channel_sum(name, inp, count):
        w = np.ones((1, count, 1, 1), dtype=np.float32)
        return conv(name, inp, w)

    # These colors exclude the clue-symbol colors, background colors, and
    # the fixed cross color. They cover the tile colors in the shown levels.
    tile_colors = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # A scalar encoding of the observed color at each display pixel.
    w = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    pixel_color = conv("pixel_color", "x0", w)

    # A clue has a colored center and eight surrounding pixels in {0,2,3}.
    # The display scales each game pixel by two, hence the offsets of 2.
    w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for color in tile_colors:
        w[0, color, 2, 2] = 1.0
    for dy, dx in directions:
        for color in (0, 2, 3):
            w[0, color, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    clue_raw = conv("clue_raw", "x0", w, [-8.0])
    clue = node("clue", "relu", [clue_raw])

    # Detect a solid 3x3 game-pixel tile, and a cross-effect tile whose
    # center and corners are its color and whose four arms are color 6.
    normal_w = np.zeros((len(tile_colors), 16, 5, 5), dtype=np.float32)
    special_w = np.zeros_like(normal_w)
    for out, color in enumerate(tile_colors):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                normal_w[out, color, 2 + 2 * dy, 2 + 2 * dx] = 1.0
                if dy == 0 and dx == 0 or (dy != 0 and dx != 0):
                    special_w[out, color, 2 + 2 * dy, 2 + 2 * dx] = 1.0
                else:
                    special_w[out, 6, 2 + 2 * dy, 2 + 2 * dx] = 1.0

    normal_raw = conv(
        "normal_raw", "x0", normal_w,
        np.full(len(tile_colors), -8.0, dtype=np.float32)
    )
    special_raw = conv(
        "special_raw", "x0", special_w,
        np.full(len(tile_colors), -8.0, dtype=np.float32)
    )
    normal_channels = node("normal_channels", "relu", [normal_raw])
    special_channels = node("special_channels", "relu", [special_raw])
    normal0 = channel_sum("normal0", normal_channels, len(tile_colors))
    special0 = channel_sum("special0", special_channels, len(tile_colors))

    # Require a neighboring background gap. This also excludes the
    # noninteractive example diagrams on the first level.
    background = node("background", "slice", ["x0"], start=4, stop=5)
    axial = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    w1 = np.zeros((4, 1, 5, 5), dtype=np.float32)
    w2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(axial):
        w1[i, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        w2[i, i, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    gap_step = conv("gap_step", background, w1)
    gap_four = conv("gap_four", gap_step, w2)
    gap_count = channel_sum("gap_count", gap_four, 4)
    ones_plane = node("ones_plane", "slice", ["coords"], start=2, stop=3)
    one = node("one", "gmax", [ones_plane])
    gap = node("gap", "min2", [gap_count, one])
    normal = node("normal", "mul", [normal0, gap])
    special = node("special", "mul", [special0, gap])

    # For each clue direction, construct:
    #   clue present, clue center color, and whether the direction says equal.
    replicate = np.ones((8, 1, 1, 1), dtype=np.float32)
    clue8 = conv("clue8", clue, replicate)
    clue_color = node("clue_color", "mul", [clue, pixel_color])
    clue_color8 = conv("clue_color8", clue_color, replicate)

    zero_plane = node("zero_plane", "slice", ["x0"], start=0, stop=1)
    w = np.zeros((8, 1, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[i, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    equal_symbols = conv("equal_symbols", zero_plane, w)
    clue_equal = node("clue_equal", "mul", [clue8, equal_symbols])
    messages = node("messages", "concat", [clue8, clue_color8, clue_equal])

    # Translate each directional message by eight display pixels. Three
    # sparse 7x7 convolutions implement shifts of 3, 3, and 2 pixels.
    previous = messages
    for stage, distance in enumerate((3, 3, 2)):
        w = np.zeros((24, 24, 7, 7), dtype=np.float32)
        for group in range(3):
            for i, (dy, dx) in enumerate(directions):
                channel = group * 8 + i
                w[channel, channel, 3 - distance * dy, 3 - distance * dx] = 1.0
        previous = conv("shift_%d" % stage, previous, w)

    present = node("present", "slice", [previous], start=0, stop=8)
    target = node("target", "slice", [previous], start=8, stop=16)
    wants_equal = node("wants_equal", "slice", [previous], start=16, stop=24)

    difference = node("difference", "sub", [pixel_color, target])
    absolute_difference = node("absolute_difference", "abs", [difference])
    w = np.zeros((8, 8, 1, 1), dtype=np.float32)
    for i in range(8):
        w[i, i, 0, 0] = -1.0
    equality_raw = conv(
        "equality_raw", absolute_difference, w,
        np.ones(8, dtype=np.float32)
    )
    equality = node("equality", "relu", [equality_raw])

    # XOR(equality, wants_equal), gated by the presence of a clue.
    both = node("both", "mul", [equality, wants_equal])
    w = np.zeros((8, 8, 1, 1), dtype=np.float32)
    for i in range(8):
        w[i, i, 0, 0] = 2.0
    twice_both = conv("twice_both", both, w)
    either = node("either", "add", [equality, wants_equal])
    mismatch0 = node("mismatch0", "sub", [either, twice_both])
    mismatch = node("mismatch", "mul", [mismatch0, present])
    errors = channel_sum("errors", mismatch, 8)

    normal_errors = node("normal_errors", "mul", [errors, normal])
    special_errors = node("special_errors", "mul", [errors, special])
    tile_mask = node("tile_mask", "add", [normal, special])
    score_inputs = node(
        "score_inputs", "concat", [normal_errors, special_errors, tile_mask]
    )
    score_w = np.array([1.0, 100.0, 0.001], dtype=np.float32).reshape(1, 3, 1, 1)
    click = conv("click", score_inputs, score_w)

    # ACTION6 is the only action used. The small baseline in the click map
    # also permits a click if the observed constraints are already satisfied.
    pooled = node("pooled", "gmax", [click])
    logits_w = np.zeros((8, 1), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    dense("logits", pooled, logits_w, logits_b)

    return {"graph": graph, "memory_size": 0, "params": params}
