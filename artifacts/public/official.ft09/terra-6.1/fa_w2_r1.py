import numpy as np


def build():
    graph = []
    params = {}

    def node(name, op, inputs, **fields):
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return name

    def conv(name, source, weights, bias=None):
        wk = "w_" + name
        params[wk] = np.asarray(weights, dtype=np.float32)
        fields = {"w": wk}
        if bias is not None:
            bk = "b_" + name
            params[bk] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bk
        return node(name, "conv", [source], **fields)

    def dense(name, source, weights, bias=None):
        wk = "w_" + name
        params[wk] = np.asarray(weights, dtype=np.float32)
        fields = {"w": wk}
        if bias is not None:
            bk = "b_" + name
            params[bk] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bk
        return node(name, "dense", [source], **fields)

    def op(name, kind, *inputs):
        return node(name, kind, inputs)

    def channel(name, source, start, stop):
        return node(name, "slice", [source], start=start, stop=stop)

    def sum_channels(name, source, count):
        return conv(name, source, np.ones((1, count, 1, 1), np.float32))

    # Each sprite is a 3x3 image rendered at scale 2. Its eight
    # neighbors in the game lattice are eight display pixels away.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    excluded = {0, 2, 3, 4, 5, 6}
    colors = [c for c in range(16) if c not in excluded]

    # A valid tile consists of one center color, with any subset of
    # its other eight pixels optionally replaced by special marker 6.
    w = np.zeros((16, 16, 5, 5), np.float32)
    b = np.full(16, -8.0, np.float32)
    for c in colors:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                iy, ix = 2 + 2 * dy, 2 + 2 * dx
                w[c, c, iy, ix] = 1.0
                if dy != 0 or dx != 0:
                    w[c, 6, iy, ix] = 1.0
    conv("tile_raw", "x0", w, b)
    op("tile_shape", "relu", "tile_raw")

    # A clue has a colored center and eight pixels drawn from 0, 2, 3.
    w = np.zeros((1, 16, 5, 5), np.float32)
    for c in colors:
        w[0, c, 2, 2] = 1.0
    for dy, dx in directions:
        for c in (0, 2, 3):
            w[0, c, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    conv("clue_raw", "x0", w, [-8.0])
    op("clue_shape", "relu", "clue_raw")

    # Require background color 4 in the four gap locations. This also
    # excludes the decorative examples on the first level.
    channel("background", "x0", 4, 5)
    w = np.zeros((4, 1, 5, 5), np.float32)
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(cardinal):
        w[i, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    conv("gap_half", "background", w)

    w = np.zeros((1, 4, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(cardinal):
        w[0, i, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    conv("gap_raw", "gap_half", w, [-3.0])
    op("gap", "relu", "gap_raw")

    op("tile_colors", "mul", "tile_shape", "gap")
    op("clues", "mul", "clue_shape", "gap")
    sum_channels("tiles", "tile_colors", 16)

    # Select one of the colors currently present as a reference.
    # With two colors, equality to that reference is a binary state.
    op("present", "gmax", "tile_colors")
    w = np.zeros((16, 16), np.float32)
    for c in range(16):
        w[c, :c] = -1.0
    dense("first_raw", "present", w, np.ones(16, np.float32))
    op("first_possible", "vrelu", "first_raw")
    op("reference", "vmul", "present", "first_possible")
    op("reference_pixels", "mul", "x0", "reference")
    sum_channels("is_reference", "reference_pixels", 16)
    conv(
        "reference_sign",
        "is_reference",
        np.asarray([[[[2.0]]]], np.float32),
        [-1.0],
    )

    # At each clue, produce eight signed desired-color votes.
    # Zero means equality to the clue's center; nonzero means inequality.
    w = np.zeros((8, 16, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[i, 0, 2 + 2 * dy, 2 + 2 * dx] = 2.0
    conv("clue_relation", "x0", w, np.full(8, -1.0, np.float32))
    op("clue_color_votes", "mul", "clue_relation", "reference_sign")
    op("clue_votes", "mul", "clue_color_votes", "clues")

    # Shift each vote from a clue to the corresponding neighboring tile.
    # Three convolutions implement a displacement of 3+3+2 pixels.
    def shift_channels(name, source, reverse):
        previous = source
        for stage, distance in enumerate((3, 3, 2)):
            k = 7 if distance == 3 else 5
            center = k // 2
            w = np.zeros((8, 8, k, k), np.float32)
            for i, (dy, dx) in enumerate(directions):
                sign = -1 if reverse else 1
                w[i, i, center + sign * distance * dy,
                  center + sign * distance * dx] = 1.0
            previous = conv(name + "_" + str(stage), previous, w)
        return previous

    shifted_votes = shift_channels("vote_shift", "clue_votes", True)
    sum_channels("desired_votes", shifted_votes, 8)

    # A positive gain means toggling this tile fixes more constraints
    # than it breaks.
    op("signed_votes", "mul", "desired_votes", "reference_sign")
    conv("negative_votes", "signed_votes",
         np.asarray([[[[-1.0]]]], np.float32))
    op("gain", "mul", "negative_votes", "tiles")

    # Read the special marker at each of the eight sprite positions.
    w = np.zeros((8, 16, 5, 5), np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[i, 6, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    conv("markers", "x0", w)
    sum_channels("marker_count", "markers", 8)
    channel("one", "coords", 2, 3)
    op("has_marker", "min2", "marker_count", "one")
    op("special_tiles", "mul", "tiles", "has_marker")
    op("ordinary_tiles", "sub", "tiles", "special_tiles")

    # Gather the gains of the eight neighboring tiles at each possible
    # special-tile click, and retain only marked directions.
    w = np.zeros((8, 1, 7, 7), np.float32)
    for i, (dy, dx) in enumerate(directions):
        w[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    conv("neighbor_first", "gain", w)

    previous = "neighbor_first"
    for stage, distance in enumerate((3, 2)):
        k = 7 if distance == 3 else 5
        center = k // 2
        w = np.zeros((8, 8, k, k), np.float32)
        for i, (dy, dx) in enumerate(directions):
            w[i, i, center + distance * dy,
              center + distance * dx] = 1.0
        previous = conv("neighbor_rest_" + str(stage), previous, w)

    op("marked_neighbor_gains", "mul", previous, "markers")
    sum_channels("neighbor_gain", "marked_neighbor_gains", 8)

    # Correct special tiles first: their effects can subsequently be
    # repaired with ordinary clicks.
    conv("special_priority", "gain",
         np.asarray([[[[128.0]]]], np.float32))
    op("special_score", "add", "special_priority", "neighbor_gain")
    op("special_choice", "mul", "special_score", "special_tiles")
    op("ordinary_choice", "mul", "gain", "ordinary_tiles")
    op("choice_score", "add", "special_choice", "ordinary_choice")

    # Ensure every detected tile outranks all non-tile pixels.
    conv("tile_base", "tiles", np.asarray([[[[4096.0]]]], np.float32))
    op("click", "add", "tile_base", "choice_score")

    # ACTION6 is the click action.
    dense(
        "logits",
        "present",
        np.zeros((8, 16), np.float32),
        np.asarray([0, 0, 0, 0, 0, 0, 1, 0], np.float32),
    )

    return {"graph": graph, "memory_size": 0, "params": params}
