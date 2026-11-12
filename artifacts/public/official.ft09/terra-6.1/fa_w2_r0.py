import numpy as np


def build():
    graph = []
    params = {}

    def p(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def n(name, op, inputs, **fields):
        entry = {"id": name, "op": op, "in": inputs}
        entry.update(fields)
        graph.append(entry)
        return name

    colors = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    p("sum10", np.ones((1, 10, 1, 1)))
    p("sum8", np.ones((1, 8, 1, 1)))
    p("decode", np.arange(16).reshape(1, 16, 1, 1))
    p("boost", np.array([[[[100.0]]]]))
    p("small", np.array([[[[0.001]]]]))
    p("neg_identity8", -np.eye(8).reshape(8, 8, 1, 1))
    p("ones8", np.ones(8))
    p("fallback_w", np.array([[-1.0]]))
    p("fallback_b", np.array([1.0]))
    p("logit_w", np.zeros((8, 1)))
    logit_bias = np.zeros(8)
    logit_bias[6] = 1
    p("logit_b", logit_bias)

    # Route eight directions, independently for each of three quantities:
    # clue presence, clue center color, and equality requirement.
    for distance, kernel, name in [(2, 5, "shift2"), (3, 7, "shift3")]:
        w = np.zeros((24, 24, kernel, kernel))
        mid = kernel // 2
        for group in range(3):
            for d, (dy, dx) in enumerate(directions):
                ch = group * 8 + d
                w[ch, ch, mid - distance * dy, mid - distance * dx] = 1
        p(name, w)

    zero = n("zero", "slice", ["x0"], start=0, stop=1)
    decoded = n("decoded", "conv", ["x0"], w="decode")

    branches = []

    # The camera commonly uses 2x scaling, but the same construction also
    # handles a 1x viewport.
    for scale in (1, 2):
        pre = "s%d_" % scale
        k = 5 if scale == 1 else 7
        mid = k // 2
        gap = 2 if scale == 1 else 3

        regular_w = np.zeros((10, 16, k, k))
        generic_w = np.zeros((10, 16, k, k))
        clue_w = np.zeros((1, 16, k, k))
        bg4_w = np.zeros((1, 16, k, k))
        pick_w = np.zeros((8, 1, 5, 5))

        for ci, color in enumerate(colors):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = mid + scale * dy
                    xx = mid + scale * dx
                    regular_w[ci, color, yy, xx] = 1
                    generic_w[ci, color, yy, xx] = 1
                    if dy or dx:
                        generic_w[ci, 6, yy, xx] = 1
            clue_w[0, color, mid, mid] = 1

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    yy = mid + scale * dy
                    xx = mid + scale * dx
                    for color in (0, 2, 3):
                        clue_w[0, color, yy, xx] = 1

        # At the first pixel of the logical center, the upper and left
        # background gaps occur at these offsets. This also anchors the
        # detector to one location inside each 2x2 display pixel.
        for dy, dx in [(-gap, 0), (0, -gap)]:
            yy, xx = mid + dy, mid + dx
            for color in (4, 5):
                regular_w[:, color, yy, xx] = 1
                generic_w[:, color, yy, xx] = 1
                clue_w[0, color, yy, xx] = 1
            bg4_w[0, 4, yy, xx] = 1

        for d, (dy, dx) in enumerate(directions):
            pick_w[d, 0, 2 + scale * dy, 2 + scale * dx] = 1

        p(pre + "regular_w", regular_w)
        p(pre + "generic_w", generic_w)
        p(pre + "clue_w", clue_w)
        p(pre + "bg4_w", bg4_w)
        p(pre + "pick_w", pick_w)
        p(pre + "bias10", np.full(10, -10.0))
        p(pre + "bias1", np.array([-10.0]))
        p(pre + "bg_bias", np.array([-1.0]))

        regular_raw = n(
            pre + "regular_raw", "conv", ["x0"],
            w=pre + "regular_w", b=pre + "bias10"
        )
        regular = n(pre + "regular", "relu", [regular_raw])

        generic_raw = n(
            pre + "generic_raw", "conv", ["x0"],
            w=pre + "generic_w", b=pre + "bias10"
        )
        generic = n(pre + "generic", "relu", [generic_raw])

        clue_raw = n(
            pre + "clue_raw", "conv", ["x0"],
            w=pre + "clue_w", b=pre + "bias1"
        )
        clue = n(pre + "clue", "relu", [clue_raw])

        bg_raw = n(
            pre + "bg_raw", "conv", ["x0"],
            w=pre + "bg4_w", b=pre + "bg_bias"
        )
        bg4 = n(pre + "bg4", "relu", [bg_raw])

        tile = n(pre + "tile", "conv", [generic], w="sum10")
        special_channels = n(pre + "special_channels", "sub", [generic, regular])
        special = n(pre + "special", "conv", [special_channels], w="sum10")
        tile4 = n(pre + "tile4", "mul", [tile, bg4])

        branches.append({
            "pre": pre,
            "scale": scale,
            "clue": clue,
            "bg4": bg4,
            "tile": tile,
            "tile4": tile4,
            "special": special,
        })

    tile4_all = n("tile4_all", "add", [branches[0]["tile4"], branches[1]["tile4"]])
    has4 = n("has4", "gmax", [tile4_all])
    fallback_raw = n("fallback_raw", "dense", [has4], w="fallback_w", b="fallback_b")
    fallback = n("fallback", "vrelu", [fallback_raw])

    scores = []
    for branch in branches:
        pre = branch["pre"]
        scale = branch["scale"]

        # Prefer the background-4 board when present, excluding the
        # background-5 explanatory illustrations in the first level.
        active = n(pre + "active", "max2", [branch["bg4"], fallback])
        tile = n(pre + "active_tile", "mul", [branch["tile"], active])
        special = n(pre + "active_special", "mul", [branch["special"], active])
        clue = n(pre + "active_clue", "mul", [branch["clue"], active])

        clue_color = n(pre + "clue_color", "mul", [clue, decoded])
        valid8 = n(pre + "valid8", "concat", [clue] * 8)
        color8 = n(pre + "color8", "concat", [clue_color] * 8)
        zero8 = n(pre + "zero8", "conv", [zero], w=pre + "pick_w")
        wanted8 = n(pre + "wanted8", "mul", [zero8, valid8])
        routed = n(pre + "routed", "concat", [valid8, color8, wanted8])

        if scale == 1:
            routed = n(pre + "shift_a", "conv", [routed], w="shift2")
            routed = n(pre + "shift_b", "conv", [routed], w="shift2")
        else:
            routed = n(pre + "shift_a", "conv", [routed], w="shift3")
            routed = n(pre + "shift_b", "conv", [routed], w="shift3")
            routed = n(pre + "shift_c", "conv", [routed], w="shift2")

        valid = n(pre + "valid", "slice", [routed], start=0, stop=8)
        target = n(pre + "target", "slice", [routed], start=8, stop=16)
        wanted = n(pre + "wanted", "slice", [routed], start=16, stop=24)

        diff = n(pre + "diff", "sub", [target, decoded])
        distance = n(pre + "distance", "abs", [diff])
        same_raw = n(
            pre + "same_raw", "conv", [distance],
            w="neg_identity8", b="ones8"
        )
        same = n(pre + "same", "relu", [same_raw])
        signed = n(pre + "signed", "sub", [same, wanted])
        mismatch = n(pre + "mismatch", "abs", [signed])
        violations = n(pre + "violations", "mul", [mismatch, valid])
        count = n(pre + "count", "conv", [violations], w="sum8")

        special_count = n(pre + "special_count", "mul", [count, special])
        boosted = n(pre + "boosted", "conv", [special_count], w="boost")
        priority = n(pre + "priority", "add", [count, boosted])
        scored_tile = n(pre + "scored_tile", "mul", [priority, tile])
        baseline = n(pre + "baseline", "conv", [tile], w="small")
        score = n(pre + "score", "add", [scored_tile, baseline])
        scores.append(score)

    n("click", "add", scores)
    n("logits", "dense", [has4], w="logit_w", b="logit_b")

    return {"graph": graph, "memory_size": 0, "params": params}
