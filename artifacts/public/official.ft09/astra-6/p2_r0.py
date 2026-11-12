import numpy as np
import random


def _parse_board(obs):
    image = np.asarray(obs, dtype=np.int16).copy()
    # The usual 2x rendering lets us recover the row covered by the HUD.
    image[63] = image[62]

    foreground = ((image != 4) & (image != 5)).ravel().tolist()
    components = []

    for start in range(4096):
        if not foreground[start]:
            continue
        foreground[start] = False
        stack = [start]
        xmin = xmax = start % 64
        ymin = ymax = start // 64
        area = 0

        while stack:
            p = stack.pop()
            y, x = divmod(p, 64)
            area += 1
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)

            if x and foreground[p - 1]:
                foreground[p - 1] = False
                stack.append(p - 1)
            if x < 63 and foreground[p + 1]:
                foreground[p + 1] = False
                stack.append(p + 1)
            if y and foreground[p - 64]:
                foreground[p - 64] = False
                stack.append(p - 64)
            if y < 63 and foreground[p + 64]:
                foreground[p + 64] = False
                stack.append(p + 64)

        w, h = xmax - xmin + 1, ymax - ymin + 1
        if area == w * h:
            components.append((xmin, ymin, w, h))

    candidates = []
    for x, y, w, h in components:
        if w != h or w % 3 or not 3 <= w <= 18:
            continue
        s = w // 3
        block = image[y:y + h, x:x + w]
        small = block[::s, ::s]
        expanded = np.repeat(np.repeat(small, s, axis=0), s, axis=1)
        if not np.array_equal(block, expanded):
            continue

        border = []
        if x:
            border.extend(image[y:y + h, x - 1].tolist())
        if x + w < 64:
            border.extend(image[y:y + h, x + w].tolist())
        if y:
            border.extend(image[y - 1, x:x + w].tolist())
        if y + h < 64:
            border.extend(image[y + h, x:x + w].tolist())

        # Tutorial examples sit on colour 5; the playable board uses 4.
        if border.count(4) == 0 or border.count(4) < border.count(5):
            continue

        pixels = [int(v) for v in small.ravel()]
        center = pixels[4]
        ring = pixels[:4] + pixels[5:]

        if all(v == center for v in pixels):
            kind = "R"
        elif center not in (0, 2, 3) and all(v in (0, 2, 3) for v in ring):
            kind = "C"
        elif center != 6 and 6 in ring and all(v in (center, 6) for v in ring):
            kind = "N"
        else:
            continue

        candidates.append({
            "x": x, "y": y, "s": s, "kind": kind,
            "pixels": pixels, "color": center
        })

    if not candidates:
        return [], [], 2

    counts = {}
    for tile in candidates:
        counts[tile["s"]] = counts.get(tile["s"], 0) + 1
    scale = max(counts, key=lambda s: (counts[s], s == 2))
    tiles = [t for t in candidates if t["s"] == scale]
    tiles.sort(key=lambda t: (t["y"], t["x"]))

    legends = []
    unit = 2 * scale
    for x, y, w, h in components:
        if w == unit and h % unit == 0 and 2 <= h // unit <= 8:
            parts = [
                image[y + k * unit:y + (k + 1) * unit, x:x + w]
                for k in range(h // unit)
            ]
        elif h == unit and w % unit == 0 and 2 <= w // unit <= 8:
            parts = [
                image[y:y + h, x + k * unit:x + (k + 1) * unit]
                for k in range(w // unit)
            ]
        else:
            continue
        if all(np.all(p == p[0, 0]) for p in parts):
            colors = [int(p[0, 0]) for p in parts]
            if len(set(colors)) == len(colors):
                legends.append(colors)

    legend = max(legends, key=len) if legends else []
    return tiles, legend, scale


def _binary_plan(effects, requirements, priorities):
    constrained = [i for i, b in enumerate(requirements) if b is not None]
    row_number = {i: r for r, i in enumerate(constrained)}
    rhs = sum(
        1 << r for r, i in enumerate(constrained)
        if requirements[i]
    )
    if rhs == 0:
        return []

    # Discard useless columns and merge equivalent actions.
    representatives = {}
    for j, affected in enumerate(effects):
        column = 0
        for i in affected:
            if i in row_number:
                column ^= 1 << row_number[i]
        if column:
            old = representatives.get(column)
            if old is None or priorities[j] > priorities[old]:
                representatives[column] = j

    columns = list(representatives)
    actions = [representatives[c] for c in columns]
    n = len(columns)
    m = len(constrained)
    if not n:
        return None

    lookup = {c: j for j, c in enumerate(columns)}
    if rhs in lookup:
        return [actions[lookup[rhs]]]
    for j, c in enumerate(columns):
        other = lookup.get(rhs ^ c)
        if other is not None:
            return [actions[j], actions[other]]

    rhs_bit = 1 << n
    rows = []
    for r in range(m):
        row = rhs_bit if (rhs >> r) & 1 else 0
        for j, column in enumerate(columns):
            if (column >> r) & 1:
                row |= 1 << j
        rows.append(row)

    pivots = []
    rank = 0
    for col in range(n):
        bit = 1 << col
        selected = next(
            (r for r in range(rank, m) if rows[r] & bit), None
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        pivot_row = rows[rank]
        for r in range(m):
            if r != rank and rows[r] & bit:
                rows[r] ^= pivot_row
        pivots.append(col)
        rank += 1
        if rank == m:
            break

    coefficient_mask = rhs_bit - 1
    for row in rows:
        if not (row & coefficient_mask) and row & rhs_bit:
            return None

    particular = 0
    for r, col in enumerate(pivots):
        if rows[r] & rhs_bit:
            particular |= 1 << col

    pivot_set = set(pivots)
    basis = []
    for free in range(n):
        if free in pivot_set:
            continue
        vector = 1 << free
        for r, pivot in enumerate(pivots):
            if (rows[r] >> free) & 1:
                vector |= 1 << pivot
        basis.append(vector)

    best = particular
    best_weight = best.bit_count()
    d = len(basis)

    # Solutions of weight 1 and 2 were already checked above.
    if best_weight > 3 and d <= 18:
        current = particular
        for t in range(1, 1 << d):
            current ^= basis[(t & -t).bit_length() - 1]
            weight = current.bit_count()
            if weight < best_weight:
                best, best_weight = current, weight
                if weight == 3:
                    break

    elif best_weight > 3 and d:
        rng = random.Random(rhs + 1000003 * n)
        for restart in range(28):
            current = best if restart % 3 == 0 else particular
            if restart:
                for vector in basis:
                    if rng.randrange(2):
                        current ^= vector

            for _ in range(10):
                weight = current.bit_count()
                flip = 0
                new_weight = weight

                for vector in basis:
                    w = (current ^ vector).bit_count()
                    if w < new_weight:
                        flip, new_weight = vector, w

                if not flip:
                    if d <= 64:
                        pairs = (
                            (a, b) for a in range(d)
                            for b in range(a + 1, d)
                        )
                    else:
                        pairs = (
                            (rng.randrange(d), rng.randrange(d))
                            for _ in range(512)
                        )
                    for a, b in pairs:
                        vector = basis[a] ^ basis[b]
                        w = (current ^ vector).bit_count()
                        if w < new_weight:
                            flip, new_weight = vector, w

                if not flip:
                    break
                current ^= flip

            weight = current.bit_count()
            if weight < best_weight:
                best, best_weight = current, weight
            if best_weight == 3:
                break

    return [actions[j] for j in range(n) if (best >> j) & 1]


def _multicolor_plan(effects, current, allowed, k):
    """Enumerate coupled actions; solve independently controllable cells directly."""
    n = len(current)
    independent = {}
    coupled = []

    for j, affected in enumerate(effects):
        if len(affected) == 1:
            independent.setdefault(affected[0], j)
        elif affected:
            coupled.append(j)

    correction = []
    for i in range(n):
        if not allowed[i]:
            return None
        correction.append([
            next(t for t in range(k) if (v + t) % k in allowed[i])
            for v in range(k)
        ])

    last = [-1] * n
    for p, j in enumerate(coupled):
        for i in effects[j]:
            if i not in independent:
                last[i] = p

    checks = [[] for _ in coupled]
    for i in range(n):
        if i in independent:
            continue
        if last[i] < 0:
            if current[i] not in allowed[i]:
                return None
        else:
            checks[last[i]].append(i)

    values = current[:]
    chosen = [0] * len(effects)
    best_plan = None
    best_cost = float("inf")
    nodes = 0
    node_limit = min(80000, max(5000, 2500000 // max(1, n)))
    independent_items = list(independent.items())

    def search(p, cost):
        nonlocal nodes, best_cost, best_plan
        nodes += 1
        if nodes > node_limit or cost >= best_cost:
            return

        if p == len(coupled):
            extra = []
            total = cost
            for i, j in independent_items:
                amount = correction[i][values[i]]
                total += amount
                if total >= best_cost:
                    return
                if amount:
                    extra.append((j, amount))
            best_cost = total
            best_plan = [
                (j, chosen[j]) for j in coupled if chosen[j]
            ] + extra
            return

        j = coupled[p]
        affected = effects[j]
        saved = [values[i] for i in affected]

        for amount in range(k):
            if cost + amount >= best_cost:
                break
            for i, old in zip(affected, saved):
                values[i] = (old + amount) % k
            if all(values[i] in allowed[i] for i in checks[p]):
                chosen[j] = amount
                search(p + 1, cost + amount)

        chosen[j] = 0
        for i, old in zip(affected, saved):
            values[i] = old

    search(0, 0)
    if best_plan is None:
        return None
    return [j for j, amount in best_plan if amount]


def act(obs, available, memory):
    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    tiles, legend, scale = _parse_board(obs)
    movable = [t for t in tiles if t["kind"] != "C"]
    clues = [t for t in tiles if t["kind"] == "C"]

    if not movable:
        return "ACTION6", 32, 32

    signature_parts = []
    for t in tiles:
        if t["kind"] == "C":
            pattern = tuple(t["pixels"])
        elif t["kind"] == "N":
            pattern = tuple(int(v == 6) for v in t["pixels"])
        else:
            pattern = ()
        signature_parts.append((t["x"], t["y"], t["kind"], pattern))
    signature = repr((scale, signature_parts))

    if memory.get("signature") != signature:
        memory.clear()
        memory.update({
            "signature": signature,
            "regular_mask": [-1] * 9,
            "observed_effects": {},
            "transitions": {},
            "colors": [],
            "step": 0,
        })

    n = len(movable)
    positions = {(t["x"], t["y"]): i for i, t in enumerate(movable)}
    spacing = 4 * scale
    offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    current_colors = [t["color"] for t in movable]

    def neighbor(x, y):
        hit = positions.get((x, y))
        if hit is not None:
            return hit
        # Also implement the simulator's point-in-sprite lookup.
        for i, t in enumerate(movable):
            if (t["x"] <= x < t["x"] + 3 * scale and
                    t["y"] <= y < t["y"] + 3 * scale):
                return i
        return None

    neighbors = []
    for t in movable:
        neighbors.append([
            neighbor(t["x"] + dx * spacing, t["y"] + dy * spacing)
            for dx, dy in offsets
        ])

    known_mask = memory["regular_mask"]
    observed = memory["observed_effects"]

    previous = memory.get("previous_colors")
    last_action = memory.get("previous_action")
    if previous is not None and len(previous) == n and last_action is not None:
        changed = [
            i for i in range(n) if previous[i] != current_colors[i]
        ]
        observed[str(last_action)] = changed
        for i in changed:
            memory["transitions"][str(previous[i])] = current_colors[i]

        if movable[last_action]["kind"] == "R":
            changed_set = set(changed)
            for p, i in enumerate(neighbors[last_action]):
                if i is not None:
                    known_mask[p] = int(i in changed_set)

    effects = []
    priorities = []
    for j, t in enumerate(movable):
        if str(j) in observed:
            affected = list(observed[str(j)])
        else:
            if t["kind"] == "N":
                mask = [
                    p == 4 or t["pixels"][p] == 6
                    for p in range(9)
                ]
            else:
                mask = [
                    bool(known_mask[p]) if known_mask[p] >= 0 else p == 4
                    for p in range(9)
                ]
            affected = sorted({
                i for p, i in enumerate(neighbors[j])
                if i is not None and mask[p]
            })
        effects.append(affected)

        information = 0
        if t["kind"] == "R" and str(j) not in observed:
            information = sum(
                i is not None and known_mask[p] < 0
                for p, i in enumerate(neighbors[j])
            )
        priorities.append((information, len(affected), -j))

    # Each link requires equality (marker 0) or inequality (other markers).
    links = []
    for clue in clues:
        for p, (dx, dy) in enumerate(offsets):
            if p == 4:
                continue
            i = neighbor(
                clue["x"] + dx * spacing,
                clue["y"] + dy * spacing
            )
            if i is not None:
                links.append((i, clue["color"], clue["pixels"][p] == 0))

    colors = []
    for c in legend + memory["colors"] + current_colors + [
        t["color"] for t in clues
    ]:
        if c not in colors:
            colors.append(c)
    memory["colors"] = colors[:]

    # Use observed transitions to reconstruct a longer colour cycle.
    if len(colors) > 2:
        transitions = memory["transitions"]
        for start in colors:
            chain = [start]
            while str(chain[-1]) in transitions:
                nxt = transitions[str(chain[-1])]
                if nxt in chain or nxt not in colors:
                    break
                chain.append(nxt)
            if len(chain) == len(colors):
                colors = chain
                break

    requirements = [None] * n
    plan = None
    allowed = None
    current_indices = None

    if len(colors) <= 2:
        consistent = True
        for i, color, equality in links:
            satisfied = (current_colors[i] == color) == equality
            needed = 0 if satisfied else 1
            if requirements[i] is not None and requirements[i] != needed:
                consistent = False
            requirements[i] = needed

        if consistent:
            plan = _binary_plan(effects, requirements, priorities)
    else:
        k = len(colors)
        color_index = {c: i for i, c in enumerate(colors)}
        current_indices = [color_index[c] for c in current_colors]
        allowed = [set(range(k)) for _ in movable]
        for i, color, equality in links:
            c = color_index[color]
            if equality:
                allowed[i].intersection_update({c})
            else:
                allowed[i].discard(c)
        plan = _multicolor_plan(effects, current_indices, allowed, k)

    if plan:
        chosen = max(plan, key=lambda j: priorities[j])
    elif plan == []:
        # A successful state is checked only after a valid cell click.
        if allowed is None:
            harmless = [
                j for j, affected in enumerate(effects)
                if all(requirements[i] is None for i in affected)
            ]
        else:
            k = len(colors)
            harmless = [
                j for j, affected in enumerate(effects)
                if all((current_indices[i] + 1) % k in allowed[i]
                       for i in affected)
            ]
        chosen = max(harmless or list(range(n)),
                     key=lambda j: priorities[j])
    else:
        # If a partially learned movement mask is inconsistent, probe the
        # cell that reveals the most still-unknown mask entries.
        probes = [j for j in range(n) if priorities[j][0] > 0]
        if probes:
            chosen = max(probes, key=lambda j: priorities[j])
        else:
            def improvement(j):
                if allowed is None:
                    gain = sum(
                        1 if requirements[i] == 1 else -1
                        for i in effects[j] if requirements[i] is not None
                    )
                else:
                    k = len(colors)
                    gain = sum(
                        int((current_indices[i] + 1) % k in allowed[i])
                        - int(current_indices[i] in allowed[i])
                        for i in effects[j]
                    )
                return gain, priorities[j], -((j - memory["step"]) % n)

            chosen = max(range(n), key=improvement)

    memory["previous_colors"] = current_colors
    memory["previous_action"] = chosen
    memory["step"] += 1

    tile = movable[chosen]
    row = tile["y"] + (3 * scale) // 2
    col = tile["x"] + (3 * scale) // 2
    return "ACTION6", int(row), int(col)
