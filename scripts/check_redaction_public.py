"""Mechanical check of the public-tier mechanics-only sections (condition M, `wsp.redact_public`).

Per game (reads runs/private/game_sections_public/<id>_M.md):
  1. static checks on the game code block:
     - it equals what wsp.redact_public.redact_source produces now (no stale section);
     - compile(); no undefined global names (symtable) beyond the original's; no `self.<attr>` that
       is read but never assigned, beyond the original's;
     - no held level named in a comment, docstring or string ("L4", "level 4", "grid4", ordinals),
       except dict keys whose value is WITHHELD and config `Allow` patterns;
     - no route or planner content: no identifier or string that only the privileged planner sidecar
       uses, no planner/route vocabulary, no int sequence that reproduces part of a frozen route
       (action ids or click coordinates of any level);
     - no comparison of a level-number expression against a held level outside a raise-only branch.
  2. functional equivalence on support levels: the official source tree is copied to a temp dir and
     the game file replaced by the code block of the section; original and redacted modules are
     loaded from file (exec of the file bytes, as the arc3_games loader does, without its hash pin).
     Both are entered at each support level k by the same direct mechanism (fresh game, RESET, then
     `set_level(k)` plus the level bookkeeping: `_score = k` and the replay
     attributes) and driven with identical sequences, recorded on the original: 5 uniform random
     (length 60, uniform click cells, RESET with p=0.05), 5 board-biased random (clicks on
     non-background pixels), the level's segment of the frozen route, and 3 route-prefix + 30 random
     steps. After every step frames (all frames of the action), state, level, score, action count and
     available actions are compared. A sequence stops at a level change or a terminal state; when the
     original enters a held level the redacted game must raise LookupError in that same step.
     Global `random`/`numpy.random` are seeded identically before each run (lf52 uses np.random).
  Statement coverage of the redacted game under these runs is recorded (functions never entered are
  listed for the manual audit).
  3. withheld levels refuse: `set_level(h)` on the redacted game raises LookupError for every held h
     (and works on the original); continuous entry (support route completing into a held level)
     raises too.
Writes runs/private/game_sections_public/check.json and check_summary.md.
Usage: check_redaction_public.py [--games g1 g2 ...] [--quick] [--jobs N]
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import io
import json
import random
import re
import shutil
import symtable
import sys
import tempfile
import time
import tokenize
import traceback
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import redact_public as RP  # noqa: E402
from wsp.redact_public_configs import Allow, config_for  # noqa: E402

OUT = RP.RUNS / "private" / "game_sections_public"
TMP_ROOT = RP.RUNS / "private" / "redaction_check_tmp"
from wsp.public import ROUTES_PATH as ROUTES  # noqa: E402  (manifests/public/official_routes.json)


# =============================================================================================
# loading and driving
# =============================================================================================
def load_class(path: Path, tag: str):
    from arcengine import ARCBaseGame
    name = f"_pubcheck_{tag}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_bytes(), str(path), "exec"), mod.__dict__)
    cls = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, ARCBaseGame)
           and v is not ARCBaseGame and v.__module__ == name]
    if len(cls) != 1:
        raise ImportError(f"expected one ARCBaseGame subclass in {path}, found {cls}")
    return cls[0]


def _ai(a, r, c):
    from arcengine import ActionInput, GameAction
    return ActionInput(id=GameAction[a], data={"x": c, "y": r} if a == "ACTION6" else {})


def fresh(cls, k: int):
    """Direct entry at level k."""
    random.seed(0)
    np.random.seed(0)
    g = cls()
    g.perform_action(_ai("RESET", None, None), raw=True)
    if k:
        g.set_level(k)
        g._score = k
        try:    # replay bookkeeping of the game package's planner (only when the package is installed)
            from arc3_games.games import planning as _pl
        except ImportError:
            _pl = None
        if _pl is not None:
            setattr(g, _pl._REPLAY_ORIGIN_ATTR, k)
            setattr(g, _pl._REPLAY_SCORE_ATTR, k)
            setattr(g, _pl._REPLAY_HISTORY_ATTR, ())
    return g


def _h(frame) -> str:
    a = np.asarray(frame).astype(np.int16)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def _obs(g, frames=None) -> dict:
    if frames is None:
        frames = [g.camera.render(g.current_level.get_sprites())]
    return {"frames": [_h(f) for f in frames], "state": g._state.name, "level": int(g._current_level_index),
            "score": int(g._score), "count": int(g._action_count),
            "avail": [int(a) for a in g._available_actions]}


def run(cls, k: int, actions) -> list:
    rec = []
    try:
        g = fresh(cls, k)
        rec.append(_obs(g))
    except Exception as exc:  # noqa: BLE001
        return [{"raise": type(exc).__name__, "msg": str(exc)[:200], "at": "entry"}]
    for a, r, c in actions:
        try:
            res = g.perform_action(_ai(a, r, c), raw=True)
            o = _obs(g, res.frame if res.frame else None)
        except Exception as exc:  # noqa: BLE001
            rec.append({"raise": type(exc).__name__, "msg": str(exc)[:200],
                        "tb": traceback.format_exc()[-600:]})
            break
        rec.append(o)
        if o["level"] != k or o["state"] in ("WIN", "GAME_OVER"):
            break
    return rec


def _avail_names(g) -> list[str]:
    from arcengine import GameAction
    return [GameAction.from_id(int(a)).name for a in g._available_actions]


def gen_random(cls, k: int, seed: str, biased: bool, prefix=(), length: int = 60) -> list:
    """Random sequence recorded on the original (stops where the run would stop)."""
    rng = random.Random(seed)
    g = fresh(cls, k)
    acts = []
    last = g.camera.render(g.current_level.get_sprites())
    for a, r, c in prefix:
        res = g.perform_action(_ai(a, r, c), raw=True)
        acts.append((a, r, c))
        if res.frame:
            last = res.frame[-1]
        if g._current_level_index != k or g._state.name in ("WIN", "GAME_OVER"):
            return acts
    for _ in range(length):
        names = [x for x in _avail_names(g) if x != "RESET"]
        if not names or rng.random() < 0.05:
            a = "RESET"
        else:
            a = rng.choice(names)
        r = c = None
        if a == "ACTION6":
            if biased:
                fr = np.asarray(last)
                vals, cnt = np.unique(fr, return_counts=True)
                ys, xs = np.nonzero(fr != vals[np.argmax(cnt)])
                if len(ys):
                    j = rng.randrange(len(ys))
                    r, c = int(ys[j]), int(xs[j])
            if r is None:
                r, c = rng.randrange(64), rng.randrange(64)
        acts.append((a, r, c))
        res = g.perform_action(_ai(a, r, c), raw=True)
        if res.frame:
            last = res.frame[-1]
        if g._current_level_index != k or g._state.name in ("WIN", "GAME_OVER"):
            break
    return acts


def compare(orig: list, red: list, held: set) -> str | None:
    """None if equivalent; else a description of the first difference."""
    for j, o in enumerate(orig):
        r = red[j] if j < len(red) else None
        if "raise" in o:
            return f"original raised at step {j}: {o}"
        if o.get("level") in held and j == len(orig) - 1:
            if r is not None and r.get("raise") == "LookupError":
                return None
            return f"step {j}: original entered held level {o['level']}, redacted gave {str(r)[:300]}"
        if r != o:
            return f"step {j}: orig={str(o)[:300]} red={str(r)[:400]}"
    if len(red) != len(orig):
        return f"length {len(orig)} vs {len(red)}: extra red {str(red[len(orig):])[:300]}"
    return None


# =============================================================================================
# static checks
# =============================================================================================
BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__spec__", "__package__",
                                      "__loader__", "__builtins__", "__annotations__", "__dict__"}


def undefined_globals(src: str) -> set[str]:
    top = symtable.symtable(src, "x", "exec")

    def walk(t):
        yield t
        for c in t.get_children():
            yield from walk(c)
    tables = list(walk(top))
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    for t in tables:
        for s in t.get_symbols():
            if s.is_declared_global() and s.is_assigned():
                bound.add(s.get_name())
    bad = set()
    for t in tables:
        for s in t.get_symbols():
            nm = s.get_name()
            if s.is_referenced() and nm not in bound and nm not in BUILTIN_NAMES and (t is top or s.is_global()):
                bad.add(nm)
    return bad


def self_attrs(src: str) -> tuple[set, set]:
    loads, stores = set(), set()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                stores.add(n.attr)
            elif isinstance(n.value, ast.Name) and n.value.id == "self":
                loads.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(n.name)
        elif isinstance(n, ast.ClassDef):
            pass
        if isinstance(n, ast.ClassDef):
            for st in n.body:
                if isinstance(st, ast.Assign):
                    stores |= {t.id for t in st.targets if isinstance(t, ast.Name)}
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    stores.add(st.target.id)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setattr" and len(n.args) >= 2 \
                and isinstance(n.args[1], ast.Constant):
            stores.add(n.args[1].value)
    return loads, stores


LEVEL_WORDS = r"(?:L|level|levels|Level|Levels|LEVEL|LEVELS|lvl|Lvl|grid|Grid|stage|Stage)"
NUM_LIST = r"(\d+(?:\s*(?:,|and|or|&|/|-|–|—|to|\.\.)\s*(?:L)?\d+)*)"
LEVEL_REF_RE = re.compile(r"(?<![A-Za-z0-9])" + LEVEL_WORDS + r"[\s_\-]?#?\s?" + NUM_LIST + r"(?![0-9])")
ORDINALS = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth".split())}
ORD_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+level\b", re.I)
VOCAB_RE = re.compile(r"solution|solve|solver|shortest|oracle|planner|witness|optimal|route|cheat|answer|"
                      r"hint|tutorial|walkthrough", re.I)


def _expand(numlist: str) -> list[int]:
    out = []
    for p in re.split(r"\s*(?:,|and|or|&|/)\s*", numlist):
        m = re.match(r"L?(\d+)\s*(?:-|–|—|to|\.\.)\s*L?(\d+)$", p.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out += list(range(a, b + 1)) if a <= b <= a + 20 else [a, b]
        else:
            out += [int(x) for x in re.findall(r"\d+", p)]
    return out


def prose_items(src: str) -> list[tuple[int, str, str]]:
    items = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            items.append((tok.start[0], "comment", tok.string))
    tree = ast.parse(src)
    withheld_keys = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(v, ast.Name) and v.id == "WITHHELD" and isinstance(k, ast.Constant):
                    withheld_keys.add(id(k))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in withheld_keys:
            items.append((n.lineno, "string", n.value))
    return items


def held_mentions(src: str, held1: set[int], allow: list) -> tuple[list, list]:
    hits, allowed = [], []
    for line, kind, text in prose_items(src):
        found = []
        for m in LEVEL_REF_RE.finditer(text):
            bad = set(_expand(m.group(1))) & held1
            if bad:
                found.append(f"line {line} [{kind}]: '{m.group(0)}' in {text[:80]!r} -> held {sorted(bad)}")
        for m in ORD_RE.finditer(text):
            if ORDINALS[m.group(1).lower()] in held1:
                found.append(f"line {line} [{kind}]: ordinal '{m.group(0)}'")
        for f in found:
            if any(re.search(a.pattern, text) for a in allow):
                allowed.append(f)
            else:
                hits.append(f)
    return hits, allowed


def vocab_hits(src: str, allow: list) -> tuple[list, list]:
    bad, ok = [], []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.COMMENT) and \
                tok.type != getattr(tokenize, "FSTRING_MIDDLE", -1):
            continue
        for m in VOCAB_RE.finditer(tok.string):
            item = f"line {tok.start[0]}: {tok.string[:100]!r}"
            (ok if any(re.search(a.pattern, tok.string) for a in allow) else bad).append(item)
    return bad, ok


def planner_tokens(game_id: str, orig_src: str) -> set[str]:
    """Identifiers and strings of the privileged planner sidecar that the game source does not use."""
    import arc3_games
    p = Path(arc3_games.__file__).parent / "games" / "public_planners" / f"{game_id}.py"
    if not p.exists():
        return set()
    tree = ast.parse(p.read_text())
    toks = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            toks.add(n.name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and 6 <= len(n.value) <= 80 \
                and re.fullmatch(r"[\w\-]+", n.value):
            toks.add(n.value)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    toks.add(t.id)
    orig_words = set(re.findall(r"[\w\-]+", orig_src))
    for f in RP.ENGINE_FILES:
        orig_words |= set(re.findall(r"[\w\-]+", (RP.ENGINE_DIR / f).read_text()))
    return {t for t in toks if len(t) >= 6 and t not in orig_words}


def int_sequences(src: str) -> list[list[int]]:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.List, ast.Tuple)):
            flat = []
            for e in ast.walk(n):
                if isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
                    flat.append(e.value)
                elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                    flat.append(-e.operand.value)
            if len(flat) >= 6:
                out.append(flat)
    return out


def route_hits(src: str, routes: list) -> list[str]:
    seqs = int_sequences(src)
    hay = [",".join(map(str, s)) for s in seqs]
    hits = []
    aid = {"RESET": 0, **{f"ACTION{i}": i for i in range(1, 8)}}
    for li, lvl in enumerate(routes):
        ids = [aid[a] for a, _, _ in lvl]
        clicks_rc = [v for a, r, c in lvl if a == "ACTION6" for v in (r, c)]
        clicks_cr = [v for a, r, c in lvl if a == "ACTION6" for v in (c, r)]
        grams = set()
        for i in range(len(ids) - 11):
            g = ids[i:i + 12]
            if len(set(g)) >= 3:
                grams.add(",".join(map(str, g)))
        for cl in (clicks_rc, clicks_cr):
            for i in range(0, len(cl) - 5, 2):
                g = cl[i:i + 6]
                if max(g) > 15:
                    grams.add(",".join(map(str, g)))
        for g in grams:
            for h in hay:
                if f",{g}," in f",{h},":
                    hits.append(f"L{li + 1} route n-gram {g}")
                    break
    return sorted(set(hits))


def missing_sprite_keys(src: str) -> list[str]:
    """`sprites["k"]` with a constant key that the (redacted) sprite dict does not define."""
    tree = ast.parse(src)
    keys = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sprites" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "sprites" \
                and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and n.slice.value not in keys:
            out.append(f"line {n.lineno}: sprites[{n.slice.value!r}]")
    return sorted(set(out))


ROUTE_WORDS_RE = re.compile(r"public_planners|official_routes|evaluate_state|direct_plan|solution_plan|"
                            r"supported_source_hash|recommended_plan|make_planner")


def static_checks(game_id: str, md: str, red: str, orig: str, rep_now: dict, routes: list) -> dict:
    cfg = config_for(game_id)
    allow = [e for e in cfg if isinstance(e, Allow)]
    held1 = set(rep_now["held_levels"])
    res: dict = {}
    try:
        compile(red, f"{game_id}.py", "exec")
        res["compile"] = True
    except SyntaxError as exc:
        res["compile"] = f"SyntaxError: {exc}"
    res["stale_section"] = red != RP.redact_source(game_id)[0]
    res["undefined_globals"] = sorted(undefined_globals(red) - undefined_globals(orig))
    rl, rs = self_attrs(red)
    ol, os_ = self_attrs(orig)
    res["dangling_self_attrs"] = sorted((rl - rs) - (ol - os_))
    hits, allowed = held_mentions(red, held1, allow)
    res["held_mentions"], res["held_mentions_allowed"] = hits, allowed
    vb, vok = vocab_hits(red, allow)
    res["vocab"], res["vocab_allowed"] = vb, vok
    toks = planner_tokens(game_id, orig)
    words = set(re.findall(r"[\w\-]+", red))
    res["planner_tokens_in_section"] = sorted(toks & words)
    res["planner_words_in_section"] = sorted(set(ROUTE_WORDS_RE.findall(md)))
    res["route_ngrams"] = route_hits(red, routes)
    res["held_level_compares"] = rep_now["level_index_uses"]["held_refs"]
    res["missing_sprite_keys"] = missing_sprite_keys(red)
    res["redaction_problems"] = rep_now["problems"]
    res["ok"] = (res["compile"] is True and not res["stale_section"] and not res["undefined_globals"]
                 and not res["dangling_self_attrs"] and not hits and not vb and not res["planner_tokens_in_section"]
                 and not res["planner_words_in_section"] and not res["route_ngrams"]
                 and not res["held_level_compares"] and not res["redaction_problems"]
                 and not res["missing_sprite_keys"])
    return res


# =============================================================================================
# per game
# =============================================================================================
def check_game(game_id: str, tree_root: str, quick: bool) -> dict:
    t0 = time.time()
    out: dict = {"game_id": game_id}
    try:
        md = (OUT / f"{game_id}_M.md").read_text()
        red = RP.parse_game_block(md, game_id)
        orig_path = RP.source_path(game_id)
        orig = orig_path.read_text()
        _, rep_now = RP.redact_source(game_id)
        n = rep_now["n_levels"]
        held = set(rep_now["held_levels_0based"])
        support = [k for k in range(n) if k not in held]
        routes_all = json.loads(ROUTES.read_text())[game_id]["levels"]
        out["static"] = static_checks(game_id, md, red, orig, rep_now, routes_all)

        red_path = RP.source_path(game_id, Path(tree_root))
        red_path.write_text(red)
        cls_o = load_class(orig_path, "orig")
        cls_r = load_class(red_path, "red")

        func = {"levels": {}, "failures": [], "n_sequences": 0, "n_steps": 0}
        covered: set[int] = set()
        rfile = str(red_path)

        def tracer(frame, event, arg):
            if frame.f_code.co_filename != rfile:
                return None
            if event in ("line", "call"):
                covered.add(frame.f_lineno)
            return tracer
        n_rand = 2 if quick else 5
        for k in support:
            seqs = []
            for i in range(n_rand):
                seqs.append(("uniform", gen_random(cls_o, k, f"{game_id}:{k}:u{i}", False)))
                seqs.append(("biased", gen_random(cls_o, k, f"{game_id}:{k}:b{i}", True)))
            route = [tuple(a) for a in routes_all[k]] if k < len(routes_all) else []
            if route:
                seqs.append(("route", route))
                rng = random.Random(f"{game_id}:{k}:prefix")
                for i in range(1 if quick else 3):
                    cut = rng.randint(1, max(1, len(route) - 1))
                    seqs.append(("route_prefix", gen_random(cls_o, k, f"{game_id}:{k}:p{i}", True,
                                                            prefix=route[:cut], length=30)))
            lv = {"sequences": 0, "steps": 0, "fail": 0, "ended": {}}
            for kind, acts in seqs:
                ro = run(cls_o, k, acts)
                sys.settrace(tracer)
                try:
                    rr = run(cls_r, k, acts)
                finally:
                    sys.settrace(None)
                diff = compare(ro, rr, held)
                lv["sequences"] += 1
                lv["steps"] += len(ro) - 1
                last = ro[-1]
                end = ("held_entry" if last.get("level") in held else "level_change" if last.get("level") != k
                       else last.get("state", "raise"))
                lv["ended"][end] = lv["ended"].get(end, 0) + 1
                if kind == "route":
                    lv["route_completes_level"] = last.get("level") != k or last.get("state") == "WIN"
                    if last.get("level") in held:
                        func.setdefault("continuous_entry_raises", {})[str(last["level"] + 1)] = diff is None
                if diff:
                    lv["fail"] += 1
                    func["failures"].append(f"L{k + 1} {kind}: {diff}")
            func["levels"][f"L{k + 1}"] = lv
            func["n_sequences"] += lv["sequences"]
            func["n_steps"] += lv["steps"]
        entry = {}
        for h in sorted(held):
            g = fresh(cls_r, 0)
            try:
                g.set_level(h)
                entry[f"L{h + 1}"] = "NO RAISE"
            except LookupError:
                entry[f"L{h + 1}"] = "LookupError"
            except Exception as exc:  # noqa: BLE001
                entry[f"L{h + 1}"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            go = fresh(cls_o, 0)
            go.set_level(h)
        func["held_entry"] = entry
        func["ok"] = not func["failures"] and all(v == "LookupError" for v in entry.values()) and \
            all(func.get("continuous_entry_raises", {}).values())
        out["functional"] = func
        out["coverage"] = coverage_report(red, covered)
        out["ok"] = out["static"]["ok"] and func["ok"]
    except Exception as exc:  # noqa: BLE001
        out["error"] = traceback.format_exc()[-2000:]
        out["ok"] = False
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def coverage_report(src: str, covered: set[int]) -> dict:
    """Statements of the redacted game executed by the support-level runs (redacted side only).
    Functions never entered are listed: code that no support level exercises (for the manual audit)."""
    tree = ast.parse(src)
    stmts, never, partly = set(), [], []

    def walk(body, prefix):
        for n in body:
            if isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + ".")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = {s.lineno for s in ast.walk(n) if isinstance(s, ast.stmt) and s is not n}
                stmts.update(lines)
                hit = lines & covered
                if not hit and n.lineno not in covered:
                    never.append(f"{prefix}{n.name} (L{n.lineno}-{n.end_lineno})")
                elif len(lines - covered) >= 1:
                    partly.append((f"{prefix}{n.name}", sorted(lines - covered)))
    walk(tree.body, "")
    return {"statements": len(stmts), "executed": len(stmts & covered),
            "fraction": round(len(stmts & covered) / max(1, len(stmts)), 3),
            "functions_never_entered": never,
            "unexecuted_lines": {name: lines for name, lines in partly}}


def _one(args):
    return check_game(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=RP.GAMES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="public_", dir=TMP_ROOT))
    tree = tmp / "official"
    shutil.copytree(RP.official_root(), tree)
    tree.chmod(0o755)
    for p in tree.rglob("*"):
        p.chmod(0o755 if p.is_dir() else 0o644)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(_one, [(g, str(tree), a.quick) for g in a.games]))
    if not a.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    path = OUT / "check.json"
    old = json.loads(path.read_text()) if path.exists() and set(a.games) != set(RP.GAMES) else {}
    allres = {**old.get("games", {}), **{r["game_id"]: r for r in results}}
    rows = ["| game | n | held | support | static | seqs | steps | func fails | held entry | cont. entry | stmt cov. | pass |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in RP.GAMES:
        r = allres.get(g)
        if not r:
            continue
        st, fn = r.get("static", {}), r.get("functional", {})
        n = RP.n_levels(g)
        held = RP.held_levels(g)
        rows.append(f"| {g} | {n} | {','.join(str(h + 1) for h in held)} | "
                    f"{','.join(str(k + 1) for k in range(n) if k not in held)} | {'ok' if st.get('ok') else 'FAIL'} | "
                    f"{fn.get('n_sequences', '-')} | {fn.get('n_steps', '-')} | {len(fn.get('failures', [])) if fn else '-'} | "
                    f"{sum(v == 'LookupError' for v in fn.get('held_entry', {}).values())}/{len(held)} | "
                    f"{sum(fn.get('continuous_entry_raises', {}).values())}/{len(fn.get('continuous_entry_raises', {}))} | "
                    f"{r.get('coverage', {}).get('fraction', '-')} | "
                    f"{'PASS' if r.get('ok') else 'FAIL'} |")
    table = "\n".join(rows)
    doc = {"description": __doc__.split("\n")[0], "quick": a.quick, "wall_s": round(time.time() - t0, 1),
           "all_pass": all(allres.get(g, {}).get("ok") for g in RP.GAMES), "table": table, "games": allres}
    path.write_text(json.dumps(doc, indent=1))
    (OUT / "check_summary.md").write_text("# Public-tier redaction check (condition M)\n\n" + table + "\n")
    print(table)
    for r in results:
        if not r.get("ok"):
            print("\n==", r["game_id"], "FAIL")
            if "error" in r:
                print(r["error"])
            st = r.get("static", {})
            for k in ("compile", "stale_section", "undefined_globals", "dangling_self_attrs", "held_mentions", "vocab",
                      "planner_tokens_in_section", "planner_words_in_section", "route_ngrams",
                      "held_level_compares", "redaction_problems", "missing_sprite_keys"):
                v = st.get(k)
                if v and v is not True:
                    print(f"  {k}: {str(v)[:1500]}")
            for f in r.get("functional", {}).get("failures", [])[:6]:
                print("  F", f[:700])
            he = r.get("functional", {}).get("held_entry")
            if he and any(v != "LookupError" for v in he.values()):
                print("  held_entry", he)


if __name__ == "__main__":
    main()
