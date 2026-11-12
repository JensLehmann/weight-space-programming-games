"""Mechanics-only writer sections (condition M) for the 25 official public ARC3 games.

The section shows (1) a header naming the withheld levels, (2) an excerpt of
the `arcengine` source the game uses, (3) the game's redacted source. The harness keeps using the
full pinned source; only the prompt text is redacted. Planner sidecars and the frozen routes are
never read here.

Source: `<root>/<game>/<rev>/<game>.py`, root = arc3_games.sources.source_root("official",
cache_dir=runs/private/arc3_cache).

Redaction of the game source = generic passes + the game's reviewed config
(`wsp.redact_public_configs`), all computed on the original text and applied right to left:
  G1  `levels`: the module-level list with one Level(...) per level. Held entries -> WITHHELD; the
      comment block directly above a held entry (e.g. "# Level 4") goes with it.
  G2  sprite library `sprites` (plus config `Library` dicts): a template is kept iff a support-level
      entry places it (`sprites["k"]`) or names it in its data, or code outside the withheld spans
      names it (dict key or its `name=`), or a config `Keep` matches; a config `Drop` overrides the
      name-based keeps (reviewed cases where code only compares names/tags). All other templates (used
      only by held levels, or by nothing) are omitted; a comment above the dict says so.
  G3  comments that name a held level ("L4", "level 4", ...) are dropped.
  C   config edits: Items (per-level tables), Branch (held-level-gated if/elif -> raise), Body (code
      whose only entry point is withheld -> raise), Sub (exact text edits).
`WITHHELD` is a sentinel defined after the imports: the engine may clone or copy it (ARCBaseGame
clones every level at start-up); any other use raises LookupError('level withheld for evaluation').
Entering a held level therefore raises (set_level reads level.grid_size).

Engine excerpt rule (EXCERPT_RULE below): unmodified arcengine source, restricted to
  * the engine classes/functions the game imports, plus the engine types of the action/render loop
    (ARCBaseGame, Camera, Level, Sprite, GameAction, GameState, ActionInput);
  * within a class: its `class` line, docstring and class-level declarations, and (a) the members the
    game executes on its support levels (recorded by scripts/trace_engine_public.py into
    runs/private/game_sections_public/engine_usage.json: 10 random sequences per support level, no
    routes), (b) members it overrides in a subclass, (c) members whose names the game uses but never
    defines itself (an unambiguous engine reference), (d) the action/render loop (CORE), `__init__`;
  * closed transitively over engine members, module functions and engine classes that included code
    references (pydantic output/validation containers FrameData, FrameDataRaw, SimpleAction,
    ComplexAction are not pulled in by the closure);
  * small enum/value classes are shown whole; omitted members are listed by name in a comment.
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import tokenize
from pathlib import Path

import arcengine

from .manifest import RUNS
from .queries import held_out_levels
from .redact_public_configs import (RAISE, Allow, Body, Branch, Drop, Items, Keep, LevelExpr, Library, Note, Sub,
                                    config_for)

CACHE = RUNS / "private" / "arc3_cache"
GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
         "sp80 su15 tn36 tr87 tu93 vc33 wa30").split()

WITHHELD_DEF = '''

class _Withheld:
    """Placeholder for an evaluation-only entry you cannot see. The engine may clone or copy it (it
    clones every level at start-up); any other use raises LookupError('level withheld for evaluation')."""

    def clone(self):
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise LookupError("level withheld for evaluation")

    def __getitem__(self, key):
        raise LookupError("level withheld for evaluation")

    def __iter__(self):
        raise LookupError("level withheld for evaluation")

    def __len__(self):
        raise LookupError("level withheld for evaluation")


WITHHELD = _Withheld()
'''

TABLES_OMITTED = "# Entries of withheld levels (and entries no level uses) are omitted from this dict.\n"
SPRITES_OMITTED = "# Sprite templates that no visible level uses (only withheld levels, or none) are omitted.\n"


# ---------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------
def official_root() -> Path:
    from arc3_games.sources import source_root
    return Path(source_root("official", cache_dir=CACHE))


def source_path(game_id: str, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else official_root()
    files = sorted((root / game_id).glob(f"*/{game_id}.py"))
    if len(files) != 1:
        raise FileNotFoundError(f"{game_id}: expected one <rev>/{game_id}.py under {root / game_id}, got {files}")
    return files[0]


def n_levels(game_id: str) -> int:
    """Number of levels of an official game (manifests/public/official_routes.json)."""
    from .public import n_levels as _n_levels, PREFIX
    return _n_levels(PREFIX + game_id)


def held_levels(game_id: str) -> list[int]:
    """0-based held-out level indices (revision-2 split)."""
    return sorted(held_out_levels(n_levels(game_id), version=2))


# ---------------------------------------------------------------------------------------------
# text/offset helpers
# ---------------------------------------------------------------------------------------------
class _Text:
    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for ln in self.lines:
            self.starts.append(self.starts[-1] + len(ln))

    def off(self, lineno: int, col_bytes: int) -> int:
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode()[:col_bytes].decode())

    def span(self, node) -> tuple[int, int]:
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def line_start(self, lineno: int) -> int:
        return self.starts[lineno - 1]

    def line_end(self, lineno: int) -> int:          # offset after the newline of `lineno`
        return self.starts[lineno]

    def starts_line(self, node) -> bool:
        line = self.lines[node.lineno - 1]
        return line[: len(line.encode()[: node.col_offset].decode())].strip() == ""

    def comment_block_above(self, lineno: int) -> int:
        """First line (1-based) of the run of comment-only lines directly above `lineno`."""
        k = lineno
        while k - 1 >= 1 and self.lines[k - 2].strip().startswith("#"):
            k -= 1
        return k


def _apply(text: str, edits: list[tuple[int, int, str, str]]) -> tuple[str, list[str]]:
    """Apply (start, end, replacement, tag) edits; nested edits are dropped, partial overlaps fail."""
    spans = sorted(set(edits), key=lambda e: (e[0], -e[1]))
    kept, problems = [], []
    for a, b, rep, tag in spans:
        if kept and a >= kept[-1][0] and b <= kept[-1][1]:
            continue
        if kept and a < kept[-1][1]:
            problems.append(f"overlapping edits: {tag} at {a}-{b} vs {kept[-1][3]} at {kept[-1][0]}-{kept[-1][1]}")
            continue
        kept.append((a, b, rep, tag))
    for a, b, rep, _tag in reversed(kept):
        text = text[:a] + rep + text[b:]
    return text, problems


def _module_assign(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node
    return None


def _find_func(tree: ast.Module, qual: str):
    parts = qual.split(".")
    scope = tree.body
    node = None
    for i, p in enumerate(parts):
        node = next((n for n in scope if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == p), None)
        if node is None:
            raise KeyError(f"{qual}: '{p}' not found")
        scope = node.body
    return node


def _find_target(tree: ast.Module, target: str):
    """`name` (module level) or `Class.method:var` -> the literal value node of the assignment."""
    if ":" in target:
        qual, var = target.split(":")
        fn = _find_func(tree, qual)
        hits = []
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tg:
                    src = ast.unparse(t)
                    if src == var:
                        hits.append(n.value)
        if len(hits) != 1:
            raise KeyError(f"{target}: {len(hits)} assignments found")
        return hits[0]
    node = _module_assign(tree, target)
    if node is None:
        raise KeyError(f"{target}: module-level assignment not found")
    return node.value


def _held_comment_re(held1: list[int]) -> re.Pattern | None:
    if not held1:
        return None
    nums = "|".join(str(h) for h in sorted(held1, reverse=True))
    return re.compile(r"\b(?:L|level|Level|LEVEL|lvl|Lvl)\s?#?\s?(?:" + nums + r")\b")


# ---------------------------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------------------------
def _sprite_refs(tree: ast.Module, excluded: list[tuple[int, int]], T: _Text) -> set[str]:
    """String constants outside the excluded offset spans."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            a, b = T.span(n)
            if any(x <= a and b <= y for x, y in excluded):
                continue
            out.add(n.value)
        elif isinstance(n, ast.JoinedStr):
            pass
    return out


def _sprite_entries(dict_node: ast.Dict) -> list[tuple[str, ast.AST, ast.AST, str | None]]:
    out = []
    for k, v in zip(dict_node.keys, dict_node.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            raise ValueError("non-constant key in sprite library")
        name = None
        if isinstance(v, ast.Call):
            for kw in v.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = kw.value.value
        out.append((k.value, k, v, name))
    return out


def _remove_dict_entry(T: _Text, k, v) -> tuple[int, int]:
    a = T.off(k.lineno, k.col_offset)
    b = T.off(v.end_lineno, v.end_col_offset)
    rest = T.text[b:]
    m = re.match(r"[ \t]*,?[ \t]*(#[^\n]*)?\n", rest)
    if T.starts_line(k) and m:
        return T.line_start(k.lineno), b + m.end()
    m = re.match(r"\s*,", rest)
    return a, b + (m.end() if m else 0)


def _raise_body(T: _Text, node: ast.If) -> tuple[int, int, str]:
    body0, body1 = node.body[0], node.body[-1]
    header_end = node.test.end_lineno
    indent = " " * body0.col_offset
    if body0.lineno > header_end:
        return T.line_end(header_end), T.line_end(body1.end_lineno), f"{indent}{RAISE}\n"
    a, _ = T.span(body0)
    _, b = T.span(body1)
    return a, b, RAISE


def _drop_branch(T: _Text, node: ast.If) -> tuple[int, int, str]:
    head = T.lines[node.lineno - 1].lstrip()
    if not head.startswith("elif"):
        if node.orelse:
            raise ValueError("drop of an `if` with an else/elif: use action='raise'")
        start = T.comment_block_above(node.lineno)
        return T.line_start(start), T.line_end(node.end_lineno), ""
    return T.line_start(node.lineno), T.line_end(node.body[-1].end_lineno), ""


def redact_source(game_id: str, src: str | None = None) -> tuple[str, dict]:
    """Redacted game source (condition M) and its audit record."""
    path = source_path(game_id)
    src = path.read_text() if src is None else src
    n = n_levels(game_id)
    held = held_levels(game_id)
    held1 = [h + 1 for h in held]
    cfg = config_for(game_id)
    T = _Text(src)
    tree = ast.parse(src)
    edits: list[tuple[int, int, str, str]] = []
    withheld_spans: list[tuple[int, int]] = []
    rep: dict = {"game_id": game_id, "source": str(path), "n_levels": n, "held_levels_0based": held,
                 "held_levels": held1, "support_levels": [l + 1 for l in range(n) if l not in held],
                 "generic": {}, "config": [], "notes": [], "problems": []}

    # ---- G1: levels list
    lv = _module_assign(tree, "levels")
    if lv is None or not isinstance(lv.value, ast.List) or len(lv.value.elts) != n:
        raise ValueError(f"{game_id}: module-level `levels` list with {n} entries not found")
    for i, el in enumerate(lv.value.elts):
        if i in held:
            a, b = T.span(el)
            edits.append((a, b, "WITHHELD", f"G1 levels[{i}]"))
            withheld_spans.append((a, b))
            if T.starts_line(el):
                c0 = T.comment_block_above(el.lineno)
                if c0 < el.lineno:
                    edits.append((T.line_start(c0), T.line_start(el.lineno), "", f"G1 comment above levels[{i}]"))
    rep["generic"]["levels_withheld"] = [f"levels[{i}] (L{i + 1})" for i in held]
    level_entries = {i: T.span(el) for i, el in enumerate(lv.value.elts)}

    # ---- C: Items (per-level tables) and Branch edits (computed before G2 so their spans are excluded)
    for e in cfg:
        if isinstance(e, Items):
            node = _find_target(tree, e.target)
            done = []
            if isinstance(node, ast.Dict):
                keys = {k.value: (k, v) for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
                want = [(e.key(h) if e.key else h, f"L{h + 1}") for h in held] + [(x, "extra") for x in e.extra]
                for key, label in want:
                    if key not in keys:
                        raise KeyError(f"{game_id}: {e.target} has no key {key!r}")
                    k, v = keys[key]
                    if e.omit:
                        a, b = _remove_dict_entry(T, k, v)
                        edits.append((a, b, "", f"Items(omit) {e.target}[{key!r}]"))
                    else:
                        a, b = T.span(v)
                        edits.append((a, b, "WITHHELD", f"Items {e.target}[{key!r}]"))
                    withheld_spans.append((a, b))
                    done.append(f"{key!r} ({label})")
                if e.omit:
                    stmt = _module_assign(tree, e.target)
                    a = T.line_start(stmt.lineno)
                    edits.append((a, a, TABLES_OMITTED, f"Items(omit) note above {e.target}"))
            elif isinstance(node, (ast.List, ast.Tuple)):
                want = [((e.key(h) if e.key else h), f"L{h + 1}") for h in held] + [(x, "extra") for x in e.extra]
                if not e.key and not e.extra and len(node.elts) != n:
                    raise ValueError(f"{game_id}: {e.target} has {len(node.elts)} entries, expected {n}")
                for idx, label in want:
                    el = node.elts[idx]
                    a, b = T.span(el)
                    edits.append((a, b, "WITHHELD", f"Items {e.target}[{idx}]"))
                    withheld_spans.append((a, b))
                    done.append(f"[{idx}] ({label})")
            else:
                raise TypeError(f"{game_id}: {e.target} is not a list/tuple/dict literal")
            rep["config"].append({"kind": "Items", "target": e.target, "withheld": done, "why": e.why,
                                  **({"extra_why": e.extra_why} if e.extra else {})})
        elif isinstance(e, Body):
            fn = _find_func(tree, e.func)
            b0, b1 = fn.body[0], fn.body[-1]
            indent = " " * b0.col_offset
            if b0.lineno <= fn.lineno:
                raise ValueError(f"{game_id}: Body {e.func}: one-line function")
            a, b = T.line_start(b0.lineno), T.line_end(b1.end_lineno)
            edits.append((a, b, f"{indent}{RAISE}\n", f"Body {e.func}"))
            withheld_spans.append((a, b))
            rep["config"].append({"kind": "Body", "func": e.func, "why": e.why})
        elif isinstance(e, Branch):
            fn = _find_func(tree, e.func)
            want = ast.unparse(ast.parse(e.test, mode="eval").body)
            hits = [x for x in ast.walk(fn) if isinstance(x, ast.If) and ast.unparse(x.test) == want]
            if len(hits) != e.n:
                raise ValueError(f"{game_id}: Branch {e.func}: `{e.test}` matched {len(hits)} times, expected {e.n}")
            for node in hits:
                a, b, r = _raise_body(T, node) if e.action == "raise" else _drop_branch(T, node)
                edits.append((a, b, r, f"Branch {e.func}: {e.test}"))
                withheld_spans.append((a, b))
            rep["config"].append({"kind": "Branch", "func": e.func, "test": e.test, "action": e.action,
                                  "matches": len(hits), "why": e.why})

    # ---- G2: sprite libraries
    libs = ["sprites"] + [e.name for e in cfg if isinstance(e, Library)]
    keep_res = [re.compile(e.pattern) for e in cfg if isinstance(e, Keep)]
    drop_res = [re.compile(e.pattern) for e in cfg if isinstance(e, Drop)]
    lib_nodes = {}
    for name in libs:
        node = _module_assign(tree, name)
        if node is None or not isinstance(node.value, ast.Dict):
            raise ValueError(f"{game_id}: sprite library `{name}` not found")
        lib_nodes[name] = node
    lib_spans = [T.span(node) for node in lib_nodes.values()]
    support_spans = [s for i, s in level_entries.items() if i not in held]
    code_spans_excluded = lib_spans + [level_entries[i] for i in level_entries] + withheld_spans
    code_refs = _sprite_refs(tree, code_spans_excluded, T)
    support_refs, support_data_refs = set(), set()     # placed via sprites["k"] / other strings
    for n_ in ast.walk(lv.value):
        if isinstance(n_, ast.Subscript) and isinstance(n_.value, ast.Name) and n_.value.id in libs \
                and isinstance(n_.slice, ast.Constant) and isinstance(n_.slice.value, str):
            a, b = T.span(n_)
            if any(x <= a and b <= y for x, y in support_spans):
                support_refs.add(n_.slice.value)
    for n_ in ast.walk(lv.value):
        if isinstance(n_, ast.Constant) and isinstance(n_.value, str):
            a, b = T.span(n_)
            if any(x <= a and b <= y for x, y in support_spans) and n_.value not in support_refs:
                support_data_refs.add(n_.value)
    # library entries can reference each other (e.g. legends naming tiles): kept entries' refs count
    g2 = {}
    for name, node in lib_nodes.items():
        entries = _sprite_entries(node.value)
        kept, dropped, by_code_only = [], [], []
        # constants inside other kept library entries also count as references (fixpoint)
        lib_internal = {}
        for key, k, v, nm in entries:
            refs = set()
            for c in ast.walk(v):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    refs.add(c.value)
            lib_internal[key] = refs - {key, nm}
        status = {}
        for key, k, v, nm in entries:
            ids = {key} | ({nm} if nm else set())
            if ids & support_refs:
                status[key] = "support"
                if any(r.search(key) for r in drop_res):
                    rep["problems"].append(f"Drop matches {key!r}, which a support level places")
            elif any(r.search(key) for r in drop_res):
                continue
            elif ids & support_data_refs:
                status[key] = "support-data"
            elif ids & code_refs:
                status[key] = "code"
            elif any(r.search(key) for r in keep_res):
                status[key] = "keep"
        changed = True
        while changed:
            changed = False
            refs = set().union(*(lib_internal[k_] for k_ in status)) if status else set()
            for key, k, v, nm in entries:
                if key not in status and ({key} | ({nm} if nm else set())) & refs \
                        and not any(r.search(key) for r in drop_res):
                    status[key] = "library"
                    changed = True
        for key, k, v, nm in entries:
            if key in status:
                kept.append(key)
                if status[key] in ("code", "keep", "library", "support-data"):
                    by_code_only.append(f"{key} ({status[key]})")
            else:
                a, b = _remove_dict_entry(T, k, v)
                edits.append((a, b, "", f"G2 {name}[{key!r}]"))
                dropped.append(key)
        if dropped:
            a = T.line_start(node.lineno)
            edits.append((a, a, SPRITES_OMITTED, f"G2 note above {name}"))
        g2[name] = {"n": len(entries), "kept": len(kept), "omitted": dropped,
                    "kept_without_support_level_use": by_code_only}
    rep["generic"]["libraries"] = g2

    # ---- G3: comments naming a held level
    held_re = _held_comment_re(held1)
    n_com = 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT or not held_re or not held_re.search(tok.string):
            continue
        i, c = tok.start
        line = T.lines[i - 1]
        if line[:c].strip() == "":
            edits.append((T.line_start(i), T.line_end(i), "", f"G3 comment line {i}"))
        else:
            a = T.line_start(i) + c
            edits.append((a, a + len(tok.string), "", f"G3 comment line {i}"))
        n_com += 1
    rep["generic"]["held_comments_removed"] = n_com

    text, problems = _apply(src, edits)
    rep["problems"] += problems

    # ---- C: Sub edits
    for e in cfg:
        if isinstance(e, Sub):
            cnt = text.count(e.old)
            if cnt != e.n:
                rep["problems"].append(f"Sub matched {cnt}x (expected {e.n}): {e.old[:80]!r}")
            text = text.replace(e.old, e.new)
            rep["config"].append({"kind": "Sub", "old": e.old[:200], "new": e.new[:200], "matches": cnt, "why": e.why})
        elif isinstance(e, (Keep, Drop, Allow, Note, LevelExpr, Library)):
            rep["notes" if isinstance(e, Note) else "config"].append(
                e.why if isinstance(e, Note) else {"kind": type(e).__name__, **e._asdict()})

    text = _insert_withheld(text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    compile(text, f"{game_id}.py", "exec")
    rep["level_index_uses"] = level_index_uses(text, cfg, held)
    rep["chars_source"] = len(src)
    rep["chars_redacted"] = len(text)
    return text, rep


def _insert_withheld(text: str) -> str:
    """Define WITHHELD right after the leading import block."""
    tree = ast.parse(text)
    end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)):
            end = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    return "".join(lines[:end]) + WITHHELD_DEF + "".join(lines[end:])


LEVEL_ATTRS0 = {"level_index", "_current_level_index"}


def level_index_uses(text: str, cfg: list, held: list[int]) -> dict:
    """Comparisons and subscripts on a level-number expression, for review. `held_refs` lists
    comparisons against a held level that are not inside a raise-only branch (must be empty unless
    allowed by the config)."""
    extra = {e.src: e.base for e in cfg if isinstance(e, LevelExpr)}
    tree = ast.parse(text)

    def base_of(node):
        s = ast.unparse(node)
        if s in extra:
            return extra[s]
        if isinstance(node, ast.Attribute) and node.attr in LEVEL_ATTRS0:
            return 0
        return None

    raise_ifs = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Raise) and \
                "level withheld" in ast.unparse(n.body[0]):
            raise_ifs.add(id(n.test))
            for c in ast.walk(n.test):
                raise_ifs.add(id(c))
    compares, subscripts, held_refs = [], [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            ops = [n.left, *n.comparators]
            for i, o in enumerate(ops):
                b = base_of(o)
                if b is None:
                    continue
                consts = []
                for other in ops[:i] + ops[i + 1:]:
                    if isinstance(other, ast.Constant) and isinstance(other.value, int):
                        consts.append(other.value)
                    elif isinstance(other, (ast.List, ast.Tuple, ast.Set)):
                        consts += [x.value for x in other.elts if isinstance(x, ast.Constant)]
                levels0 = [c - b for c in consts]
                line = f"line {n.lineno}: {ast.unparse(n)[:120]}"
                compares.append(line)
                eq_like = any(isinstance(op, (ast.Eq, ast.In, ast.NotEq, ast.NotIn)) for op in n.ops)
                if eq_like and set(levels0) & set(held) and id(n) not in raise_ifs:
                    held_refs.append(line)
        elif isinstance(n, ast.Subscript) and base_of(n.slice) is not None:
            subscripts.append(f"line {n.lineno}: {ast.unparse(n)[:120]}")
    return {"compares": compares, "subscripts": subscripts, "held_refs": held_refs}


# ---------------------------------------------------------------------------------------------
# engine excerpt
# ---------------------------------------------------------------------------------------------
ENGINE_DIR = Path(arcengine.__file__).parent
ENGINE_FILES = ["enums.py", "sprites.py", "level.py", "interfaces.py", "camera.py", "base_game.py"]
ALWAYS = ["ARCBaseGame", "Camera", "Level", "Sprite", "GameAction", "GameState", "ActionInput"]
CORE = {
    "ARCBaseGame": {"__init__", "perform_action", "_set_action", "complete_action", "is_action_complete",
                    "step", "handle_reset", "full_reset", "level_reset", "set_level", "on_set_level",
                    "next_level", "_really_set_next_level", "is_last_level", "win", "lose", "current_level",
                    "level_index", "camera"},
    "Camera": {"__init__", "render", "_raw_render", "_calculate_scale_and_offset", "resize"},
    "Level": {"__init__", "get_sprites", "clone", "grid_size"},
    "Sprite": {"__init__", "clone", "render"},
    "RenderableUserDisplay": {"render_interface"},
}
WHOLE = {"GameState", "GameAction", "ActionInput", "BlockingMode", "InteractionMode", "PlaceableArea"}
NO_CLOSURE = {"FrameData", "FrameDataRaw", "SimpleAction", "ComplexAction"}
EXCERPT_RULE = ("unmodified `arcengine` source, restricted to the engine classes the game imports or "
                "that the action/render loop needs; within a class, to the members the game executes, "
                "overrides or unambiguously names, the action/render loop, and the engine members those "
                "call; omitted members are listed by name")


def _engine_index():
    classes, funcs, consts = {}, {}, {}
    for f in ENGINE_FILES:
        src = (ENGINE_DIR / f).read_text()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (f, node, src)
            elif isinstance(node, ast.FunctionDef):
                funcs[node.name] = (f, node, src)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                tg = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(tg, ast.Name):
                    consts[tg.id] = (f, node, src)
    return classes, funcs, consts


def _names(node) -> tuple[set[str], set[str]]:
    attrs, names = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            attrs.add(n.attr)
        elif isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.keyword) and n.arg:
            attrs.add(n.arg)
    return attrs, names


def _seg_lines(src: str, node) -> str:
    lines = src.splitlines(keepends=True)
    start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
    return "".join(lines[start:node.end_lineno])


ENGINE_USAGE = RUNS / "private" / "game_sections_public" / "engine_usage.json"


def _game_defined_names(gtree: ast.Module) -> set[str]:
    """Names the game itself defines as attributes/methods/parameters (so a use of `.name` in the
    game may refer to the game's own object rather than an engine member)."""
    out = set()
    for n in ast.walk(gtree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if not isinstance(n, ast.ClassDef):
                out |= {a.arg for a in n.args.args + n.args.kwonlyargs}
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.attr)
        elif isinstance(n, ast.ClassDef):
            pass
    for n in ast.walk(gtree):
        if isinstance(n, ast.ClassDef):
            for st in n.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    out.add(st.target.id)
                elif isinstance(st, ast.Assign):
                    out |= {t.id for t in st.targets if isinstance(t, ast.Name)}
    return out


def _overrides(gtree: ast.Module, engine_classes: set[str]) -> dict[str, set[str]]:
    """Engine class -> method names that game subclasses (transitively) define."""
    bases = {n.name: [ast.unparse(b).split(".")[-1] for b in n.bases] for n in ast.walk(gtree)
             if isinstance(n, ast.ClassDef)}
    meths = {n.name: {st.name for st in n.body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}
             for n in ast.walk(gtree) if isinstance(n, ast.ClassDef)}

    def engine_ancestors(c, seen=()):
        out = set()
        for b in bases.get(c, []):
            if b in engine_classes:
                out.add(b)
            elif b in bases and b not in seen:
                out |= engine_ancestors(b, seen + (c,))
        return out
    res: dict[str, set[str]] = {}
    for c in bases:
        for e in engine_ancestors(c):
            res.setdefault(e, set()).update(meths[c])
    return res


def engine_excerpt(game_src: str, game_id: str | None = None) -> tuple[list, dict]:
    classes, funcs, consts = _engine_index()
    gtree = ast.parse(game_src)
    imported = set()
    for node in gtree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("arcengine"):
            imported |= {a.name for a in node.names}
    g_attrs, _ = _names(gtree)
    traced = None
    if game_id and ENGINE_USAGE.exists():
        traced = json.loads(ENGINE_USAGE.read_text()).get(game_id)
    overrides = _overrides(gtree, set(classes))
    if traced is not None:
        # members executed on support levels, overrides, and names the game uses that it does not
        # define itself (unambiguous engine references)
        static_names = g_attrs - _game_defined_names(gtree)
    else:
        static_names = g_attrs | {m for ms in overrides.values() for m in ms}
    traced_set = set(traced or ())
    want_cls = [c for c in ALWAYS] + sorted(c for c in imported if c in classes and c not in ALWAYS)
    members: dict[str, set[str]] = {}
    inc_funcs, inc_consts = set(), set()
    inc_funcs |= {t for t in traced_set if t in funcs}

    def add_class(c):
        if c in members or c not in classes:
            return False
        _, node, _ = classes[c]
        names = {st.name for st in node.body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if c in WHOLE:
            members[c] = set(names)
        else:
            members[c] = (names & (static_names | CORE.get(c, set()) | overrides.get(c, set())
                                   | {m for m in names if f"{c}.{m}" in traced_set})) | ({"__init__"} & names)
        return True

    for c in want_cls:
        add_class(c)
    changed = True
    while changed:
        changed = False
        for c in list(members):
            _, node, _ = classes[c]
            for st in node.body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name in members[c]:
                    attrs, names = _names(st)
                    if traced is not None:
                        # with a trace, callees that run are already included; the closure only adds
                        # the engine classes, module functions and constants that included code names
                        attrs = set()
                    for c2 in list(members):
                        _, n2, _ = classes[c2]
                        for st2 in n2.body:
                            if isinstance(st2, (ast.FunctionDef, ast.AsyncFunctionDef)) and st2.name in attrs \
                                    and st2.name not in members[c2]:
                                members[c2].add(st2.name)
                                changed = True
                    for nm in names:
                        if nm in classes and nm not in NO_CLOSURE and add_class(nm):
                            changed = True
                        if nm in funcs and nm not in inc_funcs:
                            inc_funcs.add(nm)
                            changed = True
                        if nm in consts and nm not in inc_consts:
                            inc_consts.add(nm)
                            changed = True
        for fn in list(inc_funcs):
            _, node, _ = funcs[fn]
            attrs, names = _names(node)
            for nm in names:
                if nm in classes and nm not in NO_CLOSURE and add_class(nm):
                    changed = True
                if nm in funcs and nm not in inc_funcs:
                    inc_funcs.add(nm)
                    changed = True
    parts, rep = [], {"imported": sorted(imported), "classes": {}, "functions": sorted(inc_funcs),
                      "trace_used": traced is not None}
    for f in ENGINE_FILES:
        chunk = []
        src = (ENGINE_DIR / f).read_text()
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                tg = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(tg, ast.Name) and tg.id in inc_consts:
                    chunk.append(_seg_lines(src, node))
            elif isinstance(node, ast.FunctionDef) and node.name in inc_funcs:
                chunk.append(_seg_lines(src, node))
            elif isinstance(node, ast.ClassDef) and node.name in members:
                c = node.name
                if c in WHOLE or members[c] == {st.name for st in node.body
                                                 if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}:
                    chunk.append(_seg_lines(src, node))
                    rep["classes"][c] = "whole"
                    continue
                first = node.body[0]
                head_end = (first.decorator_list[0].lineno if getattr(first, "decorator_list", None)
                            else first.lineno) - 1
                out = ["".join(lines[(node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1:
                                     head_end])]
                shown, omitted = [], []
                for st in node.body:
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if st.name in members[c]:
                            out.append(_seg_lines(src, st))
                            if st.name not in shown:
                                shown.append(st.name)
                        elif st.name not in omitted:
                            omitted.append(st.name)
                    elif st is not first or not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
                        if st.lineno - 1 >= head_end:
                            out.append(_seg_lines(src, st))
                body = []
                for piece in out:
                    body.append(piece if piece.endswith("\n") else piece + "\n")
                text = "".join(body).rstrip("\n") + "\n"
                text = re.sub(r"\n(\s*\n)+(?=    (?:def|@))", "\n\n", text)
                if omitted:
                    text += f"\n    # (members not shown: {', '.join(omitted)})\n"
                chunk.append(text)
                rep["classes"][c] = {"shown": shown, "omitted": omitted}
        if chunk:
            parts.append((f"arcengine/{f}", "\n\n".join(x.rstrip("\n") + "\n" for x in chunk)))
    return parts, rep


# ---------------------------------------------------------------------------------------------
# section
# ---------------------------------------------------------------------------------------------
def _fmt_levels(xs: list[int]) -> str:
    return ", ".join(str(x) for x in xs)


def public_section(game_id: str, condition: str = "M") -> tuple[str, dict]:
    """Markdown game section for the writer prompt (condition M only) and its audit report."""
    if condition != "M":
        raise ValueError("the public tier has only condition M")
    if game_id not in GAMES:
        raise KeyError(game_id)
    red, rep = redact_source(game_id)
    src = source_path(game_id).read_text()
    parts, erep = engine_excerpt(src, game_id)
    rep["engine"] = erep
    n = rep["n_levels"]
    held1, sup1 = rep["held_levels"], rep["support_levels"]
    body = [
        f"## The game: official ARC-AGI-3 game `{game_id}` (simulator source, redacted)",
        f"You get the game's simulator source: state, transitions, goal and loss rules, and rendering, "
        f"written against the `arcengine` package. The game has {n} levels. Levels {_fmt_levels(held1)} "
        f"(1-based) are withheld for evaluation and appear as `WITHHELD`: their entries are replaced, code "
        f"that only they use raises `LookupError('level withheld for evaluation')`, and sprite templates "
        f"that only they use are omitted. Levels {_fmt_levels(sup1)} are shown in full. The actor will "
        f"also be tested on states a few actions into a level. There is one canonical rendering (no "
        f"renderer templates or palette variants). Identifiers were obfuscated by the game's authors. "
        f"The engine excerpt is {EXCERPT_RULE}.",
    ]
    for rel, text in parts:
        body.append(f"`{rel}` (excerpt):\n```python\n{text}```")
    body.append(f"`{game_id}.py`:\n```python\n{red}```")
    md = "\n\n".join(body) + "\n"
    rep["chars_engine"] = sum(len(t) for _, t in parts)
    rep["chars_section"] = len(md)
    rep["tokens_section"] = round(len(md) / 4)
    return md, rep


def parse_game_block(md: str, game_id: str) -> str:
    m = re.search(r"^`" + re.escape(game_id) + r"\.py`:\n```python\n(.*?)^```$", md, re.S | re.M)
    if not m:
        raise ValueError(f"{game_id}: game code block not found")
    return m.group(1)
