"""Revision-2 redaction: mechanics-only source for the writer prompt.

Removed from the prompt text (the harness keeps using the full source internally):
  * held-out level data (literal entries -> WITHHELD; `if level == k` branches -> raise),
  * query renderer templates (literal entries -> WITHHELD),
  * solver / planning / oracle routines and everything only they use (call-graph closure),
  * statements that store solver output (e.g. `self.<attr> = self._<solver>(...)`),
  * comments naming solvers/oracles/optimal solutions, and comments naming held-out levels.
Kept (rules): state, transitions, goal test, rendering, level schema, support-level data, and any
level-gated mechanic (reported in the audit).

Pipeline of `game_section_v2` per family file (conditions M and MS):
  1. `redact_mechanics`: the steps above (solver removal only in M); `WITHHELD = None` is defined
     after the module docstring and `from __future__` imports and the result must compile.
     Template tuples are recognised by their real length (`template_split`: k templates, seed % k).
  2. per-family overrides from `wsp.redact2_overrides` (audit corrections, explicit move-budget
     tables, `_engine_*` hooks, MS-specific fixes), each tagged with the conditions it runs in.
  3. `strip_level_prose`: comments and prose strings inside per-level data literals are dropped.
  4. `tree_shake` over all files of the family: every top-level definition the simulator class cannot
     reach is dropped (baselines, diagnostics, probes, design metrics, held-only constants, unused
     imports). In MS the class keeps `solution_actions`, so the universal solver it drives stays.

`audit` lists everything removed and every remaining suspicious line for manual review.
"""
from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from pathlib import Path

import arc3_games
from arc3_games.games import base
from arc3_games.games.families import load

from .queries import QUERY_TEMPLATES, SUPPORT_TEMPLATES, held_out_levels
from .redact2_overrides import DOC_ROOTS, MS_SOLVER_ROOTS, overrides_for

GAMES = Path(arc3_games.__file__).parent / "games"
SOLVER_RE = re.compile(r"(solution|solve|solver|shortest|oracle|planner|plan_|_plan\b|_plans?\b|"
                       r"feasib|witness|\bbfs\b|dijkstra|astar|a_star|_search|search_|path_to|"
                       r"optimal|best_route|route_to)", re.I)
COMMENT_RE = re.compile(r"(solution|solve|solver|shortest|oracle|optimal|witness|planner|"
                        r"\bplan\b|minimum (moves|clicks|actions)|best route|depth \d)", re.I)
LEVEL_LITERAL_RE = re.compile(r"(level|layout|board|blueprint|puzzle|ladder|stage|spec|maze|map|circuit|scenario)", re.I)
TEMPLATE_RE = re.compile(r"TEMPLATES")


def family_files(family: str) -> list[Path]:
    cls = load(family)
    main = Path(inspect.getsourcefile(cls))
    pkg = main.parent.name
    files = [main]
    tree = ast.parse(main.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(f"arc3_games.games.{pkg}."):
            f = GAMES / pkg / (node.module.split(".")[-1] + ".py")
            if f.exists() and f not in files and "planner" not in f.name:
                files.insert(0, f)
    return files


def _comment_block_start(lines, idx):
    k = idx
    while k - 1 >= 0 and lines[k - 1].strip().startswith("#"):
        k -= 1
    return k


def _names_in(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _drop_other_families(src: str, family: str) -> tuple[str, list[str]]:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drops = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fam = None
            for st in node.body:
                if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "family" for t in st.targets) \
                        and isinstance(st.value, ast.Constant):
                    fam = st.value.value
            if fam and fam != family:
                drops.append((node, fam))
    for node, fam in sorted(drops, key=lambda x: x[0].lineno, reverse=True):
        l0 = _comment_block_start(lines, (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1)
        lines = lines[:l0] + [f"# class {node.name} (family {fam}, a different game) omitted\n"] + lines[node.end_lineno:]
    return "".join(lines), [f for _, f in drops]


WITHHELD_LINE = "WITHHELD = None  # placeholder for an evaluation-only entry you cannot see\n"


def _insert_withheld(text: str) -> str:
    """Define WITHHELD after the module docstring and any `from __future__` import (a
    statement before them demotes the docstring and makes `from __future__` a SyntaxError)."""
    body = ast.parse(text).body
    end = 0
    for i, node in enumerate(body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            end = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            end = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    return "".join(lines[:end]) + WITHHELD_LINE + "".join(lines[end:])


def redact_mechanics(src: str, held: set[int], held_tpl: set[int], n_levels: int,
                     remove_solvers: bool = True, n_templates: int = 10) -> tuple[str, dict]:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits: list[tuple[int, int, int, int, str]] = []
    audit = {"withheld_levels": [], "withheld_templates": [], "if_level_branches": [], "solvers_removed": [],
             "oracle_statements_removed": [], "solver_refs_kept": [], "level_gated": [], "comments_removed": 0}

    def seg(n):
        return ast.get_source_segment(src, n) or ""

    # ---- functions and their call graph
    funcs: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append(node)
    solver = {n for n in funcs if SOLVER_RE.search(n)} if remove_solvers else set()
    # closure: functions referenced only from solver functions
    changed = remove_solvers
    while changed:
        changed = False
        for name, defs in funcs.items():
            if name in solver or name.startswith("__"):
                continue
            users = set()
            for other, odefs in funcs.items():
                if other == name:
                    continue
                for d in odefs:
                    if name in _names_in(d):
                        users.add(other)
            module_level_use = any(name in _names_in(st) for st in tree.body
                                   if not isinstance(st, (ast.FunctionDef, ast.ClassDef)))
            class_level_use = any(name in _names_in(st) for c in tree.body if isinstance(c, ast.ClassDef)
                                  for st in c.body if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)))
            if users and users <= solver and not module_level_use and not class_level_use:
                solver.add(name)
                changed = True

    # ---- never remove engine API methods; never remove what mechanics still references
    protected = ({n for n in dir(base.GameSimulator) if not n.startswith("__")} | {
        "goal_reached", "is_dead", "_apply", "_init_level", "_render", "observe", "step", "frames"}) \
        - {"solution_actions", "self_test"}
    solver -= protected
    oracle_store = lambda st: isinstance(st, ast.Assign) and any(
        SOLVER_RE.search(t) for t in _names_in(st.targets[0]))
    while True:
        refs = set()
        for name, defs in funcs.items():
            if name in solver:
                continue
            for d in defs:
                for st in ast.walk(d):
                    if isinstance(st, ast.stmt) and not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and not oracle_store(st):
                        refs |= _names_in(st) if not isinstance(st, (ast.If, ast.For, ast.While, ast.With, ast.Try)) \
                            else _names_in(getattr(st, "test", getattr(st, "iter", st)))
        module_refs = set()
        for st in tree.body:
            if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_refs |= _names_in(st)
        keep = solver & (refs | module_refs)
        if not keep:
            break
        for k in keep:
            audit["solver_refs_kept"].append(f"kept (used by mechanics): {k}")
        solver -= keep

    # ---- statements outside solvers that reference solvers
    for node in ast.walk(tree):
        if remove_solvers and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in solver:
            for st in ast.walk(node):
                if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr)) and \
                        (_names_in(st) & solver or any(SOLVER_RE.search(t) for t in
                                                       _names_in(st.targets[0] if isinstance(st, ast.Assign) else st))):
                    tgt_names = _names_in(st.targets[0]) if isinstance(st, ast.Assign) else set()
                    uses_solver = bool(_names_in(st) & solver)
                    stores_oracle = any(SOLVER_RE.search(t) for t in tgt_names)
                    if stores_oracle or (uses_solver and isinstance(st, ast.Expr)):
                        audit["oracle_statements_removed"].append(seg(st)[:120])
                        edits.append((st.lineno - 1, 0, st.end_lineno, 0, ""))
                    elif uses_solver:
                        audit["solver_refs_kept"].append(f"line {st.lineno}: {seg(st)[:120]}")
    for name in sorted(solver):
        for d in funcs[name]:
            start = (d.decorator_list[0].lineno if d.decorator_list else d.lineno) - 1
            start = _comment_block_start(lines, start)
            edits.append((start, 0, d.end_lineno, 0, ""))
            audit["solvers_removed"].append(name)

    # ---- names subscripted by the level index (X[level], X[self.level], X[level_index], ...)
    level_indexed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            key = getattr(sl, "id", getattr(sl, "attr", None))
            if key in ("level", "level_index", "lvl", "idx_level") and isinstance(node.value, (ast.Name, ast.Attribute)):
                level_indexed.add(getattr(node.value, "id", getattr(node.value, "attr", None)))

    # ---- level data literals and template literals
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            target = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else None)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if not target or not isinstance(getattr(node, "value", None), (ast.Tuple, ast.List)):
            continue
        elts = node.value.elts
        is_level = len(elts) == n_levels and (LEVEL_LITERAL_RE.search(target) or target in level_indexed
                                              or target.lower().endswith(("budgets", "limits", "slack", "levels")))
        is_tpl = TEMPLATE_RE.search(target) and len(elts) == n_templates
        held_idx = held if is_level else (held_tpl if is_tpl else None)
        if held_idx is None:
            continue
        for i, el in enumerate(elts):
            if i in held_idx:
                l0 = el.lineno - 1
                if el.col_offset == len(lines[l0]) - len(lines[l0].lstrip()):
                    c0 = _comment_block_start(lines, l0)
                    if c0 < l0:
                        edits.append((c0, 0, l0, 0, ""))
                edits.append((el.lineno - 1, el.col_offset, el.end_lineno - 1, el.end_col_offset, "WITHHELD"))
                (audit["withheld_levels"] if is_level else audit["withheld_templates"]).append(f"{target}[{i}]")

    # ---- if level == k branches (any function), level-gated mechanics
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in solver:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare) and any(isinstance(x, ast.Name) and x.id == "level" or
                                                    isinstance(x, ast.Attribute) and x.attr == "level"
                                                    for x in [sub.left, *sub.comparators]):
                is_eq = isinstance(sub.ops[0], ast.Eq) and isinstance(sub.comparators[0], ast.Constant)
                if not is_eq:
                    audit["level_gated"].append(f"{node.name} line {sub.lineno}: {seg(sub)[:80]}")
            if isinstance(sub, ast.If) and isinstance(sub.test, ast.Compare) \
                    and isinstance(sub.test.left, (ast.Name, ast.Attribute)) \
                    and getattr(sub.test.left, "id", getattr(sub.test.left, "attr", None)) == "level" \
                    and isinstance(sub.test.ops[0], ast.Eq) and isinstance(sub.test.comparators[0], ast.Constant) \
                    and sub.test.comparators[0].value in held:
                body0, body1 = sub.body[0], sub.body[-1]
                indent = " " * body0.col_offset
                edits.append((sub.lineno, 0, body1.end_lineno, 0, f"{indent}raise LookupError('level withheld for evaluation')\n"))
                head = lines[sub.lineno - 1]
                if "#" in head:
                    edits.append((sub.lineno - 1, 0, sub.lineno - 1, len(head.rstrip("\n")), head.split("#", 1)[0].rstrip()))
                c0 = _comment_block_start(lines, sub.lineno - 1)
                if c0 < sub.lineno - 1:
                    edits.append((c0, 0, sub.lineno - 1, 0, ""))
                audit["if_level_branches"].append(f"{node.name}: level=={sub.test.comparators[0].value}")

    # ---- fall-through default of `if level == k` chains (covers the levels no branch matches)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in solver:
            continue
        body = node.body
        idx = [i for i, st in enumerate(body) if isinstance(st, ast.If) and isinstance(st.test, ast.Compare)
               and getattr(st.test.left, "id", getattr(st.test.left, "attr", None)) == "level"
               and isinstance(st.test.ops[0], ast.Eq) and isinstance(st.test.comparators[0], ast.Constant)
               and not st.orelse]
        if len(idx) < 2:
            continue
        matched = {body[i].test.comparators[0].value for i in idx}
        default_levels = set(range(n_levels)) - matched
        tail = body[idx[-1] + 1:]
        if not tail or not default_levels:
            continue
        if default_levels <= held:
            t0, t1 = tail[0], tail[-1]
            start = _comment_block_start(lines, t0.lineno - 1)
            indent = " " * t0.col_offset
            edits.append((start, 0, t1.end_lineno, 0, f"{indent}raise LookupError('level withheld for evaluation')\n"))
            audit["if_level_branches"].append(f"{node.name}: default branch (levels {sorted(l + 1 for l in default_levels)})")
        elif default_levels & held:
            audit["solver_refs_kept"].append(f"MANUAL: {node.name} default branch mixes seen and held levels "
                                             f"{sorted(l + 1 for l in default_levels)}")

    # ---- comments: solver vocabulary, or naming a held-out level
    held_names = {h + 1 for h in held}
    held_re = re.compile(r"\b(L|level|Level|LEVEL)\s?(" + "|".join(str(h) for h in sorted(held_names)) + r")\b") if held_names else None
    import io
    import tokenize
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT:
            continue
        comment = tok.string
        if (remove_solvers and COMMENT_RE.search(comment)) or (held_re and held_re.search(comment)):
            i, c = tok.start[0] - 1, tok.start[1]
            line = lines[i]
            if line[:c].strip() == "":
                edits.append((i, 0, i + 1, 0, ""))
            else:
                edits.append((i, c, i, c + len(comment), ""))
            audit["comments_removed"] += 1

    # convert to absolute character offsets, drop edits nested inside others, apply right-to-left
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    def off(l, c):
        return starts[min(l, len(lines))] + c
    spans = sorted({(off(l0, c0), off(l1, c1), rep) for l0, c0, l1, c1, rep in edits},
                   key=lambda e: (e[0], -e[1]))
    kept = []
    for a, b, rep in spans:
        if kept and a >= kept[-1][0] and b <= kept[-1][1]:
            continue                      # nested inside an outer edit
        if kept and a < kept[-1][1]:
            audit["solver_refs_kept"].append(f"MANUAL: overlapping edits at offset {a}")
            continue
        kept.append((a, b, rep))
    text_s = src
    for a, b, rep in reversed(kept):
        text_s = text_s[:a] + rep + text_s[b:]
    out = _insert_withheld(text_s)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    compile(out, "<redacted>", "exec")
    # remaining suspicious lines for manual review
    rem = []
    for i, line in enumerate(out.splitlines(), 1):
        if SOLVER_RE.search(line) or (held_re and held_re.search(line)) or re.search(r"\b(deque|heapq|heappush)\b", line):
            rem.append(f"{i}: {line.strip()[:120]}")
    audit["remaining_suspicious"] = rem
    audit["solvers_removed"] = sorted(set(audit["solvers_removed"]))
    return out, audit


def base_excerpt() -> str:
    src = inspect.getsource(base.GameSimulator)
    return src[:src.index("    # ---- self-test contract")]


def _top_defs(tree: ast.Module) -> list[tuple[ast.stmt, list[str]]]:
    """Top-level statements that only bind names (def/class/assign/import) and the names they bind."""
    out = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
                    elif not isinstance(n, (ast.Tuple, ast.List, ast.Starred, ast.Load, ast.Store)):
                        names = []
                        break
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if any(a.name == "*" for a in node.names):
                continue
            names = [(a.asname or a.name).split(".")[0] for a in node.names]
        if names:
            out.append((node, names))
    return out


def _loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Store)}


def tree_shake(files: dict[str, str], family: str, pkg: str, extra_root=None) -> tuple[dict[str, str], dict]:
    """Drop every top-level definition (function, class, constant, import alias) that is not reachable
    from the family's simulator class or from a top-level statement that is not a pure definition.
    Reachability follows Name loads and `from arc3_games.games.<pkg>.<mod> import ...` across the
    family's files. `extra_root(name)` marks additional roots (condition MS: solver entry points).
    The contiguous comment block directly above a dropped definition goes with it."""
    removed: dict[str, list[str]] = {rel: [] for rel in files}
    mod_of = {f"arc3_games.games.{pkg}.{Path(rel).stem}": rel for rel in files}
    trees = {rel: ast.parse(src) for rel, src in files.items()}
    defs: dict[tuple[str, str], ast.stmt] = {}
    roots: set[tuple[str, str]] = set()
    edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
    pending_roots: list[tuple[str, set[str]]] = []
    for rel, tree in trees.items():
        bound = _top_defs(tree)
        bound_ids = {id(n) for n, _ in bound}
        for node, names in bound:
            for nm in names:
                defs[(rel, nm)] = node
        for node in tree.body:
            if id(node) not in bound_ids and not (isinstance(node, ast.Expr) and isinstance(
                    getattr(node, "value", None), ast.Constant)) and not (
                    isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                pending_roots.append((rel, _loads(node)))
        for node, names in bound:
            if isinstance(node, ast.ClassDef) and any(
                    isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "family" for t in st.targets)
                    and isinstance(st.value, ast.Constant) and st.value.value == family for st in node.body):
                roots.add((rel, node.name))
            if extra_root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and extra_root(node.name):
                roots.add((rel, node.name))
    for (rel, nm), node in defs.items():
        if isinstance(node, ast.ImportFrom) and node.module in mod_of:
            tgt = mod_of[node.module]
            edges[(rel, nm)] = {(tgt, a.name) for a in node.names if (a.asname or a.name) == nm}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            edges[(rel, nm)] = set()
        else:
            edges[(rel, nm)] = {(rel, x) for x in _loads(node) if (rel, x) in defs}
    for rel, names in pending_roots:
        roots |= {(rel, x) for x in names if (rel, x) in defs}
    seen, stack = set(), list(roots)
    while stack:
        k = stack.pop()
        if k in seen or k not in defs:
            continue
        seen.add(k)
        stack.extend(edges.get(k, ()))
    out = {}
    for rel, src in files.items():
        lines = src.splitlines(keepends=True)
        drop_lines: set[int] = set()
        rewrites: dict[int, tuple[int, str]] = {}
        for node, names in _top_defs(trees[rel]):
            live = [nm for nm in names if (rel, nm) in seen]
            if len(live) == len(names):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)) and live:
                keep = [a for a in node.names if (a.asname or a.name).split(".")[0] in live]
                txt = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
                head = f"from {node.module} import " if isinstance(node, ast.ImportFrom) else "import "
                ind = " " * node.col_offset
                rewrites[node.lineno - 1] = (node.end_lineno, f"{ind}{head}{txt}\n")
                removed[rel] += [nm for nm in names if nm not in live]
                continue
            if live:
                continue                          # tuple unpack with a live name: keep whole statement
            start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
            start = _comment_block_start(lines, start)
            drop_lines |= set(range(start, node.end_lineno))
            removed[rel] += names
        new, i = [], 0
        while i < len(lines):
            if i in rewrites:
                end, txt = rewrites[i]
                new.append(txt)
                i = end
                continue
            if i not in drop_lines:
                new.append(lines[i])
            i += 1
        out[rel] = re.sub(r"\n{4,}", "\n\n\n", "".join(new))
    return out, removed


PROSE_RE = re.compile(r"[A-Za-z]{2,}\W+[A-Za-z]{2,}\W+[A-Za-z]{2,}")


def strip_level_prose(src: str, n_levels: int) -> tuple[str, dict]:
    """Level prose: inside every per-level data literal (a list/tuple with one entry per level, named
    like level data or indexed by the level), drop all comments and blank prose strings (blurbs,
    titles, design notes). The level data itself is untouched."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    level_indexed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = getattr(node.slice, "id", getattr(node.slice, "attr", None))
            if key in ("level", "level_index", "lvl", "idx_level") and isinstance(node.value, (ast.Name, ast.Attribute)):
                level_indexed.add(getattr(node.value, "id", getattr(node.value, "attr", None)))
    spans, info = [], {"literals": [], "comments": 0, "strings": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            target = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else None)
        elif isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        else:
            continue
        if not target or not isinstance(getattr(node, "value", None), (ast.Tuple, ast.List)):
            continue
        if len(node.value.elts) != n_levels or not (LEVEL_LITERAL_RE.search(target) or target in level_indexed):
            continue
        spans.append((node.value.lineno, node.value.end_lineno, node.value))
        info["literals"].append(target)
    if not spans:
        return src, info
    edits = []                                              # (line0, col0, line1, col1, text)
    inside = lambda ln: any(a <= ln <= b for a, b, _ in spans)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and inside(tok.start[0]):
            i, c = tok.start[0] - 1, tok.start[1]
            if lines[i][:c].strip() == "":
                edits.append((i, 0, i + 1, 0, ""))
            else:
                c0 = len(lines[i][:c].rstrip())
                edits.append((i, c0, i, len(lines[i].rstrip("\n")), ""))
            info["comments"] += 1
    for _a, _b, val in spans:
        for n in ast.walk(val):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and PROSE_RE.search(n.value):
                edits.append((n.lineno - 1, n.col_offset, n.end_lineno - 1, n.end_col_offset, '""'))
                info["strings"] += 1
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    spans_c = sorted({(starts[l0] + c0, starts[l1] + c1, rep) for l0, c0, l1, c1, rep in edits}, reverse=True)
    out = src
    for a, b, rep in spans_c:
        out = out[:a] + rep + out[b:]
    return out, info


def template_split(family: str) -> tuple[int, list[int]]:
    """(number of renderer templates k, withheld template indices). Seed s uses template s % k, so a
    template is withheld iff some query seed maps to it and no support seed does."""
    k = len(getattr(load(family), "family_variants", ()) or ()) or 10
    support = {t % k for t in SUPPORT_TEMPLATES}
    return k, sorted({t % k for t in QUERY_TEMPLATES} - support)


HOOK_RE = re.compile(r"^\s*def (_engine_\w+)\(", re.M)
HOOK_SENTENCE = ("Functions named `_engine_*` are provided by the engine at run time; their implementation "
                 "is withheld and their docstring states exactly what they compute.")


def game_section_v2(family: str, n_levels: int, condition: str = "M") -> tuple[str, dict]:
    """condition M = mechanics only (primary); MS = mechanics + solver (ablation).
    Both conditions withhold held-out levels, query templates and held-level descriptions; M also
    removes solver/planner/oracle code (engine-provided rules become `_engine_*` hooks)."""
    if condition not in ("M", "MS"):
        raise ValueError(condition)
    held = set(held_out_levels(n_levels, version=2))
    n_tpl, held_tpl = template_split(family)
    parts, report = [], {"family": family, "condition": condition, "held_levels": sorted(held),
                         "n_templates": n_tpl, "held_templates": held_tpl, "files": [], "hooks": [],
                         "notes": []}
    if n_tpl != 10:
        reused = sorted({t % n_tpl for t in QUERY_TEMPLATES} - set(held_tpl))
        report["notes"].append(f"{family} has {n_tpl} renderer templates (seed % {n_tpl}): query seeds "
                               + ", ".join(f"{t}" for t in QUERY_TEMPLATES if t % n_tpl in reused)
                               + " render support templates " + ", ".join(str(t) for t in reused))
    counts: dict[int, int] = {}
    for f in family_files(family):
        src = f.read_text()
        src, dropped = _drop_other_families(src, family)
        red, au = redact_mechanics(src, held, set(held_tpl), n_levels, remove_solvers=(condition == "M"),
                                   n_templates=n_tpl)
        au["other_families_dropped"] = dropped
        red, au["overrides"] = apply_overrides(family, red, condition, counts)
        red, au["level_prose"] = strip_level_prose(red, n_levels)
        compile(red, str(f.relative_to(GAMES)), "exec")
        report["files"].append({"file": str(f.relative_to(GAMES)), "audit": au})
        parts.append((str(f.relative_to(GAMES)), red))
    # drop everything the simulator cannot reach (baselines, diagnostics, probes, design metrics,
    # held-only constants, unused imports). In MS the simulator still has `solution_actions`, so the
    # universal solver it drives stays; MS_SOLVER_ROOTS can add further solver entry points.
    pkg = family_files(family)[0].parent.name
    keep = set(DOC_ROOTS.get(family, {})) | (MS_SOLVER_ROOTS.get(family, set()) if condition == "MS" else set())
    extra = (lambda name: name in keep) if keep else None
    shaken, removed = tree_shake(dict(parts), family, pkg, extra)
    parts = [(rel, shaken[rel]) for rel, _ in parts]
    for fr, (rel, red) in zip(report["files"], parts):
        fr["audit"]["tree_shaken"] = removed[rel]
        compile(red, rel, "exec")
        fr["chars"] = len(red)
        report["hooks"] += HOOK_RE.findall(red)
    ovs = overrides_for(family)
    report["override_counts"] = [{"i": i, "note": ov.note, "pattern": ov.pattern[:100], "count": counts.get(i, 0),
                                  "optional": ov.optional} for i, ov in enumerate(ovs) if condition in ov.conds]
    report["override_unmatched"] = [c for c in report["override_counts"] if not c["count"] and not c["optional"]]
    if condition == "MS" and report["hooks"]:
        raise RuntimeError(f"{family}: engine hooks in condition MS: {report['hooks']}")
    lv = ", ".join(str(l + 1) for l in report["held_levels"])
    tp = ", ".join(str(t) for t in report["held_templates"])
    body = [f"## The game: family `{family}` (simulator source, redacted)",
            f"You get the game's simulator source: state, transitions, goal test and rendering. Levels {lv} "
            f"(1-based) and renderer templates {tp} (0-based, seed % {n_tpl}) are withheld for evaluation and "
            "appear as `WITHHELD`" + ("; solver/planning code is not included" if condition == "M" else "") + ". "
            + (HOOK_SENTENCE + " " if condition == "M" and report["hooks"] else "")
            + "The actor will also be tested "
            f"on states a few actions into a level. Seed s uses renderer template s % {n_tpl}; seeds >= 10 may "
            "permute palettes. The engine base class:",
            "```python\n" + base_excerpt() + "```"]
    for rel, red in parts:
        body.append(f"`{rel}`:\n```python\n{red}```")
    return "\n\n".join(body), report


def apply_overrides(family: str, text: str, condition: str = "M",
                    counts: dict[int, int] | None = None) -> tuple[str, list[str]]:
    """Apply the family's overrides for `condition` in order (see wsp.redact2_overrides)."""
    applied = []
    for i, ov in enumerate(overrides_for(family)):
        if condition not in ov.conds:
            continue
        new, n = re.subn(ov.pattern, ov.repl, text)
        if counts is not None:
            counts[i] = counts.get(i, 0) + n
        if n:
            applied.append(f"#{i} {ov.note}: {ov.pattern[:60]} x{n}")
            text = new
    return text, applied
