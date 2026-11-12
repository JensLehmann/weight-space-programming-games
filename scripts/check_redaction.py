"""Mechanical check of the revision-2 redacted game sections (conditions M and MS).

For every family and condition:
  1. static checks on every redacted file: compile; no undefined global names (symtable); names
     imported from sibling family files exist; no dangling `self.<attr>` reads that the original
     did not also have; no solver vocabulary outside an explicit allowlist (M only: MS keeps the
     solver by design); no module-level name that is defined but never referenced; no mention of a
     held-out level or a withheld template in comments, docstrings or other prose strings; every
     non-optional override matched.
  2. functional equivalence on support levels: the redacted package is materialised (a copy of the
     installed arc3_games package without the other game packages, the family's files overwritten
     with the code blocks of the section) and driven in a subprocess; the original package is
     driven in another subprocess with the same action sequences, and every step is compared
     (observe(), frames(), state, level, returned grid/state/level/changed/solved_level,
     available_actions). Starts are direct: `level = l; _init_level(l)` on both. A level change
     (a solved support level) is intercepted on both sides before the next level is built, and the
     settled frames, level counter and state at that moment are compared; the sequence then stops.
     Sequences per (support level, template seed 0-4): the original's certified plan (its
     solution_actions() witness segment for that level if it replays from the direct start, else
     the family planner's direct plan), the manifest's S-start / S-dev prefixes, 5 uniform random
     sequences of length 60 (uniform click cells, occasional RESET) and 5 "board-biased" random
     sequences (clicks mostly on non-background cells), recorded on the original.
  3. withheld parts refuse: `_init_level(h)` raises for every held level h, and constructing (or
     observing/stepping) a seed whose template is withheld fails, for seeds 5..19.

Engine hooks (`_engine_*`, condition M only): in the redacted subprocess the original module file
is loaded a second time under an alias name (importlib.util.spec_from_file_location on the
installed file; its `arc3_games.games.base` import resolves to the byte-identical copy) and each
hook is rebound to a small adapter (ADAPTERS below) that calls the original implementation.
The adapters and the family-specific entries of the solver-vocabulary allowlist quote the private
game sources; they are read from the private module (`wsp.private`: ADAPTERS, VOCAB_ALLOW_EXTRA)
and are empty without it.

The engine base class declares `solution_actions` abstract; condition M removes it, and the
writer-visible base excerpt ends before the self-test contract. The redacted subprocess therefore
clears `__abstractmethods__` iff `solution_actions` / `self_test` are the only abstract members
left (anything else is reported as a failure).

Usage: check_redaction.py [--live] [--fams f1 f2 ...] [--conds M MS] [--quick]
  --live    build the sections in memory with game_section_v2 (default: read the regenerated
            runs/private/game_sections_rev2/{fam}_{cond}.md and .audit.json files)
  --quick   fewer random sequences and seeds (not used for the reported check)
Writes runs/private/game_sections_rev2/check.json (only in file mode, or with --out).
"""
from __future__ import annotations

import sys

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    WORKER = True
else:
    WORKER = False


# =============================================================================================
# Worker (runs in a subprocess; must not import wsp before sys.path is set)
# =============================================================================================

class _LevelLeft(Exception):
    pass


def _worker(job_path: str, out_path: str) -> None:
    import json
    job = json.loads(open(job_path).read())
    root = job.get("pkgroot")
    if root:
        sys.path.insert(0, root)
    import hashlib
    import importlib
    import importlib.util
    import random
    import traceback

    import numpy as np
    import arc3_games
    if root:
        assert arc3_games.__file__.startswith(root), arc3_games.__file__
    out: dict = {"errors": []}

    def H(grid) -> str:
        a = np.asarray(grid)
        return hashlib.sha1(str(a.shape).encode() + a.astype(np.int16).tobytes()).hexdigest()[:16]

    try:
        mod = importlib.import_module(job["module"])
    except Exception as exc:
        out["errors"].append("import: " + "".join(traceback.format_exception_only(exc)).strip()[:400])
        open(out_path, "w").write(json.dumps(out))
        return
    cands = [o for o in vars(mod).values() if isinstance(o, type) and getattr(o, "family", None) == job["family"]
             and o.__module__ == mod.__name__]
    if len(cands) != 1:
        out["errors"].append(f"class lookup: {len(cands)} candidates")
        open(out_path, "w").write(json.dumps(out))
        return
    cls = cands[0]
    if job["redacted"]:
        abstract = set(getattr(cls, "__abstractmethods__", ()))
        if abstract - {"solution_actions", "self_test"}:
            out["errors"].append(f"abstract methods left: {sorted(abstract)}")
        cls.__abstractmethods__ = frozenset()
        out["abstract_cleared"] = sorted(abstract)
        # engine hooks -> adapters calling the original implementation
        hooks = job.get("hooks", [])
        if hooks:
            spec = importlib.util.spec_from_file_location(f"_orig_{job['family']}_sim", job["orig_file"])
            orig = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = orig          # dataclasses resolve their module via sys.modules
            spec.loader.exec_module(orig)
            env = {"orig": orig, "red": mod, "cls": cls}
            exec(job["adapter"], env)
            bound = []
            for h in hooks:
                if h not in env:
                    out["errors"].append(f"no adapter for hook {h}")
                    continue
                if hasattr(mod, h):
                    setattr(mod, h, env[h])
                    bound.append(f"module.{h}")
                elif hasattr(cls, h):
                    setattr(cls, h, env[h])
                    bound.append(f"class.{h}")
                else:
                    out["errors"].append(f"hook {h} not found")
            out["hooks_bound"] = bound

    avail0 = list(cls.available_actions)

    def start(level, seed):
        sim = cls(seed)
        sim.level = level
        sim._anim = []
        sim._init_level(level)
        sim.level = level
        real_init = sim._init_level

        def guarded(lv, *args, **kwargs):
            if lv != level:
                raise _LevelLeft(lv)
            return real_init(lv, *args, **kwargs)
        sim._init_level = guarded
        return sim

    def res_attrs(sim):
        # attempt resources (base.ATTEMPT_BUDGET_ATTRS / ATTEMPT_LIVES_ATTRS), plus family-specific
        # budget-size attributes passed in by the parent (private; see RESOURCE_ATTRS_EXTRA)
        names = ("moves_remaining", "move_budget", "budget", "charge", "lives", "lives_remaining") \
            + tuple(job.get("res_attrs_extra", ()))
        return [[a, getattr(sim, a)] for a in names
                if isinstance(getattr(sim, a, None), (int, type(None))) and hasattr(sim, a)]

    def snap(sim):
        return [H(sim.observe()), sim.state, sim.level, list(sim.available_actions), res_attrs(sim)]

    def run_seq(sim, seq):
        recs = []
        for a, r, c in seq:
            try:
                ret = sim.step(a, r, c)
            except _LevelLeft as e:
                recs.append(["LEFT", int(e.args[0]), [H(f) for f in sim._anim], sim.level, sim.state,
                             getattr(sim, "solved", None)])
                break
            except Exception as exc:
                recs.append(["EXC", type(exc).__name__, str(exc)[:200]])
                break
            grid, state, level, changed, solved = ret
            recs.append([H(grid), state, int(level), bool(changed), bool(solved), [H(f) for f in sim.frames()],
                         H(sim.observe()), sim.state, sim.level, list(sim.available_actions), res_attrs(sim)])
            if sim.state in ("WIN", "GAME_OVER") or sim.level != level or solved:
                break
        return recs

    mode = job["mode"]
    if mode == "plan":
        from arc3_games.games.planners import load_planner
        plans = {}
        for level, seed in job["pairs"]:
            key = f"{level}:{seed}"
            entry = {"plan": None, "src": None, "random": []}
            # (a) witness segment for this level from continuous play of solution_actions()
            seg = []
            try:
                w = cls(seed).solution_actions()
                g = cls(seed)
                for a, r, c in w:
                    if g.level > level or g.state != "NOT_FINISHED":
                        break
                    if g.level == level:
                        seg.append([a, r, c])
                    g.step(a, r, c)
            except Exception as exc:
                entry["witness_error"] = f"{type(exc).__name__}: {exc}"[:200]

            def solves(seq):
                sim = start(level, seed)
                recs = run_seq(sim, [tuple(m) for m in seq])
                return bool(recs) and recs[-1][0] == "LEFT"
            if seg and solves(seg):
                entry["plan"], entry["src"] = seg, "witness"
            else:
                try:
                    sim = start(level, seed)
                    res = load_planner(sim).evaluate(sim, time_budget_ms=job.get("plan_ms", 4000))
                    plan = [[p.action, p.row, p.col] for p in (res.direct_plan or ())]
                    if plan and solves(plan):
                        entry["plan"], entry["src"] = plan, "planner"
                    elif seg:
                        entry["plan"], entry["src"] = seg, "witness-unsolved"
                except Exception as exc:
                    entry["planner_error"] = f"{type(exc).__name__}: {exc}"[:200]
                    if seg:
                        entry["plan"], entry["src"] = seg, "witness-unsolved"
            # (c) random sequences: uniform, and board-biased (recorded on the original)
            for k in range(job["n_uniform"] + job["n_biased"]):
                rng = random.Random(f"{job['family']}:{level}:{seed}:{k}")
                biased = k >= job["n_uniform"]
                sim = start(level, seed)
                seq = []
                for _ in range(job["seq_len"]):
                    if rng.random() < 0.04:
                        a = "RESET"
                    else:
                        a = rng.choice([x for x in avail0 if x != "RESET"])
                    r = c = None
                    if a == "ACTION6":
                        if biased and rng.random() < 0.85:
                            grid = np.asarray(sim.observe())
                            vals, cnt = np.unique(grid, return_counts=True)
                            rows, cols = np.nonzero(grid != vals[cnt.argmax()])
                            if len(rows):
                                i = rng.randrange(len(rows))
                                r, c = int(rows[i]), int(cols[i])
                        if r is None:
                            r, c = rng.randrange(64), rng.randrange(64)
                    seq.append([a, r, c])
                    recs = run_seq(sim, [(a, r, c)])
                    if not recs or recs[-1][0] in ("LEFT", "EXC") or sim.state in ("WIN", "GAME_OVER"):
                        break
                entry["random"].append(seq)
            plans[key] = entry
        out["plans"] = plans
    elif mode == "trace":
        traces = {}
        for key, seqs in job["seqs"].items():
            level, seed = map(int, key.split(":"))
            tr = []
            for seq in seqs:
                try:
                    sim = start(level, seed)
                except Exception as exc:
                    tr.append([["START-EXC", type(exc).__name__, str(exc)[:200]]])
                    continue
                tr.append([snap(sim)] + run_seq(sim, [tuple(m) for m in seq]))
            traces[key] = tr
        out["traces"] = traces
    elif mode == "refuse":
        ref = {"levels": {}, "seeds": {}}
        for h in job["held"]:
            try:
                sim = cls(0)
                sim.level = h
                sim._init_level(h)
                ref["levels"][str(h)] = "NO-RAISE"
            except Exception as exc:
                ref["levels"][str(h)] = f"{type(exc).__name__}: {str(exc)[:120]}"
        for s in job["tpl_seeds"]:
            try:
                sim = cls(s)
                sim.observe()
                sim.step("RESET")            # re-renders level 1 under the template; never changes level
                sim.observe()
                ref["seeds"][str(s)] = "NO-RAISE"
            except Exception as exc:
                ref["seeds"][str(s)] = f"{type(exc).__name__}: {str(exc)[:120]}"
        out["refuse"] = ref
    open(out_path, "w").write(json.dumps(out))


if WORKER:
    _worker(sys.argv[2], sys.argv[3])
    sys.exit(0)


# =============================================================================================
# Parent
# =============================================================================================
import argparse
import ast
import builtins
import io
import json
import re
import shutil
import subprocess
import symtable
import tempfile
import time
import tokenize
from pathlib import Path

from wsp.envs import new_game
from wsp.manifest import RUNS
from wsp.queries import QUERY_TEMPLATES, SUPPORT_TEMPLATES, held_out_levels, load_sets
from wsp import private, redact2

PY = sys.executable
GAMES = redact2.GAMES
SECTIONS = RUNS / "private" / "game_sections_rev2"
TMP_ROOT = RUNS / "private" / "redaction_check_tmp"
FAMS = "ztl wih qcn lys esm aiv nkp obn qwv ruz vvf ubr yuz bvn trx yiy crg qpv ydv mds".split()

# ---- engine-hook adapters: each `_engine_*` (M only) calls the ORIGINAL implementation -------
# family -> Python source defining the adapters (names `orig`, `red`, `cls` are in scope); private.
ADAPTERS: dict[str, str] = private.get("ADAPTERS", {})
# family-specific budget-size attributes compared in the functional check (private)
RESOURCE_ATTRS_EXTRA: tuple = tuple(private.get("RESOURCE_ATTRS_EXTRA", ()))

# ---- solver vocabulary (condition M) ---------------------------------------------------------
VOCAB_RE = re.compile(r"solution|solve|solver|shortest|oracle|planner|witness|optimal", re.I)
# (regex matched against the token text around the hit, justification)
VOCAB_ALLOW = [
    (r"\bsolved_level\b", "engine API: GameSimulator.step() returns `solved_level` (base class)"),
    (r"\bself\.solved\b|\bsolved\b", "past participle for the goal state ('level solved'); the engine's own "
                                     "`solved` counter in GameSimulator.step()"),
    (r"\bunsolved\b", "goal-state adjective; describes state, not search"),
    (r"\bresolves?\b|\bresolved\b|\bresolving\b|\bresolution\b", "ordinary English word containing the "
                                                                  "vocabulary stem; unrelated to solving"),
    (r"\bdissolve[sd]?\b", "ordinary English word containing the vocabulary stem; unrelated to solving"),
]
# family-specific entries (they quote private game sources): (regex, justification); private
VOCAB_ALLOW += list(private.get("VOCAB_ALLOW_EXTRA", []))

LEVEL_WORDS = r"(?:L|level|levels|Level|Levels|LEVEL|LEVELS|lvl|rung|rungs|Rung|Rungs|board|boards|Board|Boards)"
NUM_LIST = r"(\d+(?:\s*(?:,|and|or|&|/|-|–|—|to|\.\.)\s*(?:L)?\d+)*)"
LEVEL_REF_RE = re.compile(r"\b" + LEVEL_WORDS + r"\s?#?\s?" + NUM_LIST + r"\b")
LRANGE_RE = re.compile(r"\bL(\d+)\s*(?:-|–|—|\.\.|to)\s*L?(\d+)\b")
TPL_REF_RE = re.compile(r"\b(?:T|template|templates|Template|Templates|TEMPLATE|TEMPLATES|tpl|seed|seeds|Seed|Seeds)"
                        r"\s?#?\s?" + NUM_LIST + r"\b")
ORDINALS = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth "
                                            "eleventh twelfth".split())}
ORD_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+(?:level|board|rung|room)\b", re.I)


def _expand(numlist: str) -> list[int]:
    out = []
    parts = re.split(r"\s*(?:,|and|or|&|/)\s*", numlist)
    for p in parts:
        m = re.match(r"L?(\d+)\s*(?:-|–|—|to|\.\.)\s*L?(\d+)$", p.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b <= a + 20:
                out += list(range(a, b + 1))
            else:
                out += [a, b]
        else:
            out += [int(x) for x in re.findall(r"\d+", p)]
    return out


# ---- section parsing --------------------------------------------------------------------------
def parse_section(sec: str, files: list[str]) -> dict[str, str]:
    out = {}
    for rel in files:
        m = re.search(r"^`" + re.escape(rel) + r"`:\n```python\n(.*?)^```$", sec, re.S | re.M)
        if not m:
            raise RuntimeError(f"code block for {rel} not found")
        out[rel] = m.group(1)
    return out


# ---- static checks ---------------------------------------------------------------------------
BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__spec__", "__package__",
                                      "__loader__", "__builtins__", "__annotations__", "__dict__"}


def undefined_globals(src: str, name: str) -> list[str]:
    top = symtable.symtable(src, name, "exec")

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
            n = s.get_name()
            if not s.is_referenced() or n in bound or n in BUILTIN_NAMES:
                continue
            if t is top or s.is_global():
                bad.add(n)
    return sorted(bad)


def _module_defs(tree: ast.Module) -> dict[str, ast.AST]:
    defs = {}

    def add_target(t, node):
        if isinstance(t, ast.Name):
            defs[t.id] = node
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e, node)

    def scan(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[node.name] = node
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    add_target(t, node)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                add_target(node.target, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name == "*":
                        continue
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                        continue
                    defs[(a.asname or a.name).split(".")[0]] = node
            elif isinstance(node, (ast.If, ast.Try)):
                scan(node.body)
                scan(getattr(node, "orelse", []))
    scan(tree.body)
    return defs


def _name_loaded(tree, name) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Store) for n in ast.walk(tree))


def unused_module_names(files: dict[str, str], family: str, pkg: str, exempt=None) -> dict[str, list[str]]:
    trees = {rel: ast.parse(src) for rel, src in files.items()}
    imported_from_siblings: set[str] = set()
    for t in trees.values():
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(f"arc3_games.games.{pkg}."):
                imported_from_siblings |= {a.name for a in node.names}
    out = {}
    for rel, tree in trees.items():
        defs = _module_defs(tree)
        unused = []
        for name, node in defs.items():
            if name.startswith("__") or name in imported_from_siblings:
                continue
            if exempt and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and exempt(name):
                continue
            if isinstance(node, ast.ClassDef) and any(
                    isinstance(st, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "family" for x in st.targets)
                    for st in node.body):
                continue                          # the simulator class itself (found by the engine)
            if isinstance(node, ast.Assign) and any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets):
                siblings = [nm for nm, nd in defs.items() if nd is node]
                if any(_name_loaded(tree, nm) for nm in siblings):
                    continue                      # enum-style tuple unpack: used as a unit
            used = False
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Store):
                    # ignore self-references inside its own def/class body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
                            node.lineno <= n.lineno <= node.end_lineno:
                        continue
                    used = True
                    break
                if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store) and \
                        not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom)):
                    used = True
                    break
            if not used:
                unused.append(name)
        out[rel] = sorted(unused)
    return out


TABLE_NAME_RE = re.compile(r"level|layout|board|blueprint|stage|spec|depth|budget|limit|slack|oracle|optimal|"
                           r"solution|shortest|witness|plan|title|blurb|archetype", re.I)


def per_level_tables(src: str, n_levels: int) -> list[str]:
    """Assignments of a per-level-looking name to a literal with one entry per level that withholds
    nothing (e.g. an oracle-depth table listing the held levels too)."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [getattr(t, "id", getattr(t, "attr", "")) for t in tgts]
            val = node.value
            if isinstance(val, ast.Call) and val.args and isinstance(val.args[0], ast.GeneratorExp):
                continue
            if isinstance(val, (ast.Tuple, ast.List)) and len(val.elts) == n_levels and \
                    any(TABLE_NAME_RE.search(nm or "") for nm in names) and \
                    not any(isinstance(x, ast.Name) and x.id == "WITHHELD" for x in ast.walk(val)):
                bad.append(f"line {node.lineno}: {names} has {n_levels} entries and no WITHHELD")
    return bad


def sibling_import_errors(files: dict[str, str], pkg: str) -> list[str]:
    defs = {rel: set(_module_defs(ast.parse(src))) for rel, src in files.items()}
    errs = []
    for rel, src in files.items():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(f"arc3_games.games.{pkg}."):
                target = f"{pkg}/{node.module.split('.')[-1]}.py"
                if target in defs:
                    for a in node.names:
                        if a.name not in defs[target]:
                            errs.append(f"{rel}: imports {a.name} from {target}, which does not define it")
    return errs


def _self_attr_sets(srcs: list[str]) -> tuple[set[str], set[str]]:
    loads, stores = set(), set()
    for src in srcs:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                if isinstance(n.ctx, ast.Store) or isinstance(n.ctx, ast.Del):
                    stores.add(n.attr)
                elif isinstance(n.value, ast.Name) and n.value.id in ("self", "sim", "cls"):
                    loads.add(n.attr)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stores.add(n.name)
            elif isinstance(n, ast.ClassDef):
                pass
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                for st in n.body:
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                stores.add(t.id)
                    elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        stores.add(st.target.id)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setattr" and len(n.args) >= 2 \
                    and isinstance(n.args[1], ast.Constant):
                stores.add(n.args[1].value)
    return loads, stores


def dangling_attrs(red_srcs: list[str], orig_srcs: list[str]) -> list[str]:
    base_names = set(dir(redact2.base.GameSimulator)) | {"rng", "nprng", "seed", "level", "state", "_anim", "grid",
                                                         "solved"}
    rl, rs = _self_attr_sets(red_srcs)
    ol, os_ = _self_attr_sets(orig_srcs)
    red_missing = rl - rs - base_names
    orig_missing = ol - os_ - base_names
    return sorted(red_missing - orig_missing)


def prose_items(src: str) -> list[tuple[int, str, str]]:
    """(line, kind, text) for comments, docstrings and other string constants."""
    items = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            items.append((tok.start[0], "comment", tok.string))
    tree = ast.parse(src)
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_nodes.add(id(body[0].value))
                items.append((body[0].lineno, "docstring", body[0].value.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            items.append((node.lineno, "string", node.value))
    return items


def held_mentions(src: str, held1: set[int], held_tpl: set[int], n_tpl: int) -> list[str]:
    hits = []
    for line, kind, text in prose_items(src):
        if kind == "string" and len(text) < 3:
            continue
        for m in LEVEL_REF_RE.finditer(text):
            nums = set(_expand(m.group(1)))
            word = m.group(0).split()[0].rstrip("#0123456789").lower()
            if word in ("board", "boards") and not re.search(r"board\s?#?\s?\d", m.group(0), re.I):
                continue
            bad = nums & held1
            if bad:
                hits.append(f"line {line} [{kind}]: level ref '{m.group(0)}' -> held {sorted(bad)}")
        for m in LRANGE_RE.finditer(text):
            a, b = int(m.group(1)), int(m.group(2))
            bad = set(range(a, b + 1)) & held1 if a <= b else set()
            if bad:
                hits.append(f"line {line} [{kind}]: level range '{m.group(0)}' -> held {sorted(bad)}")
        for m in TPL_REF_RE.finditer(text):
            word = re.match(r"[A-Za-z]+", m.group(0)).group(0).lower()
            nums = set(_expand(m.group(1)))
            if word.startswith("seed"):
                bad = {s for s in nums if s % n_tpl in held_tpl}
            else:
                bad = nums & held_tpl
            if bad:
                hits.append(f"line {line} [{kind}]: template ref '{m.group(0)}' -> withheld {sorted(bad)}")
        for m in ORD_RE.finditer(text):
            if ORDINALS[m.group(1).lower()] in held1:
                hits.append(f"line {line} [{kind}]: ordinal level ref '{m.group(0)}'")
    return sorted(set(hits))


def vocab_hits(src: str) -> tuple[list[str], dict[str, int]]:
    """Solver vocabulary in identifiers, strings and comments. A hit is allowed only if an allowlist
    pattern matches a span of the same token that covers the hit."""
    bad, allowed = [], {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.COMMENT) and \
                tok.type != getattr(tokenize, "FSTRING_MIDDLE", -1):
            continue
        text = tok.string
        for m in VOCAB_RE.finditer(text):
            for pat, _why in VOCAB_ALLOW:
                if any(a.start() <= m.start() and m.end() <= a.end() for a in re.finditer(pat, text, re.I)):
                    allowed[pat] = allowed.get(pat, 0) + 1
                    break
            else:
                ws = re.search(r"[\w.]*$", text[:m.start()]).group(0)
                we = re.match(r"[\w]*", text[m.end():]).group(0)
                ctx = text[max(0, m.start() - 20):m.end() + 20]
                bad.append(f"line {tok.start[0]}: '{ws + m.group(0) + we}' in {ctx!r}")
    return bad, allowed


# ---- materialisation and workers ------------------------------------------------------------
def family_pkg_dirs() -> list[str]:
    return sorted(p.name for p in GAMES.iterdir() if p.is_dir() and (p / "sim.py").exists())


def materialise(files: dict[str, str], pkg: str, tag: str) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=TMP_ROOT))
    others = set(family_pkg_dirs()) - {pkg}
    src_pkg = GAMES.parent

    def ignore(d, names):
        drop = {"__pycache__"}
        if Path(d) == GAMES:
            drop |= others
        return [n for n in names if n in drop]
    shutil.copytree(src_pkg, root / "arc3_games", ignore=ignore)
    for rel, text in files.items():
        (root / "arc3_games" / "games" / rel).write_text(text)
    # the family package's __init__.py is not part of the shown source (it may re-export an omitted
    # sibling family); the engine imports <pkg>.sim directly, so the package init is left empty
    (root / "arc3_games" / "games" / pkg / "__init__.py").write_text("")
    return root


def run_worker(job: dict, tag: str, timeout: int = 3600) -> dict:
    job = {**job, "res_attrs_extra": list(RESOURCE_ATTRS_EXTRA)}   # same attribute list on both sides
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"job_{tag}_", dir=TMP_ROOT))
    jp, op = d / "job.json", d / "out.json"
    jp.write_text(json.dumps(job))
    proc = subprocess.run([PY, "-B", str(Path(__file__).resolve()), "--worker", str(jp), str(op)],
                          capture_output=True, text=True, timeout=timeout, cwd=str(d))
    if proc.returncode != 0 or not op.exists():
        shutil.rmtree(d, ignore_errors=True)
        return {"errors": [f"worker failed rc={proc.returncode}: {proc.stderr[-1500:]}"]}
    out = json.loads(op.read_text())
    shutil.rmtree(d, ignore_errors=True)
    return out


def compare_traces(orig: dict, red: dict) -> tuple[int, int, list[str]]:
    n_seq = n_steps = 0
    fails = []
    for key, otr in orig.items():
        rtr = red.get(key)
        if rtr is None:
            fails.append(f"{key}: missing in redacted")
            continue
        for i, (o, r) in enumerate(zip(otr, rtr)):
            n_seq += 1
            n_steps += len(o) - 1
            if o != r:
                j = next((k for k in range(min(len(o), len(r))) if o[k] != r[k]), min(len(o), len(r)))
                fails.append(f"level:seed {key} seq {i} differs at step {j}: orig={str(o[j] if j < len(o) else None)[:220]} "
                             f"red={str(r[j] if j < len(r) else None)[:220]}")
        if len(otr) != len(rtr):
            fails.append(f"{key}: {len(otr)} vs {len(rtr)} sequences")
    return n_seq, n_steps, fails


# ---- per family ------------------------------------------------------------------------------
def original_files(rels: list[str], family: str) -> dict[str, str]:
    return {rel: redact2._drop_other_families((GAMES / rel).read_text(), family)[0] for rel in rels}


def check_family(fam: str, conds: list[str], live: bool, quick: bool) -> dict:
    t0 = time.time()
    n = new_game(fam, 0).n_levels
    cls = redact2.load(fam)
    module = cls.__module__
    pkg = module.split(".")[2]
    held = sorted(held_out_levels(n, version=2))
    support = [l for l in range(n) if l not in held]
    n_tpl, held_tpl = redact2.template_split(fam)
    seeds = list(SUPPORT_TEMPLATES)[: (2 if quick else 5)]
    pairs = [[l, t] for l in support for t in seeds]
    res = {"family": fam, "n_levels": n, "held_levels_1based": [h + 1 for h in held], "n_templates": n_tpl,
           "held_templates": held_tpl, "support_levels_1based": [l + 1 for l in support]}

    # sequences from the original
    pj = {"mode": "plan", "family": fam, "module": module, "redacted": False, "pairs": pairs,
          "n_uniform": 2 if quick else 5, "n_biased": 2 if quick else 5, "seq_len": 60, "plan_ms": 4000}
    plan_out = run_worker(pj, f"{fam}_plan")
    if plan_out.get("errors"):
        res["error"] = plan_out["errors"]
        return res
    try:
        sets = load_sets(fam, rev=2)
        manifest_ok = True
    except Exception as exc:
        sets, manifest_ok = {}, False
        res["manifest"] = f"unavailable ({type(exc).__name__}); (b) skipped"
    seqs, plan_src, n_prefix = {}, {}, 0
    for l, t in pairs:
        key = f"{l}:{t}"
        e = plan_out["plans"][key]
        s = []
        if e["plan"]:
            s.append(e["plan"])
        plan_src[e["src"] or "none"] = plan_src.get(e["src"] or "none", 0) + 1
        for qset in ("S-start", "S-dev"):
            for spec in sets.get(qset, []):
                if spec.level == l and spec.seed == t:
                    s.append([list(m) for m in spec.prefix])
                    n_prefix += 1
        s += e["random"]
        seqs[key] = s
    res["plan_sources"] = plan_src
    res["manifest_prefixes"] = n_prefix
    tj = {"mode": "trace", "family": fam, "module": module, "redacted": False, "seqs": seqs}
    orig_tr = run_worker(tj, f"{fam}_otrace")
    if orig_tr.get("errors"):
        res["error"] = orig_tr["errors"]
        return res

    for cond in conds:
        r = {"failures": [], "warnings": []}
        res[cond] = r
        try:
            if live:
                sec, rep = redact2.game_section_v2(fam, n, cond)
            else:
                sec = (SECTIONS / f"{fam}_{cond}.md").read_text()
                rep = json.loads((SECTIONS / f"{fam}_{cond}.audit.json").read_text())
        except Exception as exc:
            r["failures"].append(f"section build/read: {type(exc).__name__}: {exc}"[:500])
            r["pass"] = False
            continue
        rels = [f["file"] for f in rep["files"]]
        files = parse_section(sec, rels)
        origs = original_files(rels, fam)
        r["files"] = rels
        r["hooks"] = rep.get("hooks", [])
        r["override_counts"] = {f"#{c['i']} {c['note']}": c["count"] for c in rep.get("override_counts", [])}
        if rep.get("override_unmatched"):
            r["failures"] += [f"override matched 0 times: #{c['i']} {c['note']} {c['pattern'][:80]}"
                              for c in rep["override_unmatched"]]
        if rep["held_templates"] != held_tpl or rep["held_levels"] != held:
            r["failures"].append("report held levels/templates differ from the split")
        # header
        hdr = sec.split("```python", 1)[0]
        if f"seed % {n_tpl}" not in hdr or ("seed % 10" in hdr and n_tpl != 10):
            r["failures"].append("header template selector wrong")
        if cond == "M" and r["hooks"] and redact2.HOOK_SENTENCE not in hdr:
            r["failures"].append("header lacks the engine-hook sentence")
        if cond == "MS" and r["hooks"]:
            r["failures"].append("engine hooks present in MS")
        # static checks
        held1 = {h + 1 for h in held}
        st = {"undefined": {}, "unused": {}, "held_mentions": {}, "vocab": {}, "vocab_allowed": {},
              "dangling_attrs": []}
        for rel, src in files.items():
            try:
                compile(src, rel, "exec")
            except SyntaxError as exc:
                r["failures"].append(f"{rel}: compile: {exc}")
                continue
            u = undefined_globals(src, rel)
            if u:
                st["undefined"][rel] = u
                r["failures"].append(f"{rel}: undefined global names {u}")
            pt = per_level_tables(src, n)
            if pt:
                r["failures"] += [f"{rel}: per-level table {x}" for x in pt]
            hm = held_mentions(src, held1, set(held_tpl), n_tpl)
            if hm:
                st["held_mentions"][rel] = hm
                r["failures"] += [f"{rel}: {h}" for h in hm]
            if cond == "M":
                vb, va = vocab_hits(src)
                st["vocab_allowed"][rel] = va
                if vb:
                    st["vocab"][rel] = vb
                    r["failures"] += [f"{rel}: vocab {h}" for h in vb]
        # explicit, justified keeps: redact2_overrides.DOC_ROOTS (both) and MS_SOLVER_ROOTS (MS; empty)
        ms_keep = set(redact2.DOC_ROOTS.get(fam, {})) | (redact2.MS_SOLVER_ROOTS.get(fam, set()) if cond == "MS" else set())
        exempt = (lambda nm: nm in ms_keep) if ms_keep else None
        if redact2.DOC_ROOTS.get(fam):
            r["kept_unreferenced"] = redact2.DOC_ROOTS[fam]
        for rel, names in unused_module_names(files, fam, pkg, exempt).items():
            if names:
                st["unused"][rel] = names
                r["failures"].append(f"{rel}: module-level names defined but unreferenced: {names}")
        se = sibling_import_errors(files, pkg)
        r["failures"] += se
        da = dangling_attrs(list(files.values()), list(origs.values()))
        if da:
            st["dangling_attrs"] = da
            r["failures"].append(f"self/sim attributes read but never set (new vs original): {da}")
        r["static"] = st
        # functional
        root = materialise(files, pkg, f"{fam}_{cond}")
        try:
            hooks = r["hooks"]
            orig_file = str(Path(redact2.inspect.getsourcefile(cls))) if hooks else None
            rj = {"mode": "trace", "family": fam, "module": module, "redacted": True, "pkgroot": str(root),
                  "seqs": seqs, "hooks": hooks, "adapter": ADAPTERS.get(fam, ""), "orig_file": orig_file}
            red_tr = run_worker(rj, f"{fam}_{cond}_rtrace")
            if red_tr.get("errors"):
                r["failures"] += [f"redacted worker: {e}" for e in red_tr["errors"]]
            if hooks:
                r["hooks_bound"] = red_tr.get("hooks_bound", [])
            r["abstract_cleared"] = red_tr.get("abstract_cleared")
            if "traces" in red_tr:
                ns, nst, fl = compare_traces(orig_tr["traces"], red_tr["traces"])
                r["sequences"], r["steps"] = ns, nst
                r["failures"] += fl[:20]
                if len(fl) > 20:
                    r["failures"].append(f"... {len(fl) - 20} more trace mismatches")
                r["trace_mismatches"] = len(fl)
            tpl_seeds = list(range(5, 20))
            fj = {"mode": "refuse", "family": fam, "module": module, "redacted": True, "pkgroot": str(root),
                  "held": held, "tpl_seeds": tpl_seeds, "hooks": hooks, "adapter": ADAPTERS.get(fam, ""),
                  "orig_file": orig_file}
            ref = run_worker(fj, f"{fam}_{cond}_refuse")
            if ref.get("errors"):
                r["failures"] += [f"refuse worker: {e}" for e in ref["errors"]]
            rf = ref.get("refuse", {"levels": {}, "seeds": {}})
            r["refuse"] = rf
            for h in held:
                if rf["levels"].get(str(h), "NO-RAISE") == "NO-RAISE":
                    r["failures"].append(f"_init_level({h}) (level {h + 1}) did not raise")
            for s in tpl_seeds:
                want_fail = (s % n_tpl) in held_tpl
                got = rf["seeds"].get(str(s), "NO-RAISE")
                if want_fail and got == "NO-RAISE":
                    r["failures"].append(f"seed {s} (template {s % n_tpl}, withheld) did not fail")
                if not want_fail and got != "NO-RAISE":
                    r["failures"].append(f"seed {s} (template {s % n_tpl}, shown) failed: {got}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
        r["pass"] = not r["failures"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fams", nargs="*", default=FAMS)
    ap.add_argument("--conds", nargs="*", default=["M", "MS"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for f in args.fams:
        if f not in FAMS:
            raise SystemExit(f"family {f} is not one of the 20 checked families")
    results = {}
    for fam in args.fams:
        res = check_family(fam, args.conds, args.live, args.quick)
        results[fam] = res
        for cond in args.conds:
            r = res.get(cond, {})
            status = "PASS" if r.get("pass") else "FAIL"
            print(f"{fam:4s} {cond:2s} {status}  seqs={r.get('sequences', 0):4d} steps={r.get('steps', 0):6d} "
                  f"fails={len(r.get('failures', [])):3d} plans={res.get('plan_sources')} {res.get('seconds')}s",
                  flush=True)
            for fl in r.get("failures", [])[:12]:
                print("     -", fl[:300])
            if res.get("error"):
                print("     ERROR", str(res["error"])[:600])
    out = Path(args.out) if args.out else (SECTIONS / "check.json" if not args.live else None)
    if out:
        summary = {f: {c: {"pass": results[f].get(c, {}).get("pass", False),
                           "failures": len(results[f].get(c, {}).get("failures", [])),
                           "sequences": results[f].get(c, {}).get("sequences", 0),
                           "steps": results[f].get(c, {}).get("steps", 0)} for c in args.conds} for f in results}
        out.write_text(json.dumps({"summary": summary, "families": results,
                                   "vocab_allowlist": [{"pattern": p, "justification": j} for p, j in VOCAB_ALLOW]},
                                  indent=1))
        print("wrote", out)
    print("\nfamily " + " ".join(f"{c:>6s}" for c in args.conds))
    for f in results:
        print(f"{f:6s} " + " ".join(f"{('PASS' if results[f].get(c, {}).get('pass') else 'FAIL'):>6s}" for c in args.conds))


if __name__ == "__main__":
    main()
