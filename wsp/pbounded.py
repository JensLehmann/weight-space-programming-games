"""Bounded program player (condition 'pbounded'): a NumPy program under the free-architecture (FA) computation model.

The writer supplies Python with `MEMORY_SIZE = M` and `act(obs, available, memory, prev_action)` returning
`(scores (8,), click (64, 64), memory_next (M,))`, the outputs of an FA network (logits, click, mem_next) with the
same inputs (observation, memory, previous action one-hot, availability mask). Decoding is FA's
(wsp._pb_runtime.decode = wsp.freearch.reference_step's rule). Module-level code runs once and builds the constants
(like FA's build(): any Python with numpy); `act` and every function it calls ("player code") follow FA's
computation model, checked in two layers:

  static (`analyse`, on the AST; violations are returned to the writer like an invalid network):
    * imports: only `import numpy [as np]` at module level;
    * player code: assignments to names, expression statements, a final return, `for _ in range(K)` loops with K an
      integer literal or a module-level name assigned one integer literal once, 1 <= K <= MAX_T (FA's loop T), loop
      variable unused, no nesting (also not through calls), no break/continue/else; no if, while, try, with, assert,
      raise, conditional expressions, and/or/not, in/is, chained comparisons, comprehensions, generators, lambda,
      nested functions or classes, recursion, global/nonlocal, imports, del, f-strings, dict/set displays,
      *-unpacking, assignment into subscripts or attributes, Python builtins (except range in a loop header);
    * indices and slice bounds: literals, module-level names, None, ..., np.newaxis and + - * // of these;
    * names: locals, module-level constants, module-level functions (called only), np.<allowed function/constant>,
      array methods from METHODS and the attribute .T.
  runtime (wsp._pb_runtime, in the sandbox): counted NumPy; no array becomes a Python value; shapes, axes, shifts,
  widths, dtypes are Python constants; no assignment into arrays or in-place operations; `@`/einsum need one operand
  computed from constants only; float32 results; the caps below.

Caps (FA's values, wsp.freearch):
    constants   <= MAX_PARAMS (150,000) numbers: entries of the module-level values player code uses (arrays,
                   numbers; strings count one per character) plus numeric literals in player code;
    operations  <= MAX_EVALS (10,000) per call (FA node evaluations; a loop body counts every iteration);
    work        <= MAX_MACS (1.5e9) element operations per call (entries of each result; reductions: entries of
                   the input; @ / einsum: multiply-accumulates);
    array size  <= MAX_CH * 64 * 64 = 2,097,152 entries (FA's largest tensor: 512 channels; FA vectors <= 4,096);
    memory      <= MAX_MEM (256); loop count <= MAX_T (256).
Not mirrored: FA's limits on graph size (1,000 nodes) and on the summed channels of all tensors (4,096), which bound
the network's static size and memory; the operation and work caps bound the computation here.
Because player code cannot branch on data, every call performs the same operations: the counts are measured on two
test calls at submission (an exceeded cap makes the program invalid, as FA budget errors do) and enforced on every
call during play.

FA op -> NumPy in player code:
    conv (k x k)              np.pad + constant slices + np.einsum / @ with a constant kernel matrix
    dense                     W @ v (W constant)
    relu / tanh / sigmoid / abs  np.maximum(x, 0) / np.tanh / 1 / (1 + np.exp(-x)) / np.abs
    add / sub / mul / max2 / min2  + - * np.maximum np.minimum (NumPy broadcasting)
    concat, vconcat / slice, vslice  np.concatenate (np.stack) / constant slicing
    rowmax, colmax, rowsum, colsum   x.max(axis=2, keepdims=True), ... (broadcasting, np.broadcast_to)
    gmax / gmean              x.max(axis=(1, 2)) / x.mean(axis=(1, 2))
    loop (T, tied body)       for _ in range(K) (same code every iteration)
    beyond FA's op list       comparisons, np.where(c, a, b) (select = c*a + (1-c)*b), / // % **, np.clip, np.exp,
                              np.roll / np.flip (constant permutations), np.reshape / transpose / @ along any axis
                              with constant matrices, creation of constant arrays; all elementwise or linear with
                              constant coefficients, as FA's ops are up to the activation functions.

Other choices:
  1 act also receives the previous action one-hot (FA's `aprev`; the decoded action is outside the player), as 4th
    argument; `available` is FA's float mask, not a list of names.
  2 constants: module-level code runs once at submission (sandbox, 120 s hard limit, the prompt states 60 s); the
    values player code uses are frozen (float32, read-only; lists -> tuples) and stored as round<k>.npz (arrays) +
    round<k>.pb.json (structure and the static analysis); evaluation never re-runs module-level code.
  3 one sandboxed process per episode, as for Python players; a call has a 60 s safety timeout (not a rule: the
    budget bounds the work), an episode 3,600 s of CPU.
  4 `x += y` is executed as `x = x + y`; annotations are ignored.
"""
from __future__ import annotations

import ast
import base64
import builtins
import json
import select
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import _pb_runtime as RT
from .freearch import HW, MAX_CH, MAX_EVALS, MAX_MACS, MAX_MEM, MAX_PARAMS, MAX_T

RUNNER_PB = Path(__file__).with_name("_sandbox_runner_pb.py")
MAX_ELEMS = MAX_CH * HW
CAPS = {"ops": MAX_EVALS, "work": MAX_MACS, "elements": MAX_ELEMS}
BUILD_TIMEOUT = 120
STEP_TIMEOUT = 60.0
EPISODE_CPU_S = 3600
N_ARGS = 4
BUILTIN_NAMES = frozenset(dir(builtins))


class PBError(ValueError):
    """Invalid bounded player; the message is returned to the writer."""


def budget() -> dict:
    return {"constants": MAX_PARAMS, "ops_per_call": MAX_EVALS, "work_per_call": MAX_MACS,
            "array_entries": MAX_ELEMS, "memory": MAX_MEM, "loop_max": MAX_T}


# --------------------------------------------------------------------------- static analysis
_STMT = {ast.If: "`if`", ast.While: "`while`", ast.Try: "`try`", ast.With: "`with`", ast.AsyncWith: "`async with`",
         ast.Assert: "`assert`", ast.Raise: "`raise`", ast.Import: "an import", ast.ImportFrom: "an import",
         ast.Global: "`global`", ast.Nonlocal: "`nonlocal`", ast.Delete: "`del`",
         ast.FunctionDef: "a nested function", ast.AsyncFunctionDef: "a nested function", ast.ClassDef: "a class",
         ast.Break: "`break`", ast.Continue: "`continue`", ast.Match: "`match`", ast.AsyncFor: "`async for`"}
for _n in ("TryStar", "TypeAlias"):
    if hasattr(ast, _n):
        _STMT[getattr(ast, _n)] = {"TryStar": "`try`", "TypeAlias": "`type`"}[_n]
_EXPR = {ast.BoolOp: "`and` / `or`", ast.IfExp: "a conditional expression (x if c else y)", ast.Lambda: "`lambda`",
         ast.ListComp: "a comprehension", ast.SetComp: "a comprehension", ast.DictComp: "a comprehension",
         ast.GeneratorExp: "a generator expression", ast.Await: "`await`", ast.Yield: "`yield`",
         ast.YieldFrom: "`yield`", ast.NamedExpr: "`:=`", ast.JoinedStr: "an f-string", ast.Dict: "a dict display",
         ast.Set: "a set display", ast.Starred: "*-unpacking", ast.Slice: "a slice outside indexing"}
_WHERE = "in player code (act and the functions it calls)"


def _int_literal(e):
    if isinstance(e, ast.Constant) and type(e.value) is int:
        return e.value
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)) and isinstance(e.operand, ast.Constant) \
            and type(e.operand.value) is int:
        return -e.operand.value if isinstance(e.op, ast.USub) else e.operand.value
    return None


def _literal_default(e) -> bool:
    if isinstance(e, ast.Constant):
        return isinstance(e.value, (int, float, str, bool)) or e.value is None
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        return _literal_default(e.operand)
    if isinstance(e, ast.Tuple):
        return all(_literal_default(x) for x in e.elts)
    return False


def _module_bindings(tree) -> dict:
    """name -> binding nodes at module scope (including rebinding from functions declaring it global)."""
    out = defaultdict(list)

    def visit(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[ch.name].append(ch)
                continue
            if isinstance(ch, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(ch, ast.Name) and isinstance(ch.ctx, (ast.Store, ast.Del)):
                out[ch.id].append(ch)
            elif isinstance(ch, (ast.Import, ast.ImportFrom)):
                for a in ch.names:
                    out[(a.asname or a.name).split(".")[0]].append(ch)
            elif isinstance(ch, ast.ExceptHandler) and ch.name:
                out[ch.name].append(ch)
            elif isinstance(ch, (ast.MatchAs, ast.MatchStar)) and ch.name:
                out[ch.name].append(ch)
            visit(ch)
    visit(tree)
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            g = {n for st in ast.walk(fn) if isinstance(st, ast.Global) for n in st.names}
            for nd in ast.walk(fn):
                if isinstance(nd, ast.Name) and nd.id in g and isinstance(nd.ctx, (ast.Store, ast.Del)):
                    out[nd.id].append(nd)
    return out


def _module_scope(tree):
    """Nodes evaluated in the module scope (function, class, lambda and comprehension bodies excluded)."""
    out, todo = [], list(ast.iter_child_nodes(tree))
    while todo:
        nd = todo.pop()
        out.append(nd)
        if not isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp,
                               ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            todo += list(ast.iter_child_nodes(nd))
    return out


class _Ctx:
    def __init__(self, tree, defs, aliases, binds, intconst):
        self.tree, self.defs, self.aliases, self.binds, self.intconst = tree, defs, aliases, binds, intconst
        self.errors: list[str] = []

    def err(self, node, msg):
        line = getattr(node, "lineno", None)
        self.errors.append(msg if line is None else f"line {line}: {msg}")


class _Fn:
    """Checks one player-code function."""

    def __init__(self, fd: ast.FunctionDef, ctx: _Ctx):
        self.fd, self.ctx = fd, ctx
        self.calls: list[tuple] = []           # (callee, node, inside a loop)
        self.globals: set[str] = set()
        self.literals = 0
        self.loops: list[tuple] = []           # (line, K)
        self.in_loop = False
        a = fd.args
        self.locals = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        for nd in ast.walk(fd):
            if isinstance(nd, ast.Name) and isinstance(nd.ctx, ast.Store):
                self.locals.add(nd.id)
        for n in sorted(self.locals & (set(ctx.defs) | ctx.aliases | {"range"})):
            ctx.err(fd, f"function {fd.name}: the local name {n} hides a module-level function, numpy or range")

    def run(self):
        fd, E = self.fd, self.ctx.err
        if fd.decorator_list:
            E(fd, f"function {fd.name}: decorators are not allowed {_WHERE}")
        a = fd.args
        if a.vararg or a.kwarg or a.kwonlyargs:
            E(fd, f"function {fd.name}: only positional parameters are allowed {_WHERE}")
        for d in a.defaults + [x for x in a.kw_defaults if x is not None]:
            if not _literal_default(d):
                E(d, f"function {fd.name}: default values must be literal numbers, strings, None or tuples of them")
            for c in ast.walk(d):
                if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
                    self.literals += 1
        self.block(fd.body, top=True)
        loop_vars = {st.target.id for st in ast.walk(fd) if isinstance(st, ast.For) and isinstance(st.target, ast.Name)}
        for nd in ast.walk(fd):
            if isinstance(nd, ast.Name) and isinstance(nd.ctx, ast.Load) and nd.id in loop_vars:
                E(nd, f"the loop variable {nd.id} must not be used (the loop body is the same at every iteration)")
        return self

    # statements
    def block(self, stmts, top=False):
        for i, st in enumerate(stmts):
            self.stmt(st, top and i == len(stmts) - 1)

    def stmt(self, st, last):
        E = self.ctx.err
        if isinstance(st, ast.Expr):
            self.expr(st.value)
        elif isinstance(st, ast.Assign):
            for t in st.targets:
                self.target(t)
            self.expr(st.value)
        elif isinstance(st, ast.AugAssign):
            if not isinstance(st.target, ast.Name):
                self.target(st.target)
            elif isinstance(st.op, (ast.LShift, ast.RShift)):
                E(st, f"the operators << and >> are not allowed {_WHERE}")
            self.expr(st.value)
        elif isinstance(st, ast.AnnAssign):
            if not isinstance(st.target, ast.Name) or st.value is None:
                E(st, f"annotated assignments need a name and a value {_WHERE}")
            else:
                self.expr(st.value)
        elif isinstance(st, ast.Return):
            if not last:
                E(st, "`return` is allowed only as the last statement of a function")
            if st.value is not None:
                self.expr(st.value)
        elif isinstance(st, ast.Pass):
            pass
        elif isinstance(st, ast.For):
            self.loop(st)
        else:
            E(st, f"{_STMT.get(type(st), type(st).__name__)} is not allowed {_WHERE}")

    def target(self, t):
        E = self.ctx.err
        if isinstance(t, ast.Name):
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            for x in t.elts:
                if isinstance(x, ast.Starred):
                    E(x, f"*-unpacking is not allowed {_WHERE}")
                else:
                    self.target(x)
        elif isinstance(t, ast.Subscript):
            E(t, f"assignment into arrays (x[...] = v) is not allowed {_WHERE}; build a new array (np.concatenate, "
                 "np.where, ...)")
        elif isinstance(t, ast.Attribute):
            E(t, f"assignment to attributes is not allowed {_WHERE}")
        else:
            E(t, f"this assignment target is not allowed {_WHERE}")

    def loop(self, st):
        E = self.ctx.err
        if self.in_loop:
            E(st, "loops cannot be nested")
        if st.orelse:
            E(st, "`for ... else` is not allowed")
        if not isinstance(st.target, ast.Name):
            E(st, "the loop variable must be a single name: `for _ in range(K)`")
        it, K = st.iter, None
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and len(it.args) == 1 \
                and not it.keywords:
            arg = it.args[0]
            K = _int_literal(arg)
            if K is not None:
                self.literals += 1
            elif isinstance(arg, ast.Name) and arg.id in self.ctx.intconst and arg.id not in self.locals:
                K = self.ctx.intconst[arg.id]
                self.globals.add(arg.id)
        if K is None:
            E(st, "loops must be `for _ in range(K)` with K an integer literal or a module-level name assigned an "
                  "integer literal once")
        elif not 1 <= K <= MAX_T:
            E(st, f"loop count {K} is outside 1..{MAX_T}")
        else:
            self.loops.append((st.lineno, K))
        outer, self.in_loop = self.in_loop, True
        self.block(st.body)
        self.in_loop = outer

    # expressions
    def expr(self, e):
        E, t = self.ctx.err, type(e)
        if t is ast.Constant:
            self.const(e)
        elif t is ast.Name:
            self.name(e)
        elif t is ast.Attribute:
            self.attribute(e, called=False)
        elif t is ast.Call:
            self.call(e)
        elif t is ast.BinOp:
            if isinstance(e.op, (ast.LShift, ast.RShift)):
                E(e, f"the operators << and >> are not allowed {_WHERE}")
            self.expr(e.left)
            self.expr(e.right)
        elif t is ast.UnaryOp:
            if isinstance(e.op, ast.Not):
                E(e, f"`not` is not allowed {_WHERE} (use ~ on boolean arrays, or 1 - x)")
            self.expr(e.operand)
        elif t is ast.Compare:
            if len(e.ops) != 1:
                E(e, f"chained comparisons (a < b < c) are not allowed {_WHERE}")
            if any(isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)) for op in e.ops):
                E(e, f"`in` and `is` are not allowed {_WHERE}")
            self.expr(e.left)
            for c in e.comparators:
                self.expr(c)
        elif t is ast.Subscript:
            self.expr(e.value)
            self.index(e.slice)
        elif t in (ast.Tuple, ast.List):
            for x in e.elts:
                if isinstance(x, ast.Starred):
                    E(x, f"*-unpacking is not allowed {_WHERE}")
                else:
                    self.expr(x)
        else:
            E(e, f"{_EXPR.get(t, t.__name__)} is not allowed {_WHERE}")

    def const(self, e):
        v = e.value
        if isinstance(v, (bytes, complex)):
            self.ctx.err(e, f"{type(v).__name__} literals are not allowed {_WHERE}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            self.literals += 1
            with np.errstate(over="ignore"):
                if not np.isfinite(np.float32(v)):
                    self.ctx.err(e, f"the number {v!r} is not finite in float32")

    def name(self, e):
        n, ctx = e.id, self.ctx
        if n in self.locals:
            return
        if n in ctx.aliases:
            ctx.err(e, f"`{n}` can only be used as {n}.<function or constant> {_WHERE}")
        elif n in ctx.defs:
            ctx.err(e, f"the function {n} can only be called {_WHERE}, not used as a value")
        elif n == "range":
            ctx.err(e, "range is allowed only in a loop header `for _ in range(K)`")
        elif n in ctx.binds:
            self.globals.add(n)
        elif n in BUILTIN_NAMES:
            ctx.err(e, f"the Python builtin {n} is not allowed {_WHERE}")
        else:
            ctx.err(e, f"the name {n} is not defined at module level")

    def attribute(self, e, called):
        E, v = self.ctx.err, e.value
        if isinstance(v, ast.Name) and v.id in self.ctx.aliases and v.id not in self.locals:
            a = e.attr
            if called and a in RT.NP_CONSTS:
                E(e, f"{v.id}.{a}(...) is not allowed {_WHERE} (dtypes are passed as dtype arguments)")
            elif called and a not in RT.NP_FUNCS:
                E(e, f"{v.id}.{a}() is not available {_WHERE}")
            elif called and a == "where" and (len(self._call_node.args) != 3 or self._call_node.keywords):
                E(e, f"{v.id}.where needs exactly three positional arguments (condition, x, y) {_WHERE}")
            elif not called and a not in RT.NP_CONSTS:
                E(e, f"{v.id}.{a} must be called" if a in RT.NP_FUNCS else f"{v.id}.{a} is not available {_WHERE}")
            return
        if called and e.attr not in RT.METHODS:
            E(e, f"the method .{e.attr}() is not allowed {_WHERE}")
        elif not called and e.attr not in RT.ATTRS:
            E(e, f"the attribute .{e.attr} is not allowed {_WHERE}"
                 + (" (shapes are fixed: write them as constants)" if e.attr in ("shape", "size", "ndim") else
                    " (arrays: .T and the listed methods)"))
        self.expr(v)

    def call(self, e):
        E, f = self.ctx.err, e.func
        for a in e.args:
            if isinstance(a, ast.Starred):
                E(a, f"*-unpacking is not allowed {_WHERE}")
            else:
                self.expr(a)
        for k in e.keywords:
            if k.arg is None:
                E(k.value, f"**-unpacking is not allowed {_WHERE}")
            else:
                self.expr(k.value)
        if isinstance(f, ast.Name):
            n = f.id
            if n in self.locals:
                E(e, f"{n} is a local variable; only module-level functions can be called")
            elif n in self.ctx.defs:
                self.calls.append((n, e, self.in_loop))
            elif n == "range":
                E(e, "range is allowed only in a loop header `for _ in range(K)`")
            elif n in BUILTIN_NAMES and n not in self.ctx.binds:
                E(e, f"the Python builtin {n}() is not allowed {_WHERE}")
            else:
                E(e, f"{n}() is not a module-level function defined with def")
        elif isinstance(f, ast.Attribute):
            self._call_node = e
            self.attribute(f, called=True)
        else:
            E(e, f"only module-level functions, np functions and array methods can be called {_WHERE}")

    def index(self, s):
        if isinstance(s, ast.Slice):
            for p in (s.lower, s.upper, s.step):
                if p is not None:
                    self.static(p)
        elif isinstance(s, ast.Tuple):
            for x in s.elts:
                self.index(x)
        else:
            self.static(s)

    def static(self, e):
        ctx = self.ctx
        ok = True
        if isinstance(e, ast.Constant):
            v = e.value
            ok = (type(v) is int) or v is None or v is Ellipsis or isinstance(v, str)
            if type(v) is int:
                self.literals += 1
        elif isinstance(e, ast.Name):
            ok = e.id not in self.locals and e.id in ctx.binds and e.id not in ctx.defs and e.id not in ctx.aliases
            if ok:
                self.globals.add(e.id)
        elif isinstance(e, ast.Attribute):
            ok = isinstance(e.value, ast.Name) and e.value.id in ctx.aliases and e.attr == "newaxis"
        elif isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            self.static(e.operand)
        elif isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            self.static(e.left)
            self.static(e.right)
        else:
            ok = False
        if not ok:
            ctx.err(e, "indices and slice bounds must be constants: integer literals, module-level constants, None, "
                       "... or np.newaxis (no local variables, arrays or calls)")


def _cycle(edges: dict) -> list | None:
    state, stack = {}, []

    def dfs(u):
        state[u] = 1
        stack.append(u)
        for v in edges.get(u, ()):
            if state.get(v) == 1:
                return stack[stack.index(v):] + [v]
            if v not in state:
                c = dfs(v)
                if c:
                    return c
        stack.pop()
        state[u] = 2
        return None
    for u in sorted(edges):
        if u not in state:
            c = dfs(u)
            if c:
                return c
    return None


def analyse(source: str) -> dict:
    """Static check of a bounded player. Returns the spec used by the sandbox; raises PBError (writer-facing)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PBError(f"syntax error: {exc}") from None
    ctx = _Ctx(tree, {}, set(), {}, {})
    E = ctx.err
    module_scope = set(map(id, _module_scope(tree)))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            E(node, f"only `import numpy as np` is allowed (found `from {node.module or '.'} import ...`)")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name != "numpy":
                    E(node, f"import {a.name}: numpy is the only module that may be imported")
                elif id(node) in module_scope:          # imports inside player code are rejected as statements
                    ctx.aliases.add(a.asname or "numpy")
    ctx.binds = dict(_module_bindings(tree))
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            if st.name in ctx.defs:
                E(st, f"the function {st.name} is defined twice")
            ctx.defs[st.name] = st
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = _int_literal(st.value)
            n = st.targets[0].id
            if v is not None and len(ctx.binds.get(n, ())) == 1:
                ctx.intconst[n] = v
    for a in sorted(ctx.aliases):
        if len(ctx.binds.get(a, ())) != 1:
            E(tree, f"the name {a} must be bound only by `import numpy as {a}`")
    if "range" in ctx.binds:
        E(tree, "the name range must not be redefined")
    if "MEMORY_SIZE" not in ctx.intconst:
        E(tree, f"define MEMORY_SIZE = M once at module level, with M an integer literal, 0 <= M <= {MAX_MEM}")
    elif not 0 <= ctx.intconst["MEMORY_SIZE"] <= MAX_MEM:
        E(tree, f"MEMORY_SIZE must be in 0..{MAX_MEM}, got {ctx.intconst['MEMORY_SIZE']}")
    act = ctx.defs.get("act")
    if act is None:
        E(tree, "define act(obs, available, memory, prev_action) at module level with def")
    else:
        a = act.args
        if len(a.posonlyargs) + len(a.args) != N_ARGS or a.vararg or a.kwarg or a.kwonlyargs:
            E(act, "act must take exactly four positional parameters: act(obs, available, memory, prev_action)")
    if ctx.errors:
        raise PBError("; ".join(ctx.errors[:12]))
    fns, todo = {}, ["act"]
    while todo:
        n = todo.pop()
        if n in fns:
            continue
        fns[n] = _Fn(ctx.defs[n], ctx).run()
        todo += [c for c, _, _ in fns[n].calls if c not in fns]
    edges = {n: sorted({c for c, _, _ in f.calls}) for n, f in fns.items()}
    cyc = _cycle(edges)
    if cyc:
        E(ctx.defs[cyc[0]], "recursion is not allowed: " + " -> ".join(cyc))
    has_loop = {n: bool(f.loops) or any(isinstance(nd, ast.For) for nd in ast.walk(f.fd)) for n, f in fns.items()}
    for _ in range(len(fns)):
        for n in fns:
            has_loop[n] = has_loop[n] or any(has_loop[c] for c in edges[n])
    for n, f in fns.items():
        for c, node, inside in f.calls:
            if inside and has_loop[c]:
                E(node, f"the loop body calls {c}(), which contains a loop (loops cannot be nested)")
    if ctx.errors:
        raise PBError("; ".join(ctx.errors[:12]))
    runtime = sorted(fns)
    glob = sorted(set().union(*(f.globals for f in fns.values())) - set(ctx.defs) - ctx.aliases)
    return {"runtime": runtime, "def_lines": {n: ctx.defs[n].lineno for n in runtime},
            "globals": glob, "np_aliases": sorted(ctx.aliases), "memory_size": ctx.intconst["MEMORY_SIZE"],
            "literal_numbers": int(sum(f.literals for f in fns.values())),
            "loops": [[n, line, K] for n, f in sorted(fns.items()) for line, K in f.loops]}


# --------------------------------------------------------------------------- budget
def count_of(spec: dict, built: dict) -> dict:
    pr = built["probe"]
    return {"constants": int(built["numbers"]) + int(spec["literal_numbers"]),
            "constants_module": int(built["numbers"]), "constants_literal": int(spec["literal_numbers"]),
            "ops_per_call": int(pr["ops"]), "work_per_call": int(pr["work"]), "max_array": int(pr["max_array"]),
            "memory_size": int(spec["memory_size"]), "loops": spec["loops"], "counts_equal": bool(pr["counts_equal"]),
            "call_s": [c["s"] for c in pr["calls"]]}


def budget_errors(c: dict) -> list[str]:
    errs = []
    if c["constants"] > MAX_PARAMS:
        errs.append(f"{c['constants']:,} constants > budget {MAX_PARAMS:,}")
    if c["ops_per_call"] > MAX_EVALS:
        errs.append(f"{c['ops_per_call']:,} operations per call > limit {MAX_EVALS:,} (a loop body counts every "
                    "iteration)")
    if c["work_per_call"] > MAX_MACS:
        errs.append(f"{c['work_per_call']:,} element operations per call > budget {int(MAX_MACS):,}")
    return errs


# --------------------------------------------------------------------------- sandboxed build and artifacts
def _spec_file(spec: dict) -> dict:
    return {**spec, "caps": CAPS}


def run_build(source: str, spec: dict, timeout: int = BUILD_TIMEOUT):
    """Module-level code once + test calls in the sandbox -> (trees, arrays, built info, error, wall_s)."""
    from .sandbox import _env, _limits
    with tempfile.TemporaryDirectory(prefix="wsp_pb_") as tmp:
        tmp = Path(tmp)
        (tmp / "writer_source.py").write_text(source)
        (tmp / "spec.json").write_text(json.dumps(_spec_file(spec)))
        work = tmp / "work"
        work.mkdir()
        cmd = [sys.executable, "-I", str(RUNNER_PB), "build", str(tmp / "writer_source.py"), str(tmp / "spec.json"),
               str(tmp / "out.npz"), str(tmp / "out.json")]
        t = time.time()
        try:
            proc = subprocess.run(cmd, cwd=work, env=_env(str(work)), capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits(timeout))
        except subprocess.TimeoutExpired:
            return None, None, None, f"timeout after {timeout}s", time.time() - t
        wall = time.time() - t
        lines = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not lines:
            return None, None, None, "runtime: " + proc.stderr.strip()[-1500:], wall
        info = json.loads(lines[-1])
        if not info.get("ok"):
            return None, None, None, ("runtime: " if info.get("stage") == "load" else "invalid player: ") + info["error"], \
                wall
        trees = json.loads((tmp / "out.json").read_text())
        with np.load(tmp / "out.npz") as z:
            arrays = {k: z[k] for k in z.files}
        return trees, arrays, info, None, wall


def materialize(code: str):
    """Writer program -> session.Artifact (condition 'pbounded'): params = constant arrays,
    graph = {"spec": static analysis, "consts": frozen structure}."""
    from .sandbox import audit
    from .session import Artifact
    au = audit(code, "act")
    aj = au.to_json()
    if not au.ok:
        return Artifact("pbounded", code, error="audit: " + "; ".join(au.violations[:10]), audit=aj)
    try:
        spec = analyse(code)
    except PBError as exc:
        return Artifact("pbounded", code, error=f"invalid player: {exc}", audit=aj)
    except Exception as exc:          # checker bug or unusual syntax: still an invalid artifact
        return Artifact("pbounded", code, error=f"invalid player: {type(exc).__name__}: {exc}"[:1500], audit=aj)
    trees, arrays, info, err, wall = run_build(code, spec)
    if err:
        return Artifact("pbounded", code, error=err, audit=aj, init_wall_s=wall)
    if info.get("cpu_s", 0) > 30:
        aj["flags"].append(f"module-level code used {info['cpu_s']:.0f}s CPU")
    aj["count"] = count_of(spec, info)
    aj["probe_moves"] = [c["move"] for c in info["probe"]["calls"]]
    errs = budget_errors(aj["count"])
    if errs:
        return Artifact("pbounded", code, error="invalid player: budget: " + "; ".join(errs), audit=aj,
                        init_wall_s=wall)
    return Artifact("pbounded", code, params=arrays, audit=aj, init_wall_s=wall,
                    graph={"spec": spec, "consts": trees})


def save_artifact(art, out_dir: Path, k: int) -> None:
    np.savez_compressed(Path(out_dir) / f"round{k}.npz", **art.params)
    (Path(out_dir) / f"round{k}.pb.json").write_text(json.dumps(art.graph))


def load_artifact(out_dir: Path, k: int, code: str):
    from .session import Artifact
    g = json.loads((Path(out_dir) / f"round{k}.pb.json").read_text())
    spec = analyse(code)                                  # the stored program still passes the static check
    if spec != g["spec"]:
        raise PBError(f"{out_dir}: round{k}.pb.json does not match the static analysis of round{k}.py")
    with np.load(Path(out_dir) / f"round{k}.npz") as z:
        params = {n: z[n] for n in z.files}
    return Artifact("pbounded", code, params=params, graph=g)


# --------------------------------------------------------------------------- evaluation
class BoundedPlayerPolicy:
    """One fresh sandboxed process per episode (wsp.sandbox.PythonPlayerPolicy); the process holds the memory and
    previous action, calls act and decodes as FA."""

    def __init__(self, art, step_timeout: float = STEP_TIMEOUT, episode_cpu_s: int = EPISODE_CPU_S):
        self.tmp = tempfile.TemporaryDirectory(prefix="wsp_pbplayer_")
        d = Path(self.tmp.name)
        self.files = [d / "writer_source.py", d / "spec.json", d / "consts.npz", d / "consts.json"]
        self.files[0].write_text(art.code)
        self.files[1].write_text(json.dumps(_spec_file(art.graph["spec"])))
        np.savez(self.files[2], **art.params)
        self.files[3].write_text(json.dumps(art.graph["consts"]))
        self.step_timeout = step_timeout
        self.episode_cpu_s = episode_cpu_s
        self.procs: dict[int, subprocess.Popen] = {}
        self.latency: list[float] = []
        self.ops: list[int] = []
        self.errors: list[str] = []

    def _spawn(self):
        from .sandbox import _env, _limits
        work = tempfile.mkdtemp(dir=self.tmp.name)
        return subprocess.Popen([sys.executable, "-I", str(RUNNER_PB), "player", *map(str, self.files)],
                                cwd=work, env=_env(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, preexec_fn=_limits(self.episode_cpu_s))

    def begin(self, slots):
        for i in slots:
            if i in self.procs:
                self.procs[i].kill()
            self.procs[i] = self._spawn()

    def _send(self, i, obs, avail, outputs=False):
        msg = {"obs": base64.b64encode(np.asarray(obs).astype(np.int8).tobytes()).decode(),
               "avail": [int(bool(v)) for v in avail], "outputs": outputs}
        try:
            self.procs[i].stdin.write(json.dumps(msg) + "\n")
            self.procs[i].stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _recv(self, i) -> dict:
        p = self.procs[i]
        ready, _, _ = select.select([p.stdout], [], [], self.step_timeout)
        if not ready:
            p.kill()
            return {"move": ["TIMEOUT", None, None]}
        line = p.stdout.readline()
        if not line:
            return {"move": ["CRASH", None, None]}
        reply = json.loads(line)
        if "error" in reply:
            self.errors.append(reply["error"])
            return {"move": ["ERROR", None, None]}
        self.latency.append(reply["s"])
        self.ops.append(reply["ops"])
        return reply

    def act(self, slots, obs, avail):
        for j, i in enumerate(slots):
            self._send(i, obs[j], avail[j])
        return [tuple(self._recv(i)["move"]) for i in slots]

    def close(self):
        for p in self.procs.values():
            p.kill()
        self.tmp.cleanup()


def policy(art) -> BoundedPlayerPolicy:
    return BoundedPlayerPolicy(art)


def probe_internals(art, spec, trace, step) -> dict | None:
    """Masked scores, [max, row, col] of the click scores and the memory norm at decision `step` of an episode
    (replayed with the player; the feedback analogue of wsp.freearch.probe_internals)."""
    from .envs import ACTIONS, avail_mask, build_start, obs_of
    pol = BoundedPlayerPolicy(art)
    try:
        pol.begin([0])
        g = build_start(spec)
        for k in range(step + 1):
            pol._send(0, obs_of(g), avail_mask(g), outputs=k == step)
            reply = pol._recv(0)
            if k == step:
                return {key: reply[key] for key in ("scores", "click_max_at", "memory_l2")} if "scores" in reply \
                    else None
            a, r, c = trace[k] if k < len(trace) else ("ACTION1", None, None)
            if a in ACTIONS:
                g.step(a, r, c)
    finally:
        pol.close()
    return None


# --------------------------------------------------------------------------- trusted in-process run (tests, prompt)
def run_trusted(source: str, frames=None) -> dict:
    """Harness-authored programs only (the prompt example, tests): analyse, run in this process, count."""
    spec = analyse(source)
    ns: dict = {"__name__": "trusted"}
    exec(compile(source, RT.SOURCE_NAME, "exec"), ns)
    trees, arrays, n = RT.freeze(ns, spec["globals"])
    RT.install(CAPS, hard=False)
    act = RT.load_player(source, spec, trees, arrays)
    pr = RT.probe(act, spec["memory_size"], frames or RT.probe_frames())
    return {"spec": spec, "act": act, "count": count_of(spec, {"numbers": n, "probe": pr}), "trees": trees,
            "arrays": arrays}
