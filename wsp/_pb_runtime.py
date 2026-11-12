"""Counted NumPy for the bounded program player (condition 'pbounded'; see wsp.pbounded).

Player code (act and the functions it calls) sees its numpy name bound to NP, a proxy that exposes only the
functions in NP_FUNCS and the constants in NP_CONSTS; every array it handles is a BArray. One operation is one
NP function call, one operator with an array operand (a NumPy ufunc call), one array method call or one indexing
of an array. Its element work is the number of entries of its result; reductions count the entries of their
input, `@` and einsum their multiply-accumulates. Arrays never become Python values: truth value, int(), float()
and use as an index raise, as do assignment into arrays and in-place operations. BArray._data marks values
computed from the inputs; `@` and einsum need at least one operand without it (a weight). Floating results are
float32.

This module imports only numpy and the standard library: the sandbox process (_sandbox_runner_pb.py) loads it by
path, and wsp.pbounded imports it for the prompt example and the tests.
"""
from __future__ import annotations

import ast
import math
import time
import traceback
from types import MappingProxyType

import numpy as _np

ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")
CLICK = 6
SOURCE_NAME = "writer_source.py"


class PBRuntimeError(Exception):
    """Rule violation while player code runs; the message is returned to the writer."""


class BudgetExceeded(PBRuntimeError):
    pass


# --------------------------------------------------------------------------- counter
class Counter:
    """Operations and element work of one act() call. hard=False (the test call at submission) stops only at
    10x the caps so that the exact totals can be reported; the array-size limit always applies."""

    def __init__(self, ops: int, work: float, elements: int, hard: bool = True):
        self.cap_ops, self.cap_work, self.cap_elems = int(ops), int(work), int(elements)
        self.hard = hard
        self.active = False
        self.reset()

    def reset(self):
        self.ops = 0
        self.work = 0
        self.max_array = 0

    def charge(self, work, what):
        if not self.active:
            return
        self.ops += 1
        self.work += int(work)
        k = 1 if self.hard else 10
        if self.ops > k * self.cap_ops:
            raise BudgetExceeded(f"budget: more than {k * self.cap_ops:,} operations per call (at {what})")
        if self.work > k * self.cap_work:
            raise BudgetExceeded(f"budget: more than {k * self.cap_work:,} element operations per call (at {what})")

    def size(self, n, what):
        if not self.active:
            return
        if n > self.cap_elems:
            raise PBRuntimeError(f"{what}: an array of {n:,} entries > limit {self.cap_elems:,}")
        self.max_array = max(self.max_array, int(n))


_C = Counter(10_000, 1.5e9, 2_097_152)


def install(caps: dict, hard: bool = True) -> Counter:
    """Set the caps {'ops', 'work', 'elements'} used by all player code in this process."""
    global _C
    _C = Counter(caps["ops"], caps["work"], caps["elements"], hard)
    return _C


def counter() -> Counter:
    return _C


# --------------------------------------------------------------------------- helpers
def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _plain(x):
    """BArray -> plain ndarray view; lists and tuples are converted element-wise."""
    if isinstance(x, BArray):
        return x.view(_np.ndarray)
    if isinstance(x, (list, tuple)):
        return type(x)(_plain(v) for v in x)
    return x


def _is_data(x) -> bool:
    if isinstance(x, BArray):
        return bool(x._data)
    if isinstance(x, (list, tuple)):
        return any(_is_data(v) for v in x)
    return False


def _shape(x) -> tuple:
    return tuple(x.shape) if isinstance(x, _np.ndarray) else _np.shape(x)


def _static(v, what):
    """Shapes, axes, shifts, widths, counts: Python constants only (never arrays)."""
    if isinstance(v, (_np.ndarray, _np.generic)):
        raise PBRuntimeError(f"{what} must be a Python constant (integers, tuples, None), not an array")
    if isinstance(v, (list, tuple)):
        for e in v:
            _static(e, what)
    elif isinstance(v, slice):
        for e in (v.start, v.stop, v.step):
            _static(e, what)
    return v


def _index(key):
    """Indices: integers, slices of integers, None, Ellipsis, or a tuple of these."""
    parts = key if isinstance(key, tuple) else (key,)
    for p in parts:
        ok = p is None or p is Ellipsis or (isinstance(p, int) and not isinstance(p, bool))
        if isinstance(p, slice):
            ok = all(e is None or (isinstance(e, int) and not isinstance(e, bool)) for e in (p.start, p.stop, p.step))
        if not ok:
            raise PBRuntimeError("indices must be integer constants, slices of them, None or ... (not arrays, lists "
                                 "or other values)")
    return key


DTYPES = (_np.float32, _np.int32, _np.int64, _np.bool_)


def _dtype(d, required=False):
    if d is None and not required:
        return None
    if any(d is t for t in DTYPES):
        return d
    raise PBRuntimeError("dtype must be one of np.float32, np.int32, np.int64, np.bool_")


def _wrap(r, data, what):
    if isinstance(r, tuple):
        return tuple(_wrap(v, data, what) for v in r)
    a = r if isinstance(r, _np.ndarray) else _np.asarray(r)
    k = a.dtype.kind
    if k == "f":
        if a.dtype != _np.float32:
            a = a.astype(_np.float32)
    elif k not in "biu":
        raise PBRuntimeError(f"{what}: arrays of dtype {a.dtype} are not allowed (numbers and booleans only)")
    _C.size(a.size, what)
    out = a.view(BArray)
    out._data = bool(data)
    return out


def _matmul_work(sa, sb, what):
    if len(sa) == 0 or len(sb) == 0:
        raise PBRuntimeError(f"{what}: operands must have at least one dimension")
    a = tuple(sa) if len(sa) > 1 else (1,) + tuple(sa)
    b = tuple(sb) if len(sb) > 1 else tuple(sb) + (1,)
    if a[-1] != b[-2]:
        raise PBRuntimeError(f"{what}: shapes {tuple(sa)} and {tuple(sb)} do not match")
    batch = _np.broadcast_shapes(a[:-2], b[:-2])
    return _prod(batch) * a[-2] * a[-1] * b[-1], _prod(batch) * a[-2] * b[-1]


def _contract_check(operands, what):
    if sum(_is_data(x) for x in operands) > 1:
        raise PBRuntimeError(f"{what}: at least one operand must be computed only from constants (a weight); here "
                             "both depend on the inputs")


# operators reach __array_ufunc__ as these ufuncs (** with 2, 0.5, -1 uses square, sqrt, reciprocal)
UFUNCS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "floor_divide": "//", "remainder": "%",
          "power": "**", "square": "**", "sqrt": "**", "reciprocal": "**", "matmul": "@", "negative": "unary -",
          "positive": "unary +", "invert": "~", "bitwise_and": "&", "bitwise_or": "|", "bitwise_xor": "^",
          "less": "<", "less_equal": "<=", "greater": ">", "greater_equal": ">=", "equal": "==",
          "not_equal": "!="}


def _ufunc(uf, inputs, what):
    data = any(_is_data(x) for x in inputs)
    args = [_plain(x) for x in inputs]
    if uf is _np.matmul:
        _contract_check(inputs, what)
        work, n = _matmul_work(_shape(args[0]), _shape(args[1]), what)
    else:
        try:
            n = _prod(_np.broadcast_shapes(*[_shape(a) for a in args]))
        except ValueError:
            n = 0                                   # numpy raises its own broadcasting error below
        work = n
    _C.size(n, what)
    _C.charge(work, what)
    return _wrap(uf(*args), data, what)


# --------------------------------------------------------------------------- the array type
_NO_PY = "arrays cannot be converted to Python numbers, booleans or indices in player code"


class BArray(_np.ndarray):
    """ndarray whose operations are counted (see the module docstring)."""

    _data = False

    def __array_finalize__(self, obj):
        self._data = getattr(obj, "_data", False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        what = UFUNCS.get(ufunc.__name__)
        if "out" in kwargs:
            raise PBRuntimeError("in-place operations are not allowed in player code")
        if method != "__call__" or what is None or kwargs:
            raise PBRuntimeError(f"operation {ufunc.__name__}{'' if method == '__call__' else '.' + method} is not "
                                 "available in player code")
        return _ufunc(ufunc, inputs, f"operator {what}")

    def __array_function__(self, func, types, args, kwargs):
        raise PBRuntimeError(f"np.{getattr(func, '__name__', '?')} is not available in player code")

    def __bool__(self):
        raise PBRuntimeError("an array cannot be used as a Python truth value in player code")

    def __int__(self):
        raise PBRuntimeError(_NO_PY)

    def __float__(self):
        raise PBRuntimeError(_NO_PY)

    def __complex__(self):
        raise PBRuntimeError(_NO_PY)

    def __index__(self):
        raise PBRuntimeError(_NO_PY)

    def __setitem__(self, key, value):
        raise PBRuntimeError("assignment into arrays is not allowed in player code (build a new array instead)")

    def __repr__(self):                       # shape and dtype only: no array values become Python strings
        return f"<array shape={tuple(self.shape)} dtype={self.dtype}>"

    __str__ = __repr__

    def __format__(self, spec):
        return repr(self)

    def __getitem__(self, key):
        _index(key)
        r = self.view(_np.ndarray)[key]
        n = r.size if isinstance(r, _np.ndarray) else 1
        _C.size(n, "indexing")
        _C.charge(n, "indexing")
        return _wrap(r, self._data, "indexing")

    # allowed methods (one operation each)
    def sum(self, axis=None, keepdims=False):
        return _reduce("sum", self, axis, keepdims, ".sum()")

    def max(self, axis=None, keepdims=False):
        return _reduce("max", self, axis, keepdims, ".max()")

    def min(self, axis=None, keepdims=False):
        return _reduce("min", self, axis, keepdims, ".min()")

    def mean(self, axis=None, keepdims=False):
        return _reduce("mean", self, axis, keepdims, ".mean()")

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return _same(".reshape()", self, lambda a: a.reshape(_static(tuple(shape), "the shape")))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return _same(".transpose()", self, lambda a: a.transpose(*_static(tuple(axes), "the axes")))

    @property
    def T(self):
        return _same(".T", self, lambda a: a.T)

    def astype(self, dtype):
        dt = _dtype(dtype, required=True)
        return _same(".astype()", self, lambda a: a.astype(dt))

    def copy(self):
        return _same(".copy()", self, lambda a: a.copy())

    def ravel(self):
        return _same(".ravel()", self, lambda a: a.ravel())

    def flatten(self):
        return _same(".flatten()", self, lambda a: a.flatten())

    def squeeze(self, axis=None):
        return _same(".squeeze()", self, lambda a: a.squeeze(_static(axis, "the axis")))

    def swapaxes(self, axis1, axis2):
        return _same(".swapaxes()", self, lambda a: a.swapaxes(_static(axis1, "the axis"), _static(axis2, "the axis")))

    def clip(self, a_min, a_max):
        return _clip(self, a_min, a_max, ".clip()")

    # blocked (the static check rejects them first)
    def _blocked(self, *a, **k):
        raise PBRuntimeError("this array method is not allowed in player code")

    item = tolist = argmax = argmin = nonzero = tobytes = fill = put = sort = argsort = any = all = _blocked
    cumsum = cumprod = dot = take = choose = compress = repeat = searchsorted = resize = itemset = _blocked


def _same(what, x, fn):
    """Structural op on one array: work = entries of the result."""
    a = _plain(x)
    r = fn(a)
    _C.size(r.size, what)
    _C.charge(r.size, what)
    return _wrap(r, _is_data(x), what)


def _reduce(name, x, axis, keepdims, what):
    _static(axis, "the axis")
    if not isinstance(keepdims, bool):
        raise PBRuntimeError(f"{what}: keepdims must be True or False")
    a = _np.asarray(_plain(x))
    _C.charge(a.size, what)
    r = getattr(_np, name)(a, axis=axis, keepdims=keepdims)
    return _wrap(r, _is_data(x), what)


def _clip(x, lo, hi, what):
    args = [_plain(v) for v in (x, lo, hi)]
    n = _prod(_np.broadcast_shapes(*[_shape(a) for a in args]))
    _C.size(n, what)
    _C.charge(n, what)
    return _wrap(_np.clip(*args), any(_is_data(v) for v in (x, lo, hi)), what)


# --------------------------------------------------------------------------- the np proxy
def _elementwise(name, fn, nargs):
    what = f"np.{name}"

    def f(*args):
        if len(args) != nargs:
            raise PBRuntimeError(f"{what} takes {nargs} argument{'s' if nargs > 1 else ''} in player code"
                                 + (" (condition, x, y): an elementwise select" if name == "where" else ""))
        plain = [_plain(a) for a in args]
        n = _prod(_np.broadcast_shapes(*[_shape(a) for a in plain]))
        _C.size(n, what)
        _C.charge(n, what)
        return _wrap(fn(*plain), any(_is_data(a) for a in args), what)
    f.__name__ = name
    return f


def _reduction(name):
    def f(a, axis=None, keepdims=False):
        return _reduce(name, a, axis, keepdims, f"np.{name}")
    f.__name__ = name
    return f


def _np_matmul(a, b):
    return _ufunc(_np.matmul, (a, b), "np.matmul")


def _np_einsum(subscripts, *operands):
    what = "np.einsum"
    if not isinstance(subscripts, str) or "." in subscripts or "->" not in subscripts:
        raise PBRuntimeError(f"{what}: the subscripts must be a constant string with explicit output, e.g. "
                             "'oc,chw->ohw' (no '...')")
    if not 1 <= len(operands) <= 2:
        raise PBRuntimeError(f"{what} takes one or two operands in player code")
    _contract_check(operands, what)
    plain = [_np.asarray(_plain(o)) for o in operands]
    lhs, out = subscripts.replace(" ", "").split("->")
    terms = lhs.split(",")
    if len(terms) != len(plain) or any(len(t) != p.ndim for t, p in zip(terms, plain)):
        raise PBRuntimeError(f"{what}: subscripts {subscripts!r} do not match the operand shapes "
                             f"{[p.shape for p in plain]}")
    sizes: dict = {}
    for t, p in zip(terms, plain):
        for ch, n in zip(t, p.shape):
            sizes[ch] = max(sizes.get(ch, 1), int(n))
    n_out = _prod(sizes.get(ch, 1) for ch in out)
    _C.size(n_out, what)
    _C.charge(_prod(sizes.values()), what)
    r = _np.einsum(subscripts, *plain, optimize=len(plain) > 1)
    return _wrap(r, any(_is_data(o) for o in operands), what)


def _seq(what, fn):
    def f(arrays, axis=0):
        _static(axis, "the axis")
        if not isinstance(arrays, (list, tuple)):
            raise PBRuntimeError(f"{what} takes a list or tuple of arrays")
        plain = [_np.asarray(_plain(a)) for a in arrays]
        n = sum(p.size for p in plain)
        _C.size(n, what)
        _C.charge(n, what)
        return _wrap(fn(plain, axis=axis), _is_data(arrays), what)
    return f


def _np_reshape(a, shape):
    return _same("np.reshape", a, lambda p: _np.reshape(p, _static(shape, "the shape")))


def _np_transpose(a, axes=None):
    return _same("np.transpose", a, lambda p: _np.transpose(p, _static(axes, "the axes")))


def _np_broadcast_to(a, shape):
    _static(shape, "the shape")
    n = _prod(shape if isinstance(shape, (tuple, list)) else (shape,))
    _C.size(n, "np.broadcast_to")
    return _same("np.broadcast_to", a, lambda p: _np.broadcast_to(p, shape))


def _np_expand_dims(a, axis):
    return _same("np.expand_dims", a, lambda p: _np.expand_dims(p, _static(axis, "the axis")))


def _np_squeeze(a, axis=None):
    return _same("np.squeeze", a, lambda p: _np.squeeze(p, _static(axis, "the axis")))


def _np_roll(a, shift, axis=None):
    return _same("np.roll", a, lambda p: _np.roll(p, _static(shift, "the shift"), _static(axis, "the axis")))


def _np_flip(a, axis=None):
    return _same("np.flip", a, lambda p: _np.flip(p, _static(axis, "the axis")))


def _np_pad(array, pad_width, mode="constant", constant_values=0):
    what = "np.pad"
    _static(pad_width, "the pad width")
    if mode != "constant":
        raise PBRuntimeError(f"{what}: only mode='constant' is allowed")
    if not isinstance(constant_values, (int, float)) or isinstance(constant_values, bool):
        raise PBRuntimeError(f"{what}: constant_values must be a Python number")
    p = _np.asarray(_plain(array))
    w = _np.zeros((p.ndim, 2), _np.int64)
    w[...] = _np.asarray(pad_width, dtype=_np.int64)
    _C.size(_prod(_np.asarray(p.shape) + w.sum(1)), what)
    return _same(what, array, lambda q: _np.pad(q, pad_width, mode="constant", constant_values=constant_values))


def _creation(what, fn, n):
    _C.size(n, what)
    _C.charge(n, what)
    return _wrap(fn(), False, what)


def _shape_size(shape):
    _static(shape, "the shape")
    return _prod(shape if isinstance(shape, (tuple, list)) else (shape,))


def _np_zeros(shape, dtype=None):
    dt = _dtype(dtype)
    return _creation("np.zeros", lambda: _np.zeros(shape, dt), _shape_size(shape))


def _np_ones(shape, dtype=None):
    dt = _dtype(dtype)
    return _creation("np.ones", lambda: _np.ones(shape, dt), _shape_size(shape))


def _np_full(shape, fill_value, dtype=None):
    dt = _dtype(dtype)
    if not isinstance(fill_value, (int, float)):
        raise PBRuntimeError("np.full: fill_value must be a Python number (use np.where or broadcasting for arrays)")
    return _creation("np.full", lambda: _np.full(shape, fill_value, dt), _shape_size(shape))


def _np_zeros_like(a, dtype=None):
    dt = _dtype(dtype)
    p = _np.asarray(_plain(a))
    return _creation("np.zeros_like", lambda: _np.zeros_like(p, dt), p.size)


def _np_ones_like(a, dtype=None):
    dt = _dtype(dtype)
    p = _np.asarray(_plain(a))
    return _creation("np.ones_like", lambda: _np.ones_like(p, dt), p.size)


def _np_eye(N, M=None, k=0, dtype=None):
    _static((N, M, k), "the arguments of np.eye")
    dt = _dtype(dtype)
    return _creation("np.eye", lambda: _np.eye(N, M, k, dtype=dt or float), int(N) * int(N if M is None else M))


def _np_arange(*args, dtype=None):
    _static(args, "the arguments of np.arange")
    dt = _dtype(dtype)
    r = _np.arange(*args, dtype=dt)
    return _creation("np.arange", lambda: r, r.size)


def _np_array(obj, dtype=None):
    dt = _dtype(dtype)
    what = "np.array"
    r = _np.array(_plain(obj), dtype=dt)
    _C.size(r.size, what)
    _C.charge(r.size, what)
    return _wrap(r, _is_data(obj), what)


NP_FUNCS = {
    # elementwise
    "maximum": _elementwise("maximum", _np.maximum, 2), "minimum": _elementwise("minimum", _np.minimum, 2),
    "abs": _elementwise("abs", _np.abs, 1), "tanh": _elementwise("tanh", _np.tanh, 1),
    "exp": _elementwise("exp", _np.exp, 1), "where": _elementwise("where", _np.where, 3),
    "clip": lambda a, a_min, a_max: _clip(a, a_min, a_max, "np.clip"),
    # reductions over constant axes
    "sum": _reduction("sum"), "max": _reduction("max"), "min": _reduction("min"), "mean": _reduction("mean"),
    # contractions (one operand must be constant)
    "matmul": _np_matmul, "einsum": _np_einsum,
    # structure (constant shapes, axes, shifts, widths)
    "concatenate": _seq("np.concatenate", _np.concatenate), "stack": _seq("np.stack", _np.stack),
    "reshape": _np_reshape, "transpose": _np_transpose, "broadcast_to": _np_broadcast_to,
    "expand_dims": _np_expand_dims, "squeeze": _np_squeeze, "roll": _np_roll, "flip": _np_flip, "pad": _np_pad,
    # creation
    "zeros": _np_zeros, "ones": _np_ones, "full": _np_full, "zeros_like": _np_zeros_like,
    "ones_like": _np_ones_like, "eye": _np_eye, "arange": _np_arange, "array": _np_array, "asarray": _np_array,
}
NP_CONSTS = {"float32": _np.float32, "int32": _np.int32, "int64": _np.int64, "bool_": _np.bool_,
             "newaxis": None, "pi": math.pi, "e": math.e}
METHODS = ("sum", "max", "min", "mean", "reshape", "transpose", "astype", "copy", "ravel", "flatten", "clip",
           "squeeze", "swapaxes")
ATTRS = ("T",)


class _NPProxy:
    """The `np` of player code."""

    def __init__(self, entries):
        object.__setattr__(self, "_entries", dict(entries))

    def __getattr__(self, name):
        try:
            return self._entries[name]
        except KeyError:
            raise PBRuntimeError(f"np.{name} is not available in player code") from None

    def __setattr__(self, name, value):
        raise PBRuntimeError("np cannot be modified")


NP = _NPProxy({**NP_FUNCS, **NP_CONSTS})


# --------------------------------------------------------------------------- constants
def freeze(ns: dict, names) -> tuple[dict, dict, int]:
    """Module-level values used by player code -> (JSON trees by name, arrays by key, number count).
    Floating values are cast to float32 and must be finite; lists become tuples, dicts read-only."""
    arrays: dict = {}
    count = [0]

    def enc(v, where, depth=0):
        if depth > 32:
            raise PBRuntimeError(f"{where}: nested too deeply")
        if v is None:
            return ["z"]
        if isinstance(v, (bool, _np.bool_)):
            count[0] += 1
            return ["b", bool(v)]
        if isinstance(v, (int, _np.integer)):
            count[0] += 1
            return ["i", int(v)]
        if isinstance(v, (float, _np.floating)):
            with _np.errstate(over="ignore"):
                f = float(_np.float32(v))
            if not math.isfinite(f):
                raise PBRuntimeError(f"constant {where}: non-finite value after the float32 cast")
            count[0] += 1
            return ["f", f]
        if isinstance(v, str):
            count[0] += len(v)
            return ["s", v]
        if isinstance(v, _np.ndarray):
            a = _np.asarray(v)
            if a.dtype.kind == "f":
                with _np.errstate(over="ignore"):
                    a = a.astype(_np.float32)
                if not _np.isfinite(a).all():
                    raise PBRuntimeError(f"constant {where}: non-finite values after the float32 cast")
            elif a.dtype.kind not in "biu":
                raise PBRuntimeError(f"constant {where}: arrays of dtype {a.dtype} cannot be used by player code")
            key = f"c{len(arrays)}"
            arrays[key] = _np.ascontiguousarray(a)
            count[0] += int(a.size)
            return ["a", key]
        if isinstance(v, (tuple, list)):
            return ["t", [enc(x, f"{where}[{i}]", depth + 1) for i, x in enumerate(v)]]
        if isinstance(v, (dict, MappingProxyType)):
            items = []
            for k, x in v.items():
                if isinstance(k, bool) or not isinstance(k, (str, int)):
                    raise PBRuntimeError(f"constant {where}: dict keys must be strings or integers")
                items.append([k, enc(x, f"{where}[{k!r}]", depth + 1)])
            return ["d", items]
        raise PBRuntimeError(f"player code uses the module-level value {where} of type {type(v).__name__}; values "
                             "used by player code must be numbers, strings, arrays, or tuples, lists or dicts of them")

    trees = {}
    for n in names:
        if n not in ns:
            raise PBRuntimeError(f"name {n!r} used by player code is not defined at module level")
        trees[n] = enc(ns[n], n)
    return trees, arrays, count[0]


def _const(a):
    a = _np.array(a)
    a.flags.writeable = False
    b = a.view(BArray)
    b._data = False
    return b


def thaw(tree, arrays):
    tag = tree[0]
    if tag == "z":
        return None
    if tag in ("b", "i", "f", "s"):
        return tree[1]
    if tag == "a":
        return _const(arrays[tree[1]])
    if tag == "t":
        return tuple(thaw(x, arrays) for x in tree[1])
    if tag == "d":
        return MappingProxyType({k: thaw(x, arrays) for k, x in tree[1]})
    raise ValueError(tree)


# --------------------------------------------------------------------------- loading and stepping
class _Rewrite(ast.NodeTransformer):
    """Annotations are dropped; `x op= v` becomes `x = x op v` (arrays are never modified in place)."""

    def visit_FunctionDef(self, node):
        node.returns = None
        node.decorator_list = []
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            a.annotation = None
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        name = node.target.id
        new = ast.Assign(targets=[ast.Name(name, ast.Store())],
                         value=ast.BinOp(ast.Name(name, ast.Load()), node.op, node.value))
        return ast.copy_location(new, node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def load_player(source: str, spec: dict, trees: dict, arrays: dict):
    """act() of a checked program: only its player-code functions, with the frozen constants and NP."""
    tree = ast.parse(source, SOURCE_NAME)
    keep = set(spec["runtime"])
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    mod = ast.fix_missing_locations(_Rewrite().visit(ast.Module(body=defs, type_ignores=[])))
    ns = {"__builtins__": {"range": range}, "__name__": "player"}
    for a in spec["np_aliases"]:
        ns[a] = NP
    for name, t in trees.items():
        ns[name] = thaw(t, arrays)
    exec(compile(mod, SOURCE_NAME, "exec"), ns)
    return ns["act"]


def _input(x, dtype):
    b = _np.array(x, dtype=dtype).view(BArray)
    b._data = True
    return b


def check_outputs(out, M: int):
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise PBRuntimeError("act must return a tuple (scores, click, memory_next)")
    res = []
    for v, name, shape in zip(out, ("scores", "click", "memory_next"), ((8,), (64, 64), (M,))):
        a = _np.asarray(_plain(v))
        if a.dtype.kind not in "biuf" or a.shape != shape:
            raise PBRuntimeError(f"{name} must be a numeric array of shape {shape}, got {a.dtype} {a.shape}")
        res.append(a.astype(_np.float32))
    return res


def call_act(act, M, obs, avail, memory, prev):
    """One counted call of act -> float32 (scores (8,), click (64, 64), memory_next (M,))."""
    _C.reset()
    _C.active = True
    try:
        with _np.errstate(all="ignore"):
            out = act(_input(obs, _np.int64), _input(avail, _np.float32), _input(memory, _np.float32),
                      _input(prev, _np.float32))
    finally:
        _C.active = False
    return check_outputs(out, M)


def decode(scores, click, avail):
    """As wsp.freearch.reference_step: unavailable actions get -inf, argmax over the rest (ties: lowest index);
    ACTION6 -> argmax cell of click (ties: first in row-major order); a non-finite available score -> invalid."""
    av = _np.asarray(avail) > 0
    s = _np.where(av, scores, -_np.inf)
    a = int(_np.argmax(s))
    if not _np.isfinite(s[av]).all():
        return ("NONFINITE", None, None)
    if a == CLICK:
        cell = int(_np.argmax(click))
        return (ACTIONS[a], cell // 64, cell % 64)
    return (ACTIONS[a], None, None)


def step(act, M, obs, avail, memory, prev):
    """One harness step: (move, scores, click, memory_next, prev_next)."""
    scores, click, mem_next = call_act(act, M, obs, avail, memory, prev)
    move = decode(scores, click, avail)
    prev_next = _np.zeros(len(ACTIONS), _np.float32)
    if move[0] in ACTIONS:
        prev_next[ACTIONS.index(move[0])] = 1.0
    return move, scores, click, mem_next, prev_next


def _num(x, nd=3):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def outputs_summary(scores, click, mem_next, avail) -> dict:
    """Feedback internals: masked scores, [max, row, col] of click, memory norm."""
    av = _np.asarray(avail) > 0
    idx = int(_np.argmax(click))
    return {"scores": [_num(v) if a else None for v, a in zip(scores, av)],
            "click_max_at": [_num(click.max()), idx // 64, idx % 64],
            "memory_l2": _num(_np.linalg.norm(mem_next.astype(_np.float64))) if mem_next.size else 0.0}


def describe(exc: BaseException) -> str:
    """Writer-facing error: exception text plus the last line of the writer's code involved."""
    lines = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == SOURCE_NAME]
    where = f" (line {lines[-1]})" if lines else ""
    msg = str(exc) if isinstance(exc, PBRuntimeError) else f"{type(exc).__name__}: {exc}"
    return (msg + where)[:1500]


def writer_traceback(exc: BaseException, source: str = "") -> str:
    """Traceback restricted to the writer's own frames (no harness or library paths)."""
    src = source.splitlines()
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == SOURCE_NAME]
    lines = ["Traceback (writer code):"] + [
        f'  File "{SOURCE_NAME}", line {f.lineno}, in {f.name}\n    '
        + (src[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(src) else "") for f in frames]
    return "\n".join(lines + [f"{type(exc).__name__}: {exc}"])[-1500:]


def probe(act, M, frames) -> dict:
    """Test calls of act at submission (memory and previous action chained) -> counts and timing."""
    memory = _np.zeros(M, _np.float32)
    prev = _np.zeros(len(ACTIONS), _np.float32)
    calls = []
    for obs, avail in frames:
        t = time.perf_counter()
        move, scores, click, mem_next, prev_next = step(act, M, obs, avail, memory, prev)
        calls.append({"ops": _C.ops, "work": _C.work, "max_array": _C.max_array,
                      "s": round(time.perf_counter() - t, 4), "move": list(move)})
        memory, prev = mem_next, prev_next
    return {"ops": max(c["ops"] for c in calls), "work": max(c["work"] for c in calls),
            "max_array": max(c["max_array"] for c in calls),
            "counts_equal": len({(c["ops"], c["work"]) for c in calls}) == 1, "calls": calls}


def probe_frames():
    """Two fixed test inputs: an empty frame with every action available, then a patterned frame."""
    obs1 = _np.zeros((64, 64), _np.int64)
    obs2 = ((_np.arange(64)[:, None] * 7 + _np.arange(64)[None, :] * 3) % 16).astype(_np.int64)
    return [(obs1, _np.ones(8, _np.float32)), (obs2, _np.array([1, 1, 1, 1, 0, 1, 1, 0], _np.float32))]
