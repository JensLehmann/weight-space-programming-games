"""Executed in a separate `python -I` process with an empty cwd and scrubbed environment.

Modes:
  init   <source> <out.npz>   : exec writer source, call build() -> dict of arrays, save npz
  player <source>             : exec writer source, then serve act() over JSON lines on stdin/stdout
"""
import base64
import builtins
import json
import sys
import time

import numpy as _np

ALLOWED_MODULES = {"numpy", "math", "itertools", "functools", "collections", "heapq", "random",
                   "dataclasses", "typing", "enum", "copy", "operator", "statistics", "bisect",
                   # standard modules that the prompt permits
                   "time", "array", "re", "string", "fractions", "numbers", "abc", "warnings",
                   "struct", "decimal"}
DENIED_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                   "globals", "locals", "vars", "memoryview", "exit", "quit", "help"}
_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not permitted in the sandbox")
    return _real_import(name, globals, locals, fromlist, level)


def _namespace():
    safe = {k: getattr(builtins, k) for k in dir(builtins) if k not in DENIED_BUILTINS}
    safe["__import__"] = _guarded_import
    safe["__build_class__"] = builtins.__build_class__
    return {"__builtins__": safe, "__name__": "writer_module"}


def _load(path):
    with _real_open(path) as f:
        src = f.read()
    ns = _namespace()
    exec(compile(src, "writer_source.py", "exec"), ns)
    return ns


_real_open = builtins.open


def main():
    mode, src = sys.argv[1], sys.argv[2]
    t0 = time.process_time()
    ns = _load(src)
    if mode in ("init", "edit"):
        out = sys.argv[3]
        if mode == "init":
            params = ns["build"]()
        else:
            with _np.load(sys.argv[4]) as z:
                given = {k: z[k].astype(_np.float32).copy() for k in z.files}
            params = ns["edit"](given)
        if not isinstance(params, dict):
            raise TypeError("build() must return a dict of arrays")
        arrays = {str(k): _np.asarray(v, dtype=_np.float64) for k, v in params.items()}
        _np.savez(out, **arrays)
        sys.stdout.write(json.dumps({"ok": True, "cpu_s": time.process_time() - t0}) + "\n")
        return
    act = ns["act"]
    memory = {}
    for line in sys.stdin:
        msg = json.loads(line)
        obs = _np.frombuffer(base64.b64decode(msg["obs"]), dtype=_np.int8).reshape(64, 64).astype(_np.int64)
        t = time.perf_counter()
        try:
            move = act(obs, list(msg["avail"]), memory)
            a = move[0]
            r = None if move[1] is None else int(move[1])
            c = None if move[2] is None else int(move[2])
            reply = {"move": [str(a), r, c], "s": time.perf_counter() - t}
        except Exception as exc:  # reported to the evaluator as a player error
            reply = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
