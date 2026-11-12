"""Bounded program player (pbounded) runner, executed as `python -I _sandbox_runner_pb.py <mode> ...` in a separate
process with an empty cwd and scrubbed environment (see wsp.sandbox / wsp.pbounded).

Modes:
  build  <source> <spec.json> <out.npz> <out.json>
      Runs the module-level code once in the guarded namespace of _sandbox_runner.py (restricted builtins; numpy is
      the only importable module), freezes the module-level values that player code uses (wsp._pb_runtime.freeze),
      then makes two test calls of act on a fresh player namespace (exactly as in player mode) and reports their
      operation counts. Writes the constant arrays (npz) and their structure (json); prints one JSON line.
  player <source> <spec.json> <consts.npz> <consts.json>
      Serves one episode over JSON lines: {"obs": base64 int8 (64x64), "avail": [8 x 0/1], "outputs": bool}
      -> {"move": [action, row, col], "s", "ops", "work"} (+ masked scores, click maximum and memory norm when
      "outputs" is set) or {"error": ...}. Memory and previous action live here (zeros at the start).
"""
import base64
import builtins
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as _np


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RT = _sibling("_pb_runtime")
BASE = _sibling("_sandbox_runner")


def _numpy_only_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] != "numpy":
        raise ImportError(f"import of {name!r} is not permitted (only numpy)")
    return BASE._real_import(name, globals, locals, fromlist, level)


def _construction_namespace():
    safe = {k: getattr(builtins, k) for k in dir(builtins) if k not in BASE.DENIED_BUILTINS}
    safe["__import__"] = _numpy_only_import
    safe["__build_class__"] = builtins.__build_class__
    return {"__builtins__": safe, "__name__": "writer_module"}


def _read(path):
    with BASE._real_open(path) as f:
        return f.read()


def _load_consts(npz, js):
    with _np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return json.loads(_read(js)), arrays


def build(src_path, spec_path, out_npz, out_json):
    source, spec = _read(src_path), json.loads(_read(spec_path))
    t0 = time.process_time()
    ns = _construction_namespace()
    try:
        exec(compile(source, RT.SOURCE_NAME, "exec"), ns)
    except Exception as exc:                                        # error in the writer's module-level code
        print(json.dumps({"ok": False, "stage": "load", "error": RT.writer_traceback(exc, source)}))
        return
    cpu = time.process_time() - t0
    try:
        for name in spec["runtime"]:
            f = ns.get(name)
            code = getattr(f, "__code__", None)
            if code is None or code.co_filename != RT.SOURCE_NAME or code.co_firstlineno != spec["def_lines"][name]:
                raise RT.PBRuntimeError(f"function {name} was replaced at module level after its definition")
        if ns.get("MEMORY_SIZE") != spec["memory_size"]:
            raise RT.PBRuntimeError("MEMORY_SIZE was changed by the module-level code")
        trees, arrays, n_numbers = RT.freeze(ns, spec["globals"])
    except RT.PBRuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return
    _np.savez(out_npz, **arrays)
    with BASE._real_open(out_json, "w") as f:
        json.dump(trees, f)
    RT.install(spec["caps"], hard=False)
    try:
        act = RT.load_player(source, spec, *_load_consts(out_npz, out_json))   # exactly as in player mode
        pr = RT.probe(act, spec["memory_size"], RT.probe_frames())
    except RT.BudgetExceeded as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "act failed on a test call: " + RT.describe(exc)}))
        return
    print(json.dumps({"ok": True, "cpu_s": cpu, "numbers": n_numbers, "probe": pr}))


def player(src_path, spec_path, npz, js):
    source, spec = _read(src_path), json.loads(_read(spec_path))
    RT.install(spec["caps"], hard=True)
    act = RT.load_player(source, spec, *_load_consts(npz, js))
    C = RT.counter()
    M = spec["memory_size"]
    memory = _np.zeros(M, _np.float32)
    prev = _np.zeros(len(RT.ACTIONS), _np.float32)
    for line in sys.stdin:
        msg = json.loads(line)
        obs = _np.frombuffer(base64.b64decode(msg["obs"]), dtype=_np.int8).reshape(64, 64).astype(_np.int64)
        avail = _np.asarray(msg["avail"], _np.float32)
        t = time.perf_counter()
        try:
            move, scores, click, mem_next, prev_next = RT.step(act, M, obs, avail, memory, prev)
            reply = {"move": list(move), "s": time.perf_counter() - t, "ops": C.ops, "work": C.work}
            if msg.get("outputs"):
                reply.update(RT.outputs_summary(scores, click, mem_next, avail))
            memory, prev = mem_next, prev_next
        except Exception as exc:                 # reported to the evaluator as a player error (invalid action)
            reply = {"error": RT.describe(exc)[:400]}
            prev = _np.zeros(len(RT.ACTIONS), _np.float32)
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        build(*sys.argv[2:6])
    elif mode == "player":
        player(*sys.argv[2:6])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
