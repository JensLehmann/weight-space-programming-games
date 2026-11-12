"""Free-architecture build() runner, executed as `python -I _sandbox_runner_fa.py <source> <out.npz> <out.json>`
in a separate process with an empty cwd and scrubbed environment (see wsp.sandbox / wsp.freearch).

The writer source is executed in the guarded namespace of `_sandbox_runner.py` (same module allowlist and
restricted builtins as the weights build()). build() must return
{"graph": [...], "memory_size": M, "params": {name: array}}; the graph and memory size are written as
JSON, the parameters (as float64, transport keys "p_" + name) as npz.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as _np


def _base_runner():
    spec = importlib.util.spec_from_file_location("_sandbox_runner", Path(__file__).with_name("_sandbox_runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plain(x, path):
    """Writer-built graph -> plain JSON types (numpy scalars/arrays and tuples are converted)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: dict keys must be strings, got {k!r}")
            out[k] = _plain(v, f"{path}.{k}")
        return out
    if isinstance(x, (list, tuple)):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(x)]
    if isinstance(x, _np.ndarray):
        return _plain(x.tolist(), path)
    if isinstance(x, (bool, _np.bool_)):
        return bool(x)
    if isinstance(x, (int, _np.integer)):
        return int(x)
    if isinstance(x, (float, _np.floating)):
        return float(x)
    if x is None or isinstance(x, str):
        return x
    raise TypeError(f"{path}: unsupported type {type(x).__name__} (use dict/list/str/int/float/bool/None)")


def main():
    src, out_npz, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.process_time()
    ns = _base_runner()._load(src)
    res = ns["build"]()
    if not isinstance(res, dict) or set(res) != {"graph", "memory_size", "params"}:
        got = sorted(map(str, res)) if isinstance(res, dict) else type(res).__name__
        raise TypeError(f"build() must return a dict with exactly the keys 'graph', 'memory_size', 'params' "
                        f"(got {got})")
    if not isinstance(res["params"], dict):
        raise TypeError("build()['params'] must be a dict name -> array")
    arch = {"graph": _plain(res["graph"], "graph"), "memory_size": _plain(res["memory_size"], "memory_size")}
    arrays = {"p_" + str(k): _np.asarray(v, dtype=_np.float64) for k, v in res["params"].items()}
    _np.savez(out_npz, **arrays)
    with open(out_json, "w") as f:
        json.dump(arch, f)
    sys.stdout.write(json.dumps({"ok": True, "cpu_s": time.process_time() - t0}) + "\n")


if __name__ == "__main__":
    main()
