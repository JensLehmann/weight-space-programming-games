"""Static audit and isolated execution of writer-produced code (initializers and Python players).

Threat model: non-adversarial writer models that may accidentally access data, fit parameters,
or embed lookup tables. Enforcement layers: AST allowlist, restricted builtins + import hook,
separate `python -I` process with empty cwd, scrubbed environment and CPU/wall limits.
Flags are recorded for manual classification; they do not by themselves reject an artifact.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("_sandbox_runner.py")
ALLOWED_MODULES = {"numpy", "math", "itertools", "functools", "collections", "heapq", "random",
                   "dataclasses", "typing", "enum", "copy", "operator", "statistics", "bisect",
                   # standard modules that the prompt permits
                   "time", "array", "re", "string", "fractions", "numbers", "abc", "warnings",
                   "struct", "decimal"}
DENIED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                "locals", "vars", "memoryview", "exit", "quit", "help", "getattr", "setattr",
                "delattr"}
DENIED_ATTRS = {"load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt",
                "savetxt", "genfromtxt", "memmap", "DataSource", "ctypeslib", "fromregex",
                "npyio", "format", "ctypes", "system", "popen"}
FIT_ATTRS = {"lstsq", "solve", "pinv", "inv", "svd", "eig", "eigh", "qr", "cholesky", "polyfit",
             "tensorsolve", "tensorinv"}
ENV_KEEP = {"PATH", "LANG", "LC_ALL", "TMPDIR"}


@dataclass
class Audit:
    ok: bool
    violations: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    source_bytes: int = 0
    literal_numbers: int = 0
    loops: int = 0

    def to_json(self) -> dict:
        return self.__dict__.copy()


def audit(source: str, entry: str) -> Audit:
    a = Audit(ok=True, source_bytes=len(source.encode()))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Audit(ok=False, violations=[f"syntax error: {exc}"], source_bytes=a.source_bytes)
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                a.violations.append("relative import")
            for n in names:
                if n.split(".")[0] not in ALLOWED_MODULES:
                    a.violations.append(f"import {n}")
        elif isinstance(node, ast.Name) and (node.id in DENIED_NAMES or node.id.startswith("__")):
            a.violations.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                a.violations.append(f"dunder attribute {node.attr}")
            elif node.attr in DENIED_ATTRS:
                a.violations.append(f"attribute {node.attr}")
            elif node.attr in FIT_ATTRS:
                a.flags.append(f"linear-algebra solver/decomposition: {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            a.literal_numbers += 1
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            a.loops += 1
            if isinstance(node, ast.While):
                a.flags.append("while loop")
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
    if entry not in defined:
        a.violations.append(f"missing top-level function {entry}()")
    if a.literal_numbers > 1000:
        a.flags.append(f"{a.literal_numbers} numeric literals (possible embedded table)")
    a.ok = not a.violations
    return a


def _limits(cpu_s: int):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        os.setsid()
    return fn


def _env(home: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP}
    env.update({"HOME": home, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    return env


@dataclass
class InitResult:
    params: dict | None
    error: str | None
    audit: Audit
    wall_s: float = 0.0
    cpu_s: float | None = None


def run_initializer(source: str, timeout: int = 600, base: dict | None = None) -> InitResult:
    """Run `build()` (base is None) or `edit(base_params)` and return the produced tensors."""
    au = audit(source, "build" if base is None else "edit")
    if not au.ok:
        return InitResult(None, "audit: " + "; ".join(au.violations[:10]), au)
    with tempfile.TemporaryDirectory(prefix="wsp_init_") as tmp:
        src = Path(tmp) / "writer_source.py"
        src.write_text(source)
        out = Path(tmp) / "out.npz"
        work = Path(tmp) / "work"
        work.mkdir()
        t = time.time()
        cmd = [sys.executable, "-I", str(RUNNER), "init", str(src), str(out)]
        if base is not None:
            inp = Path(tmp) / "base.npz"
            np.savez(inp, **base)
            cmd = [sys.executable, "-I", str(RUNNER), "edit", str(src), str(out), str(inp)]
        try:
            proc = subprocess.run(cmd,
                                  cwd=work, env=_env(str(work)), capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits(timeout))
        except subprocess.TimeoutExpired:
            return InitResult(None, f"timeout after {timeout}s", au, time.time() - t)
        wall = time.time() - t
        if proc.returncode != 0 or not out.exists():
            return InitResult(None, "runtime: " + proc.stderr.strip()[-1500:], au, wall)
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        with np.load(out) as z:
            params = {k: z[k] for k in z.files}
        if info.get("cpu_s", 0) > 30:
            au.flags.append(f"initializer used {info['cpu_s']:.0f}s CPU")
        return InitResult(params, None, au, wall, info.get("cpu_s"))


class PythonPlayerPolicy:
    """One fresh sandboxed process per episode (no cross-episode memory)."""

    def __init__(self, source: str, step_timeout: float = 10.0, episode_cpu_s: int = 600):
        self.audit = audit(source, "act")
        if not self.audit.ok:
            raise ValueError("audit: " + "; ".join(self.audit.violations[:10]))
        self.tmp = tempfile.TemporaryDirectory(prefix="wsp_player_")
        self.src = Path(self.tmp.name) / "writer_source.py"
        self.src.write_text(source)
        self.step_timeout = step_timeout
        self.episode_cpu_s = episode_cpu_s
        self.procs: dict[int, subprocess.Popen] = {}
        self.latency: list[float] = []

    def _spawn(self):
        work = tempfile.mkdtemp(dir=self.tmp.name)
        return subprocess.Popen([sys.executable, "-I", str(RUNNER), "player", str(self.src)],
                                cwd=work, env=_env(work), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                preexec_fn=_limits(self.episode_cpu_s))

    def begin(self, slots):
        for i in slots:
            if i in self.procs:
                self.procs[i].kill()
            self.procs[i] = self._spawn()

    def act(self, slots, obs, avail):
        from .envs import ACTIONS
        import select
        for j, i in enumerate(slots):
            names = [ACTIONS[k] for k in np.nonzero(avail[j])[0]]
            msg = {"obs": base64.b64encode(obs[j].astype(np.int8).tobytes()).decode(), "avail": names}
            try:
                self.procs[i].stdin.write(json.dumps(msg) + "\n")
                self.procs[i].stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        moves = []
        for j, i in enumerate(slots):
            p = self.procs[i]
            ready, _, _ = select.select([p.stdout], [], [], self.step_timeout)
            if not ready:
                p.kill()
                moves.append(("TIMEOUT", None, None))
                continue
            line = p.stdout.readline()
            if not line:
                moves.append(("CRASH", None, None))
                continue
            reply = json.loads(line)
            if "error" in reply:
                moves.append(("ERROR", None, None))
            else:
                self.latency.append(reply["s"])
                moves.append(tuple(reply["move"]))
        return moves

    def close(self):
        for p in self.procs.values():
            p.kill()
        self.tmp.cleanup()
