"""Prompt for the free-architecture (FA) condition: the writer designs the network (wsp.freearch language)
and supplies all weights. Game section, episode rules and support observations are byte-identical to the
revision-2 weights prompt (wsp.prompts.build_prompt_v2); only the actor spec and the task differ.
"""
from __future__ import annotations

import inspect

import numpy as np

from . import actor as A
from . import freearch as FA
from .envs import ACTIONS
from .prompts import WEIGHTS_TASK, actor_spec, build_prompt_v2, feedback_message


def fa_spec() -> str:
    ref = "\n\n".join(inspect.getsource(f) for f in (FA.np_op, FA.run_graph, FA.reference_step))
    example = inspect.getsource(FA.fixed_actor_arch)
    shapes = "\n".join(f"    {k:4s} {tuple(v)}" for k, v in FA.fixed_actor_shapes().items())
    c = FA.count(FA.fixed_actor_arch(), FA.fixed_actor_params(A.zero_params()))
    return f"""## The actor network (you design its architecture and choose all its numbers)

The actor is a neural network that you define as a static graph of tensor operations in the small
language below. At every step it receives only the current 64x64 observation (colour ids 0-15), its
own memory vector (zeros at the start of every episode), a one-hot of its previous action (zeros at
episode start) and the mask of available actions. It outputs action logits, a click map and its next
memory. The actor runs alone: no other code, search, or lookup exists at deployment. Action index
order: {list(ACTIONS)} (index 6 = ACTION6 = click).

A network is `{{"graph": [node, ...], "memory_size": M}}` plus a dict `params` of named arrays
(cast to float32). The graph must consist of JSON-like values (dicts, lists, strings, numbers).

**Tensors and inputs.** Spatial tensors have shape (C, 64, 64); vectors have shape (n,). Inputs:
- `x0` (16, 64, 64): one-hot colour planes, x0[c, i, j] = 1 if obs[i, j] == c else 0;
- `coords` (3, 64, 64): row/63, col/63, constant 1;
- `mem` (M,): your memory, 0 <= M <= {FA.MAX_MEM} chosen by you (absent if M = 0); zeros at episode start;
- `aprev` (8,): one-hot of the previous action (zeros at episode start);
- `avail` (8,): 1.0 for available actions, else 0.0.

**Nodes.** Each node is a dict `{{"id": name, "op": op, "in": [ids], ...op fields}}` (an optional
"comment" string is allowed; any other field is an error). Nodes run in list order and may only use
inputs and earlier nodes (no cycles). Ids match `[A-Za-z_][A-Za-z0-9_]*` and are unique across the
whole graph, including loop bodies and loop states. Parameters are referenced by name in the fields
"w" and "b" ("b" is optional: omit it for no bias). Every array in params must be used; one array may
be used by several nodes.

**Ops** (the NumPy reference below is the exact definition):
- `conv` (1 spatial input; w (O, C, k, k) with k in {{1, 3, 5, 7}}, optional b (O,)): cross-correlation,
  stride 1, zero padding k//2, output (O, 64, 64);
- `relu`, `tanh`, `sigmoid`, `abs` (1 spatial): elementwise;
- `add`, `sub`, `mul`, `max2`, `min2` (2 inputs, at least one spatial): elementwise with NumPy
  broadcasting, a vector (C,) acting as (C, 1, 1); channel counts must be equal or one of them 1;
- `concat` (1 or more spatial): channel concatenation; `slice` (1 spatial; integer fields "start",
  "stop"): channels start..stop-1;
- `rowmax`, `colmax`, `rowsum`, `colsum` (1 spatial): max / sum of each channel over its row (resp.
  column), broadcast back to (C, 64, 64); `rowmax` at (c, i, j) = max over j' of x[c, i, j'];
- `gmax`, `gmean` (1 spatial): max / mean over all 4096 positions, giving a vector (C,);
- `dense` (1 vector; w (m, n), optional b (m,)): w @ x + b;
- `vrelu`, `vtanh`, `vsigmoid` (1 vector); `vadd`, `vmul` (2 vectors of equal length, or one of
  length 1); `vconcat` (1 or more vectors); `vslice` (1 vector; "start", "stop");
- `loop` (fields "T", "state", "body", "next"; no "in"): runs the node list "body" T times
  (1 <= T <= {FA.MAX_T}) with the same parameters every iteration. "state" lists the carried tensors, each
  `{{"id": s, "init": id}}` (start from a tensor defined before the loop) or `{{"id": s, "zeros": [C, 64, 64]}}`
  / `{{"id": s, "zeros": [n]}}`. Body nodes may use the states, earlier body nodes and anything
  defined before the loop. "next" maps every state id to the id holding its value for the next
  iteration; all states update together. After the loop, `"<loop id>.<state id>"` is a state's final
  value (body ids are not visible outside the loop). Loops cannot be nested.
There are no other ops, no data-dependent control flow and no argmax inside the network.

**Outputs** (top-level nodes with these ids): `logits` (8,); `click` (1, 64, 64); `mem_next` (M,)
(omit it if M = 0), which becomes `mem` at the next step exactly as computed (nothing squashes it).

**Decoding** is deterministic and outside the network: unavailable actions get logit -inf and the
argmax over the available actions is taken (ties go to the lowest index). If ACTION6 wins, the click
target is the argmax cell of `click` (ties go to the first cell in row-major order). If any available
logit is NaN or infinite, the step is an invalid action.

**Budget and limits** (checked when build() returns; any violation makes the network invalid):
- at most {FA.MAX_PARAMS:,} parameters (entries of all arrays in params);
- at most {FA.MAX_MACS / 1e9:g}e9 = {FA.MAX_MACS:,.0f} multiply-accumulates (MACs) per step, counted as:
  conv O*C*k*k*4096; dense m*n; add/sub/mul/max2/min2 one per output element;
  rowmax/colmax/rowsum/colsum/gmax/gmean one per input element; vadd/vmul one per element; all other
  ops 0; a loop body counts T times;
- at most {FA.MAX_NODES:,} nodes; at most {FA.MAX_EVALS:,} node evaluations per step (a loop body counts T times);
  at most {FA.MAX_CH} channels per spatial tensor and {FA.MAX_TOTAL_CH:,} channels summed over all spatial tensors
  (inputs, nodes and loop states, a loop body counted once); vectors of at most {FA.MAX_VEC:,} entries;
- every parameter must be finite after the float32 cast; the network runs in exact float32.

Exact semantics (this NumPy reference is authoritative; deployment uses an equivalent batched
float32 PyTorch implementation; `ACTIONS` is the action list above and `CLICK` = 6):

```python
{ref}```

**Example: the study's standard fixed actor** written in this language ({c['params']:,} parameters,
{c['macs_per_step']:,} MACs per step). You may reuse it, change it, or design a different network.

```python
{example}```

Its parameter shapes (1x1 kernels are stored as (out, in, 1, 1)):
{shapes}
"""


FA_SPEC = fa_spec()

FA_TASK = """## Your task: design the network and write its weights

Write a Python function `build()` that returns `{"graph": [...], "memory_size": M, "params": {name:
array}}`: a network in the language above together with all of its weights. The network must make
the actor play this game well **by itself on levels, layouts and visual variants it has never seen**
(the evaluation uses held-out levels, held-out renderer templates and perturbed start states).
Behaviour that only replays the support levels will fail.

Rules:
- Any architecture expressible in the language within the budget is allowed (the fixed actor above,
  a modification of it, or a different design).
- Construct the weights deliberately (analytically / by design). Do not train, optimise, fit,
  search over weights, or simulate the game inside `build()`; it receives no data.
- Use only numpy and math (plus pure-Python standard modules for bookkeeping). `build()` takes no
  arguments, does not read or write anything, and must finish within 60 seconds.
- An invalid network (unknown op or field, shape mismatch, undefined or unused name, non-finite
  value, limit or budget exceeded) counts as a failed submission.

Reply with a short explanation of the architecture and mechanism you implemented (which nodes and
channels compute what), then exactly one ```python code block containing the complete program with
`build()`.
"""


def build_prompt_fa(game_section: str, support_obs) -> str:
    """build_prompt_v2(weights) with the actor spec and weights task replaced by FA_SPEC and FA_TASK."""
    base = build_prompt_v2(game_section, "weights", support_obs, variant="full")
    tail = "\n\n".join([actor_spec("full"), WEIGHTS_TASK])
    assert base.endswith(tail)
    return base[: len(base) - len(tail)] + "\n\n".join([FA_SPEC, FA_TASK])


_REVISE_W = "Revise your build() weights."
_REVISE_FA = "Revise your build() network (architecture and weights)."


def feedback_message_fa(summary: dict, failures: list[dict], round_idx: int, rounds_left: int,
                        condition: str = "freearch") -> str:
    """The weights feedback message (same table, frames and internals) with the revision request for FA."""
    txt = feedback_message(summary, failures, round_idx, rounds_left, "weights")
    head, sep, rest = txt.rpartition(_REVISE_W)
    assert sep
    return head + _REVISE_FA + rest
