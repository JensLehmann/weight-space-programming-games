"""Model-agnostic prompt templates. Only the game section differs between families.

Conditions:
  weights : writer returns `build()` constructing the fixed actor's tensors (no training/fitting)
  python  : writer returns `act(obs, available, memory)` (an ordinary program player)
"""
from __future__ import annotations

import inspect
import json

import numpy as np

from . import actor
from .envs import ACTIONS

HEX = "0123456789abcdef"


def grid_hex(grid) -> str:
    g = np.asarray(grid)
    return "\n".join("".join(HEX[int(v)] for v in row) for row in g)


VARIANT_NOTE = {
    "full": "",
    "noaxial": ("\n**Actor variant for this experiment: `noaxial`.** The row/column max broadcasts are "
                "absent: F3 = relu(W3[:, :32] @ F2 + b3); the entries W3[:, 32:] are ignored.\n"),
    "t8": ("\n**Actor variant for this experiment: `t8`.** The weight-tied block runs for only 8 "
           "iterations (not 128).\n"),
}


def actor_spec(variant: str = "full") -> str:
    ref = inspect.getsource(actor._conv) + "\n\n" + inspect.getsource(actor.reference_step)
    shapes = "\n".join(f"    {k:4s} {tuple(v)}" for k, v in actor.SCHEMA.items())
    return f"""## The actor network (fixed; you choose all its numbers)

The actor is a fixed neural network with {actor.N_PARAMS} float32 parameters. At every step it
receives only the current 64x64 observation (colour ids 0-15), its own 64-dim memory vector `m`
(zeros at the start of every episode), and a one-hot of its previous action (zeros at episode
start). It outputs one action. The actor runs alone: no other code, search, or lookup exists at
deployment. Action index order: {list(ACTIONS)} (index 6 = ACTION6 = click).

Exact semantics (this NumPy reference is authoritative; deployment uses an equivalent float32
PyTorch implementation, T = {actor.T_ITER} iterations of the tied block):

```python
{ref}```

Tensor names and shapes (row-major; conv weights are (out, in, k, k) cross-correlation kernels,
zero padding keeps 64x64; `rowmax` at (c,i,j) is the max of channel c over row i, `colmax` the max
over column j; spatial `max`/`mean` reduce over all 4096 positions):
{shapes}

Decoding is deterministic: the argmax over *available* actions (unavailable logits are -inf;
ties go to the lowest index). If ACTION6 wins, the click target is the argmax cell of the click
map (ties go to the first cell in row-major order). Inputs x0 channels 0-15 are the one-hot
colour planes, 16 = row/63, 17 = col/63, 18 = constant 1. The `variant` argument of the
reference is "full" unless stated otherwise below.
""" + VARIANT_NOTE[variant]


WEIGHTS_TASK = """## Your task: write the actor's weights

Write a Python function `build()` that returns a dict mapping every tensor name above to a NumPy
array of exactly that shape. The weights must make the actor play this game well **by itself on
levels, layouts and visual variants it has never seen** (the evaluation uses held-out levels,
held-out renderer templates and perturbed start states). Behaviour that only replays the support
levels will fail.

Rules:
- Construct the weights deliberately (analytically / by design). Do not train, optimise, fit,
  search over weights, or simulate the game inside `build()`; it receives no data.
- Use only numpy and math (plus pure-Python standard modules for bookkeeping). `build()` takes no
  arguments, does not read or write anything, and must finish within 60 seconds.
- Any tensor you do not need can be zero.

Reply with a short explanation of the mechanism you implemented (which channels/units compute
what), then exactly one ```python code block containing the complete program with `build()`.
"""

PYTHON_TASK = """## Your task: write a program player

Write a Python function `act(obs, available, memory)` that plays this game well **on levels,
layouts and visual variants it has never seen** (the evaluation uses held-out levels, held-out
renderer templates and perturbed start states).
- `obs`: NumPy int array (64, 64) of colour ids 0-15 (the current frame only).
- `available`: list of available action names, e.g. ["RESET", "ACTION1", ...].
- `memory`: a dict that persists between steps of one episode (empty at episode start).
- Return a tuple `(action_name, row, col)`; row/col are ints only for ACTION6 (else None).

Rules: use only numpy, math and pure-Python standard modules (e.g. collections, heapq,
itertools); `act` must not read or write anything outside its arguments and must return within
10 seconds. The program starts fresh for every episode.

Reply with a short explanation of your strategy, then exactly one ```python code block
containing the complete program with `act()`.
"""


def episode_rules(cap_rule: str = "3 x (shortest solution length) + 10") -> str:
    return f"""## Episodes and scoring

An episode starts at the beginning of one level (or at a state reached by a few actions into
it) and ends when that level is completed (success), on GAME_OVER, or after {cap_rule}
actions. Invalid actions count as wasted steps. RESET restarts the current level and counts as an
action. Success rate over many held-out episodes is the score.
"""


def build_prompt(game_section: str, condition: str, support_obs: list[tuple[str, np.ndarray]]) -> str:
    parts = [
        "# Writing a compact neural game-playing agent" if condition == "weights"
        else "# Writing a game-playing program",
        game_section,
        episode_rules(),
        "## Example observations (support set)\n\nEach grid is 64 rows of 64 hexadecimal colour ids.\n"
        + "\n\n".join(f"### {name}\n```\n{grid_hex(g)}\n```" for name, g in support_obs),
    ]
    if condition == "weights":
        parts += [actor_spec(), WEIGHTS_TASK]
    else:
        parts += [PYTHON_TASK]
    return "\n\n".join(parts)


def feedback_message(summary: dict, failures: list[dict], round_idx: int, rounds_left: int,
                     condition: str) -> str:
    lines = [f"## Evaluation of your submission (round {round_idx}) on the support episodes",
             f"Solved {summary['solved']}/{summary['n']} support episodes. "
             f"{rounds_left} revision(s) remain.",
             "", "| episode | template | level | start | result | actions |", "|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        lines.append(f"| {r['i']} | {r['seed']} | {r['level'] + 1} | {r['start']} | {r['result']} | {r['actions']} |")
    for f in failures:
        lines += ["", f"### Failed episode {f['i']}: {f['why']}",
                  f"First actions: {' '.join(f['trace'][:40])}{' ...' if len(f['trace']) > 40 else ''}",
                  f"Frame at step {f['step']} (where the problem first shows):",
                  "```\n" + grid_hex(f["frame"]) + "\n```"]
        if f.get("internals"):
            lines.append("Network internals at that frame: " + json.dumps(f["internals"]))
        if f.get("error"):
            lines.append(f"Error: {f['error']}")
    what = {"weights": "build() weights", "python": "act() program", "pyreact": "act() program",
            "edit": "edit() program (it is always applied to the original trained weights)"}[condition]
    lines += ["", f"Revise your {what}. Reply with a short explanation and exactly one ```python "
              "block with the complete program."]
    return "\n".join(lines)


EDIT_TASK = """## Your task: repair a trained actor by editing its weights

The actor above already has trained weights (it was trained with reinforcement learning on the
support episodes). Its current performance and failures are described below. Write a Python
function `edit(params)` that receives the current tensors as a dict of float32 NumPy arrays and
returns the edited dict (same names and shapes). The goal is an actor that plays well on levels,
layouts and visual variants it has never seen.

Rules:
- Edit deliberately: you may read the given weights inside `edit()` (e.g. to find unused units)
  and change any entries, but do not train, optimise, fit, search over weights, or simulate the
  game; `edit()` receives no observations or data other than the weights.
- Use only numpy and math (plus pure-Python standard modules for bookkeeping). `edit()` must not
  read or write anything and must finish within 60 seconds.
- Smaller, targeted edits are preferred, but any edit is allowed.

Reply with a short explanation of what you changed and why, then exactly one ```python code
block containing the complete program with `edit()`.
"""


def weight_summary(params: dict) -> str:
    rows = []
    for k, v in params.items():
        v = np.asarray(v)
        rows.append(f"    {k:4s} shape={tuple(v.shape)} l2={np.linalg.norm(v):.3f} "
                    f"max|w|={np.abs(v).max():.3f} nonzero={int(np.count_nonzero(v))}")
    return "\n".join(rows)


def build_edit_prompt(game_section: str, support_obs, base_params: dict, diagnosis: str) -> str:
    parts = ["# Repairing a compact neural game-playing agent", game_section, episode_rules(),
             "## Example observations (support set)\n\nEach grid is 64 rows of 64 hexadecimal colour ids.\n"
             + "\n\n".join(f"### {name}\n```\n{grid_hex(g)}\n```" for name, g in support_obs),
             actor_spec(),
             "## The current trained weights (summary; the full arrays are passed to edit())\n\n"
             + weight_summary(base_params),
             "## Current behaviour on the support episodes\n\n" + diagnosis,
             EDIT_TASK]
    return "\n\n".join(parts)


PYREACT_TASK = """## Your task: write a reactive program player

Write a Python function `act(obs, available, memory)` that plays this game well **on levels,
layouts and visual variants it has never seen** (the evaluation uses held-out levels, held-out
renderer templates and perturbed start states).
- `obs`: NumPy int array (64, 64) of colour ids 0-15 (the current frame only).
- `available`: list of available action names, e.g. ["RESET", "ACTION1", ...].
- `memory`: a dict that persists between steps of one episode (empty at episode start).
- Return a tuple `(action_name, row, col)`; row/col are ints only for ACTION6 (else None).

This player must be **reactive**: decide each action from the current frame (and your small
memory) with a fixed amount of computation, like a neural network would. Do not re-implement the
game's transition function, do not simulate future states, and do not search over action
sequences; image analysis such as finding objects, distances or connected regions in the
*current* frame is allowed. Each call must return within 50 milliseconds.

Rules: use only numpy, math and pure-Python standard modules; `act` must not read or write
anything outside its arguments. The program starts fresh for every episode.

Reply with a short explanation of your strategy, then exactly one ```python code block
containing the complete program with `act()`.
"""


def build_prompt_v2(game_section: str, condition: str, support_obs, variant: str = "full") -> str:
    """Revision-2 prompt: identical structure to revision 1 plus the pyreact condition and variants."""
    parts = ["# Writing a compact neural game-playing agent" if condition == "weights"
             else "# Writing a game-playing program",
             game_section, episode_rules(),
             "## Example observations (support set)\n\nEach grid is 64 rows of 64 hexadecimal colour ids.\n"
             + "\n\n".join(f"### {name}\n```\n{grid_hex(g)}\n```" for name, g in support_obs)]
    if condition == "weights":
        parts += [actor_spec(variant), WEIGHTS_TASK]
    elif condition == "pyreact":
        parts += [PYREACT_TASK]
    else:
        parts += [PYTHON_TASK]
    return "\n\n".join(parts)
