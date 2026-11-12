"""Prompt for the bounded program player (pbounded, wsp.pbounded): the writer writes a NumPy program under the
free-architecture computation model. Game section, episode rules and support observations are byte-identical to
the revision-2 prompts (wsp.prompts.build_prompt_v2; the FA prompt shares them, wsp.prompts_fa); the title is the
program-player title; only the player spec and the task differ. Feedback is the FA W2 feedback (same table,
frames and per-frame outputs) with the revision request of a program.
"""
from __future__ import annotations

import textwrap

from . import _pb_runtime as RT
from . import pbounded as PB
from .envs import ACTIONS
from .freearch import MAX_MACS, MAX_MEM, MAX_PARAMS, MAX_T
from .prompts import PYTHON_TASK, build_prompt_v2, feedback_message

EXAMPLE = '''import numpy as np

MEMORY_SIZE = 8
STEPS = 32                                    # iterations of the loop in act

# Constants, built once when the program is loaded (any Python and NumPy may be used here).
COLOURS = np.arange(16).reshape(16, 1, 1)     # colour ids, for one-hot planes
PREFER = np.zeros((8, 16))                    # colour frequencies -> action scores
PREFER[1, :] = 1.0                            # ACTION1 always scores the sum of the frequencies (1.0)
PREFER[6, 3] = 4.0                            # ACTION6 wins when colour 3 covers more than a quarter of the frame
MIX = 0.5 * np.eye(MEMORY_SIZE, 16)           # memory input from the colour frequencies


def neighbours(x):
    """Maximum over the 4 neighbours of every cell of x (C, 64, 64); zero outside the grid."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    return np.maximum(np.maximum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]),
                      np.maximum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]))


def act(obs, available, memory, prev_action):
    onehot = (obs[None] == COLOURS).astype(np.float32)         # (16, 64, 64)
    free = 1.0 - onehot[0:1]                                   # cells that are not colour 0
    reach = onehot[3:4]                                        # start at the colour-3 cells
    for _ in range(STEPS):                                     # spread through cells that are not colour 0
        reach = np.maximum(reach, neighbours(reach) * free)
    freq = onehot.mean(axis=(1, 2))                            # colour frequencies (16,)
    scores = PREFER @ freq + 0.5 * prev_action                 # small bonus for repeating the last action
    rows = reach[0].sum(axis=1, keepdims=True)                 # reached cells per row (64, 1)
    click = np.where(reach[0] > 0, rows, -1.0)                 # a reached cell in the fullest row
    memory_next = np.tanh(0.5 * memory + MIX @ freq)
    return scores, click, memory_next
'''


def _names(prefix, names):
    return textwrap.fill(", ".join(f"`{prefix}{n}`" for n in names), width=100, initial_indent="  ",
                         subsequent_indent="  ", break_on_hyphens=False)


def pb_spec() -> str:
    c = PB.run_trusted(EXAMPLE)["count"]
    funcs = _names("np.", RT.NP_FUNCS)
    consts = _names("np.", RT.NP_CONSTS)
    methods = _names(".", RT.METHODS)
    return f"""## The player (a bounded NumPy program; you write its code and choose all its constants)

The player is a Python program with a fixed amount of computation per step, like a neural network:
every call runs the same array operations, and nothing in it branches on the data. At every step it
receives only the current 64x64 observation (colour ids 0-15), its own memory vector (zeros at the
start of every episode), a one-hot of its previous action (zeros at episode start) and the mask of
available actions. It outputs action scores, click scores and its next memory. The player runs
alone: no other code, search, or lookup exists at deployment. Action index order: {list(ACTIONS)}
(index 6 = ACTION6 = click).

**Program.** `import numpy as np` is the only import. The module-level code runs once, when the
program is loaded, and builds the constants (it is not player code: any Python with NumPy). It defines
`MEMORY_SIZE = M` (an integer literal assigned once at module level, 0 <= M <= {MAX_MEM}) and
`act(obs, available, memory, prev_action)`, which is called once per step with
- `obs` (64, 64) int64: the colour ids;
- `available` (8,) float32: 1.0 for available actions, else 0.0;
- `memory` (M,) float32: your memory (zeros at episode start);
- `prev_action` (8,) float32: one-hot of the previous action (zeros at episode start).
It returns `(scores, click, memory_next)`: `scores` (8,), `click` (64, 64) and `memory_next` (M,),
which becomes `memory` at the next step exactly as computed (nothing squashes it; for M = 0 return
`memory`). Floating-point values are float32: the module-level constants used by player code are
cast to float32, and every floating-point result is float32.

**Decoding** is deterministic and outside the player: unavailable actions get score -inf and the
argmax over the available actions is taken (ties go to the lowest index). If ACTION6 wins, the click
target is the argmax cell of `click` (ties go to the first cell in row-major order). If any
available score is NaN or infinite, the step is an invalid action.

**Player code** is `act` and every function it calls, directly or indirectly. These functions are
defined at module level with `def` (positional parameters; defaults only literal numbers). Player
code may contain only:
- assignments to names (`x = ...`, `a, b = ...`; `x += y` means `x = x + y`), expression
  statements and a final `return`;
- loops `for _ in range(K)`, with K an integer literal or a module-level name assigned an integer
  literal once, 1 <= K <= {MAX_T}. The loop variable is not used; no `break`, `continue` or `else`.
  Loops cannot be nested, and a loop body cannot call a function that contains a loop;
- the array operators `+ - * / // % ** @`, the comparisons `< <= > >= == !=` (not chained),
  `& | ^ ~` and unary `-`;
- indexing and slicing with constants: integers, slices, `None` and `...`, written with integer
  literals and module-level constants (no variables, no arrays);
- the NumPy functions (`np.where` with three arguments, an elementwise select)
{funcs};
- the NumPy constants
{consts};
- the array methods
{methods}
  and the attribute `.T`;
- local variables, the parameters, module-level functions (called) and module-level constants. The
  constants are read-only; they may be numbers, strings, arrays, or tuples, lists or dicts of them
  (lists become tuples).
Shapes, axes, shifts, pad widths, reshape/transpose arguments and dtypes are Python constants, never
arrays. `@` and `np.einsum` need at least one operand computed only from constants (a weight
matrix, as in a network layer); `np.einsum` takes one or two operands and explicit subscripts with
`->` (no `...`); `np.pad` pads only with a constant.
Nothing else is allowed in player code, in particular `if`, `while`, `try`, `with`, `assert`,
`raise`, conditional expressions, `and`/`or`/`not`, `in`/`is`, comprehensions, generators, `lambda`,
nested functions, recursion, `global`, imports, assignments into arrays or attributes, Python
builtins (except `range` in a loop header), and anything that turns array values into Python values
or indices (`int()`, `float()`, `bool()`, `len()`, `.item()`, `.tolist()`, `np.argmax`, `np.nonzero`,
`np.where` with one argument, indexing with arrays).

**Budget and limits** (checked on a test call when the program is submitted; any violation makes the
program invalid). Player code cannot branch on the data, so every call runs the same operations:
- at most {MAX_PARAMS:,} constants: the entries of the module-level arrays and numbers that player
  code uses (a string counts one per character), plus the numbers written in player code;
- at most {PB.MAX_EVALS:,} operations per call. One operation is one NumPy function call, one operator
  with an array operand, one array method call or `.T`, or one indexing of an array (unpacking
  `a, b = x` indexes twice); a loop body counts every iteration;
- at most {MAX_MACS / 1e9:g}e9 = {MAX_MACS:,.0f} element operations per call: each operation counts the
  entries of its result; the reductions `sum`, `max`, `min`, `mean` count the entries of their input,
  `@` and `np.einsum` their multiply-accumulates;
- at most {PB.MAX_ELEMS:,} (= 512 x 64 x 64) entries per array;
- constants and numbers in player code must be finite after the float32 cast.

**Example: a trivial player** ({c['ops_per_call']:,} operations and {c['work_per_call']:,} element operations per
call, {c['constants']:,} constants). It only illustrates the rules; it does not play any game well.

```python
{EXAMPLE}```
"""


PB_SPEC = pb_spec()

PB_TASK = """## Your task: write a bounded NumPy player

Write a Python program with `MEMORY_SIZE` and `act(obs, available, memory, prev_action)` that follows
the rules above and makes the player play this game well **by itself on levels, layouts and visual variants it has never seen**
(the evaluation uses held-out levels, held-out renderer templates and perturbed start states).
Behaviour that only replays the support levels will fail.

Rules:
- Construct the constants deliberately (analytically / by design). Do not train, optimise, fit,
  search over constants, or simulate the game in the module-level code; it receives no data.
- Use only numpy. The program does not read or write anything, and loading it (the module-level
  code) must finish within 60 seconds.
- An invalid program (a rule violation, an error on the test call, wrong output shapes, a non-finite
  constant, a limit or budget exceeded) counts as a failed submission.

Reply with a short explanation of the mechanism you implemented (which arrays and steps compute
what), then exactly one ```python code block containing the complete program.
"""


def build_prompt_pb(game_section: str, support_obs) -> str:
    """build_prompt_v2(python) with the Python task replaced by PB_SPEC and PB_TASK."""
    base = build_prompt_v2(game_section, "python", support_obs, variant="full")
    assert base.endswith("\n\n" + PYTHON_TASK)
    return base[: len(base) - len(PYTHON_TASK)] + "\n\n".join([PB_SPEC, PB_TASK])


_INTERNALS = "Network internals at that frame: "
_OUTPUTS = "Player outputs at that frame: "


def feedback_message_pb(summary: dict, failures: list[dict], round_idx: int, rounds_left: int,
                        condition: str = "pbounded") -> str:
    """The FA W2 feedback message (same table, frames and per-frame outputs) asking to revise the act() program."""
    return feedback_message(summary, failures, round_idx, rounds_left, "python").replace(_INTERNALS, _OUTPUTS)
