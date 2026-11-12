"""Record which arcengine functions each official game executes on its support levels.

Input to the engine-excerpt rule of wsp.redact_public (members actually executed are shown even when
the game's own classes reuse the member name). Sequences: per support level, 5 uniform and 5
board-biased random sequences of length 60 (the generator of check_redaction_public.py; no frozen
routes, no planner). Writes runs/private/game_sections_public/engine_usage.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_redaction_public import gen_random, load_class, run  # noqa: E402
from wsp import redact_public as RP  # noqa: E402

ENG = str(RP.ENGINE_DIR)
out = {}
for g in sys.argv[1:] or RP.GAMES:
    cls = load_class(RP.source_path(g), "trace")
    used: set[str] = set()

    def prof(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(ENG):
            used.add(frame.f_code.co_qualname.split(".<locals>")[0])
    held = set(RP.held_levels(g))
    for k in range(RP.n_levels(g)):
        if k in held:
            continue
        for i in range(5):
            for biased in (False, True):
                acts = gen_random(cls, k, f"{g}:{k}:{'b' if biased else 'u'}{i}", biased)
                sys.setprofile(prof)
                try:
                    run(cls, k, acts)
                finally:
                    sys.setprofile(None)
    out[g] = sorted(used)
    print(g, len(used))
path = RP.RUNS / "private" / "game_sections_public" / "engine_usage.json"
old = json.loads(path.read_text()) if path.exists() else {}
path.write_text(json.dumps({**old, **out}, indent=1))
