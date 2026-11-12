"""Writer smoke test (prompt pilot) on the instrumentation-only toy game (not a study result)."""
import inspect, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from wsp import toy
from wsp.envs import build_start, obs_of, summarize
from wsp.manifest import RUNS
from wsp.prompts import build_prompt
from wsp.queries import reviewed_start
from wsp.session import run_session, load_final, evaluate

SUPPORT = [reviewed_start("toy_reach", s, l, "S-start") for s in range(5) for l in range(4)]
QUERY = [reviewed_start("toy_reach", s, l, "Q") for s in range(100, 140) for l in range(4)]
GAME = ("## The game\n\nThe complete simulator source code is below. Levels are generated from the seed; "
        "evaluation uses unseen seeds.\n\n```python\n" + inspect.getsource(toy) + "```\n")

def one(model, condition, mode, effort):
    out = RUNS / "smoke_toy" / f"{model}_{condition}_{mode}_{effort}"
    if (out / "session.json").exists():
        rec = json.loads((out / "session.json").read_text())
    else:
        sup_obs = [(f"seed {s.seed}, level {s.level + 1}", obs_of(build_start(s))) for s in SUPPORT[:4]]
        prompt = build_prompt(GAME, condition, sup_obs)
        rec = run_session(model, condition, mode, prompt, SUPPORT, out, rounds=2, effort=effort)
    art = load_final(out, condition)
    q = summarize(evaluate(art, QUERY)) if art else {"n": 0}
    line = {"model": model, "cond": condition, "mode": mode, "effort": effort, "wall_s": round(rec["wall_s"]),
            "tok_out": rec["tokens_out"], "tok_in": rec["tokens_in"],
            "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"), c["call"]["status"],
                        round(c["call"]["wall_s"])) for c in rec["candidates"]],
            "query_solve": q.get("solve_rate")}
    print(json.dumps(line), flush=True)
    return line

if __name__ == "__main__":
    jobs = [tuple(j.split(":")) for j in sys.argv[1:]]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in [ex.submit(one, *j) for j in jobs]:
            f.result()
