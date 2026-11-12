import numpy as np
from wsp.sandbox import audit, run_initializer, PythonPlayerPolicy
from wsp.actor import SCHEMA, validate
from wsp.envs import EpisodeSpec, run_episodes, summarize
from wsp.queries import reviewed_start

GOOD = """
import numpy as np
SHAPES = %r
def build():
    return {k: np.zeros(s) for k, s in SHAPES.items()}
""" % ({k: v for k, v in SCHEMA.items()},)

def test_good_initializer():
    r = run_initializer(GOOD, timeout=60)
    assert r.error is None, r.error
    validate(r.params)

def test_rejections():
    bad = {
        "import os\ndef build(): return {}": "import os",
        "def build():\n    return open('/etc/hosts').read()": "name open",
        "import numpy as np\ndef build():\n    return np.load('x.npy')": "attribute load",
        "def build():\n    return ().__class__.__bases__": "dunder",
        "x = 1": "missing top-level function build",
    }
    for src, why in bad.items():
        a = audit(src, "build")
        assert not a.ok and any(why in v for v in a.violations), (src, a.violations)

def test_runtime_import_guard():
    # dynamic import attempts are blocked at runtime even if audit were bypassed
    src = "def build():\n    m = __builtins__['__import__']('os')\n    return {}"
    assert not audit(src, "build").ok

def test_timeout():
    src = "def build():\n    x = 0\n    for i in range(10**12):\n        x += i\n    return {}"
    r = run_initializer(src, timeout=3)
    assert r.params is None and ("timeout" in r.error or "runtime" in r.error)

def test_player_runs():
    src = "def act(obs, available, memory):\n    memory['n'] = memory.get('n', 0) + 1\n    return ('ACTION1', None, None)"
    specs = [reviewed_start("toy_reach", 154, 0, "t")]   # toy target straight above the player (3 x ACTION1)
    pol = PythonPlayerPolicy(src)
    res = run_episodes(specs, pol)
    pol.close()
    assert res[0].success and res[0].actions == specs[0].ref_len
