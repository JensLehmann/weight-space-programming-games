import numpy as np, torch, time
from wsp.actor import (Actor, NeuralPolicy, reference_step, random_params, zero_params, validate,
                       N_PARAMS, MEM, decode)
from wsp.envs import ACTIONS, new_game, obs_of, avail_mask

def _setup(seed=0):
    g = new_game("esm", 3); obs = obs_of(g); av = avail_mask(g)
    return obs, av

def test_param_count():
    assert 50_000 < N_PARAMS < 70_000

def test_torch_matches_numpy_reference():
    obs, av = _setup()
    for s in range(3):
        p = random_params(s, scale=0.05)
        mem = np.random.default_rng(s).uniform(-1, 1, MEM).astype(np.float32)
        a_prev = np.eye(8, dtype=np.float32)[s]
        move_np, logits_np, mem_np = reference_step(p, obs, mem, a_prev, av)
        actor = Actor(p)
        with torch.no_grad():
            lt, ct, mt = actor(torch.as_tensor(obs[None]), torch.as_tensor(mem[None]),
                               torch.as_tensor(a_prev[None]), torch.as_tensor(av[None]))
        fin = np.isfinite(logits_np)
        assert np.allclose(lt[0].numpy()[fin], logits_np[fin], rtol=1e-4, atol=1e-4)
        assert np.allclose(mt[0].numpy(), mem_np, atol=1e-5)
        assert decode(lt, ct)[0] == move_np

def test_validate_rejects_bad():
    import pytest
    p = zero_params(); p["W1"] = np.zeros((3, 3))
    with pytest.raises(ValueError): validate(p)
    p = zero_params(); p["bo"] = np.full(8, np.nan)
    with pytest.raises(ValueError): validate(p)
    p = zero_params(); del p["Wc"]
    with pytest.raises(ValueError): validate(p)
