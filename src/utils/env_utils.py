
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,

)

import gym
import numpy as np

from seals.util import AbsorbAfterDoneWrapper
from imitation.util.util import make_vec_env
from gym.wrappers.time_limit import TimeLimit
def make_fixed_horizon_venv(
    env_name: str,
    max_episode_steps: int,
    n_envs: int = 8,
    absorb_reward: float = 0.0,
    absorb_obs: Optional[np.ndarray] = None,
    seed: int = 0,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
):
    def f(env, i):
        return TimeLimit(AbsorbAfterDoneWrapper(env, absorb_reward, absorb_obs), max_episode_steps)
    if not post_wrappers:
        post_wrappers = [f]
    else:
        post_wrappers = list(post_wrappers).append(f)
    return make_vec_env(env_name, n_envs, seed, parallel, log_dir, max_episode_steps, post_wrappers, env_make_kwargs)
