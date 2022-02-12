
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,

)

import gym
import numpy as np

from seals.util import AbsorbAfterDoneWrapper, AutoResetWrapper
from imitation.util.util import make_vec_env
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import *


def make_fixed_horizon_venv(
    env_name: str,
    max_episode_steps: int,
    n_envs: int = 8,
    absorb_reward: float = 0.0,
    absorb_obs: Optional[np.ndarray] = None,
    seed: int = 0,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    post_wrappers=None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
):
    def f(env, i):
        # return TimeLimit(AbsorbAfterDoneWrapper(env, absorb_reward, absorb_obs), max_episode_steps)
        return AbsorbAfterDoneWrapper(env, absorb_reward, absorb_obs)
    if post_wrappers:
        post_wrappers.append(f)
    else:
        post_wrappers = [f]
    return make_vec_env(env_name, n_envs, seed, parallel, log_dir, max_episode_steps, post_wrappers, env_make_kwargs)

class SpaceInvadersEnv:
    def __init__(self, env_name='SpaceInvadersNoFrameskip-v4', num_env=1, wrappers=None, max_timestemp=np.inf,
                 transpose=True):

        def super_wrap(a, b):
            return SupperAtariWrapper(a)
        self.wrappers = [super_wrap]
        self.num_env = num_env
        if wrappers:
            self.wrappers.extend(wrappers)
        self.max_timestemp = max_timestemp
        self.env_name = env_name
        self.transpose = transpose

    def make_venv(self):

        venv = make_vec_env(self.env_name, n_envs=self.num_env, post_wrappers=self.wrappers) if self.max_timestemp == np.inf else \
               make_fixed_horizon_venv(self.env_name, max_episode_steps=self.max_timestemp, n_envs=self.num_env,
                                       post_wrappers=self.wrappers)
        if self.transpose:
            venv = VecTransposeImage(venv)
        return venv

class SuperWrapWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
             low=0, high=255, shape=(self.height, self.width, 4), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.moveaxis(np.array([frame] * 4), 0, -1)

class SupperAtariWrapper(gym.Wrapper):
    def __init__(
                self,
                env: gym.Env,
                noop_max: int = 30,
                frame_skip: int = 4,
                screen_size: int = 84,
                terminal_on_life_loss: bool = True,
                clip_reward: bool = True,
    ):

        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = SuperWrapWrapper(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(SupperAtariWrapper, self).__init__(env)

