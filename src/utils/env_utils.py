
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
                 transpose=True, use_history=True, **kwargs):

        depth = 1 if use_history else 4
        # def super_wrap(a, b):
        #     return SupperAtariWrapper(a, depth=depth)
        self.wrappers = [lambda a, b: SupperAtariWrapper(a, depth=depth, **kwargs)]
        if use_history:
            self.wrappers.append(lambda a, b: HistoryWrapper(a, horizon=4))
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
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, depth=1):
        gym.ObservationWrapper.__init__(self, env)
        self.depth = depth
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
             low=0, high=255, shape=(self.height, self.width, depth), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.moveaxis(np.array([frame] * self.depth), 0, -1)

class SupperAtariWrapper(gym.Wrapper):
    def __init__(
                self,
                env: gym.Env,
                noop_max: int = 30,
                frame_skip: int = 4,
                screen_size: int = 84,
                terminal_on_life_loss: bool = True,
                clip_reward: bool = True,
                depth=1
    ):

        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = SuperWrapWrapper(env, width=screen_size, height=screen_size, depth=depth)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(SupperAtariWrapper, self).__init__(env)

class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.

    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 4):
        assert isinstance(env.observation_space, gym.spaces.Box)
        wrapped_obs_space = env.observation_space
        self.low = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        self.high = np.repeat(wrapped_obs_space.high, horizon, axis=-1)
        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=self.low, high=self.high, dtype=wrapped_obs_space.dtype)
        super(HistoryWrapper, self).__init__(env)
        self.horizon = horizon
        self.obs_history = np.zeros(self.low.shape, self.low.dtype)

    def _create_obs_from_history(self):
        return self.obs_history

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history(), reward, done, info
