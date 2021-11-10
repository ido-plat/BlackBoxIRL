import numpy as np

from src.utils.imitation_connector import to_im_traj
from imitation.src.imitation.algorithms.adversarial import airl
from imitation.src.imitation.data import rollout
from imitation.src.imitation.util import logger
import stable_baselines3 as sb3
import tempfile
import pathlib

def airl(samples, venv,policy_training_steps,total_timesteps , batch_size=32, logger=None):
    if type(samples) == np.array:
        samples = to_im_traj(samples)
        samples = rollout.flatten_trajectories(samples)
    airl_trainer = airl.AIRL(
        venv=venv,
        demonstrations=samples,
        demo_batch_size=batch_size,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=policy_training_steps),
        custom_logger=logger,
    )
    airl_trainer.train(total_timesteps=total_timesteps)
    return airl_trainer.reward_test().predict