import numpy as np

from src.utils.imitation_connector import to_im_traj
from imitation.algorithms.adversarial import airl as im_airl
from imitation.data import rollout
import stable_baselines3 as sb3


def airl(samples, venv, policy_training_steps, total_timesteps, disc_updates=4, batch_size=1024, logger=None,
         return_disc=False):
    #   disc big is fake, low is expert
    if type(samples) == np.array:
        samples = to_im_traj(samples)
        samples = rollout.flatten_trajectories(samples)
    airl_trainer = im_airl.AIRL(
        venv=venv,
        demonstrations=samples,
        demo_batch_size=batch_size,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=policy_training_steps),
        custom_logger=logger, n_disc_updates_per_round=disc_updates
    )
    airl_trainer.train(total_timesteps=total_timesteps)
    if return_disc:
        return airl_trainer._reward_net.base.predict, airl_trainer.logits_gen_is_high
    return airl_trainer._reward_net.base.predict
