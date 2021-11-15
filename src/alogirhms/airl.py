import numpy as np
import torch as th
from src.utils.imitation_connector import to_im_traj
from imitation.algorithms.adversarial import airl as im_airl
from imitation.data import rollout
import stable_baselines3 as sb3


def airl(samples, venv, policy_training_steps, total_timesteps, disc_updates=4, batch_size=1024, logger=None,
         return_disc=False, save_reward_func_path=None, save_disc_path=None):
    #   disc big is fake, low is expert
    if isinstance(samples, np.ndarray):
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
    if save_disc_path:
        th.save(airl_trainer._reward_net, save_disc_path)
    if save_reward_func_path:
        th.save(airl_trainer._reward_net.base, save_reward_func_path)
    if return_disc:
        return airl_trainer._reward_net.base.predict, airl_trainer.logits_gen_is_high
    return airl_trainer._reward_net.base.predict


def load_reward_net(path, device='cuda:0'):
    net = th.load(path).to(device)
    return net.predict


def load_disc_func(path, device='cuda:0'):
    net = th.load(path).to(device)

    def f(state, action, next_state, done, log_policy_act_prob):
        """Compute the discriminator's logits for each state-action sample."""
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        with th.no_grad():
            reward_output_train = net(state.float(), action.float(), next_state.float(), done.float())
            return log_policy_act_prob - reward_output_train
    return f
