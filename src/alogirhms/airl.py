import numpy as np
import torch as th
from src.utils.imitation_connector import to_im_traj
from imitation.algorithms.adversarial import airl as im_airl
from imitation.data import rollout, types
from src.networks.reward_nets import ClassificationShapedRewardNet
import stable_baselines3 as sb3
from imitation.rewards.reward_nets import BasicShapedRewardNet

def airl(samples, venv, policy_training_steps_for_iteration, total_timesteps, iagent_args, disc_updates=4,
         batch_size=1024, logger=None, return_disc=False, save_reward_func_path=None, save_disc_path=None,
         save_iagent_path=None, allow_variable_horizon=False, gen_train_timesteps=None, reward_net=None,
         evalDB=None,
         **kwargs
         ):
    if isinstance(samples, np.ndarray):
        raise ValueError('give transitions to airl - not np array')
    if reward_net is None:
        reward_net = BasicShapedRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

    airl_trainer = im_airl.AIRL(
        venv=venv, reward_net=reward_net,
        demonstrations=samples,
        demo_batch_size=batch_size,
        gen_algo=sb3.PPO(env=venv, n_steps=policy_training_steps_for_iteration, **iagent_args),
        gen_train_timesteps=gen_train_timesteps,
        custom_logger=logger, n_disc_updates_per_round=disc_updates, allow_variable_horizon=allow_variable_horizon,
        evalDB=evalDB,
        **kwargs
    )
    airl_trainer.train(total_timesteps=total_timesteps)
    if save_iagent_path:
        airl_trainer.gen_algo.save(save_iagent_path)
    if save_disc_path:
        th.save(airl_trainer._reward_net, save_disc_path)
    if save_reward_func_path:
        th.save(airl_trainer._reward_net.base, save_reward_func_path)
    if return_disc:
        return airl_trainer._reward_net.base.predict, airl_trainer.logits_gen_is_high
    return airl_trainer._reward_net.base.predict


def load_reward_net(path, device='cuda'):
    net = th.load(path).to(device)
    return net.predict


def load_disc_func(path, device='cuda'):
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
