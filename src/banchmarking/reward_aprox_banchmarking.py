import numpy as np
import torch as th
import matplotlib.pyplot as plt
from imitation.data.types import Transitions
from imitation.data.rollout import generate_trajectories, flatten_trajectories_with_rew, make_min_timesteps
from src.alogirhms.airl import airl
from src.utils.imitation_connector import *
from stable_baselines3 import DQN, A2C, PPO
from src.utils.agent_utils import generate_trajectory_footage
from imitation.rewards.reward_wrapper import *


def check_reward_distribution(agent, alg, venv, num_traj, plot_hist=False, norm=True, args=None, test_num_traj=0,
                              test_agent=None):
    if not args:
        args = {}
    if test_num_traj == 0:
        test_num_traj = num_traj
    if not test_agent:
        test_agent = agent

    traj = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_traj)))
    test_traj = flatten_trajectories_with_rew(generate_trajectories(test_agent, venv, make_min_timesteps(test_num_traj)))

    print('created trajectories')
    reward_func = alg(traj, venv, **args)
    real_reward = test_traj.rews
    fake_reward = reward_func(test_traj.obs, test_traj.acts, test_traj.next_obs, test_traj.dones)
    print('real reward shape: ' + str(real_reward.shape))
    print('fake reward shape: ' + str(fake_reward.shape))
    if plot_hist:
        plt.hist(fake_reward, label='fake rewards')
        plt.hist(real_reward, label='real rewards')
        plt.legend()
        plt.show()

    if norm:
        real_reward = real_reward - real_reward.mean()
        fake_reward = fake_reward - fake_reward.mean()
    return real_reward, fake_reward


def train_agent_learnt_reward(samples, venv, model_type, learning_time_step, save_model_path=None, model_arg=None,
                              airl_args=None, return_disc=False):  # could need a lot more arguments
    if not airl_args:
        airl_args = {}
    if not model_arg:
        model_arg = {'policy': 'CnnPolicy'}
    disc = None
    if return_disc:
        reward_func, disc = airl(samples, venv, return_disc=True, **airl_args)
    else:
        reward_func = airl(samples, venv, **airl_args)  # check airl arguments
    new_env = RewardVecEnvWrapper(
        venv=venv,
        reward_fn=reward_func,
    )
    model = train_agent(new_env,model_type, learning_time_step, model_arg)
    if save_model_path:
        model.save(save_model_path)
    if return_disc:
        return model, disc
    return model


def get_agent_avg_reward(agent, venv, num_samples):
    traj = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_samples)))
    return traj.rews.mean()


def compare_agents(agent1, agent2, venv, num_samples):
    #   >0 => agent 1 is better, <0 => agent 2 is better
    traj1 = flatten_trajectories_with_rew(generate_trajectories(agent1, venv, make_min_timesteps(num_samples)))
    traj2 = flatten_trajectories_with_rew(generate_trajectories(agent2, venv, make_min_timesteps(num_samples)))
    return get_agent_avg_reward(agent1, venv, num_samples) - get_agent_avg_reward(agent2, venv, num_samples)


def traj_confidence(samples: Transitions, disc_func, agent, action_space_size):
    discriminator = discriminator_conversion(disc_func, agent, action_space_size)
    confidence = discriminator(samples.obs, samples.acts, samples.next_obs, samples.dones)
    return confidence


def train_agent(env, rl_algo, total_timesteps, rl_algo_args):
    model = rl_algo(env=env, verbose=1, **rl_algo_args)
    model.learn(total_timesteps=total_timesteps)
    return model

def eval_single_traj(venv, agent, action_space_size, save_path=None, samples=None, discriminator=None, args_for_airl=None,
                     plot_confidence_hist=False):
    if not discriminator:
        if not samples:
            raise ValueError("Pass samples if there's no discriminator!!!")
        _, disc_func = airl(samples, venv, return_disc=True, **args_for_airl)
        discriminator = discriminator_conversion(disc_func, agent, action_space_size)
    past_obs, action, next_obs, dones, _ = generate_trajectory_footage(agent, venv, save_path)
    confidence = discriminator(past_obs, action, next_obs, dones)
    if plot_confidence_hist:
        plt.hist(confidence)
        print('confidence mean: '+str(confidence.mean()))
        plt.show()
    return confidence, discriminator

