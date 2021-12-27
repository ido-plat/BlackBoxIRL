import numpy as np
import torch as th
import matplotlib.pyplot as plt
from imitation.data.types import Transitions
from imitation.data.rollout import generate_trajectories, flatten_trajectories_with_rew, make_min_timesteps
from src.alogirhms.airl import airl
from src.utils.imitation_connector import *
from stable_baselines3 import DQN, A2C, PPO
from imitation.rewards.reward_wrapper import *


def check_reward_distribution(agent, reward_func, venv, num_traj, plot_hist=False, num_bins=None, xlim=None):
    if not num_bins:
        num_bins = int(np.sqrt(num_traj))
    traj = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_traj)))
    real_reward = traj.rews
    fake_reward = reward_func(traj.obs, traj.acts, traj.next_obs, traj.dones)
    n_real = real_reward - real_reward.mean()
    n_fake = fake_reward - fake_reward.mean()
    diff = np.abs(n_fake - n_real)
    if plot_hist:
        plt.hist(fake_reward, bins=num_bins, alpha=0.5,  label='Fake rewards')
        plt.hist(real_reward, bins=num_bins, alpha=0.5, label='Real rewards')
        plt.hist(diff, bins=num_bins, alpha=0.5, label='Difference')
        if xlim:
            plt.xlim(xlim)
        plt.legend()
        plt.show()
    return real_reward, fake_reward, diff


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
    if isinstance(agent, list):
        rewards = [flatten_trajectories_with_rew(generate_trajectories(a, venv, make_min_timesteps(num_samples))).rews.mean()
                   for a in agent]
        return rewards
    traj = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_samples)))
    return traj.rews.mean()


def compare_agents(agent1, agent2, venv, num_samples):
    #   >0 => agent 1 is better, <0 => agent 2 is better
    return get_agent_avg_reward(agent1, venv, num_samples) - get_agent_avg_reward(agent2, venv, num_samples)


def train_agent(env, rl_algo, total_timesteps, rl_algo_args):
    print("create model")
    model = rl_algo(env=env, verbose=1, **rl_algo_args)
    print("start learning")
    model.learn(total_timesteps=total_timesteps)
    print("finish learning")
    return model


