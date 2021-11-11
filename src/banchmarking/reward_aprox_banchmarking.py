import numpy as np
import torch as th
import matplotlib.pyplot as plt

from imitation.data.rollout import generate_trajectories, flatten_trajectories_with_rew, make_min_timesteps
from src.alogirhms.airl import airl
from stable_baselines3 import DQN, A2C, PPO
from src.utils.agent_utils import generate_trajectory_footage


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


def train_agent_learnt_reward(samples, venv, model_type, alg, model_args=None):  # could need a lot more arguments
    # reward_func = alg()
    # build env with reward_func
    # train agent on env
    return PPO.load('temp')  # temp just to put the right return type


def compare_agents(agent1, agent2, venv, num_samples):
    #   >0 => agent 1 is better, <0 => agent 2 is better
    traj1 = flatten_trajectories_with_rew(generate_trajectories(agent1, venv, make_min_timesteps(num_samples)))
    traj2 = flatten_trajectories_with_rew(generate_trajectories(agent2, venv, make_min_timesteps(num_samples)))
    return traj1.rews.mean() - traj2.rews.mean()


def provide_footage_from_fake_agent(samples, venv, save_path, agent=None, discriminator=None, args_for_airl=None,
                                    plot_confidence_hist=False):
    # return confidance level (has to use airl)
    if not agent:
        agent = train_agent_learnt_reward(samples, venv)
    if not discriminator:
        _, disc_func = airl(samples, venv, return_disc=True, **args_for_airl)

        def f(states, actions, next_states, done) -> np.ndarray:
            states = th.tensor(states)
            actions = th.tensor(actions)
            next_states = th.tensor(next_states)
            done = th.tensor(done)
            _, log_prob, _ = agent.policy.evaluate_actions(states, actions)
            disc_result = disc_func(states, actions, next_states, done, log_prob)
            return 1/(1 + th.exp(-disc_result)).numpy()
        discriminator = f
    past_obs, action, next_obs, dones, _ = generate_trajectory_footage(agent, venv, save_path)
    confidence = discriminator(past_obs, action, next_obs, dones)
    if plot_confidence_hist:
        plt.hist(confidence)
        plt.show()
    print('confidence mean: '+str(confidence.mean()))
    return confidence, discriminator

