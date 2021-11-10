
from imitation.data.rollout import generate_trajectories, flatten_trajectories_with_rew, make_min_timesteps
import numpy as np
import matplotlib.pyplot as plt


def check_reward_distribution(agent, alg, venv, num_traj, plot_hist=False, norm=True, args=None):
    if not args:
        args = {}
    traj = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_traj)))
    print('created trajectories')
    reward_func = alg(traj, venv, **args)
    real_reward = traj.rews
    fake_reward = reward_func(traj.obs, traj.acts, traj.next_obs, traj.dones)
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
    return 0


def compare_agents(agent1, agent2, venv):
    pass


def provide_footage_from_fake_agent(samples, venv, agent=None, discrim=None) -> float:
    # return confidance level (has to use airl)
    if not agent:
        agent = train_agent_learnt_reward(samples, venv)
    return 0
