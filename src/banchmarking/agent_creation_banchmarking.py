import matplotlib.pyplot as plt
from imitation.data.rollout import flatten_trajectories_with_rew, generate_trajectories, make_min_timesteps
from imitation.data.types import *

from src.utils.imitation_connector import *
from src.alogirhms.airl import *
from typing import Sequence, List
from src.utils.agent_utils import generate_trajectory_footage
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback


def fake_agent_classification(agent, disc_func, agents_to_asses: Sequence, labels: Sequence, action_space_size,
                              venv, num_trajectories, num_bins=100, label_size=20):

    assert len(agents_to_asses) == len(labels)
    confidences = []
    for i, assessed_agent in enumerate(agents_to_asses):
        traj = flatten_trajectories_with_rew(generate_trajectories(assessed_agent, venv, make_min_timesteps(num_trajectories)))
        confidences.append(traj_confidence(traj, disc_func, agent, action_space_size))
        print('finished assessing ' + str(labels[i]) + ' - avg confidence: ' + str(confidences[i].mean()))
    bins = np.linspace(0, 1, num_bins)
    for i, confidence in enumerate(confidences):
        plt.hist(confidence, alpha=0.5, bins=bins, label=labels[i], density=True)
    plt.legend()
    plt.xticks(fontsize=label_size)
    plt.xlabel('Confidence', fontsize=label_size + 2)
    # plt.ylim(0, 100)
    plt.show()


def traj_confidence(samples: Transitions, disc_func, agent, action_space_size):
    discriminator = discriminator_conversion(disc_func, agent, action_space_size)
    confidence = discriminator(samples.obs, samples.acts, samples.next_obs, samples.dones)
    return confidence


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


def generate_fake_agents(venv, algos: List, num_agents_per_algo, algo_kwargs: List, dictionary_save_path,
                         stopping_points, initial_name='Fake_agent_real_reward', max_timestep=pow(2, 17)) -> List:
    model_list = []
    curr_agent = 0
    for i in range(num_agents_per_algo):
        for agent in algos:
            file_path = dictionary_save_path + initial_name + "_" + (curr_agent % len(algos))
            model_list.append(generate_agent(venv, agent, algo_kwargs[curr_agent % len(algos)]
                                             , stopping_points[curr_agent], file_path, max_timestep))
            curr_agent = curr_agent + 1
    return model_list


def generate_agent(venv, algo, algo_kwards, stopping_point, model_path, max_timestep):
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=stopping_point, verbose=0)
    eval_callback = EvalCallback(venv, callback_on_new_best=callback_on_best, verbose=0)
    model = algo(env=venv, verbose=1, **algo_kwards)
    model.learn(total_timesteps=max_timestep, callback=eval_callback)
    if model_path:
        model.save(model_path)
    return model
