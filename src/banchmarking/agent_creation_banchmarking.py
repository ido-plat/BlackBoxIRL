import matplotlib.pyplot as plt
from imitation.data.rollout import flatten_trajectories_with_rew, generate_trajectories, make_min_timesteps
from imitation.data.types import *

from src.utils.imitation_connector import *
from src.alogirhms.airl import *
from typing import Sequence
from src.utils.agent_utils import generate_trajectory_footage


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
