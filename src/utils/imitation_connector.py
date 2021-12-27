from imitation.data.rollout import TrajectoryAccumulator
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Union


def to_im_traj(arr):
    # return list of trajections, arr = 4xN np array
    # arr[0] = actions, arr[1] = observations, arr[2] = rewards, arr[3] = dones
    return TrajectoryAccumulator().add_steps_and_auto_finish(arr[0], arr[1], arr[2], arr[3], [])


def discriminator_conversion(disc, agent, action_space_size, def_device='cuda:0'):
    def f(states, actions, next_states, done) -> Union[th.Tensor, np.ndarray]:
        was_np = isinstance(states, np.ndarray)
        if was_np:
            states = th.tensor(states, device=def_device)
            actions = th.tensor(actions, device=def_device)
            next_states = th.tensor(next_states, device=def_device)
            done = th.tensor(done, device=def_device)

        log_prob = log_prob_calc(states, actions, agent, action_space_size, def_device)
        if action_space_size is not np.inf: # discreate
            actions = discrete_action_conversion(actions, action_space_size)
        disc_result = disc(states.to(def_device), actions.to(def_device), next_states.to(def_device),
                           done.to(def_device), log_prob)
        if was_np:
            return 1 / (1 + th.exp(-disc_result)).to('cpu').numpy()
        return 1 / (1 + th.exp(-disc_result))
    return f


def log_prob_calc(states, actions, agent, action_space_size, def_device='cuda:0'):
    if not agent:
        return 0
    if isinstance(agent.policy, ActorCriticPolicy):
        _, log_prob, _ = agent.policy.evaluate_actions(states.to(def_device), actions.to(def_device))
    else:
        prob = th.ones(len(states))
        prob *= agent.exploration_rate / action_space_size
        np_states = states.to('cpu').numpy()
        predicted_actions = agent.predict(np_states, deterministic=True)[0]
        assert len(predicted_actions) == len(actions)
        indecies = th.tensor(predicted_actions, device=def_device) == actions
        prob[indecies] += 1 - agent.exploration_rate
        log_prob = th.log(prob)
    return log_prob.to(def_device)


# imitation used a different way to define actions in discrete space, instead of 3 -> it turns to [0, 0, 0, 1]
def discrete_action_conversion(action, action_space_len):
    action_to_return = th.zeros((len(action)), action_space_len)
    for i, val in enumerate(action.int()):  # maybe theres a way to make it a one liner, couldnt think of one.
        action_to_return[i][val] = 1
    return action_to_return
