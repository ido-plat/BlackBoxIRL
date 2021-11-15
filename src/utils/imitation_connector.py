from imitation.data.rollout import TrajectoryAccumulator
import numpy as np
import torch as th

def to_im_traj(arr):
    # return list of trajections, arr = 4xN np array
    # arr[0] = actions, arr[1] = observations, arr[2] = rewards, arr[3] = dones
    return TrajectoryAccumulator().add_steps_and_auto_finish(arr[0], arr[1], arr[2], arr[3], [])


def discriminator_conversion(disc, agent, action_space_size=None):
    def_device = 'cuda:0'

    def f(states, actions, next_states, done) -> np.ndarray:
        was_np = isinstance(states, np.ndarray)
        if was_np:
            states = th.tensor(states, device=def_device)
            actions = th.tensor(actions, device=def_device)
            next_states = th.tensor(next_states, device=def_device)
            done = th.tensor(done, device=def_device)
        if not agent:
            if action_space_size == np.inf:
                log_prob = 1
            else:
                log_prob = np.log(1/action_space_size)
        else:
            _, log_prob, _ = agent.policy.evaluate_actions(states.to(def_device), actions.to(def_device))
        disc_result = disc(states.to(def_device), actions.to(def_device), next_states.to(def_device),
                           done.to(def_device), log_prob)
        if was_np:
            return 1 / (1 + th.exp(-disc_result)).to('cpu').numpy()
        return 1 / (1 + th.exp(-disc_result))
    return f
