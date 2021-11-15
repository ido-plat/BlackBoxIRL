from imitation.data.rollout import TrajectoryAccumulator
import numpy as np
import torch as th

def to_im_traj(arr):
    # return list of trajections, arr = 4xN np array
    # arr[0] = actions, arr[1] = observations, arr[2] = rewards, arr[3] = dones
    return TrajectoryAccumulator().add_steps_and_auto_finish(arr[0], arr[1], arr[2], arr[3], [])


def im_disc_to_paper_disc(disc, agent, action_space_size=None):
    def f(states, actions, next_states, done) -> np.ndarray:
        if isinstance(states, np.ndarray):
            states = th.tensor(states)
            actions = th.tensor(actions)
            next_states = th.tensor(next_states)
            done = th.tensor(done)
        if not agent:
            if action_space_size == np.inf:
                log_prob = 1
            else:
                log_prob = np.log(1/action_space_size)
        else:
            _, log_prob, _ = agent.policy.evaluate_actions(states, actions)
        disc_result = disc(states, actions, next_states, done, log_prob)
        if isinstance(states, np.ndarray):
            return 1 / (1 + th.exp(-disc_result)).numpy()
        return 1 / (1 + th.exp(-disc_result))
    return f
