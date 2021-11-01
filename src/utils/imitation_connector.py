from imitation.data.rollout import TrajectoryAccumulator


def to_im_traj(arr):
    # return list of trajections, arr = 4xN np array
    # arr[0] = actions, arr[1] = observations, arr[2] = rewards, arr[3] = dones
    return TrajectoryAccumulator().add_steps_and_auto_finish(arr[0], arr[1], arr[2], arr[3], [])


