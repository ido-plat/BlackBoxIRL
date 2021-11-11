import numpy as np
import imageio


def generate_trajectory_footage(agent, env, save_path=None):
    # returns (obj, actions, next_obs, dones, rewards) of the trajectory
    done = False
    obs = env.reset()
    past_obs, past_rewards, next_obs, actions, dones = [obs], [], [], [], []
    while not done:
        action, _ = agent.predict(obs)
        obs, rewards, done, info = env.step(action)
        next_obs.append(obs)
        past_rewards.append(rewards)
        past_obs.append(obs)
        actions.append(action)
        dones.append(done)
    if save_path:
        imageio.mimsave(save_path, [img for img in np.array(past_obs)], fps=29)
    return np.array(past_obs[:-1]), np.array(actions), np.array(next_obs), np.array(dones), np.array(past_rewards)
