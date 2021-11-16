import numpy as np
import imageio


def generate_trajectory_footage(agent, env, save_path=None, length=1, subsampling=True):
    # returns (obj, actions, next_obs, dones, rewards) of the trajectory
    done = False
    obs = env.reset()
    past_obs, past_rewards, next_obs, actions, dones, frames = [obs], [], [], [], [], []
    while not done:
        if not agent:
            action = [env.action_space.sample()]
        else:
            action, _ = agent.predict(obs)
        obs, rewards, done, info = env.step(action)
        next_obs.append(obs)
        past_rewards.append(rewards)
        past_obs.append(obs)
        actions.append(action)
        dones.append(done)
        if save_path:
            frames.append(env.render(mode='rgb_array'))
    if save_path:
        fps = 60
        sampling = int(len(frames)/(length * fps))
        imageio.mimsave(save_path, [img for i, img in enumerate(np.array(frames)) if not subsampling or i %
                                    sampling == 0],
                        fps=fps)
    return np.array(past_obs[:-1]).squeeze(), np.array(actions).squeeze(), np.array(next_obs).squeeze(),\
           np.array(dones).squeeze(), np.array(past_rewards).squeeze()
