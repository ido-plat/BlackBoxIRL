import unittest
import gym
import cv2
import numpy as np
import imageio
from stable_baselines3 import DQN, A2C, PPO
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3.common.evaluation import evaluate_policy
import atari_py.import_roms
from stable_baselines3.common.env_util import make_atari_env

class MyTestCase(unittest.TestCase):

    def test_agent_training_render(self):
        model_path = 'rl-baselines3-zoo/rl-trained-agents/a2c/LunarLander-v2_1/LunarLander-v2.zip'
        from_file = True
        game = 'LunarLander-v2'
        gif_path = None  # put None to not save
        learning_time_stemp = int(1e6)
        render = False
        render_frames = 5000
        gif_subsampling = 20 #   once every n frames will be in the gif
        model_type = A2C
        # buffer_size = int(1e6)
        n_env = 1
        # env = make_atari_env(game, n_env, seed=0)
        env = gym.make(game)
        imgs = []
        # Instantiate the agent
        if from_file:
            model = model_type.load(model_path, env=env)
        else:
            model = model_type('CnnPolicy', env, verbose=1)
            model.learn(total_timesteps=learning_time_stemp)
            model.save(model_path)
        print('generated model')
        # Evaluate the agent
        # NOTE: If you use wrappers with your environment that modify rewards,
        #       this will be reflected here. To evaluate with original rewards,
        #       wrap environment in a "Monitor" wrapper before other wrappers.
        # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        obs = env.reset()
        reward_past = []
        t = 0
        for i in trange(render_frames):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if gif_path is not None and i % gif_subsampling == 1:
                imgs.append(env.render(mode='rgb_array'))
            elif render:
                env.render()
                print(rewards)
            reward_past.append(rewards)
            if dones:
                obs = env.reset()
        reward_past = np.array(reward_past)
        print(reward_past.sum())
        plt.hist(reward_past, bins=500, density=True)
        plt.show()

        if gif_path is not None:
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(imgs) if i % 2 == 0], fps=29)


if __name__ == '__main__':
    unittest.main()
