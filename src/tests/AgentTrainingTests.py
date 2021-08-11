import unittest
import gym
import cv2
import numpy as np
import imageio
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


class MyTestCase(unittest.TestCase):

    def test_agent_training_render(self):
        model_path = 'src/tests/temp/dqn_lunar'  # make sure to run from root
        from_file = True
        game = 'LunarLander-v2'
        gif_path = 'src/tests/temp/lunar_gif.gif'  # put None to not save
        learning_time_stemp = int(2e5)
        render_frames = 10000
        gif_subsampling = 20 #   once every n frames will be in the gif
        env = gym.make(game)
        imgs = []
        # Instantiate the agent
        if from_file:
            model = DQN.load(model_path, env=env)
        else:
            model = DQN('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=learning_time_stemp)
            model.save(model_path)

        # Evaluate the agent
        # NOTE: If you use wrappers with your environment that modify rewards,
        #       this will be reflected here. To evaluate with original rewards,
        #       wrap environment in a "Monitor" wrapper before other wrappers.
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        obs = env.reset()
        for i in range(render_frames):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if gif_path is not None and i % gif_subsampling == 1:
                imgs.append(env.render(mode='rgb_array'))
            else:
                env.render()
        if gif_path is not None:
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(imgs) if i % 2 == 0], fps=29)


if __name__ == '__main__':
    unittest.main()
