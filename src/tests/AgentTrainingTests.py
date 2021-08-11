import unittest
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


class MyTestCase(unittest.TestCase):

    def test_agent_training_render(self):
        model_path = 'src/tests/temp/dqn_lunar'  # make sure to run from root
        from_file = False
        learning_time_stemp = int(2e5)
        render_frames = 10000
        env = gym.make('LunarLander-v2')

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
            print(rewards)
            env.render()


if __name__ == '__main__':
    unittest.main()
