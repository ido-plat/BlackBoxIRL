from src.banchmarking.reward_aprox_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType

class BenchMarkTest(unittest.TestCase):
    def test_reward_hist(self):
        env = 'CartPole-v0'
        num_env = 1
        agent_path = 'data/agents/CartPole-v1_dqn.zip'
        agent_training_alg = DQN
        num_traj = 1e4
        venv = util.make_vec_env(env, n_envs=num_env)
        agent = agent_training_alg.load(agent_path, venv)
        args = {'dense_type': DensityType.STATE_DENSITY}
        real, fake = check_reward_distribution(agent, density_aprox, venv, num_traj, plot_hist=True, args=args)

if __name__ == '__main__':
    unittest.main()
