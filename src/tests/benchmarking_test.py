from src.banchmarking.reward_aprox_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import airl

class BenchMarkTest(unittest.TestCase):

    def test_reward_hist(self):
        env = 'Pendulum-v0'
        num_env = 1
        agent_path = 'data/agents/Pendulum-v0_ppo.zip'
        agent_training_alg = PPO
        num_traj = 1e6
        args = {'policy_training_steps': 1024, 'total_timesteps': 2048}
        # def airl(samples, venv,policy_training_steps,total_timesteps , batch_size=32, logger=None):
        algo = airl
        venv = util.make_vec_env(env, n_envs=num_env)
        agent = agent_training_alg.load(agent_path, venv)
        real, fake = check_reward_distribution(agent, algo, venv, num_traj, plot_hist=True, args=args)

if __name__ == '__main__':
    unittest.main()
