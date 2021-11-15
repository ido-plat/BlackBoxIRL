from src.banchmarking.reward_aprox_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import airl
from src.utils.agent_utils import generate_trajectory_footage


class BenchMarkTest(unittest.TestCase):
    class Config:
        env = 'Pendulum-v0'
        num_transitions = 1e5
        irl_alo = airl
        agent_training_algo = PPO
        airl_args = {
                        'policy_training_steps': 1024,
                        'total_timesteps': int(1e6)

                     }

    def test_reward_hist(self):
        config = self.Config
        env = config.env
        num_env = 1
        agent_path = 'data/agents/Pendulum-v0_ppo.zip'
        agent_training_alg = config.agent_training_algo
        num_traj = config.num_transitions
        args = config.airl_args
        algo = config.irl_alo
        venv = util.make_vec_env(env, n_envs=num_env)
        agent = agent_training_alg.load(agent_path, venv)
        real, fake = check_reward_distribution(agent, algo, venv, num_traj, plot_hist=True, args=args)

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/pend_gif.gif'
        env = self.Config.env
        num_env = 1
        agent_path = 'data/agents/Pendulum-v0_ppo.zip'
        agent_training_alg = self.Config.agent_training_algo
        venv = util.make_vec_env(env, n_envs=num_env)
        agent = agent_training_alg.load(agent_path, venv)
        obs, act, next_obs, done, rewards = generate_trajectory_footage(agent, venv, None)
        pass

    def test_fake_agent_creation(self):
        # create fake agent, disc
        # test agent with avg reward with benchmarking.compare_agents
        # save agent, disc
        pass

    def test_fake_agent_classification(self):
        # load disc
        # load fake agent
        # generate traj from fake agent
        # generate traj from real agent (expecting disc to give high values)
        # generate traj from noise (expecting disc to give low values)
        # test trajectories on disc
        # show hist
        pass

if __name__ == '__main__':
    unittest.main()
