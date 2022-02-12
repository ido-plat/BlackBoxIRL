import numpy as np

from src.banchmarking.reward_aprox_banchmarking import *
from src.banchmarking.agent_creation_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import *
from src.utils.agent_utils import generate_trajectory_footage
from src.utils.env_utils import make_fixed_horizon_venv, make_vec_env
from src.config import Config
from imitation.data.rollout import rollout_stats, make_min_episodes
from src.tests.fake_agent_test_eval import generate_fake_list
from stable_baselines3.common.atari_wrappers import *
from src.utils.confidence_plots import *
from stable_baselines3.common.vec_env import VecTransposeImage
import gym
from tqdm import trange
from src.utils.env_utils import SpaceInvadersEnv
class BenchMarkTest(unittest.TestCase):
    def setUp(self) -> None:
        venv_generator = SpaceInvadersEnv(Config.env, Config.num_env, None, Config.env_max_timestep, True)
        self.venv = venv_generator.make_venv()
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)
        self.noise = None


    def get_agent_traj_len(self, n, agent):
        traj = [rollout_stats(generate_trajectories(agent, self.venv, make_min_episodes(3)))['len_mean'] for _ in trange(n)]
        print("done generating traj")
        return traj

    def test_plot_duration(self):
        big_num = 1000
        n_bins = 31
        lens = self.get_agent_traj_len(big_num, self.expert)
        f = open('src/tests/temp/means.txt', 'w')
        print(np.array(lens).mean(), file=f)
        plt.hist(np.array(lens), n_bins)
        f.close()
        plt.show()

def print_make_min_episodes(n: int):
    """Terminate after collecting n episodes of data.

    Args:
        n: Minimum number of episodes of data to collect.
            May overshoot if two episodes complete simultaneously (unlikely).

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1
    def t(trajectories):
        if len(trajectories) % 10 == 0:
            print(f"im at {len(trajectories)}")
        return len(trajectories) >= n
    return t

