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
import glob


class FakeAgentTestEval(unittest.TestCase):
    def setUp(self) -> None:
        if Config.env_max_timestep is not np.inf:
            self.venv = make_fixed_horizon_venv(Config.env, max_episode_steps=Config.env_max_timestep,
                                                n_envs=Config.num_env)
        else:
            self.venv = make_vec_env(Config.env, Config.num_env)
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv)
        self.noise = None
        self.algo_list = [DQN, PPO]
        self.num_algo = len(self.algo_list)
        self.save_dictionary_path = 'src/tests/temp/fake_agents/'

    def test_fake_agents_creation(self):

        kwargs_alg = [Config.all_model_training_args[alg] for alg in self.algo_list]
        num_agents_per = 3
        # expert_mean_reward = 0.278
        expert_mean_reward = -0.3
        noise_mean = -0.33
        num_agents = num_agents_per * self.num_algo

        stopping_points = [round((expert_mean_reward - noise_mean) * (i + 1) / (num_agents + 2) + noise_mean, 4)
                           for i in range(num_agents)]

        generate_fake_agents(self.venv, self.algo_list, num_agents_per, kwargs_alg, self.save_dictionary_path,
                             stopping_points, max_timestep=pow(2, 15))

    def test_fake_agent_eval(self):
        agent_path = 'src/tests/temp/LunarLander-v2_fake_agent1'
        disc_func_path = 'src/tests/temp/disc_func2'
        fakes_path = glob.glob(self.save_dictionary_path)
        disc_func = load_disc_func(disc_func_path)
        agent = Config.agent_training_algo.load(agent_path, self.venv)
        fakes = [self._path_to_algo(path).load(path, self.venv) for path in fakes_path]
        labels = [self._path_to_label(path) for path in fakes_path]
        fakes.append(agent)
        fakes.append(self.expert)
        labels.append('Real Agent')
        labels.append('Expert')
        fake_agent_classification(agent, disc_func, fakes, labels, Config.env_action_space_size, self.venv,
                                  Config.num_transitions)

    def test_mean_fake_score(self):
        fakes_path = glob.glob(self.save_dictionary_path)
        fakes = [self._path_to_algo(path).load(path, self.venv) for path in fakes_path]
        for i in range(len(fakes)):
            avg = get_agent_avg_reward(fakes[i], self.venv, Config.num_transitions)
            print(self._path_to_label(fakes_path[i]) + ' mean rewards ' + str(avg))

    def _path_to_label(self, string):
        ending = '.zip'
        return string[len(self.save_dictionary_path):-len(ending)]

    def _path_to_algo(self, string):
        for agent in self.algo_list:
            if agent.__name__ in string:
                return agent
        raise ValueError('%s Did not contain known algorithm' % string)
