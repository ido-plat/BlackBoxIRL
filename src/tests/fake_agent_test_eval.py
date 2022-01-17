from src.banchmarking.reward_aprox_banchmarking import *
from src.banchmarking.agent_creation_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
import os
from stable_baselines3 import DQN, A2C, PPO
from imitation.util import util
from imitation.algorithms.density import DensityType
from src.banchmarking.reward_aprox_banchmarking import *
from src.banchmarking.agent_creation_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from src.alogirhms.airl import *
from src.utils.agent_utils import generate_trajectory_footage
from src.utils.env_utils import make_fixed_horizon_venv, make_vec_env
from src.config import Config
from src.utils.confidence_plots import plot_bar_mean
from src.utils.env_utils import SpaceInvadersEnv

class FakeAgentTestEval(unittest.TestCase):
    def setUp(self) -> None:
        # if Config.env_max_timestep is not np.inf:
        #     self.venv = make_fixed_horizon_venv(Config.env, max_episode_steps=Config.env_max_timestep,
        #                                         n_envs=Config.num_env)
        # else:
        #     self.venv = make_vec_env(Config.env, Config.num_env)
        venv_generator = SpaceInvadersEnv(Config.env, Config.num_env, None, Config.env_max_timestep, True)
        self.venv = venv_generator.make_venv()
        self.co = Config.expert_custom_objects
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv, custom_objects=self.co)
        self.noise = None
        self.algo_list = [DQN, PPO]
        self.num_algo = len(self.algo_list)
        self.save_dictionary_path = 'data/SpaceInvadersNoFrameskip-v4/fakes/'
        self.plot_function = plot_bar_mean

    def test_fake_agents_creation(self):
        assert Config.env_max_timestep == np.inf
        kwargs_alg = [Config.all_model_training_args[Config.env][alg] for alg in self.algo_list]
        num_agents_per = 3
        expert_mean_reward = 0.03624605
        noise_mean = 0.018624349
        num_agents = num_agents_per * self.num_algo
        expert_len = 360
        noise_len = 168
        int_len = [round((expert_len - noise_len) * (i + 1) / (num_agents + 2) + noise_mean, 4)
                           for i in range(num_agents)]
        int_mean = [round((expert_mean_reward - noise_mean) * (i + 1) / (num_agents + 2) + noise_mean, 4)
                           for i in range(num_agents)]
        stopping_points = [a*b for a,b in zip(int_len, int_mean)]
        big_number = 1e10
        # stopping_points = [-big_number for _ in range(num_agents)]
        generate_fake_agents(self.venv, self.algo_list, num_agents_per, kwargs_alg, self.save_dictionary_path,
                             stopping_points, max_timestep=pow(2, 17))

    def test_fake_agent_eval(self):
        agent_path = 'data/agents/our_agents/LunarLander-v2_agent1'
        disc_func_path = 'data/disc_functions/disc_func2'
        disc_setting_agent_path = 'data/agents/our_agents/LunarLander-v2_agent2'
        iagent_path = 'data/iagents/LunarLander-v2_iterative_agent2'
        fakes_path = [self.save_dictionary_path + path for path in os.listdir(self.save_dictionary_path)]
        disc_func = load_disc_func(disc_func_path)
        iagent = Config.iterative_agent_training_algo.load(iagent_path, self.venv, custom_objects=self.co)
        agent = Config.agent_training_algo.load(agent_path, self.venv, custom_objects=self.co)
        disc_setting_agent = Config.agent_training_algo.load(disc_setting_agent_path, self.venv, custom_objects=self.co)
        fakes = [self._path_to_algo(path).load(path, self.venv, custom_objects=self.co) for path in fakes_path]
        labels = [self._path_to_label(path) for path in fakes_path]
        fakes.append(iagent)
        fakes.append(agent)
        fakes.append(self.expert)
        labels.append('Iterative agent')
        labels.append('Real Agent')
        labels.append('Expert')
        fake_agent_classification(disc_setting_agent, disc_func, fakes, labels, Config.env_action_space_size, self.venv,
                                  Config.num_transitions, plot_function=self.plot_function, agent_color='r')

    def test_mean_fake_score(self):
        fakes_path = [self.save_dictionary_path + path for path in os.listdir(self.save_dictionary_path)]
        fakes = [self._path_to_algo(path).load(path, self.venv) for path in fakes_path]
        f = open('src/tests/temp/means.txt', 'w')
        for i in range(len(fakes)):
            avg = get_agent_avg_reward(fakes[i], self.venv, Config.num_transitions)
            print(self._path_to_label(fakes_path[i]) + ' mean rewards ' + str(avg), file=f)
        f.close()

    def _path_to_label(self, string):
        ending = '.zip'
        return string[len(self.save_dictionary_path):-len(ending)]

    def _path_to_algo(self, string):
        for agent in self.algo_list:
            if agent.__name__ in string:
                return agent
        raise ValueError('%s Did not contain known algorithm' % string)
def generate_fake_list():
    t = FakeAgentTestEval()
    t.setUp()
    fakes_path = [t.save_dictionary_path + path for path in os.listdir(t.save_dictionary_path)]
    fakes = [t._path_to_algo(path).load(path, t.venv, custom_objects=t.co) for path in fakes_path]
    labels = [t._path_to_label(path) for path in fakes_path]
    return fakes, labels