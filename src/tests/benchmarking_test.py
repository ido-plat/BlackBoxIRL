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
from src.tests.fake_agent_test_eval import generate_fake_list
from src.utils.confidence_plots import *
class BenchMarkTest(unittest.TestCase):
    def setUp(self) -> None:
        if Config.env_max_timestep is not np.inf:
            self.venv = make_fixed_horizon_venv(Config.env, max_episode_steps=Config.env_max_timestep, n_envs=Config.num_env)
        else:
            self.venv = make_vec_env(Config.env, Config.num_env)
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)
        self.noise = None

    def test_reward_hist(self):
        config = Config
        num_traj = config.num_transitions
        reward_func_path = 'data/reward_functions/LunarLander-v2_reward_func2'
        reward_func = load_reward_net(reward_func_path)
        real, fake, diff = check_reward_distribution(self.expert, reward_func, self.venv, num_traj, plot_hist=True,
                                                     xlim=(-5, 5))

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/pend_gif2.gif'
        config = Config
        agent_path = 'src/tests/temp/real_reward_agent.zip'
        agent_training_alg = config.agent_training_algo
        agent = agent_training_alg.load(agent_path, self.venv)
        obs, act, next_obs, done, rewards = generate_trajectory_footage(agent, self.venv, gif_path, subsampling=False)
        # plt.hist(act)
        # plt.show()

    def fake_agent_creation(self, save_agent_path, save_disc_func_path, save_iterated_agent_path, save_reward_function):
        config = Config
        agent_training_alg = config.agent_training_algo
        airl_arg = config.airl_args.copy()
        airl_arg['save_disc_path'] = save_disc_func_path
        airl_arg['save_reward_func_path'] = save_reward_function
        airl_arg['save_iagent_path'] = save_iterated_agent_path
        samples1 = flatten_trajectories_with_rew(generate_trajectories(self.expert, self.venv, make_min_timesteps(config.airl_num_transitions)))
        print('generated samples')
        train_agent_learnt_reward(samples1, self.venv, agent_training_alg,
                                  learning_time_step=config.model_total_training_steps,
                                  model_arg=config.model_training_args, save_model_path=save_agent_path,
                                  return_disc=True, airl_args=airl_arg)

    def test_compare_expart_agent_noise(self):
        agent_path = 'data/agents/our_agents/LunarLander-v2_agent1.zip'
        agent2_path = 'data/agents/our_agents/LunarLander-v2_agent1.zip'
        iagent1_path = 'data/iagents/LunarLander-v2_iterative_agent1.zip'
        iagent2_path = 'data/iagents/LunarLander-v2_iterative_agent2.zip'
        agent = Config.agent_training_algo.load(agent_path, self.venv, custom_objects=Config.expert_custom_objects)
        agent2 = Config.agent_training_algo.load(agent2_path, self.venv, custom_objects=Config.expert_custom_objects)
        iagent = Config.iterative_agent_training_algo.load(iagent1_path, self.venv,
                                                           custom_objects=Config.expert_custom_objects)
        iagent2 = Config.iterative_agent_training_algo.load(iagent2_path, self.venv,
                                                           custom_objects=Config.expert_custom_objects)
        agents = [agent, agent2, iagent, iagent2]
        labels = ["Agent 1", 'Agent 2', 'Iagent 1', "Iagent 2"]
        for a, l in zip(agents, labels):
            reward = get_agent_avg_reward(a, self.venv, Config.num_transitions)
            print(l + ' mean reward ' + str(reward))


    def test_train_model(self):
        config = Config
        gif_path = 'src/tests/temp/training_agent.gif'
        rl_args = config.model_training_args
        rl_algo = config.agent_training_algo
         # agent = train_agent(self.venv, rl_algo, config.model_total_training_steps, rl_args)
        # agent_save_path = 'src/tests/temp/real_reward_agent'
        # agent.save(agent_save_path)
        agent_load_path = 'src/tests/temp/real_reward_agent2.zip'
        agent = config.agent_training_algo.load(agent_load_path)
        avg_rewards = get_agent_avg_reward(agent, self.venv, config.num_transitions)
        print('avg reward: '+str(avg_rewards))
        generate_trajectory_footage(agent, self.venv, gif_path)

    def test_partial_pipelie(self):
        agent_path = 'data/agents/our_agents/LunarLander-v2_fake_agent1.zip'
        agent2_path = 'data/agents/our_agents/LunarLander-v2_fake_agent2.zip'
        iagent_path = 'data/iagents/LunarLander-v2_iterative_agent2.zip'
        disc_func_path = 'data/disc_functions/disc_func1'
        disc_func2_path = 'data/disc_functions/disc_func2'

        agent = Config.agent_training_algo.load(agent_path, self.venv, custom_objects=Config.expert_custom_objects)
        agent2 = Config.agent_training_algo.load(agent2_path, self.venv, custom_objects=Config.expert_custom_objects)
        iagent = Config.iterative_agent_training_algo.load(iagent_path, self.venv,
                                                           custom_objects=Config.expert_custom_objects)
        fake_agent_classification(agent2, load_disc_func(disc_func_path), [agent, self.expert, iagent],
                                  ['Agent', 'Expert', 'iterative agent'], Config.env_action_space_size,
                                  self.venv, Config.num_transitions)

    def test_full_pipeline(self):
        print("starting full pipeline")
        agent1_save_path = 'data/agents/our_agents/LunarLander-v2_agent1'
        agent2_save_path = 'data/agents/our_agents/LunarLander-v2_agent2'
        iagent1_save_path = 'data/iagents/LunarLander-v2_iterative_agent1'
        iagent2_save_path = 'data/iagents/LunarLander-v2_iterative_agent2'
        reward1_func_path = 'data/reward_functions/LunarLander-v2_reward_func1'
        reward2_func_path = 'data/reward_functions/LunarLander-v2_reward_func2'
        disc1_save_path = 'data/disc_functions/disc_func1'
        disc2_save_path = 'data/disc_functions/disc_func2'
        self.fake_agent_creation(agent1_save_path, disc1_save_path, iagent1_save_path, reward1_func_path)
        print('finished creating first agent, starting second')
        self.fake_agent_creation(agent2_save_path, disc2_save_path, iagent2_save_path, reward2_func_path)
        #                                  finished pipline, creating result visualisation
        save_dir = 'src/data/result_plots/'
        agent_list_path = [agent1_save_path, agent2_save_path, iagent1_save_path, iagent2_save_path]
        label_list = ["Agent1", "Agent2", "Iagent1", "Iagent2"]
        algo_list = [Config.agent_training_algo, Config.agent_training_algo, Config.iterative_agent_training_algo,
                     Config.iterative_agent_training_algo]
        disc_func_lst = [disc1_save_path, disc2_save_path]
        interesting_agents = [0, 1]
        self._analyze_results(agent_list_path, label_list, algo_list, disc_func_lst, save_dir, interesting_agents)

    def _analyze_results(self, agents_path, agents_label, algo_list, disc_function_path_list, save_dir,
                         distribution_agents_index):
        num_agents = len(agents_path)
        agents = [algo_list[i].load(agents_path[i]) for i in range(num_agents)]
        for n_disc, disc_function_path in enumerate(disc_function_path_list):
            disc_func = load_disc_func(disc_function_path)
            fakes, labels = generate_fake_list()
            for i in range(num_agents):
                for k in range(num_agents):
                    if i != k:
                        temp_labels = labels + [labels[k], "Expert"]
                        temp_agents = fakes + [agents[k], self.expert]
                        path = save_dir + "confidence_mean_plot_disc_" + str(n_disc) + "disc_setting_agent_" +\
                               agents_label[i] + "_agent" + agents_label[k] + ".png"
                        fake_agent_classification(agents[i], disc_func, temp_agents, temp_labels,
                                                  Config.env_action_space_size, self.venv, Config.num_transitions,
                                                  plot_function=plot_bar_mean, agent_color='r', save_path=path)
                        if k in distribution_agents_index:
                            temp_labels = [temp_labels[i], temp_labels[k], "Expert"]
                            temp_agents = [agents[i], agents[k], self.expert]
                            path = save_dir + "confidence_distribution_plot_disc" + str(n_disc) + "disc_setting_agent_" + \
                                   agents_label[i] + "_agent" + agents_label[k] + ".png"
                            fake_agent_classification(agents[i], disc_func, temp_agents, temp_labels,
                                                      Config.env_action_space_size, self.venv, Config.num_transitions,
                                                      plot_function=plot_distribution, save_path=path)
            for k in range(num_agents):
                temp_labels = labels + [labels[k], "Expert"]
                temp_agents = fakes + [agents[k], self.expert]
                path = save_dir + "confidence_mean_plot_disc_" + str(n_disc) + "disc_setting_agent_NONE_agent" \
                       + agents_label[k] + ".png"
                fake_agent_classification(None, disc_func, temp_agents, temp_labels,
                                          Config.env_action_space_size, self.venv, Config.num_transitions,
                                          plot_function=plot_bar_mean, agent_color='r', save_path=path)
                if k in distribution_agents_index:
                    temp_labels = [temp_labels[k], "Expert"]
                    temp_agents = [agents[k], self.expert]
                    path = save_dir + "confidence_distribution_plot_disc" + str(n_disc) + "disc_setting_agent_NONE_agent"\
                           + agents_label[k] + ".png"
                    fake_agent_classification(None, disc_func, temp_agents, temp_labels,
                                              Config.env_action_space_size, self.venv, Config.num_transitions,
                                              plot_function=plot_distribution, save_path=path)


if __name__ == '__main__':
    t = BenchMarkTest()
    t.test_full_pipeline()
