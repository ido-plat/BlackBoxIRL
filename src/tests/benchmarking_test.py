import matplotlib.pyplot as plt

from src.banchmarking.reward_aprox_banchmarking import *
from src.banchmarking.agent_creation_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
import os
from sb3_contrib import QRDQN
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import *
from sb3_contrib import QRDQN
from src.utils.agent_utils import generate_trajectory_footage
from src.utils.env_utils import SpaceInvadersEnv
from src.config import Config, print_to_cfg_log
from src.tests.fake_agent_test_eval import generate_fake_list
from src.utils.confidence_plots import *
from src.transitions.db_transitions import make_db_using_config, TransitionsDB
from src.transitions.airl_eval_db import make_eval_db_from_config


class BenchMarkTest(unittest.TestCase):
    def setUp(self) -> None:
        # if Config.env_max_timestep is not np.inf:
        #     self.venv = make_fixed_horizon_venv(Config.env, max_episode_steps=Config.env_max_timestep, n_envs=Config.num_env)
        # else:
        #     self.venv = make_vec_env(Config.env, Config.num_env)
        venv_generator = SpaceInvadersEnv(max_timestemp=Config.env_max_timestep)
        self.venv = venv_generator.make_venv()
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)
        self.noise = None
        other_expert_path = ['data/SpaceInvadersNoFrameskip-v4/agents/other_experts/SpaceInvadersNoFrameskip-v4_A2C.zip',
                             'data/SpaceInvadersNoFrameskip-v4/agents/other_experts/SpaceInvadersNoFrameskip-v4_PPO.zip',
                             'data/SpaceInvadersNoFrameskip-v4/agents/other_experts/SpaceInvadersNoFrameskip-v4_QRDQN.zip'
                             ]
        other_expert_algo = [A2C, PPO, QRDQN]
        self.other_experts = [a.load(b, custom_objects=Config.expert_custom_objects)
                              for a, b in zip(other_expert_algo, other_expert_path)]
        self.other_experts_labels = ['A2C', 'PPO', "QRDQN"]
        self.reward_net = ClassificationShapedRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            n_classes=2
        )

    def test_reward_hist(self):
        config = Config
        num_traj = config.num_transitions
        reward_func_path = 'data/reward_functions/LunarLander-v2_reward_func2'
        reward_func = load_reward_net(reward_func_path)
        real, fake, diff = check_reward_distribution(self.expert, reward_func, self.venv, num_traj, plot_hist=True,
                                                     xlim=(-5, 5))

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/space.gif'
        agent_path = 'src/tests/temp/real_reward_agent.zip'
        # agent_training_alg = config.agent_training_algo
        # agent = agent_training_alg.load(agent_path, self.venv)

        obs, act, next_obs, done, rewards = generate_trajectory_footage(self.expert, self.venv, gif_path, subsampling=False)
        print(len(done))
        # plt.hist(act)
        # plt.show()

    def fake_agent_creation(self, save_agent_path, save_disc_func_path, save_iterated_agent_path, save_reward_function,
                            use_db, db_filename='', index=0, rewrite_db_file=False, eval_db_path='', eval_result_path='',
                            mode='', other_experts=()):
        config = Config
        agent_training_alg = config.agent_training_algo
        airl_arg = config.airl_args.copy()
        airl_arg['save_disc_path'] = save_disc_func_path
        airl_arg['save_reward_func_path'] = save_reward_function
        airl_arg['save_iagent_path'] = save_iterated_agent_path
        airl_arg['reward_net'] = self.reward_net
        print_to_cfg_log('starting to generate samples')
        samples = make_db_using_config(db_filename, index, rewrite_db_file, self.expert, self.venv) if use_db else \
                  flatten_trajectories_with_rew(generate_trajectories(self.expert, self.venv, make_min_timesteps(config.airl_num_transitions)))

        print_to_cfg_log('generated samples')
        db = None
        if eval_db_path:
            db = make_eval_db_from_config(eval_db_path, eval_result_path, self.expert, self.venv, mode,
                                          other_expert=other_experts)
            print_to_cfg_log('Made eval DB')
        train_agent_learnt_reward(samples, self.venv, agent_training_alg,
                                  learning_time_step=config.model_total_training_steps,
                                  model_arg=config.model_training_args, save_model_path=save_agent_path,
                                  return_disc=True, airl_args=airl_arg, evalDB=db)
        if eval_db_path:
            if mode != 'train':
                db.plot_result()
            db.close()
        if use_db:
            samples.close()

    def test_compare_expart_agent_noise(self):
        print_to_cfg_log("Starting test compare")
        agent_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent1.zip'
        agent2_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent2.zip'
        iagent1_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent1.zip'
        iagent2_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent2.zip'
        expert_path = 'data/SpaceInvadersNoFrameskip-v4/agents/SpaceInvadersNoFrameskip-v4_DQN_Expert.zip'
        # agent = Config.agent_training_algo.load(agent_path, self.venv, custom_objects=Config.expert_custom_objects)
        # agent2 = Config.agent_training_algo.load(agent2_path, self.venv, custom_objects=Config.expert_custom_objects)
        expert = Config.expert_training_algo.load(expert_path, self.venv, custom_objects=Config.expert_custom_objects)
        # iagent = Config.iterative_agent_training_algo.load(iagent1_path, self.venv,
        #                                                    custom_objects=Config.expert_custom_objects)
        # iagent2 = Config.iterative_agent_training_algo.load(iagent2_path, self.venv,
        #                                                    custom_objects=Config.expert_custom_objects)
        # agents = [expert, self.noise, agent, agent2, iagent, iagent2]
        # labels = ["expert", "noise", "Agent 1", 'Agent 2', 'Iagent 1', "Iagent 2"]
        agents, labels = [expert] + self.other_experts, ["Expert"] + self.other_experts_labels
        f = open(f'{os.path.dirname(Config.log_file)}/rewards.txt', 'w')
        for a, l in zip(agents, labels):
            reward = get_agent_avg_reward(a, self.venv, Config.num_transitions)
            print(l + ' mean reward ' + str(reward))
            print(l + ' mean reward ' + str(reward), file=f)
        f.close()

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
        print_to_cfg_log('avg reward: '+str(avg_rewards))
        generate_trajectory_footage(agent, self.venv, gif_path)

    def test_single_classification(self):
        print_to_cfg_log('Starting test classification')
        agent_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent2'
        disc_path = 'data/SpaceInvadersNoFrameskip-v4/disc_functions/disc_func1'
        reference_agent_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent1'
        save_path = 'data/SpaceInvadersNoFrameskip-v4/result_plots/res3.png'
        agent_label = 'Agent 1'
        num_chunks = 100
        agent = Config.agent_training_algo.load(agent_path)
        ref_agent = Config.agent_training_algo.load(reference_agent_path)
        disc_func = load_disc_func(disc_path)
        fakes, labels = self.other_experts, self.other_experts_labels
        fakes += [agent, self.expert]
        labels += [agent_label, "Expert"]
        fake_agent_classification(ref_agent, disc_func, fakes, labels,
                                  Config.env_action_space_size, self.venv, Config.num_transitions,
                                  plot_function=plot_bar_mean, save_path=save_path, device='cuda', out_file='out.txt',
                                  print_assesement=True, agent_color='r', expert_color='g', num_chunks=num_chunks)

    def test_partial_pipeline(self):
        print_to_cfg_log("starting partial pipeline")
        save_dir = 'data/SpaceInvadersNoFrameskip-v4/result_plots/'
        save_dir = os.path.abspath(save_dir)
        agent1_save_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent1'
        agent2_save_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent2'
        iagent1_save_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent1'
        iagent2_save_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent2'
        disc1_save_path = 'data/SpaceInvadersNoFrameskip-v4/disc_functions/disc_func1'
        disc2_save_path = 'data/SpaceInvadersNoFrameskip-v4/disc_functions/disc_func2'
        agent_list_path = [agent1_save_path, agent2_save_path, iagent1_save_path, iagent2_save_path]
        label_list = ["Agent1", "Agent2", "Iagent1", "Iagent2"]
        algo_list = [Config.agent_training_algo, Config.agent_training_algo, Config.iterative_agent_training_algo,
                     Config.iterative_agent_training_algo]
        disc_func_lst = [disc1_save_path, disc2_save_path]
        interesting_agents = [0, 1]
        self._analyze_results(agent_list_path, label_list, algo_list, disc_func_lst, save_dir, interesting_agents, num_chunks=100)

    def test_full_pipeline(self):
        print_to_cfg_log("starting full pipeline")
        agent1_save_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent1'
        agent2_save_path = 'data/SpaceInvadersNoFrameskip-v4/agents/our_agents/SpaceInvaders-v4_agent2'
        iagent1_save_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent1'
        iagent2_save_path = 'data/SpaceInvadersNoFrameskip-v4/iagents/SpaceInvaders-v4_iterative_agent2'
        reward1_func_path = 'data/SpaceInvadersNoFrameskip-v4/reward_functions/SpaceInvaders-v4_reward_func1'
        reward2_func_path = 'data/SpaceInvadersNoFrameskip-v4/reward_functions/SpaceInvaders-v4_reward_func2'
        disc1_save_path = 'data/SpaceInvadersNoFrameskip-v4/disc_functions/disc_func1'
        disc2_save_path = 'data/SpaceInvadersNoFrameskip-v4/disc_functions/disc_func2'
        db_file = 'data/SpaceInvadersNoFrameskip-v4/transitions_db/DB_SpaceInvadersNoFrameskip-v4.h5'
        eval_db_path = 'data/SpaceInvadersNoFrameskip-v4/eval_db/evalDB_SpaceInvadersNoFrameskip-v4.h5'
        eval_result_path = 'data/SpaceInvadersNoFrameskip-v4/result_plots/eval_result.png'
        eval_result_path = os.path.abspath(eval_result_path)
        print_to_cfg_log("About to start making first agent")
        self.fake_agent_creation(agent1_save_path, disc1_save_path, iagent1_save_path, reward1_func_path, Config.use_db, db_file,
                                 0, False, eval_db_path, eval_result_path, 'train', other_experts=self.other_experts)
        print_to_cfg_log('finished creating first agent, starting second')
        self.fake_agent_creation(agent2_save_path, disc2_save_path, iagent2_save_path, reward2_func_path, Config.use_db, db_file,
                                 1, False, eval_db_path, eval_result_path, 'eval', other_experts=self.other_experts)

    def _analyze_results(self, agents_path, agents_label, algo_list, disc_function_path_list, save_dir,
                         distribution_agents_index, use_fakes=True, device='cuda', num_chunks=1):
        num_agents = len(agents_path)
        agents = [algo_list[i].load(agents_path[i]) for i in range(num_agents)]
        for n_disc, disc_function_path in enumerate(disc_function_path_list):
            disc_func = load_disc_func(disc_function_path, device)
            fakes, labels = generate_fake_list()
            for i in range(num_agents):
                for k in range(num_agents):
                    if i != k:
                        temp_labels = labels + [agents_label[k], "Expert"]
                        temp_agents = fakes + [agents[k], self.expert]
                        path = save_dir + "confidence_mean_plot_disc_" + str(n_disc) + "disc_setting_agent_" +\
                               agents_label[i] + "_agent" + agents_label[k] + ".png"
                        fake_agent_classification(agents[i], disc_func, temp_agents, temp_labels,
                                                  Config.env_action_space_size, self.venv, Config.num_transitions,
                                                  plot_function=plot_bar_mean, agent_color='r', expert_color='g',
                                                  save_path=path,
                                                  print_assesement=False, device=device, num_chunks=num_chunks)
                        print_to_cfg_log('finished creating ' + path)
                        if k in distribution_agents_index:
                            temp_labels = [agents_label[i], agents_label[k], "Expert"]
                            temp_agents = [agents[i], agents[k], self.expert]
                            path = save_dir + "confidence_distribution_plot_disc" + str(n_disc) + "disc_setting_agent_" + \
                                   agents_label[i] + "_agent" + agents_label[k] + ".png"
                            fake_agent_classification(agents[i], disc_func, temp_agents, temp_labels,
                                                      Config.env_action_space_size, self.venv, Config.num_transitions,
                                                      plot_function=plot_distribution, save_path=path,
                                                      print_assesement=False, device=device, num_chunks=num_chunks)
                            print_to_cfg_log('finished creating ' + path)
            for k in range(num_agents):
                temp_labels = labels + [agents_label[k], "Expert"]
                temp_agents = fakes + [agents[k], self.expert]
                path = save_dir + "confidence_mean_plot_disc_" + str(n_disc) + "disc_setting_agent_NONE_agent" \
                       + agents_label[k] + ".png"
                fake_agent_classification(None, disc_func, temp_agents, temp_labels,
                                          Config.env_action_space_size, self.venv, Config.num_transitions,
                                          plot_function=plot_bar_mean, agent_color='r', expert_color='g',
                                          save_path=path,
                                          print_assesement=False, device=device, num_chunks=num_chunks)
                print_to_cfg_log('finished creating ' + path)
                if k in distribution_agents_index:
                    temp_labels = [agents_label[k], "Expert"]
                    temp_agents = [agents[k], self.expert]
                    path = save_dir + "confidence_distribution_plot_disc" + str(n_disc) + "disc_setting_agent_NONE_agent"\
                           + agents_label[k] + ".png"
                    fake_agent_classification(None, disc_func, temp_agents, temp_labels,
                                              Config.env_action_space_size, self.venv, Config.num_transitions,
                                              plot_function=plot_distribution, save_path=path,
                                              print_assesement=False, device=device, num_chunks=num_chunks)
                    print_to_cfg_log('finished creating ' + path)


    def test_make_plots(self, run_indx=0):
        files = ['run_logs/other_expert_disc_eval.txt', 'run_logs/reward_func_acc.txt']
        labels = ['other expert acc', 'reward function acc']
        for i, path in enumerate(files):
            with open(path) as f:
                lines = f.readlines()
                index = [ind for ind, ele in enumerate(lines) if ele.startswith('-')]
                index = index[0] if len(index) > 0 else 10000
                if run_indx == 0:
                    data = [float(item.strip()) for item in lines[:index]]
                else:
                    data = [float(item.strip()) for item in lines[index+1:]]
                plt.plot(data, label=labels[i])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    t = BenchMarkTest()
    t.test_full_pipeline()
