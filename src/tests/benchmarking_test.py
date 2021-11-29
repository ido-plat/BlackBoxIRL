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

class BenchMarkTest(unittest.TestCase):
    def setUp(self) -> None:
        if Config.env_max_timestep is not np.inf:
            self.venv = make_fixed_horizon_venv(Config.env, max_episode_steps=Config.env_max_timestep, n_envs=Config.num_env)
        else:
            self.venv = make_vec_env(Config.env, Config.num_env)
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       costum_object=Config.expert_custom_objects)
        self.noise = None


    def test_reward_hist(self):
        config = Config
        env = config.env
        num_env = config.num_env
        agent_path = config.expert_path
        agent_training_alg = config.agent_training_algo
        num_traj = config.num_transitions
        args = config.airl_args
        algo = config.irl_alo
        real, fake = check_reward_distribution(self.expert, algo, self.venv, num_traj, plot_hist=True, args=args)

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/pend_gif2.gif'
        config = Config
        agent_path = 'src/tests/temp/real_reward_agent.zip'
        agent_training_alg = config.agent_training_algo
        agent = agent_training_alg.load(agent_path, self.venv)
        obs, act, next_obs, done, rewards = generate_trajectory_footage(agent, self.venv, gif_path, subsampling=False)
        # plt.hist(act)
        # plt.show()

    def fake_agent_creation(self, save_agent_path, save_disc_func_path):
        config = Config
        agent_training_alg = config.agent_training_algo
        num_traj = int(config.num_transitions)
        airl_arg = config.airl_args.copy()
        airl_arg['save_disc_path'] = save_disc_func_path
        samples1 = flatten_trajectories_with_rew(generate_trajectories(self.expert, self.venv, make_min_timesteps(num_traj)))
        print('generated samples')
        train_agent_learnt_reward(samples1, self.venv, agent_training_alg,
                                  learning_time_step=config.model_total_training_steps,
                                  model_arg=config.model_training_args, save_model_path=save_agent_path,
                                  return_disc=True, airl_args=airl_arg)

    def test_compare_expart_agent_noise(self):
        config = Config
        agent_path = 'src/tests/temp/LunarLander-v2_fake_agent1'
        agent_training_alg = config.agent_training_algo
        agent = agent_training_alg.load(agent_path, self.venv)
        expert_avg_reward = get_agent_avg_reward(self.expert, self.venv, config.num_transitions)
        agent_avg_reward = get_agent_avg_reward(agent, self.venv, config.num_transitions)
        noise_avg_reward = get_agent_avg_reward(self.noise, self.venv, config.num_transitions)
        print('Expert mean rewards ' + str(expert_avg_reward))
        print('Agent mean rewards ' + str(agent_avg_reward))
        print('Noise mean rewards ' + str(noise_avg_reward))


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
        agent_path = 'src/tests/temp/LunarLander-v2_fake_agent1'
        disc_func_path = 'src/tests/temp/disc_func2'
        agent = Config.agent_training_algo.load(agent_path, self.venv)
        agent2_path = 'src/tests/temp/LunarLander-v2_fake_agent2'
        agent2 = Config.agent_training_algo.load(agent2_path, self.venv)
        fake_agent_classification(agent, load_disc_func(disc_func_path), [agent, self.expert, agent2],
                                  ['fake agent', 'expert', 'fake agent2'], Config.env_action_space_size,
                                  self.venv, Config.num_transitions)

    def test_full_pipeline(self):
        agent1_save_path = 'src/tests/temp/DEBUGGING_AGENT'
        agent2_save_path = 'src/tests/temp/LunarLander-v2_fake_agent2'
        disc1_save_path = 'src/tests/temp/DEBUGGING_DISC'
        disc2_save_path = 'src/tests/temp/disc_func2'
        self.fake_agent_creation(agent1_save_path, disc1_save_path)
        self.fake_agent_creation(agent2_save_path, disc2_save_path)
        agent1 = Config.agent_training_algo.load(agent1_save_path, self.venv)
        agent2 = Config.agent_training_algo.load(agent2_save_path, self.venv)
        fake_agent_classification(agent1, load_disc_func(disc2_save_path), [agent1, agent2, self.expert, self.noise],
                                  ['fake agent 1', 'fake agent 2 ', 'expert', 'noise'], Config.env_action_space_size,
                                  self.venv, Config.num_transitions)

if __name__ == '__main__':
    unittest.main()
