from src.banchmarking.reward_aprox_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import *
from src.utils.agent_utils import generate_trajectory_footage
from src.utils.env_utils import make_fixed_horizon_venv
from gym.wrappers.time_limit import TimeLimit
import gym
from seals.util import AbsorbAfterDoneWrapper

class BenchMarkTest(unittest.TestCase):
    class Config:
        # env configs
        env = 'LunarLander-v2'
        env_action_space_size = 4
        num_env = 1
        env_max_timestep = 500

        # expert configs
        expert_path = 'data/agents/LunarLander-v2_dqn_expert.zip'
        expert_training_algo = DQN

        # agent configs
        agent_training_algo = DQN
        model_training_args = {
                    # PPO
                    # 'policy': 'MlpPolicy',
                    # 'n_steps': 1024

                    # DQN
                    'policy': 'MlpPolicy',
                    'batch_size': 128,
                    'target_update_interval': 250,
                    'buffer_size': 50000,
                    'exploration_final_eps': 0.1,
                    'exploration_fraction': 0.12,
                    'gamma': 0.99,
                    'gradient_steps': -1,
                    'learning_rate': 0.00063,
                    'learning_starts': 0,
                    'train_freq': 4,
                    'policy_kwargs': dict(net_arch=[256, 256])
        }
        model_total_training_steps = int(pow(2, 17))

        # airl configs
        irl_alo = airl
        airl_iterations = 800
        airl_model_training_steps = int(pow(2, 13))
        airl_args = {
                        'policy_training_steps': airl_model_training_steps,
                        'total_timesteps': airl_iterations * airl_model_training_steps
                     }
        # misc
        num_transitions = int(4e5)



    def test_reward_hist(self):
        config = self.Config
        env = config.env
        num_env = config.num_env
        agent_path = config.expert_path
        agent_training_alg = config.agent_training_algo
        num_traj = config.num_transitions
        args = config.airl_args
        algo = config.irl_alo
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        expert = config.expert_training_algo.load(agent_path, venv)
        real, fake = check_reward_distribution(expert, algo, venv, num_traj, plot_hist=True, args=args)

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/pend_gif2.gif'
        config = self.Config
        agent_path = 'src/tests/temp/real_reward_agent.zip'
        agent_training_alg = config.agent_training_algo
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        agent = agent_training_alg.load(agent_path, venv)
        obs, act, next_obs, done, rewards = generate_trajectory_footage(agent, venv, gif_path, subsampling=False)
        # plt.hist(act)
        # plt.show()

    def fake_agent_creation(self, save_agent_path, save_disc_func_path):
        config = self.Config
        expert_path = config.expert_path
        agent_training_alg = config.agent_training_algo
        num_traj = int(config.num_transitions)
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        expert = config.expert_training_algo.load(expert_path, venv)
        airl_arg = config.airl_args.copy()
        airl_arg['save_disc_path'] = save_disc_func_path
        samples1 = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(num_traj)))
        print('generated samples')
        train_agent_learnt_reward(samples1, venv, agent_training_alg,
                                  learning_time_step=config.model_total_training_steps,
                                  model_arg=config.model_training_args, save_model_path=save_agent_path,
                                  return_disc=True, airl_args=airl_arg)

    def test_compare_expart_agent_noise(self):
        config = self.Config
        agent_path = 'src/tests/temp/real_reward_agent.zip'
        expert_path = config.expert_path
        agent_training_alg = config.agent_training_algo
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        expert = config.expert_training_algo.load(expert_path, venv)
        agent = agent_training_alg.load(agent_path, venv)
        expert_avg_reward = get_agent_avg_reward(expert, venv, config.num_transitions)
        agent_avg_reward = get_agent_avg_reward(agent, venv, config.num_transitions)
        noise_avg_reward = get_agent_avg_reward(None, venv, config.num_transitions)
        print('Expert mean rewards ' + str(expert_avg_reward))
        print('Agent mean rewards ' + str(agent_avg_reward))
        print('Noise mean rewards ' + str(noise_avg_reward))

    def fake_agent_classification(self, agent_path, disc_path, num_bins=100):
        config = self.Config
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        num_traj = config.num_transitions
        expert_path = config.expert_path
        disc_func = load_disc_func(disc_path)
        agent = config.agent_training_algo.load(agent_path)
        expert = config.expert_training_algo.load(expert_path)

        agent_trans = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_traj)))
        expert_trans = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(num_traj)))
        noise_trans = flatten_trajectories_with_rew(generate_trajectories(None, venv, make_min_timesteps(num_traj)))
        print('generated transitions')

        agent_confidence = traj_confidence(agent_trans, disc_func, agent, config.env_action_space_size)
        expert_confidence = traj_confidence(expert_trans, disc_func, agent, config.env_action_space_size)
        noise_confidence = traj_confidence(noise_trans, disc_func, None, config.env_action_space_size)

        bins = np.linspace(0, 1, num_bins)
        label_size = 20
        plt.hist(noise_confidence, alpha=0.5, bins=bins, label="Noise", density=True)
        plt.hist(agent_confidence, alpha=0.5, bins=bins, label="Agent", density=True)
        plt.hist(expert_confidence, alpha=0.5, bins=bins, label="Expert", density=True)
        plt.legend()
        plt.xticks(fontsize=label_size)
        plt.xlabel('Confidence', fontsize=label_size+2)
        plt.ylim(0, 100)
        plt.show()

    def test_train_model(self):
        config = self.Config
        rl_args = config.model_training_args
        rl_algo = config.agent_training_algo
        venv = make_fixed_horizon_venv(config.env, max_episode_steps=config.env_max_timestep, n_envs=config.num_env)
        agent = train_agent(venv, rl_algo, config.model_total_training_steps, rl_args)
        gif_path = 'src/tests/temp/training_agent.gif'
        agent_save_path = 'src/tests/temp/real_reward_agent'
        agent.save(agent_save_path)
        avg_rewards = get_agent_avg_reward(agent, venv, config.num_transitions)
        print('avg reward: '+str(avg_rewards))
        generate_trajectory_footage(agent, venv, gif_path)

    def test_partial_pipelie(self):
        agent_path = 'src/tests/temp/Pnedulum-v0_fake1_ppo.zip'
        disc_func_path = 'src/tests/temp/disc_2'
        self.fake_agent_classification(agent_path, disc_func_path)

    def test_full_pipeline(self):
        agent1_save_path = 'src/tests/temp/LunarLander-v2_fake_agent1'
        agent2_save_path = 'src/tests/temp/LunarLander-v2_fake_agent2'
        disc1_save_path = 'src/tests/temp/disc_func1'
        disc2_save_path = 'src/tests/temp/disc_func2'
        self.fake_agent_creation(agent1_save_path, disc1_save_path)
        self.fake_agent_creation(agent2_save_path, disc2_save_path)
        self.fake_agent_classification(agent1_save_path, disc2_save_path)

if __name__ == '__main__':
    unittest.main()
