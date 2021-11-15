from src.banchmarking.reward_aprox_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import *
from src.utils.agent_utils import generate_trajectory_footage


class BenchMarkTest(unittest.TestCase):
    class Config:
        env = 'Pendulum-v0'
        num_transitions = int(1e3)
        irl_alo = airl
        agent_training_algo = PPO
        expert_training_algo = PPO
        airl_args = {
                        'policy_training_steps': 1024,
                        'total_timesteps': int(1e6)
                     }
        model_training_args = {
                    'policy': 'MlpPolicy',
                    'n_steps': 1024
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
        expert = config.expert_training_algo.load(agent_path, venv)
        real, fake = check_reward_distribution(expert, algo, venv, num_traj, plot_hist=True, args=args)

    def test_agent_footage(self):
        gif_path = 'src/tests/temp/pend_gif.gif'
        env = self.Config.env
        num_env = 1
        agent_path = 'src/tests/temp/Pnedulum-v0_fake1_ppo.zip'
        agent_training_alg = self.Config.agent_training_algo
        venv = util.make_vec_env(env, n_envs=num_env)
        agent = agent_training_alg.load(agent_path, venv)
        obs, act, next_obs, done, rewards = generate_trajectory_footage(agent, venv, gif_path)
        pass

    def test_fake_agent_creation(self):
        config = self.Config
        env = config.env
        num_env = 1
        agent_path = "data/agents/Pendulum-v0_ppo"
        save_model_path = 'src/tests/temp/Pnedulum-v0_fake1_ppo.zip'
        agent_training_alg = config.agent_training_algo
        num_traj = int(config.num_transitions)
        venv = util.make_vec_env(env, n_envs=num_env)
        expert = config.expert_training_algo.load(agent_path, venv)
        airl_arg = config.airl_args.copy()
        airl_arg['save_disc_path'] = 'src/tests/temp/disc_1'

        samples1 = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(num_traj)))
        samples2 = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(num_traj)))
        print('generated samples')
        train_agent_learnt_reward(samples1, venv, agent_training_alg,
                                  learning_time_step=config.model_training_args['n_steps'],
                                  model_arg=config.model_training_args, model_path=save_model_path,
                                  return_disc=True, airl_args=airl_arg)
        print('trained first')
        airl_arg['save_disc_path'] = 'src/tests/temp/disc_2'
        save_model_path = 'src/tests/temp/Pnedulum-v0_fake2_ppo.zip'
        train_agent_learnt_reward(samples2, venv, agent_training_alg,
                                  learning_time_step=config.model_training_args['n_steps'],
                                  model_arg=config.model_training_args, model_path=save_model_path,
                                  return_disc=True, airl_args=airl_arg)


    def test_compare_expart_agent_noise(self):
        config = self.Config
        env = config.env
        num_env = 1
        agent_path = 'src/tests/temp/Pnedulum-v0_fake1_ppo.zip'
        expert_path = 'data/agents/Pendulum-v0_ppo.zip'
        num_traj = config.num_transitions
        agent_training_alg = config.agent_training_algo
        venv = util.make_vec_env(env, n_envs=num_env)
        expert = config.expert_training_algo.load(expert_path, venv)
        agent = agent_training_alg.load(agent_path, venv)
        expert_noise_compare = compare_agents(expert, None, venv=venv, num_samples=num_traj)
        expert_agent_compare = compare_agents(expert, agent, venv=venv, num_samples=num_traj)
        print('Expert-noise mean rewards diff ' + str(expert_noise_compare))
        print('Expert-agent mean rewards diff ' + str(expert_agent_compare))

    def test_fake_agent_classification(self):
        config = self.Config
        num_env = 1

        venv = util.make_vec_env(config.env, n_envs=num_env)
        num_traj = config.num_transitions
        num_bins = 100
        disc_path = 'src/tests/temp/disc_2'
        agent_path = 'src/tests/temp/Pnedulum-v0_fake1_ppo.zip'
        expert_path = 'data/agents/Pendulum-v0_ppo.zip'
        disc_func = load_disc_func(disc_path)
        agent = config.agent_training_algo.load(agent_path)
        expert = config.expert_training_algo.load(expert_path)
        agent_trans = flatten_trajectories_with_rew(generate_trajectories(agent, venv, make_min_timesteps(num_traj)))
        expert_trans = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(num_traj)))
        noise_trans = flatten_trajectories_with_rew(generate_trajectories(None, venv, make_min_timesteps(num_traj)))
        print('generated transitions')
        agent_confidence = traj_confidence(agent_trans, disc_func, agent, np.inf)
        expert_confidence = traj_confidence(expert_trans, disc_func, agent, np.inf)
        noise_confidence = traj_confidence(noise_trans, disc_func, None, np.inf)
        bins = np.linspace(0, 1, num_bins)
        plt.hist(noise_confidence, alpha=0.5, bins=bins, label="Noise")
        plt.hist(agent_confidence, alpha=0.5, bins=bins, label="Agent")
        plt.hist(expert_confidence, alpha=0.5, bins=bins, label="Expert")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
