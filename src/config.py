from stable_baselines3 import DQN, A2C, PPO
from src.alogirhms.airl import airl


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
    num_transitions = int(5e4)
