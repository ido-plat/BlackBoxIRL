
import numpy as np
import tables as tb
from stable_baselines3 import DQN, A2C, PPO
from src.alogirhms.airl import airl


class Config:
    # misc
    num_transitions = int(2e5)
    in_lab = True
    use_db = True
    # env configs
    env = 'SpaceInvadersNoFrameskip-v4'
    env_action_space_size = 6
    num_env = 1
    env_max_timestep = np.inf
    env_obs_shape = (4, 84, 84)
    env_action_shape = ()
    env_dones_shape = ()
    env_obs_dtype = tb.UInt8Col
    env_act_dtype = tb.Int64Col
    env_dones_dtype = tb.BoolCol
    # db configs
    batch_size = 1024
    maximum_batches_in_memory = 10

    # expert configs
    expert_path = 'data/SpaceInvadersNoFrameskip-v4/agents/SpaceInvadersNoFrameskip-v4_DQN_Expert.zip'
    expert_training_algo = DQN
    expert_custom_objects = {
        "learning_rate": lambda x: .003,
        "clip_range": lambda x: .02
    }   # if in_lab else None  # need to sync python 3.8 and 3.7

    # agent configs
    agent_training_algo = DQN
    iterative_agent_training_algo = PPO
    model_total_training_steps = int(pow(2, 17))
    all_model_training_args = {

        'LunarLander-v2': {
            DQN: {
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
            },
            PPO: {
                'policy': 'MlpPolicy',
                'n_steps': 1024,
                'ent_coef': 0.01,
                'gae_lambda': 0.98,
                'gamma': 0.999
            },
            A2C: {
                'policy': 'MlpPolicy',
                'learning_rate': 0.00083,
                'gamma': 0.995,
                'ent_coef': 1.0e-05
            }
        },

        'SpaceInvadersNoFrameskip-v4': {
            DQN: {
                'policy': 'CnnPolicy',
                'batch_size': 32,
                'buffer_size': 10000,
                'exploration_final_eps': 0.01,
                'exploration_fraction': 0.1,
                'gradient_steps': 1,
                'learning_rate': 0.0001,
                'learning_starts': 100000,
                'optimize_memory_usage': True,
                'target_update_interval': 1000,
                'train_freq': 4
            },
            PPO: {
                'policy': 'CnnPolicy',
                'batch_size': 256,
                'clip_range': 0.1,  # was something weird
                'ent_coef': 0.01,
                'learning_rate': 2.5e-4,
                'vf_coef': 0.5
            }
        }

    }
    model_training_args = all_model_training_args[env][agent_training_algo]

    # airl configs
    irl_alo = airl
    airl_num_transitions = batch_size * 1000    # roughly int(1e7)
    airl_iterations = 5000
    airl_model_training_steps = int(pow(2, 10))
    airl_args = {
        'batch_size': batch_size,
        'policy_training_steps_for_iteration': airl_model_training_steps,
        'total_timesteps': airl_iterations * airl_model_training_steps,
        "allow_variable_horizon": env_max_timestep is np.inf,
        'disc_updates': 8,
        'iagent_args': all_model_training_args[env][iterative_agent_training_algo]
    }
