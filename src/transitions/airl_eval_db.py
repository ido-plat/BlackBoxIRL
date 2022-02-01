
import tables as tb
import numpy as np
from imitation.data.rollout import make_min_timesteps, flatten_trajectories_with_rew, generate_trajectories
import random
import torch as th
import matplotlib.pyplot as plt
import tqdm


class EvalDB:
    def __init__(self, file_name, result_plot_path, num_transitions_per_agent, num_traj_per_disc_eval, venv, expert,
                 observation_shape, observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype,
                 num_airl_iteration, every_n_agent_make_traj, mode, obs):
        self.obs = obs
        self.mode = mode
        self.num_traj = num_traj_per_disc_eval
        self.batch_size = num_transitions_per_agent
        self.file_name = file_name
        self.result_path = result_plot_path
        self.venv = venv
        self.results = []
        expected_rows = int(num_airl_iteration / every_n_agent_make_traj)
        self.every_n_agent_make_traj = every_n_agent_make_traj

        class TransitionBatch(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=2)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)
        if mode == 'train':
            self.db = tb.open_file(self.file_name, 'w')
            self.agent_table = self.db.create_table(self.db.root, "Agent_transitions", TransitionBatch,
                                                    expectedrows=expected_rows)
            self.expert_table = self.db.create_table(self.db.root, "Expert_transitions", TransitionBatch,
                                                     expectedrows=expected_rows + 1)

            for _ in tqdm.tqdm(range(expected_rows + 1), desc="Expert Row"):
                self.fill_row(expert, self.expert_table)
        else:
            self.db = tb.open_file(self.file_name, 'r')
            self.agent_table = self.db.root['Agent_transitions']
            self.expert_table = self.db.root['Expert_transitions']

    def activate(self, disc, gen_algo, round_number):
        if self.mode == 'train':
            if round_number % self.every_n_agent_make_traj == 0:
                self.fill_row(gen_algo)
        else:
            self.eval_disc(disc, gen_algo)

    def fill_row(self, agent, table=None):
        def create_transition_batch():
            transitions = flatten_trajectories_with_rew(
                generate_trajectories(agent, self.venv, make_min_timesteps(self.batch_size)))
            return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size], \
                   transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]
        if not table:
            table = self.agent_table
        func = create_transition_batch if not self.obs else None
        table.append([func()])
        table.flush()

    def eval_disc(self, disc, gen_algo):
        def row_to_confidence(row):
            obs = row['obs']
            acts = row['act']
            next_obs = row['next_obs']
            dones = row['done']
            with th.no_grad():
                _, log_prob, _ = gen_algo.policy.evaluate_actions(obs, acts)
                disc_result = disc(obs, acts, next_obs, dones, log_prob)
            return 1 / (1 + th.exp(-disc_result))

        num_rows = self.agent_table.nrows
        indx = range(num_rows) if self.num_traj is None else random.sample(range(num_rows), self.num_traj)
        agent_traj = self.agent_table[indx]
        expert_traj = self.expert_table[indx]
        result = 0
        for i in range(len(indx)):
            curr_agent_row = agent_traj[i]
            curr_exprt_row = expert_traj[i]
            agent_confidence = row_to_confidence(curr_agent_row)
            expert_confidence = row_to_confidence(curr_exprt_row)
            result += expert_confidence.mean() < agent_confidence.mean()

        result = result/len(indx)
        self.results.append(result)
        return result

    def plot_result(self):
        plt.plot(np.array(self.results))
        plt.savefig(self.result_path)
        plt.clf()

    def close(self):
        self.db.close()


def make_eval_db_from_config(path, result_plot_path, expert, venv, mode):
    from src.config import Config
    return EvalDB(path, result_plot_path, Config.num_transitions_per_eval, Config.num_traj_disc_eval, venv, expert,
                  Config.env_obs_shape, Config.env_obs_dtype, Config.env_action_shape, Config.env_act_dtype,
                  Config.env_dones_shape, Config.env_dones_dtype,
                  Config.airl_iterations, Config.every_n_agent_eval, mode, Config.use_obs)
