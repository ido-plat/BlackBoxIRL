
import tables as tb
import numpy as np
from imitation.data.rollout import make_min_timesteps, flatten_trajectories_with_rew, generate_trajectories
import random
import torch as th
import matplotlib.pyplot as plt
import tqdm, datetime

from src.utils.imitation_connector import discrete_action_conversion


class EvalDB:
    def __init__(self, file_name, result_plot_path, num_transitions_per_agent, num_traj_per_disc_eval, venv, expert,
                 observation_shape, observation_dtype, action_shape, action_space,
                 action_dtype, dones_shape, dones_dtype,
                 num_airl_iteration, every_n_agent_make_traj, mode, obs, log_file):
        self.action_space = action_space
        self.log_file = log_file
        self.obs = obs
        self.num_iter = num_airl_iteration
        self.mode = mode
        self.num_traj = num_traj_per_disc_eval
        self.batch_size = num_transitions_per_agent
        self.file_name = file_name
        self.result_path = result_plot_path
        self.venv = venv
        self.results = []
        self.result_timing = []
        expected_rows = int(num_airl_iteration / every_n_agent_make_traj)
        self.every_n_agent_make_traj = every_n_agent_make_traj
        print(f"Dtype {action_dtype} , Shape : {action_shape} , batch : {self.batch_size} summation : {(self.batch_size,) + action_shape}")

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
                self.fill_row(expert, True)
        else:
            self.db = tb.open_file(self.file_name, 'r+')
            self.agent_table = self.db.root['Agent_transitions']
            self.expert_table = self.db.root['Expert_transitions']

    def activate(self, disc, gen_algo, round_number):
        if self.mode == 'train':
            if round_number % self.every_n_agent_make_traj == 0:
                self.fill_row(gen_algo)
        else:
            self.eval_disc(disc, gen_algo)
        self.log()

    def log(self):
        curr_iter = len(self.result_timing) + 1
        now = datetime.datetime.now()
        time_diff = "?" if curr_iter == 1 else (now-self.result_timing[-1])
        total_time_left = "?" if curr_iter == 1 else \
            (now + datetime.timedelta(seconds=((self.num_iter - curr_iter) * time_diff.seconds))).strftime('%d/%m, %H:%M:%S')
        self.result_timing.append(now)
        with open(self.log_file, 'a') as f:
            f.write(f"{curr_iter}) Took {time_diff} seconds, finished at {now.strftime('%m/%d, %H:%M:%S')} And should finish the entire run at {total_time_left}")
            if self.mode != "train":
                f.write(" Curr result is %.4f" % self.results[-1])
            f.write('\n')

    def fill_row(self, agent, is_expert=False):
        def create_transition_batch():
            transitions = flatten_trajectories_with_rew(
                generate_trajectories(agent, self.venv, make_min_timesteps(self.batch_size)))
            return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size], \
                   transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]
        table = self.expert_table if is_expert else self.agent_table
        func = create_transition_batch if not self.obs else None
        table.append([func()])
        table.flush()

    def eval_disc(self, disc, gen_algo):
        def row_to_confidence(row):
            def torchify(x, type=None):
                return th.as_tensor(x, device=th.device('cuda:0'), dtype=type)
            obs = torchify(row['obs'], th.float)
            acts = torchify(row['acts'])
            next_obs = torchify(row['next_obs'], th.float)
            dones = torchify(row['dones'])
            with th.no_grad():
                _, log_prob, _ = gen_algo.policy.evaluate_actions(obs, acts)
                acts = discrete_action_conversion(acts, self.action_space)
                disc_result = disc(obs, acts, next_obs, dones, log_prob)
            return 1 / (1 + th.exp(-disc_result))

        num_rows = self.agent_table.nrows
        indx = list(range(num_rows)) if self.num_traj is None else random.sample(range(num_rows), self.num_traj)
        # print(f"AGENT LEN : {self.agent_table.nrows} , EXPERT LEN : {self.expert_table.nrows}")
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

    def plot_result(self, label_size=20):
        plt.plot(np.array(self.results))
        plt.title('Discriminator accuracy', fontsize=label_size + 4)
        plt.xticks(fontsize=label_size)
        plt.xlabel('Iteration', fontsize=label_size + 2)
        plt.yticks(fontsize=label_size)
        plt.ylabel('Accuracy', fontsize=label_size + 2)
        plt.savefig(self.result_path)
        plt.clf()

    def close(self):
        self.db.close()


def make_eval_db_from_config(path, result_plot_path, expert, venv, mode):
    from src.config import Config
    return EvalDB(path, result_plot_path, Config.num_transitions_per_eval, Config.num_traj_disc_eval, venv, expert,
                  observation_shape=Config.env_obs_shape, observation_dtype=Config.env_obs_dtype,
                  action_shape=Config.env_action_shape, action_dtype=Config.env_act_dtype,
                  action_space=Config.env_action_space_size,
                  dones_shape=Config.env_dones_shape, dones_dtype=Config.env_dones_dtype,
                  num_airl_iteration=Config.airl_iterations, every_n_agent_make_traj=Config.every_n_agent_eval,
                  mode=mode, obs=Config.use_obs, log_file=Config.log_file)
