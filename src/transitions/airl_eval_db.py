
import tables as tb
import numpy as np
from imitation.data.rollout import make_min_timesteps, flatten_trajectories_with_rew, generate_trajectories
import random
import torch as th
import matplotlib.pyplot as plt
import tqdm, datetime
import os
from src.utils.imitation_connector import discrete_action_conversion


class EvalDB:
    def __init__(self, file_name, result_plot_path, num_transitions_per_agent,
                 num_traj_per_disc_eval_agent, num_traj_per_disc_eval_other_experts, num_traj_per_reward_eval,
                 venv, expert, other_expert_list,
                 observation_shape, observation_dtype, action_shape, action_space,
                 action_dtype, dones_shape, dones_dtype,
                 num_airl_iteration, every_n_agent_make_traj, mode, obs, log_file,
                 do_reward_func_eval, log_dir=None):

        self.log_dir = os.path.dirname(log_file) if not log_dir else log_dir
        self.other_expert = other_expert_list
        self.do_reward_eval = do_reward_func_eval
        self.action_space = action_space
        self.log_file = log_file
        self.obs = obs
        self.num_iter = num_airl_iteration
        self.mode = mode
        self.num_traj = num_traj_per_disc_eval_agent
        self.num_traj_other_experts = num_traj_per_disc_eval_other_experts
        self.num_traj_reward = num_traj_per_reward_eval
        self.batch_size = num_transitions_per_agent
        self.file_name = file_name
        self.result_path = result_plot_path
        self.venv = venv
        self.results = []
        self.result_timing = []
        self.diff_expert_acc = []
        self.reward_acc = []
        expected_rows = int(num_airl_iteration / every_n_agent_make_traj)
        self.every_n_agent_make_traj = every_n_agent_make_traj

        class TransitionBatch(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=2)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)

        class TransitionBatchWithReward(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            rewards = action_dtype(shape=(self.batch_size,) + action_dtype, pos=2)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=4)

        expert_db_type = TransitionBatchWithReward if do_reward_func_eval else TransitionBatch
        if mode == 'train':
            self.db = tb.open_file(self.file_name, 'w')
            self.agent_table = self.db.create_table(self.db.root, "Agent_transitions", TransitionBatch,
                                                    expectedrows=expected_rows)
            self.expert_table = self.db.create_table(self.db.root, "Expert_transitions", expert_db_type,
                                                     expectedrows=expected_rows + 1)

            for _ in tqdm.tqdm(range(expected_rows + 1), desc="Expert Row"):
                self.fill_row(expert, True)
        else:
            self.db = tb.open_file(self.file_name, 'r+')
            self.agent_table = self.db.root['Agent_transitions']
            self.expert_table = self.db.root['Expert_transitions']

    def activate(self, disc, gen_algo, reward_func, round_number):
        if self.do_reward_eval:
            self.eval_reward_func_acc(reward_func)
        if len(self.other_expert) > 0:
            self.eval_against_other_expert(disc, gen_algo)
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
            f.write(f"{curr_iter}) Took {time_diff} seconds, finished at {now.strftime('%d/%m, %H:%M:%S')} And should finish the entire run at {total_time_left}")
            if len(self.other_expert) > 0:
                f.write(" Curr other expert acc is %.4f" % self.diff_expert_acc[-1])
            if self.do_reward_eval:
                f.write(" Curr reward func acc is is %.4f" % self.reward_acc[-1])
            if self.mode != "train":
                f.write(" Curr iterative agent acc is %.4f" % self.results[-1])
            f.write('\n')

    def _create_transition_batch(self, agent, use_rewards):
        transitions = flatten_trajectories_with_rew(
            generate_trajectories(agent, self.venv, make_min_timesteps(self.batch_size)))
        if use_rewards:
            return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size], \
                   transitions.rews[:self.batch_size], \
                   transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]
        return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size], \
               transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]

    def fill_row(self, agent, is_expert=False):
        table = self.expert_table if is_expert else self.agent_table
        table.append([self._create_transition_batch(agent, self.do_reward_eval and is_expert)])
        table.flush()

    def eval_disc(self, disc, gen_algo):
        def row_to_confidence(row):
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
        with open(f"{self.log_dir}/iterative_agent_disc_eval.txt", "a") as f:
            print(f"{result} ", file=f)
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

    def eval_reward_func_acc(self, reward_func):
        num_rows = self.expert_table.nrows
        indx = list(range(num_rows)) if self.num_traj_reward is None else random.sample(range(num_rows), self.num_traj_reward)
        acc = 0
        for i in indx:
            relevent_row = self.expert_table[i]
            true_reward = relevent_row['rewards']
            with th.no_grad():
                fake_reward = reward_func(torchify(relevent_row['obs'], th.float),
                                          relevent_row['acts'], torchify(relevent_row['next_obs'], th.float))
            acc += (true_reward == fake_reward).mean()
        acc = acc / len(indx)
        self.reward_acc.append(acc)
        with open(f"{self.log_dir}/reward_func_acc.txt", "a") as f:
            print(f"{acc} ", file=f)
        return acc

    def eval_against_other_expert(self, disc, gen_algo):
        def row_to_confidence(row, obs=None, acts=None, next_obs=None, dones=None):
            obs = obs if obs is not None else row['obs']
            obs = torchify(obs, th.float)
            acts = acts if acts is not None else row['acts']
            acts = torchify(acts)
            next_obs = next_obs if next_obs is not None else row['next_obs']
            next_obs = torchify(next_obs, th.float)
            dones = dones if dones is not None else row['dones']
            dones = torchify(dones)
            with th.no_grad():
                _, log_prob, _ = gen_algo.policy.evaluate_actions(obs, acts)
                acts = discrete_action_conversion(acts, self.action_space)
                disc_result = disc(obs, acts, next_obs, dones, log_prob)
            return 1 / (1 + th.exp(-disc_result))
        num_rows = self.expert_table.nrows
        indx = list(range(num_rows)) if self.num_traj_other_experts is None else random.sample(range(num_rows),
                                                                                        self.num_traj_other_experts)
        expert_traj = self.expert_table[indx]
        result = 0
        for i in range(len(indx)):
            curr_exprt_row = expert_traj[i]
            curr_expert_confidence = row_to_confidence(curr_exprt_row)
            temp_res = 0
            for agent in self.other_expert:
                curr_acts, curr_dones, curr_obs, curr_next_obs = self._create_transition_batch(agent, False)
                curr_fake_confidence = row_to_confidence(None, curr_obs, curr_acts, curr_next_obs, curr_dones)
                temp_res += curr_expert_confidence.mean() < curr_fake_confidence.mean()
            temp_res = temp_res / len(self.other_expert)
            result += temp_res

        result = result/len(indx)
        self.diff_expert_acc.append(result)
        with open(f"{self.log_dir}/other_expert_disc_eval.txt", "a") as f:
            print(f"{result} ", file=f)
        return result

    def close(self):
        self.db.close()


def torchify(x, type=None):
    return th.as_tensor(x, device=th.device('cuda:0'), dtype=type)


def make_eval_db_from_config(path, result_plot_path, expert, venv, mode, other_expert):
    from src.config import Config
    return EvalDB(path, result_plot_path, expert=expert, venv=venv, mode=mode, other_expert_list=other_expert,
                  num_traj_per_disc_eval_agent=Config.num_traj_disc_eval,
                  num_transitions_per_agent=Config.num_transitions_per_eval,
                  num_traj_per_disc_eval_other_experts=Config.num_traj_other_expert,
                  num_traj_per_reward_eval=Config.num_traj_reward_func, do_reward_func_eval=Config.do_reward_func_eval,
                  observation_shape=Config.env_obs_shape, observation_dtype=Config.env_obs_dtype,
                  action_shape=Config.env_action_shape, action_dtype=Config.env_act_dtype,
                  action_space=Config.env_action_space_size,
                  dones_shape=Config.env_dones_shape, dones_dtype=Config.env_dones_dtype,
                  num_airl_iteration=Config.airl_iterations, every_n_agent_make_traj=Config.every_n_agent_eval,
                  obs=Config.use_obs, log_file=Config.log_file)
