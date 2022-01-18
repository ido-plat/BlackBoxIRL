from typing import Iterable
import tables as tb
import os
import gc
from imitation.data.rollout import make_min_timesteps, flatten_trajectories_with_rew, generate_trajectories


class TransitionsDB(Iterable):
    def __init__(self, batch_size, num_transitions, max_transition_per_run, file_name, venv, expert, observation_shape,
                 observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype, rewrite_db_file=False):

        self.file_name = file_name
        self.batch_size = batch_size
        if os.path.isfile(self.file_name) and not rewrite_db_file:
            self.db = tb.open_file(self.file_name, 'r')
            self.table = self.db.root['Database']
        else:
            self._make_db(num_transitions, max_transition_per_run, venv, expert, observation_shape,
                          observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype)

    def _make_db(self, num_transitions, max_transition_per_run, venv, expert, observation_shape,
                 observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype):
        class TransitionBatch(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=2)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)


        def create_transition_batch():
            transitions = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(self.batch_size)))
            return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size],\
                   transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]

        num_batches = int(num_transitions/self.batch_size)
        num_batches_per_append = int(max_transition_per_run/self.batch_size)
        self.db = tb.open_file(self.file_name, 'w')
        self.table = self.db.create_table(self.db.root, "Database", TransitionBatch,
                                          expectedrows=num_batches)
        batches_inserted = 0
        while batches_inserted < num_batches:
            num_batches_to_create = min(num_batches_per_append, num_batches - batches_inserted)
            to_append = [create_transition_batch() for _ in range(num_batches_to_create)]
            self.table.append(to_append)
            self.table.flush()
            del to_append
            gc.collect()    # to make sure we free the data
            batches_inserted += num_batches_to_create
            print(f"Created {batches_inserted} out of {num_batches} batches, this iteration made "
                  f"{num_batches_to_create}")

    def __iter__(self):
        for item in self.table:
            yield item

    def close(self):
        self.db.close()


def make_db_using_config(file_name,  rewrite_file, expert, venv):
    from src.config import Config
    return TransitionsDB(Config.batch_size, Config.airl_num_transitions,
                         Config.maximum_batches_in_memory * Config.batch_size, file_name, venv, expert,
                         Config.env_obs_shape, Config.env_obs_dtype, Config.env_action_shape, Config.env_act_dtype,
                         Config.env_dones_shape, Config.env_dones_dtype, rewrite_file)

