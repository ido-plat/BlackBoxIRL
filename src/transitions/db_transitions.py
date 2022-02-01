from typing import Iterable
import tables as tb
import numpy as np
import os
import gc
from imitation.data.rollout import make_min_timesteps, flatten_trajectories_with_rew, generate_trajectories


class TransitionsDB(Iterable):
    def __init__(self, batch_size, num_transitions, max_transition_per_run, file_name, venv, expert, observation_shape,
                 observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype, result_shape,
                 result_dtype, max_num_result=100, rewrite_db_file=False, num_tables=2, starting_index=0,
                 convert_from=None, use_observation=False):

        self.index = starting_index
        self.file_name = file_name
        if convert_from:
            self._convert_from(convert_from, num_transitions, observation_shape, observation_dtype, action_shape,
                               dones_shape, dones_dtype, result_shape, result_dtype, max_num_result)
            return
        if os.path.isfile(self.file_name) and not rewrite_db_file:
            self.db = tb.open_file(self.file_name, 'r')
            self.tables = []
            for i in range(num_tables):
                self.tables.append(self.db.root[f'Database{i}'])
            self.result_plots = self.db.root['Results']
            self.batch_size = self.tables[0].attr.batch_size
            self.result_shape = self.result_plots.attrs.image_shape
            self.max_num_result = self.result_plots.attrs.max_images

        else:
            self._make_db(batch_size, num_transitions, max_transition_per_run, venv, expert, observation_shape,
                          observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype, result_shape,
                          result_dtype, max_num_result, num_tables, use_observation)

    def change_index(self, ind):
        self.index = ind

    def _convert_from(self, convert, num_transitions, observation_shape,
                      observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype, result_shape,
                      result_dtype, max_num_result=100, ):

        self.db = tb.open_file(self.file_name, 'w')

        class TransitionBatch(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=2)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)

        num_batches = int(num_transitions / self.batch_size)
        self.tables = []
        for i in range(2):
            old_table = tb.open_file(convert[i], 'r')['Database']
            table = self.db.create_table(self.db.root, f"Database{i}", TransitionBatch, expectedrows=num_batches)
            for row in old_table:
                table.append(row)
            table.attrs.batch_size = self.batch_size
            self.tables.append(table)

        class Result(tb.IsDescription):
            images = result_dtype(shape=(max_num_result, ) + result_shape, pos=0)
            config = tb.StringCol(pos=1)
            date = tb.StringCol(pos=2)
            actual_num = tb.UInt8Col(pos=3)
            labels = tb.StringCol(pos=4, shape=max_num_result)

        self.result_plots = self.db.create_table(self.db.root, "Results", Result)
        self.result_shape = result_shape
        self.result_plots.attrs.image_shape = result_shape
        self.result_plots.attrs.max_images = max_num_result
        self.max_num_result = max_num_result

    def _make_db(self, batch_size, num_transitions, max_transition_per_run, venv, expert, observation_shape,
                 observation_dtype, action_shape, action_dtype, dones_shape, dones_dtype, result_shape,
                 result_dtype, max_num_result=100, num_databases_to_make=2, use_observation=False):

        class TransitionBatch(tb.IsDescription):
            acts = action_dtype(shape=(self.batch_size,) + action_shape, pos=0)
            dones = dones_dtype(shape=(self.batch_size,) + dones_shape, pos=1)
            obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=2)
            next_obs = observation_dtype(shape=(self.batch_size,) + observation_shape, pos=3)

        self.batch_size = batch_size
        num_batches = int(num_transitions/self.batch_size)
        num_batches_per_append = int(max_transition_per_run/self.batch_size)
        self.db = tb.open_file(self.file_name, 'w')
        self.tables = []
        for i in range(num_databases_to_make):
            table = self.db.create_table(self.db.root, f"Database{i}", TransitionBatch, expectedrows=num_batches)
            self._fil_table(table, num_batches, num_batches_per_append, expert, venv)
            table.attrs.batch_size = self.batch_size
            self.tables.append(table)
            print(f"Finished making table n{i}")

        class Result(tb.IsDescription):
            images = result_dtype(shape=(max_num_result, ) + result_shape, pos=0)
            config = tb.StringCol(pos=1)
            date = tb.StringCol(pos=2)
            actual_num = tb.UInt8Col(pos=3)
            labels = tb.StringCol(pos=4, shape=max_num_result)

        self.result_plots = self.db.create_table(self.db.root, "Results", Result)
        self.result_shape = result_shape
        self.result_plots.attrs.image_shape = result_shape
        self.result_plots.attrs.max_images = max_num_result
        self.max_num_result = max_num_result

    def __iter__(self):
        for item in self.tables[self.index]:
            yield item

    def close(self):
        self.db.close()

    def _fil_table(self, table, num_batches, num_batches_per_append, expert, venv, obs=False):
        def create_transition_batch():
            transitions = flatten_trajectories_with_rew(generate_trajectories(expert, venv, make_min_timesteps(self.batch_size)))
            return transitions.acts[:self.batch_size], transitions.dones[:self.batch_size],\
                   transitions.obs[:self.batch_size], transitions.next_obs[:self.batch_size]

        func = create_transition_batch if not obs else None
        batches_inserted = 0
        while batches_inserted < num_batches:
            num_batches_to_create = min(num_batches_per_append, num_batches - batches_inserted)
            to_append = [func() for _ in range(num_batches_to_create)]
            table.append(to_append)
            table.flush()
            del to_append
            gc.collect()    # to make sure we free the data
            batches_inserted += num_batches_to_create
            print(f"Created {batches_inserted} out of {num_batches} batches, this iteration made "
                  f"{num_batches_to_create}")

    def add_visualisation_images(self, config, images, labels):
        curr_len = len(images)
        if curr_len > self.max_num_result:
            print(f"Max result len is {self.max_num_result} but asked to make {curr_len} results")
        from datetime import datetime
        time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        to_insert_images = np.zeroes((self.max_num_result, ) + self.result_shape)
        to_insert_images[:curr_len] += np.array(images)
        new_labels = ['']*self.max_num_result
        for i, label in enumerate(labels):
            new_labels[i] = label
        to_append = (to_insert_images, str(config), time, curr_len, new_labels)
        self.result_plots.append([to_append])
        self.result_plots.flush()

    def spill_images_to_folder(self, index, path):
        import cv2
        row = self.result_plots[index]
        for i in range(row['actual_num']):
            cv2.imwrite(path+row['label'][i], row['images'][i])


def make_db_using_config(file_name, index, rewrite_file, expert, venv, convert_from=None):
    from src.config import Config
    return TransitionsDB(Config.batch_size, Config.airl_num_transitions,
                         Config.maximum_batches_in_memory * Config.batch_size, file_name, venv, expert,
                         Config.env_obs_shape, Config.env_obs_dtype, Config.env_action_shape, Config.env_act_dtype,
                         Config.env_dones_shape, Config.env_dones_dtype, rewrite_db_file=rewrite_file,
                         starting_index=index, convert_from=convert_from)

