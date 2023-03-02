import os.path

import tables as tb
from src.banchmarking.reward_aprox_banchmarking import *
from src.banchmarking.agent_creation_banchmarking import *
from src.alogirhms.density_approximate import density_aprox
from imitation.util import util
import unittest
from stable_baselines3 import DQN, A2C, PPO
from imitation.algorithms.density import DensityType
from src.alogirhms.airl import *
from src.utils.agent_utils import generate_trajectory_footage
from src.utils.env_utils import SpaceInvadersEnv
from src.config import Config
from src.tests.fake_agent_test_eval import generate_fake_list
from src.utils.confidence_plots import *
from src.transitions.db_transitions import TransitionsDB, make_db_using_config

class BenchMarkTest(unittest.TestCase):
    def setUp(self, use_db=True) -> None:
        self.file_name = 'data/SpaceInvadersNoFrameskip-v4/transitions_db/DB_SpaceInvadersNoFrameskip-v4.h5'

        venv_generator = SpaceInvadersEnv(Config.env, Config.num_env, None, Config.env_max_timestep, True)
        self.venv = venv_generator.make_venv()
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)
        self.use_db = use_db
        if use_db:
            self.db = make_db_using_config(self.file_name, 0, False, self.expert, self.venv)

    def doCleanups(self, close=False) -> None:
        if self.use_db or close:
            self.db.close()

    def test_merge(self):
        db1 = 'data/SpaceInvadersNoFrameskip-v4/transitions_db/DB1_SpaceInvadersNoFrameskip-v4.h5'
        db2 = 'data/SpaceInvadersNoFrameskip-v4/transitions_db/DB2_SpaceInvadersNoFrameskip-v4.h5'
        convert_from = (db1, db2)
        self.db = make_db_using_config(self.file_name, 0, True, self.expert, self.venv, convert_from)


    def test_photoes(self):
        import cv2
        path_dir = 'data/SpaceInvadersNoFrameskip-v4/result_plots/'
        images = os.listdir(path_dir)
        for i in images:
            print(f"starting {i}")
            img = cv2.imread(path_dir+i)
            if img is not None:
                print(f"{i} : {img.shape} : {img.dtype}")
                # cv2.imshow(i, img)
                # cv2.waitKey()
            else:
                print(f"{i} was none")

    def test_arrange(self):
        def f(t):
            return t[12:]
        path_dir = 'data/SpaceInvadersNoFrameskip-v4/result_plots/'
        self.db.spill_folder_to_db(path_dir, Config())

    def test_spill(self):
        path = '/home/user_109/PycharmProjects/BlackBoxIRL/src/tests/temp/vis/'
        self.db.spill_images_to_folder(1, path)
        
    def details(self):
        print(f"nrows normal0 : {self.db.db.root['Database0'].nrows} normal1 : {self.db.db.root['Database1'].nrows}  Result : {self.db.db.root['Results'].nrows}")



