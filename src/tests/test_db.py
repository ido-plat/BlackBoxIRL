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
from src.transitions.db_transitions import TransitionsDB

class BenchMarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.file_name = 'src/tests/temp/test_db.h5'
        # self.db = tb.open_file(self.file_name, 'w')
        venv_generator = SpaceInvadersEnv(Config.env, Config.num_env, None, Config.env_max_timestep, True)
        self.venv = venv_generator.make_venv()
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)

    def doCleanups(self) -> None:
        # self.db.close()
        pass

    def test_make_db(self):
        # shape = (100, )
        #
        # class BaseM(tb.IsDescription):
        #     zero = tb.Float32Col(shape=shape)
        #     # ones = tb.Float32Col(shape=shape)
        #
        # table = self.db.create_table(self.db.root, "table", description=BaseM)
        #
        # zeroes = np.zeros(shape, dtype=np.float)
        # ones = np.ones(shape, dtype=np.float)
        # table.append([(zeroes,), (ones,), (ones*2,), (ones*3,)])
        # for t in table:
        #     print(t['zero'].mean())
        self.db = TransitionsDB(Config.batch_size, Config.airl_num_transitions,
                                Config.maximum_batches_in_memory * Config.batch_size,
                                self.file_name, self.venv, self.expert, Config.env_obs_shape,
                                Config.env_obs_dtype, Config.env_action_shape, Config.env_act_dtype,
                                Config.env_dones_shape, Config.env_dones_dtype, True)

        for i, item in enumerate(self.db):
            print(f"{i} has shape of {len(item['obs'])}")
        self.db.close()



