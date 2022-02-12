import torch as th
import numpy as np
import unittest
from src.config import Config
from src.utils.confidence_plots import *
from src.utils.env_utils import SpaceInvadersEnv
from src.networks.reward_nets import ClassificationShapedRewardNet
class NetWorkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.venv = SpaceInvadersEnv(max_timestemp=Config.env_max_timestep, use_history=True).make_venv()
        self.expert = Config.expert_training_algo.load(Config.expert_path, self.venv,
                                                       custom_objects=Config.expert_custom_objects)
        self.noise = None
        self.net = ClassificationShapedRewardNet(self.venv.observation_space, self.venv.action_space, 2)


    def test_zero_out(self):
        batch_size = 4
        obs = th.zeros((batch_size, ) + Config.env_obs_shape)
        next_obs = th.zeros((batch_size,) + Config.env_obs_shape)
        dones = th.zeros(batch_size)
        res = self.net(obs, None, next_obs, dones)
        print(res)
        print(res.size())


