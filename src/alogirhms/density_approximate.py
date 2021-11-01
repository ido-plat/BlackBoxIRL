from src.utils.imitation_connector import to_im_traj
from imitation.algorithms.density import *


def density_aprox(samples, venv, dense_type):
    alg = DensityAlgorithm(demonstrations=samples, venv=venv, density_type=dense_type)
    alg.train()
    # use alg(obs_b: np.ndarray,
    #         act_b: np.ndarray,
    #         next_obs_b: np.ndarray,
    #         dones: np.ndarray,) all batches
    return alg
