from src.utils.imitation_connector import to_im_traj
from imitation.src.imitation.algorithms.adversarial import airl
from imitation.src.imitation.data import rollout
from imitation.src.imitation.util import logger
import stable_baselines3 as sb3
import tempfile
import pathlib

def airl(samples, venv):
    trajectories = to_im_traj(samples)
    transitions = rollout.flatten_trajectories(trajectories)
    tempdir = tempfile.TemporaryDirectory(prefix="AIRL")
    tempdir_path = pathlib.Path(tempdir.name)
    airl_logger = logger.configure(tempdir_path / "AIRL/")
    airl_trainer = airl.AIRL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=32,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        custom_logger=airl_logger,
    )
    airl_trainer.train(total_timesteps=2048)
