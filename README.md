# BlackBoxIRL
 
In this project we will analyze and imitate an expert given a set of choices (transitions) it makes using Inverse Reinforcement Learning tools.
We use GAN based method to generate the underlying reward function explaining the expert, and train a policy on said reward function.
We implemented a learnt measure between an expert and other policies that capture the intent of the expert regardless of the RL algorithm or reward function they were trained on.

# Usage

In order to run BlackBoxIRL:

 1. Change the relavent `src/config.py` to match the correct environment/params
 2. Run `python ./make_fakes.py` to generate the fake agents
 3. Change the relavent path in `src/benchmarking_test.py`
 4. run `python ./running_script.py`

This will generate two databeses for two different runs, and will generate at the specified result dir:

 1. Confidence plots (using each of the trained agents as reference agent against 
 agents)
 2. Discriminators
 3. Reward functions
