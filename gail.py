from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import load
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

import gym
from gym.wrappers import TimeLimit
import numpy as np

ALGO_ID = "PPO"
algo = {"PPO": PPO, "SAC": SAC}

EXPERT_ID = ALGO_ID+"-500"

ENV_ID = "HalfCheetah-v3"
env_factory = lambda: TimeLimit(gym.make(ENV_ID), 1000) # variable horizon should be disabled for imitation

venv = make_vec_env(env_factory, n_envs=4)

learner = algo[ALGO_ID]("MlpPolicy", venv) 
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm
)

rollouts = load('demonstrations/'+ENV_ID+'-'+EXPERT_ID)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

reward, _ = evaluate_policy(learner, venv, 10)
print("Avg reward before training:", reward)
gail_trainer.train(300000)
reward, _ = evaluate_policy(learner, venv, 10)
print("Avg reward after training:", reward)