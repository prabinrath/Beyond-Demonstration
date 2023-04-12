from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.reward_nets import RewardEnsemble, BasicRewardNet

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC

import gym
from gym.wrappers import TimeLimit
import torch

from .custom_rw import SquashRewardNet, get_ensemble_members

ALGO_ID = "PPO"    
algo = {"PPO": PPO, "SAC": SAC}
ENV_ID = "HalfCheetah-v3"
# variable horizon should be disabled for sampling equal length trajectories
# env_factory = lambda: TimeLimit(gym.make(ENV_ID, terminate_when_unhealthy=False), 1000)
env_factory = lambda: TimeLimit(gym.make(ENV_ID), 1000)
env = env_factory()

# Load reward model
N_REWARD_MODELS = 3 # ensemble reward models
reward_net = RewardEnsemble(
    env.observation_space, 
    env.action_space, 
    members=get_ensemble_members(SquashRewardNet, N_REWARD_MODELS, env)
)
reward_net.load_state_dict(torch.load('checkpoints/drex_reward_net/DREX-'+ENV_ID+'.pth'))

# Train RL
venv = make_vec_env(env_factory, n_envs=4)
learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)
learner = algo[ALGO_ID](policy="MlpPolicy", 
                        env=learned_reward_venv,
                        # n_steps=4096,
                        verbose=1
                        ) 
learner.learn(2000000, callback=learned_reward_venv.make_log_callback())
reward, _ = evaluate_policy(learner, venv, 10)
print("Avg reward after training:", reward)