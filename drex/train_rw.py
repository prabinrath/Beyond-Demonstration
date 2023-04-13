from imitation.data.types import load
from imitation.algorithms.bc import BC
from imitation.rewards.reward_nets import RewardEnsemble
from imitation.algorithms.preference_comparisons import (
        PreferenceModel, 
        EnsembleTrainer,
        CrossEntropyRewardLoss
    )

import gym
from gym.wrappers import TimeLimit
import torch
import numpy as np
rng = np.random.default_rng(12345)

import argparse
from .drex import DREX
from .custom_rw import SquashRewardNet, get_ensemble_members

''' TODO
IDEAS:
- Fit rewards networks across stages (epochs of BC) and choose the appropriate reward with learnable parameters
- Try SAC
- Try shaped rewards

OTHER POSSIBLE IMPROVEMENTS
- Early stopping (imitation.testing.reward_improvement)
- Use custom rewards (more hidden units, rnn, attention)
- A better preference loss (aLRP?)

IDEAS IMPLEMENTED:
- Luce preference with discount_factor, noise_prob, clipped reward differences (DRLHP)
- Mixed sampling
- Fixed horizon rollouts for ranked_trajectories
- Input normalization in reward function

PPT:
- GT vs NN reward comparison
- Video wrapper for showcasing test results
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="HalfCheetah-v3", type=str)
    parser.add_argument('--expert', default="PPO-10", type=str)
    args = parser.parse_args()

    EXPERT_ID = args.expert
    ENV_ID = args.env
    # variable horizon should be disabled for sampling equal length trajectories
    env_factory = lambda: TimeLimit(gym.make(ENV_ID, terminate_when_unhealthy=False), 1000)
    # env_factory = lambda: TimeLimit(gym.make(ENV_ID), 1000)
    env = env_factory()

    demo_path = 'demonstrations/sub-optimal/'+ENV_ID+'-'+EXPERT_ID

    K = 5 # rollouts per noise level
    N_NOISE_LEVELS = 20 # noise levels
    N_REWARD_MODELS = 3 # ensemble reward models
    N_EPOCHS = 10 # reward training epochs
    FRAGMENT_LEN = 50 # length of trajectory fragments
    N_PAIRS = 5000 # batch size for each training epoch
    NOISE_PREF_GAP = 0.3 # min noise gap between trajectory pairs

    # Behavior cloning policy
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=load(demo_path),
        rng=rng,
    )
    bc_trainer.train(n_epochs=10)

    # Reward model
    reward_net = RewardEnsemble(
        env.observation_space, 
        env.action_space, 
        members=get_ensemble_members(SquashRewardNet, N_REWARD_MODELS, env)
    )

    # Luce-Shephard preference model
    preference_model = PreferenceModel(reward_net, noise_prob=0.1, discount_factor=0.99)
    reward_trainer = EnsembleTrainer(
        preference_model=preference_model,
        loss=CrossEntropyRewardLoss(),
        batch_size=64,
        lr=1e-4,
        rng=rng,
    )

    # Train DREX
    drex_trainer = DREX(
        expert=bc_trainer.policy,
        reward_trainer=reward_trainer,
        env_factory=env_factory,
        n_noise_levels=N_NOISE_LEVELS,
        k=K,
        n_pairs=N_PAIRS,
        noise_pref_gap=NOISE_PREF_GAP,
        fragment_len=FRAGMENT_LEN,
        rng=rng
    )
    reward_loss, reward_accuracy = drex_trainer.train(N_EPOCHS)
    print(f"Reward Loss: {reward_loss}, Reward Acc: {reward_accuracy}")
    torch.save(reward_net.state_dict(), 'checkpoints/drex_reward_net/DREX-'+ENV_ID+'.pth')

    # reward_net.load_state_dict(torch.load('checkpoints/drex_reward_net/DREX-'+ENV_ID+'.pth'))
    for noise_level in drex_trainer.ranked_trajectories:
        rollouts = drex_trainer.ranked_trajectories[noise_level]
        for roll in rollouts:
            true_reward = np.sum(roll.rews)
            predicted_reward = []
            for i in range(roll.acts.shape[0]):
                obs, act, next_obs, done = roll.obs[i][None,:], roll.acts[i][None,:], roll.obs[i+1][None,:], np.array([False])
                predicted_reward.append(reward_net.predict(obs, act, next_obs, done))
            predicted_reward = np.sum(predicted_reward)
            print(f'Noise: {noise_level}, True reward: {true_reward}, Predicted reward: {predicted_reward}')