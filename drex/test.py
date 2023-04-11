from imitation.data.types import load
from imitation.algorithms.bc import BC
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import RewardEnsemble, BasicRewardNet
from imitation.algorithms.preference_comparisons import (
        PreferenceModel, 
        EnsembleTrainer,
        CrossEntropyRewardLoss
    )

import gym
from gym.wrappers import TimeLimit
import numpy as np
rng = np.random.default_rng(0)

from .drex import DREX

def main():
    EXPERT_ID = "PPO-35"
    ENV_ID = "Hopper-v3"
    # variable horizon should be disabled for sampling equal length trajectories
    env_factory = lambda: TimeLimit(gym.make(ENV_ID, terminate_when_unhealthy=False), 1000)
    # env_factory = lambda: TimeLimit(gym.make(ENV_ID), 1000)

    demo_path = 'demonstrations/sub-optimal/'+ENV_ID+'-'+EXPERT_ID

    K = 5 # rollouts per noise level
    N_NOISE_LEVELS = 20
    N_REWARD_MODELS = 3
    N_EPOCHS = 5 # reward training epochs
    FRAGMENT_LEN = 50
    N_PAIRS = 5000
    NOISE_PREF_GAP = 0.3

    # Behavior cloning policy
    env = env_factory()
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=load(demo_path),
        rng=rng,
    )
    bc_trainer.train(n_epochs=5)

    # Reward model
    reward_members = [BasicRewardNet(
                        env.observation_space,
                        env.action_space,
                        use_action=False, # TREX has state only reward functions
                        normalize_input_layer=RunningNorm)
                        for _ in range(N_REWARD_MODELS)]
    reward_net = RewardEnsemble(
        env.observation_space, 
        env.action_space, 
        members=reward_members
    )

    # Luce-Shephard preference model
    preference_model = PreferenceModel(reward_net)
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

main()