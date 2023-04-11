from imitation.rewards.reward_nets import RewardEnsemble, BasicRewardNet
from imitation.data import rollout
from imitation.algorithms.base import BaseImitationAlgorithm
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import load, TrajectoryWithRew
from imitation.algorithms.bc import BC
from imitation.util.networks import RunningNorm
from imitation.algorithms.preference_comparisons import (
        PreferenceModel, 
        EnsembleTrainer,
        CrossEntropyRewardLoss,
        PreferenceDataset
    )

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
rng = np.random.default_rng(0)


class NoiseInjectedPolicyWrapper(BasePolicy):
    def __init__(self, policy, action_noise_type, noise_level):
        super().__init__(policy.observation_space, policy.action_space)
        self.policy = policy
        self.action_noise_type = action_noise_type

        if action_noise_type == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mean=mu, sigma=std)
        elif action_noise_type == 'ou':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = OrnsteinUhlenbeckActionNoise(mean=mu, sigma=std)
        elif action_noise_type == 'epsilon':
            self.epsilon = noise_level
        else:
            assert False, "no such action noise type: %s" % (action_noise_type)
    
    def _predict(self, observation, deterministic = False): # copied from sb3 templates
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        if self.action_noise_type == 'epsilon':
            if np.random.random() < self.epsilon:
                return self.action_space.sample()
            else:
                action, _ = self.policy.predict(observation, deterministic)
        else:
            action, _ = self.policy.predict(observation, deterministic)
            action += self.action_noise()

        return np.clip(action, self.action_space.low, self.action_space.high), state

    def reset_noise(self):
        self.action_noise.reset()


class DREX(BaseImitationAlgorithm):
    def __init__(self, 
                 demo_path,
                 env_factory,
                 n_noise_levels,
                 k,
                 n_reward_models,
                 n_pairs,
                 noise_pref_gap,
                 fragment_len
            ):

        self.n_pairs = n_pairs
        self.fragment_len = fragment_len
        self.noise_pref_gap = noise_pref_gap

        # Behavior Cloning
        env = env_factory()
        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=load(demo_path),
            rng=rng,
        )
        bc_trainer.train(n_epochs=5)
        noise_schedule = np.linspace(0,1,n_noise_levels)
        self.ranked_trajectories = self.generate_ranked_trajectories(noise_schedule, k, env, 
                                                        bc_trainer.policy, action_noise_type='normal',
                                                        rng=rng)
        self.log_rankings(self.ranked_trajectories)

        # TREX
        venv = make_vec_env(env_factory, n_envs=4)
        reward_members = [BasicRewardNet(
                            venv.observation_space,
                            venv.action_space,
                            use_action=False, # TREX has state only reward functions
                            normalize_input_layer=RunningNorm)
                            for _ in range(n_reward_models)]
        self.reward_net = RewardEnsemble(
            venv.observation_space, 
            venv.action_space, 
            members=reward_members
        )
        preference_model = PreferenceModel(self.reward_net)
        self.reward_trainer = EnsembleTrainer(
            preference_model=preference_model,
            loss=CrossEntropyRewardLoss(),
            batch_size=64,
            lr=1e-4,
            rng=rng,
        )
        self.dataset = PreferenceDataset(max_size=n_pairs) # allow infinite queue size

    def generate_ranked_trajectories(self, noise_schedule, k, env, expert, action_noise_type, rng):
        ranked_trajectories = {}
        for noise_level in noise_schedule:
            rollouts = rollout.rollout(
                NoiseInjectedPolicyWrapper(policy=expert, action_noise_type=action_noise_type, noise_level=noise_level),
                DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
                rollout.make_sample_until(min_timesteps=None, min_episodes=k),
                rng=rng,
            )
            ranked_trajectories[noise_level] = rollouts
        
        return ranked_trajectories

    def generate_fragments(self, ranked_trajectories, n_pairs, fragment_len, noise_pref_gap):
        noise_schedule = list(ranked_trajectories.keys())
        fragments = []
        for _ in range(n_pairs):
            idx_1, idx_2 = np.random.choice(len(noise_schedule), size=2, replace=False)
            while abs(noise_schedule[idx_1]-noise_schedule[idx_2])<noise_pref_gap:
                idx_1, idx_2 = np.random.choice(len(noise_schedule), size=2, replace=False)

            # idx_1 is always preferred
            if noise_schedule[idx_1] > noise_schedule[idx_2]:
                idx_1, idx_2 = idx_2, idx_1
            
            trajectories_1 = ranked_trajectories[noise_schedule[idx_1]]
            trajectories_2 = ranked_trajectories[noise_schedule[idx_2]]
            idx_1, idx_2 = np.random.choice(len(trajectories_1)), np.random.choice(len(trajectories_2))
            trajectory_1, trajectory_2 = trajectories_1[idx_1], trajectories_2[idx_2]
            assert len(trajectory_1) > fragment_len and len(trajectory_2) > fragment_len
            idx_1, idx_2 = np.random.choice(len(trajectory_1)-fragment_len-1), np.random.choice(len(trajectory_2)-fragment_len-1)
            fragments.append((
                TrajectoryWithRew(trajectory_1.obs[idx_1:idx_1+fragment_len+1,:], 
                                trajectory_1.acts[idx_1:idx_1+fragment_len,:],
                                trajectory_1.infos,
                                trajectory_1.terminal,
                                trajectory_1.rews[idx_1:idx_1+fragment_len]),
                TrajectoryWithRew(trajectory_2.obs[idx_2:idx_2+fragment_len+1,:], 
                                trajectory_2.acts[idx_2:idx_2+fragment_len,:],
                                trajectory_2.infos,
                                trajectory_2.terminal,
                                trajectory_2.rews[idx_2:idx_2+fragment_len])                              
                ))
        return fragments

    def log_rankings(self, ranked_trajectories):
        for noise_level in ranked_trajectories:
            rewards = []
            samples = 0
            rollouts = ranked_trajectories[noise_level]
            for roll in rollouts:
                rewards.append(np.mean(np.sum(roll.rews)))
                samples += roll.obs.shape[0]

            print('Noise: ', noise_level)
            print('#Samples: ', samples)
            print('Best: ', max(rewards))
            print('Worst: ', min(rewards))
            print('Avg: ', np.mean(np.array(rewards)))
            print('-----------------------------------')

    def train(self, n_epochs):
        for _ in range(n_epochs):
            fragments_batch = self.generate_fragments(self.ranked_trajectories, 
                                        self.n_pairs,
                                        self.fragment_len,
                                        self.noise_pref_gap)
            preferences_batch = np.ones((self,n_epochs,), dtype=np.float32)
            self.dataset.push(fragments_batch, preferences_batch) # should evict previous batch as it acts like a deque
            self.reward_trainer.train(self.dataset)