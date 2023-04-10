from imitation.algorithms.preference_comparisons import TrajectoryDataset, PreferenceModel, BasicRewardTrainer
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import load
from imitation.algorithms.bc import BC

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
rng = np.random.default_rng(0)

import gym

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
    
    def _predict(self, observation, deterministic = False):
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
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

def generate_ranked_trajectories(noise_schedule, k, env, expert, action_noise_type, rng):
    ranked_trajectories = {}
    for noise_level in noise_schedule:
        rollouts = rollout.rollout(
            NoiseInjectedPolicyWrapper(policy=expert, action_noise_type=action_noise_type, noise_level=noise_level),
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=k),
            rng=np.random.default_rng(0),
        )
        ranked_trajectories[noise_level] = (TrajectoryDataset(rollouts, rng=rng), rollouts)
    
    return ranked_trajectories

def log_rankings(ranked_trajectories):
    for noise_level in ranked_trajectories:
        rewards = []
        samples = 0
        rollouts = ranked_trajectories[noise_level][1]
        for roll in rollouts:
            rewards.append(np.mean(np.sum(roll.rews)))
            samples += roll.obs.shape[0]

        print('Noise: ', noise_level)
        print('#Samples: ', samples)
        print('Best: ', max(rewards))
        print('Worst: ', min(rewards))
        print('Avg: ', np.mean(np.array(rewards)))
        print('-----------------------------------')

def main():
    EXPERT_ID = "PPO-35"
    ENV_ID = "Hopper-v3"
    env = gym.make(ENV_ID) # variable horizon is true for fair evaluation

    K = 5 # rollouts per noise level
    N_NOISE_LEVELS = 20

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=load('demonstrations/sub-optimal/'+ENV_ID+'-'+EXPERT_ID),
        rng=rng,
    )
    bc_trainer.train(n_epochs=5)

    ranked_trajectories = generate_ranked_trajectories(np.linspace(0,1,N_NOISE_LEVELS), K, env, 
                                                       bc_trainer.policy, action_noise_type='normal',
                                                       rng=rng)
    
    log_rankings(ranked_trajectories)
        
main()