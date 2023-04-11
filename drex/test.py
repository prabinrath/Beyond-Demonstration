import gym
from gym.wrappers import TimeLimit
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

    drex_trainer = DREX(
        demo_path=demo_path,
        env_factory=env_factory,
        n_noise_levels=N_NOISE_LEVELS,
        k=K,
        n_reward_models=N_REWARD_MODELS,
        n_pairs=N_PAIRS,
        noise_pref_gap=NOISE_PREF_GAP,
        fragment_len=FRAGMENT_LEN
    )
    # drex_trainer.train(N_EPOCHS)

main()