from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import time

ALGO_ID = "PPO"
algo = {"PPO": PPO, "SAC": SAC}

ENV_ID = "Hopper-v3"
expert = algo[ALGO_ID].load('checkpoints/expert_policies/'+ENV_ID+'-'+ALGO_ID)

env = gym.make(ENV_ID)
reward, _ = evaluate_policy(expert, env, 10)
print("Avg reward:", reward)

for i in range(5):
    obs = env.reset()
    done = False
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)