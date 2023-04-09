from imitation.data.types import load
import numpy as np

EXPERT_ID = "SAC-10"
ENV_ID = "Ant-v3"

optimality = "sub-optimal/"
# optimality = ""
rollouts = load('demonstrations/'+optimality+ENV_ID+'-'+EXPERT_ID)

rewards = []
samples = 0
for roll in rollouts:
    rewards.append(np.mean(np.sum(roll.rews)))
    samples += roll.obs.shape[0]

print('#Samples: ', samples)
print('Best: ', max(rewards))
print('Worst: ', min(rewards))
print('Avg: ', np.mean(np.array(rewards)))