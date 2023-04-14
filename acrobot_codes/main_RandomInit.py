from random import Random
import numpy as np
from agents.RandomInit import *
import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
parser.add_argument('--train_task_count', type=int, help='An optional integer argument')
parser.add_argument('--crpo_step_count', type=int, help='An optional integer argument')
parser.add_argument('--crpo_episode_count', type=int, help='An optional integer argument')
parser.add_argument('--run', type=int, help='An optional integer argument')
args = parser.parse_args()

RUN = args.run

TRAIN_TASK_COUNT = args.train_task_count

CRPO_STEP_COUNT = args.crpo_step_count
CRPO_EPISODE_COUNT = args.crpo_episode_count
CG_ITER_COUNT = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.1
LIMIT_RANGE = [40, 42]


random_init = RandomInit(INPUT_SIZE, OUTPUT_SIZE)
np.random.seed(RUN)
noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))
# for i in range(TRAIN_TASK_COUNT-1, TRAIN_TASK_COUNT):
#     print("Task #{}".format(i))
#     noise = noises[i]
#     limit = limits[i]
#     random_init.step(noise=noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=i%2)

#     plt.plot(random_init.rewards_by_task[-1], label="Reward")
#     plt.plot(random_init.cost_1s_by_task[-1], label="Cost 1")
#     plt.plot(random_init.cost_2s_by_task[-1], label="Cost 2")
#     plt.hlines(y=limit, xmin=0, xmax=CRPO_STEP_COUNT, colors="black", linestyles="--", label="Limit")
#     plt.legend(loc="upper right")
#     plt.title(f'Performance of Random Init on Task {i}')
#     plt.xlabel("CRPO Runs")
#     plt.ylabel("Performance")
#     plt.savefig(f'results/RandomInit/run_{RUN}/plots/plot_{i}')
#     plt.close()

#     performance = np.array([random_init.rewards_by_task, random_init.cost_1s_by_task, random_init.cost_2s_by_task])
#     np.save(f"results/RandomInit/run_{RUN}/performance_data/performance_{i}.npy", performance)

#     torch.save(random_init.policy, f"results/RandomInit/run_{RUN}/models/model_{i}")
#     torch.save(random_init.value_function, f"results/RandomInit/run_{RUN}/models/value_function_{i}")
#     torch.save(random_init.cost_value_function_1, f"results/RandomInit/run_{RUN}/models/cost_value_function_1_{i}")
#     torch.save(random_init.cost_value_function_2, f"results/RandomInit/run_{RUN}/models/cost_value_function_2_{i}")

## Test time
print("########## Test Time ##########")
TEST_TASK_COUNT = 10

CRPO_STEP_COUNT = 5
CRPO_EPISODE_COUNT = 5
CG_ITER_COUNT = 4

test_noises = np.random.normal(0.0, VARIANCE, size=(TEST_TASK_COUNT, 4))
test_limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TEST_TASK_COUNT))
for i in range(TEST_TASK_COUNT):
    print("Test Task #{}".format(i))
    test_noise = test_noises[i]
    test_limit = test_limits[i]
    random_init.step(noise=test_noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=test_limit, limit_2=test_limit, direction=1)
    
    test_performance = np.array([random_init.rewards_by_task, random_init.cost_1s_by_task, random_init.cost_2s_by_task])
    np.save(f"results/RandomInit/run_{RUN}/performance_data/test_performance_{i}.npy", test_performance)