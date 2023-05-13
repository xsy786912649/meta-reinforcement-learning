import numpy as np
from agents.SimpleAveraging import *
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


simple_averaging = SimpleAveraging(INPUT_SIZE, OUTPUT_SIZE)
np.random.seed(RUN)
noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))


## Test time
print("########## Test Time ##########")
TEST_TASK_COUNT = 10

CRPO_STEP_COUNT = 5
CRPO_EPISODE_COUNT = 10
CG_ITER_COUNT = 5

test_noises = np.random.normal(0.0, VARIANCE, size=(TEST_TASK_COUNT, 4))
test_limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TEST_TASK_COUNT))
###
# torch.load()
temp_list = []
for i in range(1, 5):
    simple_averaging.policy = torch.load(f"simple_averaging_models/model_{i}")
    temp_list.append(deepcopy(simple_averaging.policy))

temp_state_dict = simple_averaging.policy.state_dict()
for key in temp_state_dict:
    temp_state_dict[key] = sum([temp_list[j].state_dict()[key] for j in range(len(temp_list))]) / len(temp_list)
simple_averaging.policy.load_state_dict(temp_state_dict)
###
for i in range(TEST_TASK_COUNT):
    print("Test Task #{}".format(i))
    test_noise = test_noises[i]
    test_limit = test_limits[i]
    simple_averaging.step(noise=test_noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=test_limit, limit_2=test_limit, direction=i%2)
    
    test_performance = np.array([simple_averaging.rewards_by_task, simple_averaging.cost_1s_by_task, simple_averaging.cost_2s_by_task])
    np.save(f"results/SimpleAveraging/run_{RUN}/performance_data/test_performance_{i}.npy", test_performance)