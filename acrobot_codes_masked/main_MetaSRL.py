import numpy as np
from agents.MetaSRL import *
import matplotlib.pyplot as plt
import argparse
import random
import math
from torch.nn import functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

setup_seed(1)

TRAIN_TASK_COUNT = 10# args.train_task_count

CRPO_STEP_COUNT = 10#args.crpo_step_count
CRPO_EPISODE_COUNT = 10#args.crpo_episode_count
CG_ITER_COUNT = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.5

eps=0.5
LIMIT_RANGE = [40, 42]
metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
np.random.seed(0)
noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))
height=None
for i in range(TRAIN_TASK_COUNT):
    RUN = i
    # print("Task #{}".format(i))
    noise = noises[i]
    limit = limits[i]
    height=-0.5
    # if height is None:
    #     height=-0.5+0.001*i
    # else:
    #     height += np.random.normal(0.05,0.001)
    metasrl.step(eps,height, noise=noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=i%2)
    
    plt.plot(metasrl.rewards_by_task[-1], label="Reward")
    plt.plot(metasrl.cost_1s_by_task[-1], label="Cost 1")
    plt.plot(metasrl.cost_2s_by_task[-1], label="Cost 2")
    #plt.hlines(y=limit, xmin=0, xmax=CRPO_STEP_COUNT, colors="black", linestyles="--", label="Limit 1")
    plt.legend(loc="upper right")
    plt.title("Performance of MetaSRL on Task {}".format(i))
    plt.xlabel("CRPO Runs")
    plt.ylabel("Performance")
    plt.show()

    

    