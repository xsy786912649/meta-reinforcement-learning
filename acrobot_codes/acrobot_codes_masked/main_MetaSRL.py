import numpy as np
from agents.MetaSRL import *
import matplotlib.pyplot as plt
import argparse
# import matplotlib.pyplot as pl

# parser = argparse.ArgumentParser(description='Optional app description')
# # Optional argument
# parser.add_argument('--train_task_count', type=int, help='An optional integer argument')
# parser.add_argument('--crpo_step_count', type=int, help='An optional integer argument')
# parser.add_argument('--crpo_episode_count', type=int, help='An optional integer argument')
# parser.add_argument('--run', type=int, help='An optional integer argument')
# args = parser.parse_args()

# RUN = args.run
RUN = 1
TRAIN_TASK_COUNT = 2# args.train_task_count

CRPO_STEP_COUNT = 2#args.crpo_step_count
CRPO_EPISODE_COUNT = 5#args.crpo_episode_count
CG_ITER_COUNT = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.0
LIMIT_RANGE = [40, 42]

eps=0.05
H=100
metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
np.random.seed(0)
noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))
for i in range(TRAIN_TASK_COUNT):
    print("Task #{}".format(i))
    noise = noises[i]
    limit = limits[i]

    metasrl.step(eps,noise=noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=i%2)
    
    plt.plot(metasrl.rewards_by_task[-1], label="Reward")
    plt.plot(metasrl.cost_1s_by_task[-1], label="Cost 1")
    plt.plot(metasrl.cost_2s_by_task[-1], label="Cost 2")
    plt.hlines(y=limit, xmin=0, xmax=CRPO_STEP_COUNT, colors="black", linestyles="--", label="Limit 1")
    plt.legend(loc="upper right")
    plt.title("Performance of MetaSRL on Task {}".format(i))
    plt.xlabel("CRPO Runs")
    plt.ylabel("Performance")
    plt.savefig("results/MetaSRL/run_{}/plots/plot_{}.png".format(RUN,i))
    plt.close()

    performance = np.array([metasrl.rewards_by_task, metasrl.cost_1s_by_task, metasrl.cost_2s_by_task])
    np.save("results/MetaSRL/run_{}/performance_data/performance_{}.npy".format(RUN,i), performance)

    torch.save(metasrl.policy, "results/MetaSRL/run_{}/models/model_{}.png".format(RUN,i))
    torch.save(metasrl.value_function, "results/MetaSRL/run_{}/models/value_function_{}.png".format(RUN,i))
    torch.save(metasrl.cost_value_function_1, "results/MetaSRL/run_{}/models/cost_value_function_1_{}.png".format(RUN,i))
    torch.save(metasrl.cost_value_function_2, "results/MetaSRL/run_{}/models/cost_value_function_2_{}.png".format(RUN,i))


## Test time
print("########## Test Time ##########")
TEST_TASK_COUNT = 5

CRPO_STEP_COUNT = 5
CRPO_EPISODE_COUNT = 100
CG_ITER_COUNT = 5

test_noises = np.random.normal(0.0, VARIANCE, size=(TEST_TASK_COUNT, 4))
test_limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TEST_TASK_COUNT))
for i in range(TEST_TASK_COUNT):
    print("Test Task #{}".format(i))
    test_noise = test_noises[i]
    test_limit = test_limits[i]
    metasrl.evaluate( eps,noise=test_noise, crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=test_limit, limit_2=test_limit, direction=i%2)
    
    test_performance = np.array([metasrl.test_rewards_by_task, metasrl.test_cost_1s_by_task, metasrl.test_cost_2s_by_task,metasrl.test_average_violations_by_task])
    np.save("results/MetaSRL/run_{}/performance_data/test_performance_{}.npy".format(RUN,i), test_performance)
    
    plt.figure(1)
    plt.plot(metasrl.test_average_violations_by_task)
    plt.savefig('test_average_violation_acrobot'+str(eps)+'.png')