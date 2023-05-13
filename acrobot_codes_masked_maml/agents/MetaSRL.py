import numpy as np
from utils.torch_utils import use_cuda, Tensor, Variable, ValueFunctionWrapper
from utils.CRPO import *
from utils.DQNSoftmax import *
from utils.DQNRegressor import *
from acrobot import *
from copy import deepcopy


class MetaSRL:
    def __init__(
        self,
        input_size,
        output_size,
        alpha = 0.15,
        value_function_lr = 1.0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.value_function_lr = value_function_lr
        ## Initial neural network
        self.policy = DQNSoftmax(input_size, output_size)
        self.value_function = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        #self.cost_value_function_1 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        #self.cost_value_function_2 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)

        self.rewards_by_task = []
        self.cost_1s_by_task = []
        self.cost_2s_by_task = []

        self.test_rewards_by_task = []
        self.test_cost_1s_by_task = []
        self.test_cost_2s_by_task = []
        self.average_violations_by_task=[]
        self.test_average_violations_by_task=[]

    def step(self, eps, height,noise: np.array, crpo_step = 10, crpo_episodes = 10, cg_iters = 10, limit_1 = 50, limit_2 = 50, direction = 0):
        env = AcrobotEnv(noise)
        #self.update_alpha()
        # print(self.cost_value_function_1,self.cost_value_function_2)
        # print(crpo_episodes)
        crpo = CRPO(env, eps,self.input_size, self.output_size, self.policy, self.value_function, None, None,
                    height=height,
                    direction=direction,
                    max_kl=self.alpha,
                    cg_iters=cg_iters,
                    episodes=crpo_episodes,
                    limit_1=limit_1,
                    limit_2=limit_2)
        # print(self.cost_value_function_1,self.cost_value_function_2)
        rewards = []
        cost_1s = []
        cost_2s = []
        violations = []
        for _ in range(crpo_step):
            # print('before step: ',crpo.cost_value_function_1,crpo.cost_value_function_2)
            reward, cost_1, cost_2, average_violations = crpo.step()
            rewards.append(reward)
            cost_1s.append(cost_1)
            cost_2s.append(cost_2)
            violations.append(average_violations)
            print("Reward: {:.2f} - Cost 1: {:.2f} - Cost 2: {:.2f} - Violation: {:.2f}".format(reward, cost_1, cost_2, average_violations))
        self.rewards_by_task.append(rewards)
        self.cost_1s_by_task.append(cost_1s)
        self.cost_2s_by_task.append(cost_2s)
        self.average_violations_by_task.append(violations)
        


