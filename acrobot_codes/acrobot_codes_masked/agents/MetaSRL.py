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
        value_function_lr = 1,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.value_function_lr = value_function_lr
        ## Initial neural network
        self.policy = DQNSoftmax(input_size, output_size)
        self.value_function = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        self.cost_value_function_1 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        self.cost_value_function_2 = ValueFunctionWrapper(DQNRegressor(input_size), value_function_lr)
        ## Cuda identity
        # self.policy.cuda()
        # self.value_function.cuda()
        # self.cost_value_function_1.cuda()
        # self.cost_value_function_2.cuda()

        self.prev_policies = []
        self.prev_value_functions = []
        self.prev_cost_value_functions_1 = []
        self.prev_cost_value_functions_2 = []

        self.prev_kl_mean_divergences = []

        self.rewards_by_task = []
        self.cost_1s_by_task = []
        self.cost_2s_by_task = []

        self.test_rewards_by_task = []
        self.test_cost_1s_by_task = []
        self.test_cost_2s_by_task = []
        self.average_violations_by_task=[]
        self.test_average_violations_by_task=[]
        self.task_count = 0


    def step(self, eps, noise: np.array, crpo_step = 10, crpo_episodes = 10, cg_iters = 10, limit_1 = 50, limit_2 = 50, direction = 0):
        env = AcrobotEnv(noise)
        self.update_alpha()
        # print(self.cost_value_function_1,self.cost_value_function_2)
        print(crpo_episodes)
        crpo = CRPO(env, eps,self.input_size, self.output_size, self.policy, self.value_function, self.cost_value_function_1, self.cost_value_function_2,
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
        
        self.prev_policies.append(deepcopy(crpo.policy))
        self.prev_value_functions.append(deepcopy(crpo.value_function))
        self.prev_cost_value_functions_1.append(deepcopy(crpo.cost_value_function_1))
        self.prev_cost_value_functions_2.append(deepcopy(crpo.cost_value_function_2))
        self.prev_kl_mean_divergences.append(self.mean_kl_divergence(deepcopy(crpo.policy), deepcopy(crpo.all_observations)))

        self.task_count += 1
        self.update()

    def reptile_update(self):
        ## Reptile style update
        temp_state_dict = self.policy.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + self.alpha * (self.prev_policies[-1].state_dict()[key] - temp_state_dict[key])
        self.policy.load_state_dict(temp_state_dict)

        temp_state_dict = self.value_function.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + self.alpha * (self.prev_value_functions[-1].state_dict()[key] - temp_state_dict[key])
        self.value_function.load_state_dict(temp_state_dict)

        temp_state_dict = self.cost_value_function_1.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + self.alpha * (self.prev_cost_value_functions_1[-1].state_dict()[key] - temp_state_dict[key])
        self.cost_value_function_1.load_state_dict(temp_state_dict)

        temp_state_dict = self.cost_value_function_2.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = temp_state_dict[key] + self.alpha * (self.prev_cost_value_functions_2[-1].state_dict()[key] - temp_state_dict[key])
        self.cost_value_function_2.load_state_dict(temp_state_dict)

    def update(self):
        ## Meta Updates
        # temp_state_dict = self.policy.state_dict()
        # for key in temp_state_dict:
        #     temp_state_dict[key] = sum([self.prev_policies[j].state_dict()[key] for j in range(len(self.prev_policies))]) / len(self.prev_policies)
        # self.policy.load_state_dict(temp_state_dict)

        # temp_state_dict = self.value_function.state_dict()
        # for key in temp_state_dict:
        #     temp_state_dict[key] = sum([self.prev_value_functions[j].state_dict()[key] for j in range(len(self.prev_policies))]) / len(self.prev_policies)
        # self.value_function.load_state_dict(temp_state_dict)

        # temp_state_dict = self.cost_value_function_1.state_dict()
        # for key in temp_state_dict:
        #     temp_state_dict[key] = sum([self.prev_cost_value_functions_1[j].state_dict()[key] for j in range(len(self.prev_policies))]) / len(self.prev_policies)
        # self.cost_value_function_1.load_state_dict(temp_state_dict)

        # temp_state_dict = self.cost_value_function_2.state_dict()
        # for key in temp_state_dict:
        #     temp_state_dict[key] = sum([self.prev_cost_value_functions_2[j].state_dict()[key] for j in range(len(self.prev_policies))]) / len(self.prev_policies)
        # self.cost_value_function_2.load_state_dict(temp_state_dict)

        ## Meta Updates
        temp_state_dict = self.policy.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = self.prev_policies[-1].state_dict()[key]
        self.policy.load_state_dict(temp_state_dict)

        temp_state_dict = self.value_function.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = self.prev_value_functions[-1].state_dict()[key]
        self.value_function.load_state_dict(temp_state_dict)

        temp_state_dict = self.cost_value_function_1.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = self.prev_cost_value_functions_1[-1].state_dict()[key]
        self.cost_value_function_1.load_state_dict(temp_state_dict)

        temp_state_dict = self.cost_value_function_2.state_dict()
        for key in temp_state_dict:
            temp_state_dict[key] = self.prev_cost_value_functions_2[-1].state_dict()[key]
        self.cost_value_function_2.load_state_dict(temp_state_dict)

        # print(self.cost_value_function_1,self.cost_value_function_2)

    def evaluate(self,  eps, noise: np.array, crpo_step = 5, crpo_episodes = 50, cg_iters = 10, limit_1 = 50, limit_2 = 50, direction = 0):
        env = AcrobotEnv(noise)
        crpo = CRPO(env,  eps, self.input_size, self.output_size, deepcopy(self.policy), deepcopy(self.value_function), deepcopy(self.cost_value_function_1), deepcopy(self.cost_value_function_2),
                    direction=direction,
                    cg_iters=cg_iters,
                    episodes=crpo_episodes,
                    limit_1=limit_1,
                    limit_2=limit_2)
        rewards = []
        cost_1s = []
        cost_2s = []
        violations = []
        for _ in range(crpo_step):
            reward, cost_1, cost_2, average_violations = crpo.step()
            rewards.append(reward)
            cost_1s.append(cost_1)
            cost_2s.append(cost_2)
            violations.append(average_violations)
            print("Reward: {:.2f} - Cost 1: {:.2f} - Cost 2: {:.2f} - Violation rate: {:2f}".format(reward, cost_1, cost_2,average_violations))
        self.test_rewards_by_task.append(rewards)
        self.test_cost_1s_by_task.append(cost_1s)
        self.test_cost_2s_by_task.append(cost_2s)
        self.test_average_violations_by_task.append(violations)
    def mean_kl_divergence(self, suboptimal_policy, observations):
        observations_tensor = torch.cat(
            [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
        actprob = suboptimal_policy(observations_tensor).detach() + 1e-8
        old_actprob = self.policy(observations_tensor)
        loss = torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean().detach().cpu().numpy()
        return loss

    def update_alpha(self):
        if self.task_count > 0:
            self.alpha = 1/(5**(1/4))*(1/self.task_count**(1/2)+np.array(self.prev_kl_mean_divergences).mean())**(1/2)
        else:
            self.alpha = 1e-2