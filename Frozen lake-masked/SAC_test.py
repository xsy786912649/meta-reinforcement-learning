# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:08:13 2023

@author: yzyja
"""
from CRPO_frozenlake import *

import scipy.io as scio
import torchvision
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
import time
import numpy as np 
import pandas as pd
import random
import math
from torch.nn import functional as F

nA=4
nS=4*4
eps=0.0
Backward=backward_state(eps,nA,nS,False)

map_name_list=[]
for i in range(200):
  map_name = np.load('maps/map'+str(i)+'.npy')
  map_name = map_name.tolist()
  map_name_list.append(map_name)

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.params = [
                torch.Tensor(128, 16).uniform_(-1./math.sqrt(16), 1./math.sqrt(8)).requires_grad_(),
                torch.Tensor(128).zero_().requires_grad_(),

                torch.Tensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.Tensor(128).zero_().requires_grad_(),

                torch.Tensor(64, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.Tensor(64).zero_().requires_grad_(),

            ]
    self.soft=torch.nn.Softmax(dim=1)

  def dense(self, x, params):
    y = F.linear(x, params[0], params[1])
    y = F.relu(y)

    y = F.linear(y, params[2], params[3])
    y = F.relu(y)

    y = F.linear(y, params[4], params[5])

    return y
  
  def forward(self, map_index, params):
    output=self.dense(self.input_process(map_index), params)
    output=output.reshape(16,4)
    output=self.soft(output)
    return output
  
  def input_process(self, map_index):
    map_name=map_name_list[map_index]
    map_input=[list(map_row) for map_row in map_name]
    map_vector=[]
    for row in map_input:
        for k in row:
            if k=='H':
                map_vector.append(1.0)
            else:
                map_vector.append(0.0)
    map_vector=np.array(map_vector)
    map_tensor=torch.FloatTensor(map_vector)
    return map_tensor

def Test(num_tasks, meta_parameter ,episodes=5):

    global eps
    
    map_name = np.load('maps/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=False)
    Unsafe_states, Unsafe_actions=unsafe_states_actions(env,Backward)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    gamma = 0.9             # Discount factor  
    length = 100  #50       # Length of the sample trajectories, maybe we can change this

    STEP = 10
    H = 100
    print('epsilon = ', eps, 'H = ', H)
    eps_new = eps/length
    d_threshold = 0.3

    # Initialize the Q-tables and the policy for the task at hand
    qtable_reward = np.zeros((16, 4))
    qtable_cost = np.zeros((16, 4))

    policy_model=meta_parameter

    alpha = 0.03

    N_0 = []

    ## Training the policy
    iterations = []
    results = []
    violations = []
    results_avg = []
    violations_avg = []
    result_avg = 0
    violation_avg = 0
    
    print(map_name)
    for iteration in range(STEP):
      
      result, violation, policy_model_out, value_model, cost_model = sample_trajectories(env, gamma, beta, episodes, length,
                                                                                         policy_model, qtable_reward, qtable_cost, 
                                                                                         d_threshold, N_0, alpha,
                                                                                         Unsafe_states, Unsafe_actions,eps_new,H)

      policy_model = policy_model_out
      qtable_cost = cost_model
      qtable_reward = value_model

      iterations.append(iteration)
      results.append(result)
      violations.append(violation)

      result_avg = (iteration/(iteration+1)) * result_avg + (1/(iteration+1)) * result
      violation_avg = (iteration / (iteration + 1)) * violation_avg + (1 / (iteration + 1)) * (violation)
      

      results_avg.append(result_avg)
      violations_avg.append(violation_avg)

    ###############--------------plot--------------------############################################
    '''
    constraint1 = [d_threshold for i in range(STEP)]
    # constraint2 = [d_threshold for i in range(STEP)]

    plt.figure()
    plt.plot(iterations, results, color='r', linestyle='-', label='TRPO reward')
    plt.plot(iterations, violations, color='b', linestyle='-', label='TRPO cost')
    
    # plt.plot(iterations, results_avg, color='r', linestyle='-.')
    # plt.plot(iterations, violations_avg, color='b', linestyle='-.')
    # # plt.plot(iterations, violations2_avg, color='g', linestyle='-.', label='C-TRPO average cost2')

    plt.plot(iterations, constraint1, color='b', linestyle='--')
    # plt.plot(iterations, constraint2, color='g', linestyle='--')
    plt.legend(loc='upper left')
    plt.xlabel('# of Episodes')
    plt.ylabel('Reward/Cost')
    plt.title('Task'+str(num_tasks))
    plt.show()
    '''
    print(results)

    return policy_model_out, results, violations


if __name__ == '__main__':

  num_tasks =100

  Meta_map=torch.load("pth/meta_parameter_map_epho99.pth")
  for i in range(num_tasks): 
    task_index=i+101
    print(task_index)

    meta_parameter=Meta_map.forward(task_index-1,Meta_map.params)
    meta_parameter=meta_parameter.data.numpy()
    #meta_parameter=np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,1,1,0]])

    policy_model_test, results_test, violations_test = Test(task_index,meta_parameter,episodes=5)
    np.save('maps/Test_task_data/SAC/rewards_test'+str(i)+'.npy', results_test)
    np.save('maps/Test_task_data/SAC/costs_test'+str(i)+'.npy', violations_test)


  Meta_map=torch.load("pth/meta_parameter_map_epho0.pth")
  for i in range(num_tasks): 
    task_index=i+101
    print(task_index)

    meta_parameter=Meta_map.forward(task_index-1,Meta_map.params)
    meta_parameter=meta_parameter.data.numpy()
    #meta_parameter=np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,1,1,0]])

    policy_model_test, results_test, violations_test = Test(task_index,meta_parameter,episodes=5)
    np.save('maps/Test_task_data/SAC/rewards_test_initial'+str(i)+'.npy', results_test)
    np.save('maps/Test_task_data/SAC/costs_test_initial'+str(i)+'.npy', violations_test)

