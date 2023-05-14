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

def setup_seed(seed):
  torch.manual_seed(seed)
  #torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  #torch.backends.cudnn.deterministic = True

setup_seed(1)

nA=4
nS=4*4
eps=0.0
gamma = 0.9             # Discount factor
Backward=backward_state(eps,nA,nS,False)

map_name_list=[]
for i in range(200):
  map_name = np.load('maps/map'+str(i)+'.npy')
  map_name = map_name.tolist()
  map_name_list.append(map_name)


def run(num_tasks, meta_parameter, episodes=5):

    global eps
    global gamma
    
    env = gym.make("FrozenLake-v1",desc= map_name_list[num_tasks], is_slippery=False)
    Unsafe_states, Unsafe_actions=unsafe_states_actions(env,Backward)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    #episodes = 5 #10        
    length = 100  #50       # Length of the sample trajectories, maybe we can change this
  
    STEP = 1
    H = 100
    #print('epsilon = ', eps, 'H = ', H)
    eps_new = eps/length
    # Set the threshold
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

  batch_size_task=10
  epoch_when_each_new=5
  sample_number=10
  nosiy_scale=0.1

  meta_init_data=np.random.normal(loc=1.0, scale=0.2, size=(16,4)) 
  meta_parameter=torch.FloatTensor(meta_init_data).requires_grad_()
  meta_parameter_list=[meta_parameter]
  optimizer1=torch.optim.Adam(meta_parameter_list,lr=0.001,weight_decay=0.0)

  for revealed_task_num in range(100):
    print("------------------------------------")
    print(revealed_task_num)
    print("------------------------------------")
    torch.save(meta_parameter, "pth/meta_parameter_map_epho"+str(revealed_task_num)+".pth")
    for epoch in range(epoch_when_each_new):
      num_tasks_list=list(range(revealed_task_num+1))
      if revealed_task_num<batch_size_task:
        num_tasks_list=list(range(revealed_task_num+1))
      else:
        random.shuffle(num_tasks_list)
        num_tasks_list1=num_tasks_list[0:batch_size_task]
        num_tasks_list=num_tasks_list1
        
      optimizer1.zero_grad()
      
      for task_index in num_tasks_list: 
        meta_parameter_data=meta_parameter.detach().data.numpy()
        for i in range(sample_number):
          noisy=np.random.normal(loc=0.0, scale=nosiy_scale, size=(16,4)) 
          meta_parameter_add_noisy=meta_parameter_data+noisy 
          policy_model_out, results, violations = run(task_index,meta_parameter_add_noisy,episodes=5) 
          data_pair1=(noisy/nosiy_scale,results[-1])

          meta_parameter_add_noisy=meta_parameter_data-noisy 
          policy_model_out, results, violations = run(task_index,meta_parameter_add_noisy,episodes=5) 
          data_pair2=(-noisy/nosiy_scale,results[-1])
          grade=(torch.FloatTensor(data_pair1[0])*data_pair1[1] +torch.FloatTensor(data_pair2[0])*data_pair2[1])/2/sample_number/len(num_tasks_list)
          
          if meta_parameter_list[0].grad== None:
            meta_parameter_list[0].grad=-grade
          else:
            meta_parameter_list[0].grad+=-grade

      optimizer1.step()

      

    

