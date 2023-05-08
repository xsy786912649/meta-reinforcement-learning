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
eps=0.05
gamma = 0.9             # Discount factor
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
                torch.FloatTensor(128, 16).uniform_(-1./math.sqrt(16), 1./math.sqrt(8)).requires_grad_(),
                torch.FloatTensor(128).zero_().requires_grad_(),

                torch.FloatTensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.FloatTensor(128).zero_().requires_grad_(),

                torch.FloatTensor(64, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.FloatTensor(64).zero_().requires_grad_(),

            ]

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


class Qfunction(torch.nn.Module):
  def __init__(self):
    super(Qfunction, self).__init__()
    self.params = [
                torch.FloatTensor(128, 16+64).uniform_(-1./math.sqrt(16), 1./math.sqrt(16)).requires_grad_(),
                torch.FloatTensor(128).zero_().requires_grad_(),

                torch.FloatTensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.FloatTensor(128).zero_().requires_grad_(),

                torch.FloatTensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.FloatTensor(128).zero_().requires_grad_(),

                torch.FloatTensor(1, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                torch.FloatTensor(1).zero_().requires_grad_(),

            ]

  def dense(self, x, params):
    y = F.linear(x, params[0], params[1])
    y = F.relu(y)

    y = F.linear(y, params[2], params[3])
    y = F.relu(y)

    y = F.linear(y, params[4], params[5])
    y = F.relu(y)

    y = F.linear(y, params[6], params[7])

    return y
  
  def forward(self, map_index, action, params):
    output=self.dense(self.input_process(map_index,action), params)
    
    return output
  
  def input_process(self, map_index, action):
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
    action_reshape=action.reshape(64)
    return torch.cat((map_tensor, action_reshape), 0)

def run(num_tasks, meta_parameter, episodes=5):

    global eps
    global gamma
    
    env = gym.make("FrozenLake-v1",desc= map_name_list[num_tasks], is_slippery=False)
    Unsafe_states, Unsafe_actions=unsafe_states_actions(env,Backward)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    #episodes = 5 #10        
    length = 100  #50       # Length of the sample trajectories, maybe we can change this
  
    STEP = 3
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

def add_pair(replay_buffer,item,replay_buffer_size):
  if len(replay_buffer)<replay_buffer_size:
    replay_buffer.append(item)
  else:
    replay_buffer.append(item)
    replay_buffer=replay_buffer[:-1]
  return replay_buffer

def sample_pair(replay_buffer,sample_size):
  if len(replay_buffer)<sample_size:
    return replay_buffer[:]
  else:
    sample_index=random.sample(range(len(replay_buffer)), sample_size)
    replay_buffer_sample=[replay_buffer[i] for i in sample_index]
    return replay_buffer_sample
   
if __name__ == '__main__':

  Meta_map=Model()
  Q_meta=Qfunction()

  target_meta=Model()
  target_meta.params=[pa.clone().detach().requires_grad_() for pa in Meta_map.params]
  target_Q_meta=Qfunction()
  target_Q_meta.params=[pa.clone().detach().requires_grad_() for pa in Q_meta.params]

  replay_buffer=[]
  replay_buffer_size=20000

  batch_size_task=10
  batch_size_point=10
  epoch_when_each_new=5

  optimizer_Q=torch.optim.Adam(Q_meta.params,lr=0.001,weight_decay=0.0)
  optimizer_action=torch.optim.Adam(Meta_map.params,lr=0.00003,weight_decay=0.0)

  nosiy_scale=0.005
  noisy=np.random.normal(loc=0.0, scale=nosiy_scale, size=(16,4)) 
  
  for revealed_task_num in range(100):
    print("------------------------------------")
    print(revealed_task_num)
    print("------------------------------------")
    torch.save(Meta_map, "pth/meta_parameter_map_epho"+str(revealed_task_num)+".pth")
    for epoch in range(epoch_when_each_new):
      num_tasks_list=list(range(revealed_task_num+1))
      if revealed_task_num<batch_size_task:
        num_tasks_list=list(range(revealed_task_num+1))
      else:
        random.shuffle(num_tasks_list)
        num_tasks_list1=num_tasks_list[0:batch_size_task]
        num_tasks_list=num_tasks_list1
        
      for task_index in num_tasks_list: 
        meta_parameter_tensor=Meta_map.forward(task_index,Meta_map.params)
        meta_parameter=meta_parameter_tensor.data.numpy()
        meta_parameter_add_noisy=meta_parameter+noisy 
        policy_model_out, results, violations = run(task_index+1,meta_parameter_add_noisy) 
        #print(meta_parameter)
        data_pair=(task_index,torch.FloatTensor(meta_parameter_add_noisy),results[-1]-0.5,task_index+1)
        replay_buffer=add_pair(replay_buffer,data_pair,replay_buffer_size)

      list_sample=sample_pair(replay_buffer,batch_size_point)

      optimizer_Q.zero_grad()
      optimizer_action.zero_grad()

      for sample in list_sample:
        task_index_sample=sample[0]
        meta_parameter_tensor_sample=sample[1]
        results_sample=sample[2]
        task_index_sample_next=sample[3]
        next_action=target_meta.forward(task_index_sample_next,target_meta.params)

        q_value=results_sample+0.9*target_Q_meta.forward(task_index_sample_next,next_action,target_Q_meta.params) 
        loss_Q=torch.pow((Q_meta.forward(task_index_sample,meta_parameter_tensor_sample,Q_meta.params)-q_value),2)/float(batch_size_point)
        loss_Q.backward()

        action_now=meta_parameter_tensor_sample.detach().clone().requires_grad_()
        Q_action=Q_meta.forward(task_index_sample,action_now,Q_meta.params)
        gradient_Q_action=torch.autograd.grad(outputs=Q_action,inputs=action_now)
        gradient_Q_action=[aa.detach().clone() for aa in gradient_Q_action]
        action_policypara=-Meta_map.forward(task_index_sample,Meta_map.params)/float(batch_size_point)
        action_policypara.backward(gradient_Q_action)

      optimizer_Q.step()
      optimizer_action.step()

      target_Q_meta_paras = [target_Q_meta.params[i]*0.99+Q_meta.params[i]*0.01 for i in range(len(target_Q_meta.params))]
      target_Q_meta_paras = [para.detach().clone().requires_grad_() for para in target_Q_meta_paras]
      target_Q_meta.params=target_Q_meta_paras
      target_meta_para = [target_meta.params[i]*0.99+Meta_map.params[i]*0.01 for i in range(len(target_meta.params))]
      target_meta_para = [para.detach().clone().requires_grad_() for para in target_meta_para] 
      target_meta.params=target_meta_para

    

