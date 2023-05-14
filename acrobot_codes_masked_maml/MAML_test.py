# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:08:13 2023

@author: yzyja
"""

from agents.MetaSRL import *
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
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True

setup_seed(20)

TRAIN_TASK_COUNT = 50# args.train_task_count

CRPO_STEP_COUNT = 4#args.crpo_step_count
CRPO_EPISODE_COUNT = 10#args.crpo_episode_count
CG_ITER_COUNT = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.2

NETWORK_NEURAL_NUMBER=64

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.params = [
                    torch.FloatTensor(128, 4).uniform_(-1./math.sqrt(4), 1./math.sqrt(4)).requires_grad_(),
                    torch.FloatTensor(128).zero_().requires_grad_(),

                    torch.FloatTensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.FloatTensor(128).zero_().requires_grad_(),

                    torch.FloatTensor(NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER+OUTPUT_SIZE, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.FloatTensor(NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER+OUTPUT_SIZE).zero_().requires_grad_(),

            ]

    def dense(self, x, params):
        y = F.linear(x, params[0], params[1])
        y = F.relu(y)

        y = F.linear(y, params[2], params[3])
        y = F.relu(y)

        y = F.linear(y, params[4], params[5])

        return y
    
    def forward1(self, noise, params):
        output=self.dense(self.input_process(noise), params)
        return output

    def forward(self, noise, params):
        output=self.dense(self.input_process(noise), params)
        layer1_weight=output[0:NETWORK_NEURAL_NUMBER*INPUT_SIZE]
        layer1_weight=layer1_weight.reshape(NETWORK_NEURAL_NUMBER,INPUT_SIZE)
        layer1_bias=output[NETWORK_NEURAL_NUMBER*INPUT_SIZE:NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER]
        layer2_weight=output[NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER:NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER]
        layer2_weight=layer2_weight.reshape(NETWORK_NEURAL_NUMBER,NETWORK_NEURAL_NUMBER)
        layer2_bias=output[NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER:NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER]
        layer3_weight=output[NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER:NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER]
        layer3_weight=layer3_weight.reshape(OUTPUT_SIZE,NETWORK_NEURAL_NUMBER)
        layer3_bias=output[NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER:NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER+OUTPUT_SIZE]
        return [layer1_weight,layer1_bias,layer2_weight,layer2_bias,layer3_weight,layer3_bias]
    
    def input_process(self, noise):
        map_vector=np.array(noise)
        map_tensor=torch.FloatTensor(map_vector)
        return map_tensor


if __name__ == '__main__':

    TRAIN_TASK_COUNT =50
    eps=0.5
    height=-0.5
    LIMIT_RANGE = [40, 42]
    
    noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
    limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))

    '''
    for task_index in range(TRAIN_TASK_COUNT): 
        limit = limits[task_index]
        print(task_index)
        metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
        metasrl.step(eps,height, noise=noises[task_index], crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=0)
        results=metasrl.rewards_by_task[-1]
        violations_test=metasrl.cost_1s_by_task[-1]
        violations_test2=metasrl.cost_2s_by_task[-1]
        print(results)
        np.save('results/rewards_test_nonn'+str(task_index)+'.npy', results)
        np.save('results/costs_test_nonn'+str(task_index)+'.npy', violations_test)
        np.save('results/costs2_test_initial_nonn'+str(task_index)+'.npy', violations_test2)
    '''
    meta_parameter_tensor=torch.load("pth/meta_parameter_map_epho98.pth")
    for task_index in range(TRAIN_TASK_COUNT): 
        limit = limits[task_index]
        print(task_index)
        metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
        meta_parameter=[tensor_meta.detach().data.numpy() for tensor_meta in meta_parameter_tensor]
        meta_parameter_add=[torch.FloatTensor(meta_parameter[i]) for i in range(6)]
        metasrl.policy.fc1.weight.data=meta_parameter_add[0]
        metasrl.policy.fc1.bias.data=meta_parameter_add[1]
        metasrl.policy.fc2.weight.data=meta_parameter_add[2]
        metasrl.policy.fc2.bias.data=meta_parameter_add[3]
        metasrl.policy.head.weight.data=meta_parameter_add[4]
        metasrl.policy.head.bias.data=meta_parameter_add[5]
        
        metasrl.step(eps,height, noise=noises[task_index], crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=0)
        results=metasrl.rewards_by_task[-1]
        violations_test=metasrl.cost_1s_by_task[-1]
        violations_test2=metasrl.cost_2s_by_task[-1]
        print(results)
        np.save('results/rewards_test'+str(task_index)+'.npy', results)
        np.save('results/costs_test'+str(task_index)+'.npy', violations_test)
        np.save('results/costs2_test_initial'+str(task_index)+'.npy', violations_test2)


    meta_parameter_tensor=torch.load("pth/meta_parameter_map_epho0.pth")
    for task_index in range(TRAIN_TASK_COUNT): 
        limit = limits[task_index]
        print(task_index)
        metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
        meta_parameter=[tensor_meta.detach().data.numpy() for tensor_meta in meta_parameter_tensor]
        meta_parameter_add=[torch.FloatTensor(meta_parameter[i]) for i in range(6)]
        metasrl.policy.fc1.weight.data=meta_parameter_add[0]
        metasrl.policy.fc1.bias.data=meta_parameter_add[1]
        metasrl.policy.fc2.weight.data=meta_parameter_add[2]
        metasrl.policy.fc2.bias.data=meta_parameter_add[3]
        metasrl.policy.head.weight.data=meta_parameter_add[4]
        metasrl.policy.head.bias.data=meta_parameter_add[5]
        
        metasrl.step(eps,height, noise=noises[task_index], crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=0)
        results=metasrl.rewards_by_task[-1]
        violations_test=metasrl.cost_1s_by_task[-1]
        violations_test2=metasrl.cost_2s_by_task[-1]
        print(results)
        np.save('results/rewards_test_initial'+str(task_index)+'.npy', results)
        np.save('results/costs_test_initial'+str(task_index)+'.npy', violations_test)
        np.save('results/costs2_test_initial'+str(task_index)+'.npy', violations_test2)

