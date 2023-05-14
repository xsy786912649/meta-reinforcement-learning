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

TRAIN_TASK_COUNT = 100# args.train_task_count

CRPO_STEP_COUNT = 5#args.crpo_step_count
CRPO_EPISODE_COUNT = 10#args.crpo_episode_count
CG_ITER_COUNT = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 3
VARIANCE = 0.2

NETWORK_NEURAL_NUMBER=64

   
if __name__ == '__main__':

    eps=0.5
    height=-0.5
    LIMIT_RANGE = [40, 42]
    
    noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
    limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))

    replay_buffer=[]
    replay_buffer_size=10000

    batch_size_task=10
    epoch_when_each_new=6
    sample_number=20
    nosiy_scale=0.001
    nosiy_scale1=0.0003

    aaa=MetaSRL(INPUT_SIZE, OUTPUT_SIZE).policy.parameters()
    meta_parameter_model = [para.detach().clone().requires_grad_() for para in aaa]
    optimizer1=torch.optim.Adam(meta_parameter_model,lr=0.001,weight_decay=0.0)

    for revealed_task_num in range(TRAIN_TASK_COUNT-1):
        limit = limits[revealed_task_num]
        print("------------------------------------")
        print(revealed_task_num)
        print("------------------------------------")
        torch.save(meta_parameter_model, "pth/meta_parameter_map_epho"+str(revealed_task_num)+".pth")
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
                meta_parameter_data=[tensor_meta.detach().data.numpy() for tensor_meta in meta_parameter_model]
                for i in range(sample_number):

                    noisy=[0,0,0,0,0,0]
                    noisy[0]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER,INPUT_SIZE)) 
                    noisy[1]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER)) 
                    noisy[2]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER,NETWORK_NEURAL_NUMBER)) 
                    noisy[3]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER)) 
                    noisy[4]=np.random.normal(loc=0.0, scale=nosiy_scale1, size=(OUTPUT_SIZE,NETWORK_NEURAL_NUMBER)) 
                    noisy[5]=np.random.normal(loc=0.0, scale=nosiy_scale1, size=(OUTPUT_SIZE)) 
                    
                    meta_parameter_add_noisy=[torch.FloatTensor(meta_parameter_data[i]+noisy[i]) for i in range(6)]
                    metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
                    metasrl.policy.fc1.weight.data=meta_parameter_add_noisy[0]
                    metasrl.policy.fc1.bias.data=meta_parameter_add_noisy[1]
                    metasrl.policy.fc2.weight.data=meta_parameter_add_noisy[2]
                    metasrl.policy.fc2.bias.data=meta_parameter_add_noisy[3]
                    metasrl.policy.head.weight.data=meta_parameter_add_noisy[4]
                    metasrl.policy.head.bias.data=meta_parameter_add_noisy[5]
                    print(noises[task_index])
                    metasrl.step(eps,height, noise=noises[task_index], crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=epoch%2)
                    results=metasrl.rewards_by_task[-1]
                    print(results)
                    noisy_normlization=[noisy[0]/nosiy_scale,noisy[1]/nosiy_scale,noisy[2]/nosiy_scale,noisy[3]/nosiy_scale,noisy[4]/nosiy_scale1,noisy[5]/nosiy_scale1]
                    data_pair1=(noisy_normlization,results[-1])

                    meta_parameter_add_noisy=[torch.FloatTensor(meta_parameter_data[i]-noisy[i]) for i in range(6)]
                    metasrl = MetaSRL(INPUT_SIZE, OUTPUT_SIZE)
                    metasrl.policy.fc1.weight.data=meta_parameter_add_noisy[0]
                    metasrl.policy.fc1.bias.data=meta_parameter_add_noisy[1]
                    metasrl.policy.fc2.weight.data=meta_parameter_add_noisy[2]
                    metasrl.policy.fc2.bias.data=meta_parameter_add_noisy[3]
                    metasrl.policy.head.weight.data=meta_parameter_add_noisy[4]
                    metasrl.policy.head.bias.data=meta_parameter_add_noisy[5]
                    print(noises[task_index])
                    metasrl.step(eps,height, noise=noises[task_index], crpo_step=CRPO_STEP_COUNT, crpo_episodes=CRPO_EPISODE_COUNT, cg_iters=CG_ITER_COUNT, limit_1=limit, limit_2=limit, direction=epoch%2)
                    results=metasrl.rewards_by_task[-1]
                    print(results)
                    noisy_normlization=[-noisy[0]/nosiy_scale,-noisy[1]/nosiy_scale,-noisy[2]/nosiy_scale,-noisy[3]/nosiy_scale,-noisy[4]/nosiy_scale1,-noisy[5]/nosiy_scale1]
                    data_pair2=(noisy_normlization,results[-1])

                    grade=[(torch.FloatTensor(data_pair1[0][k])*data_pair1[1] +torch.FloatTensor(data_pair2[0][k])*data_pair2[1])/2/sample_number/len(num_tasks_list) for k in range(6)]
        
                    if meta_parameter_model[0].grad== None:
                        meta_parameter_model[0].grad=-grade[0]
                        meta_parameter_model[1].grad=-grade[1]
                        meta_parameter_model[2].grad=-grade[2]
                        meta_parameter_model[3].grad=-grade[3]
                        meta_parameter_model[4].grad=-grade[4]
                        meta_parameter_model[5].grad=-grade[5]
                    else:
                        meta_parameter_model[0].grad+=-grade[0]
                        meta_parameter_model[1].grad+=-grade[1]
                        meta_parameter_model[2].grad+=-grade[2]
                        meta_parameter_model[3].grad+=-grade[3]
                        meta_parameter_model[4].grad+=-grade[4]
                        meta_parameter_model[5].grad+=-grade[5]

            optimizer1.step()