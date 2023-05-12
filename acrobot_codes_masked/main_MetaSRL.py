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

CRPO_STEP_COUNT = 10#args.crpo_step_count
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


class Qfunction(torch.nn.Module):
    def __init__(self):
        super(Qfunction, self).__init__()
        layer_number=4+NETWORK_NEURAL_NUMBER*INPUT_SIZE+NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER*NETWORK_NEURAL_NUMBER+NETWORK_NEURAL_NUMBER+OUTPUT_SIZE*NETWORK_NEURAL_NUMBER+OUTPUT_SIZE
        self.params = [
                    torch.FloatTensor(256, layer_number).uniform_(-1./math.sqrt(layer_number), 1./math.sqrt(layer_number)).requires_grad_(),
                    torch.FloatTensor(256).zero_().requires_grad_(),

                    torch.FloatTensor(128, 256).uniform_(-1./math.sqrt(256), 1./math.sqrt(256)).requires_grad_(),
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
    
    def forward(self, noise, action, params):
        output=self.dense(self.input_process(noise,action), params)
        return output
    
    def input_process(self, noise, action):
        map_vector=np.array(noise)
        map_tensor=torch.FloatTensor(map_vector)
        return torch.cat((map_tensor, torch.flatten(action[0]),torch.flatten(action[1]),torch.flatten(action[2]),torch.flatten(action[3]),torch.flatten(action[4]),torch.flatten(action[5])), 0)


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

    eps=0.5
    height=-0.5
    LIMIT_RANGE = [40, 42]
    
    noises = np.random.normal(0.0, VARIANCE, size=(TRAIN_TASK_COUNT, 4))
    limits = np.random.randint(low=LIMIT_RANGE[0], high=LIMIT_RANGE[1], size=(TRAIN_TASK_COUNT))

    Meta_map=Model()
    Q_meta=Qfunction()
    Q2_meta=Qfunction()

    target_meta=Model()
    target_meta.params=[pa.clone().detach().requires_grad_() for pa in Meta_map.params]
    target_Q_meta=Qfunction()
    target_Q_meta.params=[pa.clone().detach().requires_grad_() for pa in Q_meta.params]
    target_Q2_meta=Qfunction()
    target_Q2_meta.params=[pa.clone().detach().requires_grad_() for pa in Q2_meta.params]

    replay_buffer=[]
    replay_buffer_size=10000

    batch_size_task=10
    batch_size_point=10
    epoch_when_each_new=6
    nosiy_scale=0.001#*0.2
    nosiy_scale1=0.0003#*0.2

    optimizer_Q=torch.optim.Adam(Q_meta.params,lr=0.001,weight_decay=0.0)
    optimizer_Q2=torch.optim.Adam(Q2_meta.params,lr=0.001,weight_decay=0.0)
    optimizer_action=torch.optim.Adam(Meta_map.params,lr=0.00008,weight_decay=0.0)

    for revealed_task_num in range(TRAIN_TASK_COUNT-1):
        limit = limits[revealed_task_num]
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
                meta_parameter_tensor=Meta_map.forward(noises[task_index],Meta_map.params) 
                meta_parameter=[tensor_meta.data.numpy() for tensor_meta in meta_parameter_tensor]

                noisy=[0,0,0,0,0,0]
                noisy[0]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER,INPUT_SIZE)) 
                noisy[1]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER)) 
                noisy[2]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER,NETWORK_NEURAL_NUMBER)) 
                noisy[3]=np.random.normal(loc=0.0, scale=nosiy_scale, size=(NETWORK_NEURAL_NUMBER)) 
                noisy[4]=np.random.normal(loc=0.0, scale=nosiy_scale1, size=(OUTPUT_SIZE,NETWORK_NEURAL_NUMBER)) 
                noisy[5]=np.random.normal(loc=0.0, scale=nosiy_scale1, size=(OUTPUT_SIZE)) 
                meta_parameter_add_noisy=[torch.FloatTensor(meta_parameter[i]+noisy[i]) for i in range(6)]
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

                data_pair=(noises[task_index],meta_parameter_add_noisy,(results[-1]+results[-2]+results[-3])/50.0-1.0,noises[task_index+1])
                replay_buffer=add_pair(replay_buffer,data_pair,replay_buffer_size)

            list_sample=sample_pair(replay_buffer,batch_size_point)

            optimizer_Q.zero_grad()
            optimizer_Q2.zero_grad()
            optimizer_action.zero_grad()

            for sample in list_sample:
                task_index_sample=sample[0]
                meta_parameter_tensor_sample=sample[1]
                results_sample=sample[2]
                task_index_sample_next=sample[3]
                next_action=target_meta.forward(task_index_sample_next,target_meta.params)

                valueq1=target_Q_meta.forward(task_index_sample_next,next_action,target_Q_meta.params)
                valueq2=target_Q2_meta.forward(task_index_sample_next,next_action,target_Q2_meta.params)
                q_value=results_sample+0.8*torch.minimum(valueq1.clone().detach(),valueq2.clone().detach())
                loss_Q=torch.pow((Q_meta.forward(task_index_sample,meta_parameter_tensor_sample,Q_meta.params)-q_value),2)/float(batch_size_point)
                loss_Q.backward()
                loss_Q2=torch.pow((Q2_meta.forward(task_index_sample,meta_parameter_tensor_sample,Q2_meta.params)-q_value),2)/float(batch_size_point)
                loss_Q2.backward()

                action_now=[ttt.detach().clone().requires_grad_() for ttt in meta_parameter_tensor_sample]
                Q_action=Q_meta.forward(task_index_sample,action_now,Q_meta.params)
                gradient_Q_action=torch.autograd.grad(outputs=Q_action,inputs=action_now)
                gradient_Q_action_flatten=torch.cat((gradient_Q_action[0].flatten(),gradient_Q_action[1].flatten(),gradient_Q_action[2].flatten(),gradient_Q_action[3].flatten(),gradient_Q_action[4].flatten(),gradient_Q_action[5].flatten()))
                gradient_Q_action_flatten=gradient_Q_action_flatten.detach().clone()
                action_policypara=-Meta_map.forward1(task_index_sample,Meta_map.params)/float(batch_size_point)
                action_policypara.backward(gradient_Q_action_flatten)

            optimizer_Q.step()
            optimizer_Q2.step()
            if epoch%2==0:
                optimizer_action.step()
                target_Q_meta_paras = [target_Q_meta.params[i]*0.99+Q_meta.params[i]*0.01 for i in range(len(target_Q_meta.params))]
                target_Q_meta_paras = [para.detach().clone().requires_grad_() for para in target_Q_meta_paras]
                target_Q_meta.params=target_Q_meta_paras
                target_Q2_meta_paras = [target_Q2_meta.params[i]*0.99+Q2_meta.params[i]*0.01 for i in range(len(target_Q2_meta.params))]
                target_Q2_meta_paras = [para.detach().clone().requires_grad_() for para in target_Q2_meta_paras]
                target_Q2_meta.params=target_Q2_meta_paras
                target_meta_para = [target_meta.params[i]*0.99+Meta_map.params[i]*0.01 for i in range(len(target_meta.params))]
                target_meta_para = [para.detach().clone().requires_grad_() for para in target_meta_para] 
                target_meta.params=target_meta_para




