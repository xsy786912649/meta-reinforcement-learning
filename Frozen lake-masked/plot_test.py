#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:40:41 2023

@author: robotics
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.figure(1)
# num_task=
R=()
eps=0.15
for i in range(10):
    # f=open('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy','r')
    # with open('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy', 'rb') as f:
    results_test=np.load('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy')
    # R=R+results_test
    # with open('maps/Test_task_data/WFAL+adaptive/costs_test'+str(i+1)+'.npy', 'rb') as g:
    #     costs_test=np.load('maps/Test_task_data/WFAL+adaptive/costs_test'+str(i+1)+'.npy')
        

    plt.plot(results_test)
    # plt.plot(costs_test)
# r=np.vstack(R)

plt.savefig('rewards_plot.png')


plt.figure(2)    
C=()
for i in range(10):
    # with open('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy', 'rb') as f:
    #     results_test=np.load('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy')
    # with open('maps/Test_task_data/WFAL+adaptive/costs_test'+str(i+1)+'.npy', 'rb') as g:
    costs_test=np.load('maps/Test_task_data/WFAL+adaptive/costs_test'+str(i+1)+'_'+str(eps)+'.npy')
    # C=C+costs_test    
    

    plt.plot(costs_test)
plt.savefig('cost_plot'+str(eps)+'.png')