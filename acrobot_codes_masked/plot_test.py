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
for i in range(50):

    results_test=np.load('results/rewards_test'+str(i)+'.npy')

    plt.plot(results_test)

plt.savefig('rewards_plot.png')
#plt.show()


plt.figure(2)    
C=()
for i in range(50):
    # with open('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy', 'rb') as f:
    #     results_test=np.load('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(i+1)+'.npy')
    # with open('maps/Test_task_data/WFAL+adaptive/costs_test'+str(i+1)+'.npy', 'rb') as g:
    costs_test=np.load('results/costs_test'+str(i)+'.npy')
    # C=C+costs_test    
    

    plt.plot(costs_test)
plt.savefig('cost_plot'+'.png')


plt.figure(3)

results_test_list=[]
results_test_list_ini=[]
costs_test_list=[]
for i in range(50):
    results_test=np.load('results/rewards_test'+str(i)+'.npy')
    results__test_initial=np.load('results/rewards_test_initial'+str(i)+'.npy')
    results_test_list.append(results_test)
    results_test_list_ini.append(results__test_initial)

results_test_mean=np.mean(np.array(results_test_list),axis=0)
results_test_ini_mean=np.mean(np.array(results_test_list_ini),axis=0)
plt.plot(results_test_mean,"b") 
plt.plot(results_test_ini_mean,"r")
plt.savefig('rewards_plot_ave.png')

plt.figure(4)    
for i in range(50):
    costs_test=np.load('results/costs_test'+str(i)+'.npy')
    costs_test_list.append(costs_test)
    
costs_test_mean=np.mean(np.array(costs_test_list),axis=0)
plt.plot(costs_test_mean) 
plt.savefig('cost_plot_ave.png')

plt.show()
