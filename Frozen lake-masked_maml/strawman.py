# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:04:59 2023

@author: yzyja
"""
import numpy as np
import matplotlib.pyplot as plt



def main(num_tasks, policies):
    # map_name = generate_random_map()
    # Giving a custom map
    map_name = np.load('maps_high_sim/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=True)

    # np.save('map'+str(num_tasks)+'.npy', map_name)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    alpha = 0.03  # Keeping a constant learning rate
    gamma = 0.98             # Discount factor
    episodes = 3 #10        
    length = 100  #50       # Length of the sample trajectories, maybe we can change this

    STEP = 20

    # Set the threshold
    d_threshold = 0.3

    # Initialize the Q-tables and the policy for the task at hand
    qtable_reward = np.zeros((16, 4))
    qtable_cost = np.zeros((16, 4))

    if len(policies) == 0:
      policy_model = np.ones((16, 4))
    else:
      policy_model = policies[-1]


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
                                                                                         policy_model, qtable_reward, qtable_cost, d_threshold, N_0, alpha)

      
      
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


    constraint1 = [d_threshold for i in range(STEP)]
    # constraint2 = [d_threshold for i in range(STEP)]

    plt.figure()
    plt.plot(iterations, results, color='r', linestyle='-', label='C-TRPO reward')
    plt.plot(iterations, violations, color='b', linestyle='-', label='C-TRPO cost')
    
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
    # np.save('data_high_sim/rewards'+str(num_tasks)+'.npy',results)
    # np.save('data_high_sim/costs'+str(num_tasks)+'.npy',violations)

    orig_policy_model = softmax_policy_model_return(policy_model_out)

    return policy_model_out, results, violations


num_tasks = 11

policies = []
for i in range(num_tasks):
  if __name__ == '__main__':
    policy_model_out, results, violations = main(i+1, policies)
    policies.append(policy_model_out)

    # Run the policy on the test task for 10 runs to get the variance plots
    if (i+1) == 11:
      policies = policies[:-1]
      for j in range(10):
        policy_model_test, results_test, violations_test = main(i+1, policies)
        np.save('data_high_sim/Test_task_data/Strawman/rewards_test'+str(j+1)+'.npy', results_test)
        np.save('data_high_sim/Test_task_data/Strawman/costs_test'+str(j+1)+'.npy', violations_test)