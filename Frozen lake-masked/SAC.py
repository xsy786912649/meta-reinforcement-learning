# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:08:13 2023

@author: yzyja
"""
from CRPO_frozenlake import *

nA=4
nS=4*4
eps=0.05
Backward=backward_state(eps,nA,nS,False)

def run(num_tasks, meta_parameter, episodes):

    global eps
    
    map_name = np.load('maps/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=False)
    Unsafe_states, Unsafe_actions=unsafe_states_actions(env,Backward)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    gamma = 0.9             # Discount factor
    episodes = 10 #10        
    length = 100  #50       # Length of the sample trajectories, maybe we can change this

    STEP = 3
    H = 100
    print('epsilon = ', eps, 'H = ', H)
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

    return policy_model_out, results, violations

if __name__ == '__main__':

  num_tasks =11
  meta_parameter = np.ones((16, 4))

  for i in range(num_tasks): 
    print(i)
    policy_model_out, results, violations = run(i+1,meta_parameter, episodes=10)

    with open('meta_parameter.npy', 'wb') as f:
      np.save(f,meta_parameter)



