# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:08:13 2023

@author: yzyja
"""
from CRPO_frozenlake import *


def sample_trajectories_wo_update(env, gamma, beta, episodes, length, policy_model, qtable_reward, 
                        qtable_cost, d_threshold, N_0, alpha):
    # Sample trajectories
    paths = []
    episodes_so_far = 0
    k = 0

    while episodes_so_far < episodes:
        k += 1
        
        episodes_so_far += 1
        states, actions, rewards, costs = [], [], [], []
        state_action_buffer = []
        state,_ = env.reset()
        length_so_far = 0
        done = False
        hole = False
        
        while not done:
        # while length_so_far < length:
          states.append(state)
          action = sample_actions(state, policy_model)
          new_state, done, hole = step(action, env)
          new_action = sample_actions(new_state, policy_model)
          actions.append(action)
  
          state_action_pair = [state, action]
          state_action_buffer.append(state_action_pair)
            
          reward = reward_function(new_state, env)
          cost = constraint_I(new_state, env)
  
          rewards.append(reward)
          costs.append(cost)
  
          # Update Q(s,a) for the reward
          qtable_reward[state, action] = qtable_reward[state, action] + \
                                  beta * (reward + gamma * qtable_reward[new_state, new_action] - qtable_reward[state, action])
  
          # Update Q(s,a) for the cost
          qtable_cost[state, action] = qtable_cost[state, action] + \
                                  beta * (cost + gamma * qtable_cost[new_state, new_action] - qtable_cost[state, action])
  
          # Update our current state
          state = new_state
          
          if done: 
            states.append(state)
            # alpha = alpha/(k**2)
            state,_ = env.reset()

        path = {"observations": states,
                "actions": actions,
                "rewards": rewards,
                "costs": costs}
        paths.append(path)

        

    observations = flatten([path["observations"] for path in paths])
    discounted_rewards = flatten([math_utils.discount(path["rewards"], gamma) for path in paths])
    total_reward = sum(flatten([path["rewards"] for path in paths])) / episodes
    ## add for cost
    discounted_costs = flatten([math_utils.discount(path["costs"], gamma) for path in paths])
    total_cost = sum(flatten([path["costs"] for path in paths])) / episodes
    # discounted_costs2 = flatten([math_utils.discount(path["costs2"], gamma) for path in paths])
    # total_cost2 = sum(flatten([path["costs2"] for path in paths])) / episodes


    actions = flatten([path["actions"] for path in paths])

    return total_reward, total_cost, policy_model, observations
    

def get_weights(observations):

  counts = np.zeros(16)
  for i in range(len(observations)):
    val = observations[i]
    counts[val] = counts[val] + 1

  weights = counts/(len(observations))
  return weights

def get_weighted_policy(policies_arr, w, num_tasks):

  weighted_policy = np.zeros((16,4))
  for i in range(num_tasks-1):
    w[i] = w[i].reshape((16,1))
    weighted_policy = weighted_policy + (w[i]*policies_arr[i])
  
  w = np.array(w)
  # print(np.shape(w))
  for i in range(16):
    den = np.sum(w[:,i], axis=0)
    if den[0] == 0:
      weighted_policy[i,:] = weighted_policy[i,:]
    else:
      weighted_policy[i,:] = weighted_policy[i,:]/den[0]

  return weighted_policy





def main(num_tasks, policies, w):
    # map_name = generate_random_map()
    # Giving a custom map
    map_name = np.load('maps/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    
    eps=0.05
    nA=4
    nS=4*4
    # Backward=backward_state(eps,nA,nS)
    
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=True)
    # Unsafe_states, Unsafe_actions=unsafe_states_actions(env,Backward)
    # np.save('map'+str(num_tasks)+'.npy', map_name)

    ## Hyperparameter
    beta = 0.3    # value function learning rate for the critic
    gamma = 0.98             # Discount factor
    episodes = 10 #10        
    length = 100  #50       # Length of the sample trajectories, maybe we can change this

    STEP = 2

    # Set the threshold
    d_threshold = 0.3

    # Initialize the Q-tables and the policy for the task at hand
    qtable_reward = np.zeros((16, 4))
    qtable_cost = np.zeros((16, 4))

    if len(policies) == 0:
      policy_model = np.ones((16, 4))
      alpha = 0.03
    else:
      policies_arr = np.array(policies)
      # policy_model = np.mean(policies_arr, axis = 0)
      # policy_model = np.average(policies_arr, axis = 0, weights = w)
      policy_model = get_weighted_policy(policies_arr, w, num_tasks)
      # Get the weighted average policy

      # Get the learning rate for this specific task
      M = STEP*episodes
      a= np.sqrt(1/num_tasks)
      b = np.sqrt(np.sqrt(1/M))
      # kl_divergence = np.mean(kl_data)
      # print('kl_divrgence = ', kl_divergence)
      alpha = (b)*np.sqrt(a)/1


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
                                                                                         d_threshold, N_0, alpha)

      # print(result)
      
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
    # plt.savefig('books_read.png')
    # plt.show()
    np.save('maps/Training_task_data/rewards'+str(num_tasks)+'.npy',results)
    np.save('maps/Training_task_data/costs'+str(num_tasks)+'.npy',violations)

    _, _, _, observations = sample_trajectories_wo_update(env, gamma, beta, 300, length,
                                                          policy_model_out, qtable_reward, qtable_cost, d_threshold, N_0, alpha)

    # Compute the weights for the weighted average
    weights = get_weights(observations)

    return policy_model_out, weights, results, violations

num_tasks = 11

policies = []
w = []
for i in range(num_tasks):
  if __name__ == '__main__':
    policy_model_out, weights, results, violations = main(i+1, policies, w)
    # print(i)
    w.append(weights)
    policies.append(policy_model_out)

    # Run the policy on the test task for 10 runs to get the variance plots
    if (i+1) == num_tasks:
      policies = policies[:-1]
      w = w[:-1]
      for j in range(10):
        policy_model_test, weights, results_test, violations_test = main(i+1, policies, w)
        # print(results_test.type())
        np.save('maps/Test_task_data/WFAL+adaptive/rewards_test'+str(j+1)+'.npy', results_test)
        np.save('maps/Test_task_data/WFAL+adaptive/costs_test'+str(j+1)+'.npy', violations_test)