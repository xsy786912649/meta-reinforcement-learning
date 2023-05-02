# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:59:52 2023

@author: yzyja
"""

import collections
import copy
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import math_utils 

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
nrow=4
ncol=4
# Generate random map
def generate_random_map(size=4, p=0.7):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    res=[]
    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    
    return ["".join(x) for x in res]

size = 4
def is_valid(res):
  frontier, discovered = [], set()
  frontier.append((0, 0))
  while frontier:
      r, c = frontier.pop()
      if not (r, c) in discovered:
          discovered.add((r, c))
          directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
          for x, y in directions:
              r_new = r + x
              c_new = c + y
              if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                  continue
              if res[r_new][c_new] == "G":
                  return True
              if res[r_new][c_new] != "H":
                  frontier.append((r_new, c_new))
  return False


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def softmax_policy_model(observation, policy_model):
    """Compute softmax values for each sets of scores in x."""
    # print(observation)
    x = policy_model[observation,:]
    # print(x)
    # return np.exp(x) / np.sum(np.exp(x), axis=0)
    probs = softmax(x)
    probs = probs/sum(probs)
    # print(probs)
    return probs

def softmax_policy_model_return(policy_model):
  orig_policy_model = np.zeros((16,4))
  for i in range(16):
    orig_policy_model[i,:] = softmax_policy_model(i, policy_model)

  return orig_policy_model

def sample_actions(observation, policy_model):
  # print(observation)
  x = softmax_policy_model(observation, policy_model)
  probabilities = x.tolist()
  # probabilities = probabilities * masking(observation,probabilities,Unsafe_states,Unsafe_actions,eps)
  action = np.random.choice(np.arange(0, 4), p = probabilities)
  return action



def flatten(l):
    return [item for sublist in l for item in sublist]

def reward_function(observation, env):
  # The agent is rewarded +2 to reach the goal, -1 for the hole and 1 for the F.

  r = int(observation/4)   # Which row is the agent in?
  c = observation%4        # Which column is the agent in?
  desc = env.desc          # Getting the map of the current configuration

  if desc[r,c] == b'S':
    reward = 0.0
  elif desc[r,c] == b'F':
    reward = 0.0            # Reward is 1 when the agent lands on the frozen lake part
  elif desc[r,c] == b'H':
    reward = 0.0           
  elif desc[r,c] == b'G':
    reward = 2.0             

  return reward

def constraint_I(observation, env):
  # cost if the agent lands in the hole
    r = int(observation/4)
    c = observation%4
    desc = env.desc

    if desc[r,c] == b'H':
      cost = 1
    else:
      cost = 0.0

    return cost

def step(action, env):
  new_state, _,_, done, _ = env.step(action)
  hole = False
  r = int(new_state/4)
  c = new_state%4
  desc = env.desc
  
  # If the agent reaches the hole or the goal, the environment is reset
  if desc[r,c] == b'H':
    done = True
    hole = True
  elif desc[r,c] == b'G':
    done = True
  else:
    done = False

  return new_state, done, hole


def sample_trajectories(env, gamma, beta, episodes, length, policy_model, qtable_reward, 
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
        
        while not done and length_so_far < length:

          states.append(state)
          
          action = sample_actions(state, policy_model)
          new_state, done, hole = step(action, env)
          # print(new_state)
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
              # alpha = alpha/(k**2)
              state = env.reset()

              # Calculate the Q-value of all the state actions in the buffer
              q_cost = 0
              rho_j = 1/(len(state_action_buffer))
                
              for j in range(len(state_action_buffer)):
                state_value = state_action_buffer[j][0]
                action_value = state_action_buffer[j][1]
                q_cost += rho_j*qtable_cost[state_value, action_value]
              state_action_buffer = []
              if q_cost <= d_threshold:    # Update the policy to maximize the reward  
                policy_model = policy_model + alpha*(qtable_reward/(1-gamma))
              else:    # Update the policy to minimize the cost
                policy_model = policy_model - alpha*(qtable_cost/(1-gamma))

          length_so_far += 1
        
        
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

    discounted_total_reward= 0
    for path in [path["rewards"] for path in paths]:
      discounted_total_reward_tem=0
      for j,reward in enumerate(path):
          discounted_total_reward_tem+=reward*(gamma**j)
      discounted_total_reward+= discounted_total_reward_tem
    discounted_total_reward=discounted_total_reward/episodes

    actions = flatten([path["actions"] for path in paths])

    return discounted_total_reward, total_cost, policy_model, qtable_reward, qtable_cost
