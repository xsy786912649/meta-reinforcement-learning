#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:46:23 2023

@author: robotics
"""
import numpy as np
import gym
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
nrow=4
ncol=4
def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)
        
def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            # newletter = desc[newrow, newcol]
            # terminated = bytes(newletter) in b"GH"
            # reward = float(newletter == b"G")
            return newstate
def to_s(row, col):
            return row * 4 + col
def backward_state(eps,nA,nS):
    P={s: {a: [] for a in range(nA)} for s in range(nS)}
    for row in range(4):
            for col in range(4):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    # if is_slippery:
                    # letter = desc[row, col]
                    if  row==3 and col==3:
                        li.append((1.0, s))
                    for b in [(a - 1) % 4, a, (a + 1) % 4]:
                        li.append(
                            (1.0 / 3.0, update_probability_matrix(row, col, b))
                        )
                    # else:
                    #     li.append((1.0, *update_probability_matrix(row, col, a)))
    print(P[4])
    Backward={s: {a: set() for a in range(nA)} for s in range(nS)}
    for s in P.keys():
        for a in P[s].keys():
            transitions = P[s][a]
            # i = categorical_sample([t[0] for t in transitions], env.np_random)
            for (p, s_) in  transitions:
                if p>eps:
                    Backward[s_][a].add(s)
    return Backward

def unsafe_states_actions(env,Backward):
    Unsafe_states=set()
    DESC=env.desc
   
    for row in range(4):
            for col in range(4):
                s = to_s(row, col)
                for a in range(4):
                    # li = self.P[s][a]
                    letter = env.desc[row, col]
                    if letter in b"H":
                        Unsafe_states.add(s)
                        DESC[row,col]=b'U'
                    
        
      
    Unsafe_actions={s:set() for s in range(ncol*nrow)}
    # while Flag:
    for s_ in Unsafe_states:
        for a in range(4):
            for s in Backward[s_][a]:        
                Unsafe_actions[s].add(a)
                print(s_,s,a)
                if len(Unsafe_actions[s])==4 and s not in Unsafe_states:
                    Unsafe_states.add(s)
    # print(len(Unsafe_states)) 
    return Unsafe_states, Unsafe_actions,DESC

map_name = np.load('maps/map'+str(1)+'.npy')
# print(map_name)
map_name = map_name.tolist()

eps=0.05
nA=4
nS=4*4
Backward=backward_state(eps,nA,nS)

env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=True)
print(env.desc)
Unsafe_states, Unsafe_actions,DESC=unsafe_states_actions(env,Backward)
print(DESC)
print(Unsafe_states)
print(Unsafe_actions)