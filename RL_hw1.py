#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:23:20 2018

@author: Yiming Peng
"""

import time
import numpy as np
import matplotlib.pyplot as plt


start_program = time.time()
#==============================================================================
K = 15
S = 200
cf = 100
ch = 2
gamma = 0.95

# =============================================================================
# enumeration
T = 500
V = np.zeros(S+1) # vector of value functions
for t in range(T+1):
   V_new = np.zeros(S+1) # value functions at time t
   for s in range(S+1):
       # a_t = 1
       if s > 15:
           r = -(s - K) * ch - cf
           s1 = s - K + 1
           s2 = s - K + 2
           s3 = s - K + 3
           s4 = s - K + 4
           s5 = s - K + 5
           v1 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       else:
           r = -cf
           v1 = r + gamma * (V[1]+V[2]+V[3]+V[4]+V[5]) / 5
           
       # a_t = 0
       r = -s * ch
       s1 = min(S, s + 1)
       s2 = min(S, s + 2)
       s3 = min(S, s + 3)
       s4 = min(S, s + 4)
       s5 = min(S, s + 5)
       v0 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       
       # Bellman equation
       V_new[s] = max(v1, v0)
       
   # update
   V = V_new

plt.figure(figsize=(9,6), dpi=100)        
plt.plot(np.arange(S+1), V)
plt.xlabel("number of people waiting")
plt.ylabel("value function")
plt.title("Enumeration")
# =============================================================================
# value iteration
eps = 10**(-10)
V = np.zeros(S+1) # initial value functions
while True:
   V_new = np.zeros(S+1) # new value functions 
   for s in range(S+1):
       # a = 1
       if s > 15:
           r = -(s - K) * ch - cf
           s1 = s - K + 1
           s2 = s - K + 2
           s3 = s - K + 3
           s4 = s - K + 4
           s5 = s - K + 5
           v1 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       else:
           r = -cf
           v1 = r + gamma * (V[1]+V[2]+V[3]+V[4]+V[5]) / 5
           
       # a = 0
       r = -s * ch
       s1 = min(S, s + 1)
       s2 = min(S, s + 2)
       s3 = min(S, s + 3)
       s4 = min(S, s + 4)
       s5 = min(S, s + 5)
       v0 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       
       # Bellman equation
       V_new[s] = max(v1, v0)
   
   # check convergence
   if np.linalg.norm(V_new - V, ord=np.inf) < eps:
       V = V_new
       break

   # update
   V = V_new      

plt.figure(figsize=(9,6), dpi=100)        
plt.plot(np.arange(S+1), V)
plt.xlabel("number of people waiting")
plt.ylabel("value function")
plt.title("Value Iteration")
# =============================================================================
# policy iteration
eps = 10**(-4) 
Pi = np.zeros(S+1) # initial policy
V = np.zeros(S+1) # value of initial policy
while True:
   # policy evaluation
   V_new = np.zeros(S+1)
   while True:
       for s in range(S+1):
           if Pi[s] == 1:
               if s > 15:
                   r = -(s - K) * ch - cf
                   s1 = s - K + 1
                   s2 = s - K + 2
                   s3 = s - K + 3
                   s4 = s - K + 4
                   s5 = s - K + 5
                   V_new[s] = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
               else:
                   r = -cf
                   V_new[s] = r + gamma * (V[1]+V[2]+V[3]+V[4]+V[5]) / 5
                   
           else:
               r = -s * ch
               s1 = min(S, s + 1)
               s2 = min(S, s + 2)
               s3 = min(S, s + 3)
               s4 = min(S, s + 4)
               s5 = min(S, s + 5)
               V_new[s] = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
               
       # check convergence
       if np.linalg.norm(V_new - V, ord=np.inf) < eps:
           V = V_new
           break
   
       # update
       V = V_new    
       
   # policy improvement
   Pi_new = np.zeros(S+1) # new policy
   for s in range(S+1):
       # a = 1
       if s > 15:
           r = -(s - K) * ch - cf
           s1 = s - K + 1
           s2 = s - K + 2
           s3 = s - K + 3
           s4 = s - K + 4
           s5 = s - K + 5
           v1 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       else:
           r = -cf
           v1 = r + gamma * (V[1]+V[2]+V[3]+V[4]+V[5]) / 5
           
       # a = 0
       r = -s * ch
       s1 = min(S, s + 1)
       s2 = min(S, s + 2)
       s3 = min(S, s + 3)
       s4 = min(S, s + 4)
       s5 = min(S, s + 5)
       v0 = r + gamma * (V[s1]+V[s2]+V[s3]+V[s4]+V[s5]) / 5
       
       if v1 > v0:
           Pi_new[s] = 1
       else:
           Pi_new[s] = 0
           
   # check convergence
   if sum(abs(Pi - Pi_new)):
       Pi = Pi_new
       break
   
   # update
   Pi = Pi_new

plt.figure(figsize=(9,6), dpi=100)        
plt.plot(np.arange(S+1), Pi)
plt.xticks(np.arange(0,201,10))
plt.xlabel("number of people waiting")
plt.ylabel("optimal policy")
plt.title("Policy Iteration")
print(np.where(Pi > 0))
#==============================================================================
end_program = time.time()
print('This program takes', (end_program - start_program)/60, 'minutes to run.')
