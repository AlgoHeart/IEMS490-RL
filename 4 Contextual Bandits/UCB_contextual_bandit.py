#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:28:55 2018

LinUCB for contextual bandits

@author: Yiming Peng
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import statsmodels.api as sm


start_program = time.time()
#==============================================================================
def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
    file_name: Route of file containing the modified Jester dataset.
    context_dim: Context dimension (i.e. vector with some ratings from a user).
    num_actions: Number of actions (number of joke ratings to predict).
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
        
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset, opt_rewards, opt_actions

# sample data
dataset, opt_rewards, opt_actions = sample_jester_data("jester_data_40jokes_19181users.npy")

# tuning parameter
Cs = [2**i for i in range(-5, 5)]
C = 1

# training
d = dataset.shape[1]
l = 32 # dim of context
k = d - l # dim of action
T = 18000 
contexts = [np.array([]) for _ in range(k)]
rewards = [np.array([]) for _ in range(k)]
thetas = [np.zeros(l) for _ in range(k)]
As = [np.eye(l) for _ in range(k)]
predicts = [None for _ in range(k)]

regrets = []
opt_rewards_train = opt_rewards[:T]
for C in Cs:  
    real_rewards = []
    for t in range(T-1):
        context = dataset[t, :l].reshape(-1,1)
        for i in range(k):
            predicts[i] = thetas[i].T @ context + C * (context.T @
                    (As[i] @ context))**0.5    
        # choose action
        a = np.argmax(predicts)
        # observe reward
        reward = dataset[t, l + a]
        real_rewards.append(reward)
        rewards[a] = np.append(rewards[a], reward)
        contexts[a] = np.append(contexts[a], context)
        # fit linear model
        y = rewards[a].reshape(-1,1)
        X = contexts[a].reshape(-1,l)
        if X.shape[0] > l:
            beta = np.linalg.solve(X.T@X, X.T@y)
            eps = y - X @ beta
            sigma2 = eps.T @ eps / (X.shape[0] - l)
            thetas[a] = beta
            As[a] = sigma2 * np.linalg.inv(X.T@X)
      
    regret = np.sum(opt_rewards_train) - np.sum(real_rewards)
    regrets.append(regret)

# optimal tuning param
Copt = Cs[np.argmin(regrets)]
regret_opt = regrets[np.argmin(regrets)]
print("Optimal C=", Copt, "and optimal regret=", regret_opt)
    
# testing
opt_rewards_test = opt_rewards[T:]
real_rewards = []
for t in range(T, dataset.shape[0]):
    context = dataset[t, :l].reshape(-1,1)
    for i in range(k):
        predicts[i] = thetas[i].T @ context + Copt * (context.T @
                (As[i] @ context))**0.5    
    # choose action
    a = np.argmax(predicts)
    # observe reward
    reward = dataset[t, l + a]
    real_rewards.append(reward)


# regret
regret = np.cumsum(opt_rewards_test) - np.cumsum(real_rewards)

# plot
plt.figure(figsize=(9,6), dpi=100)
plt.plot(regret)  
plt.ylabel("regret")
plt.title("UCB")

#==============================================================================
end_program = time.time()
print('This program takes', (end_program - start_program)/60, 'minutes to run.')
