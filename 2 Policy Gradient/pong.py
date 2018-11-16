#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Gradient for Pong

@author: Yiming Peng
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import time

start_program = time.time()
#==============================================================================
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # training data
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []


    def build_network(self):
        with tf.name_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.n_features])
            self.actions = tf.placeholder(tf.int32, [None, ], name="actions")
            self.returns = tf.placeholder(tf.float32, [None, ], name="returns")
        
        # input layer
        input_layer = tf.layers.dense(
            inputs=self.obs,
            units=200,
            activation=tf.nn.relu,  
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name='input_layer'
        )

        # hidden layer 1
        hidden_layer_1 = tf.layers.dense(
            inputs=input_layer,
            units=self.n_actions,
            activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name='hidden_layer_1'
        )
   
        # # hidden layer 2
        # hidden_layer_2 = tf.layers.dense(
        #     inputs=hidden_layer_1,
        #     units=self.n_actions,
        #     activation=None,
        #     kernel_initializer = tf.contrib.layers.xavier_initializer(),
        #     name='hidden_layer_2'
        # )
        
        # action probabilities
        self.act_prob = tf.nn.softmax(hidden_layer_1, name='act_prob') 

        with tf.name_scope('log-likelihood'):
            logll = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden_layer_1, labels=self.actions)   
        
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(logll * self.returns)  

        with tf.name_scope('train_network'):
            self.train_network_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def sample_action(self, observation):
        prob_weights = self.sess.run(self.act_prob, feed_dict={self.obs: observation[np.newaxis, :]})
        return np.random.choice([2,3], p=prob_weights.ravel()) 
        # return np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) 


    def store_history(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)


    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_sum = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_sum = running_sum * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_sum

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


    def train_network(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self.discount_and_norm_rewards()

        # train_network on episode
        self.sess.run(self.train_network_op, feed_dict={
             self.obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.actions: np.array(self.ep_as),  # shape=[None, ]
             self.returns: discounted_ep_rs_norm,  # shape=[None, ]
        })
        
        # reset episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  
    

# make game environment
env = gym.make('Pong-v0')

# initialize agent
agent = PolicyGradient(
    n_actions=2,
    n_features=80*80,
    learning_rate = 0.01,
    reward_decay=0.99   
)

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 1D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()

# training loop
reward_hist = []
num_ep = 100000
for i_ep in range(num_ep):
    observation = env.reset()
    observation = preprocess(observation)
    while True:
        action = agent.sample_action(observation)
        observation_, reward, done, info = env.step(action)
        observation_ = preprocess(observation_)
        agent.store_history(observation, action-2, reward)

        if done:
            ep_reward = sum(agent.ep_rs) 
            reward_hist.append(ep_reward)
            agent.train_network()
            # output rewards
            np.savetxt("pong.txt", reward_hist)
            
            print("episode:", i_ep, "  reward:", ep_reward)
#            if i_ep % 10 == 0:
#                print("episode:", i_ep, "  reward:", ep_reward)
           
            break

        observation = observation_

# # output rewards
# np.savetxt("pong.txt", reward_hist)

# plot average reward
reward_mean = [np.mean(reward_hist[i-99:i+1]) for i in range(len(reward_hist))]
plt.figure(figsize=(9,6), dpi=100)
plt.plot(range(1, num_ep+1), reward_mean)
plt.xlabel("epochs")
plt.ylabel("average episode reward")
plt.title("Pong")
plt.savefig("Pong.png")
#==============================================================================
end_program = time.time()
print('This program takes', (end_program - start_program)/60, 'minutes to run.')
