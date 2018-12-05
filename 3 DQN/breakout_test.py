#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN for Breakout: testing 

@author: Yiming Peng
"""

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# Global param
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="4"
env = gym.make('Breakout-v0')
EPISODE_NUM = 1000
IMAGE_SEQUENCE_SIZE = 4
EPSILON = 0.1


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])


def Q_net(image):
    """forward pass of Q-network"""
    image = tf.reshape(image, shape=[-1, 80, 80, 4])
    # hidden layers
    h_conv1_Q = tf.nn.relu(tf.nn.conv2d(image, W_conv1_Q, [1, 4, 4, 1], padding = "VALID") + b_conv1_Q)
    h_conv2_Q = tf.nn.relu(tf.nn.conv2d(h_conv1_Q, W_conv2_Q, [1, 2, 2, 1], "VALID") + b_conv2_Q)
    h_conv2_flat_Q = tf.reshape(h_conv2_Q,[-1, 2048])
    h_fc1_Q = tf.nn.relu(tf.matmul(h_conv2_flat_Q, W_fc1_Q) + b_fc1_Q)
    # output layer with actions
    out = tf.matmul(h_fc1_Q,W_fc2_Q) + b_fc2_Q
    return out


def Target_net(image):
    """forward pass of target network"""
    image = tf.reshape(image, shape=[-1, 80, 80, 4])
    # hidden layers
    h_conv1_T = tf.nn.relu(tf.nn.conv2d(image, W_conv1_T, [1, 4, 4, 1], padding = "VALID") + b_conv1_T)
    h_conv2_T = tf.nn.relu(tf.nn.conv2d(h_conv1_T, W_conv2_T, [1, 2, 2, 1], "VALID") + b_conv2_T)
    h_conv2_flat_T = tf.reshape(h_conv2_T,[-1, 2048])
    h_fc1_T = tf.nn.relu(tf.matmul(h_conv2_flat_T, W_fc1_T) + b_fc1_T)
    # output layer with actions
    out = tf.matmul(h_fc1_T,W_fc2_T) + b_fc2_T
    return out


def encode_action(val):
    '''one-hot encoding of action'''
    action = np.zeros(env.action_space.n)
    action[val] = 1
    return action


# Create Q-network and Target Network
graph = tf.Graph()
image_sequence = list()

image_input = tf.placeholder(tf.float32, shape=(None, 4, 80, 80))

W_conv1_Q = tf.Variable(tf.truncated_normal([8,8,4,16], stddev = 0.01))
b_conv1_Q = tf.Variable(tf.constant(0.01, shape = [16]))
W_conv1_T = tf.Variable(tf.truncated_normal([8,8,4,16], stddev = 0.01))
b_conv1_T = tf.Variable(tf.constant(0.01, shape = [16]))

W_conv2_Q = tf.Variable(tf.truncated_normal([4,4,16,32], stddev = 0.01))
b_conv2_Q = tf.Variable(tf.constant(0.01, shape = [32]))
W_conv2_T = tf.Variable(tf.truncated_normal([4,4,16,32], stddev = 0.01))
b_conv2_T = tf.Variable(tf.constant(0.01, shape = [32]))

W_fc1_Q = tf.Variable(tf.truncated_normal([2048, 256], stddev = 0.01))
b_fc1_Q = tf.Variable(tf.constant(0.01, shape = [256]))
W_fc1_T = tf.Variable(tf.truncated_normal([2048, 256], stddev = 0.01))
b_fc1_T = tf.Variable(tf.constant(0.01, shape = [256]))

W_fc2_Q = tf.Variable(tf.truncated_normal([256, env.action_space.n], stddev = 0.01))
b_fc2_Q = tf.Variable(tf.constant(0.01, shape = [env.action_space.n]))
W_fc2_T = tf.Variable(tf.truncated_normal([256, env.action_space.n], stddev = 0.01))
b_fc2_T = tf.Variable(tf.constant(0.01, shape = [env.action_space.n]))

Q = Q_net(image_input)
Target = Target_net(image_input)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# Start games
with tf.Session() as sess:
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state('./tmp/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, './tmp/model.ckpt')
        print("Model restored.")


    rewards = []
    for episode in range(EPISODE_NUM):
        observation = env.reset()
        image_sequence = list()
        reward_sum = 0
        
        while True:
            image = preprocess(observation)
            image_sequence.append(image)

            if len(image_sequence) <= IMAGE_SEQUENCE_SIZE:
                action = env.action_space.sample()

            else:
                image_sequence.pop(0)
                current_state = np.stack([image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])
                # epsilon-greedy action
                if np.random.rand(1) < EPSILON:
                    action = env.action_space.sample()
                else:
                    QValue = sess.run(Q, feed_dict={image_input: [current_state]})
                    action = np.argmax(QValue)

            observation_, reward, done, info = env.step(action)
            observation = observation_
            reward_sum += reward

            if done:
                episode += 1
                print('Episode', episode, 'Rewards:', reward_sum)
                rewards.append(reward_sum)
                np.savetxt('test.txt', rewards)
                break



