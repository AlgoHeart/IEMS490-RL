#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN for Breakout: training

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
EPISODE_NUM = 100000
BUFFER_SIZE = 10000 
BATCH_SIZE = 50
IMAGE_SEQUENCE_SIZE = 4
BURN_IN = 1000
GAMMA = 0.95
EPSILON = 0.1
EPS_DECAY = 0.999
TAU = 0.999
REWARD_DECAY = 0.95
LR = 0.01


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


def update_target(tau):
    """Update target network with rate=tau"""
    W_conv1_T.assign(tau * W_conv1_T + (1-tau) * W_conv1_Q)
    b_conv1_T.assign(tau * b_conv1_T + (1-tau) * b_conv1_Q)
    W_conv2_T.assign(tau * W_conv2_T + (1-tau) * W_conv2_Q)
    b_conv2_T.assign(tau * b_conv2_T + (1-tau) * b_conv2_Q)
    W_fc1_T.assign(tau * W_fc1_T + (1-tau) * W_fc1_Q)
    b_fc1_T.assign(tau * b_fc1_T + (1-tau) * b_fc1_Q)
    W_fc2_T.assign(tau * W_fc2_T + (1-tau) * W_fc2_Q)
    b_fc2_T.assign(tau * b_fc2_T + (1-tau) * b_fc2_Q)


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

# Training
action_vec = tf.placeholder("float", [None, env.action_space.n])
y_target = tf.placeholder("float", [None]) # target value
Q_value = tf.reduce_sum(tf.multiply(Q, action_vec), reduction_indices = 1)
error = tf.reduce_mean(tf.square(Q_value - y_target))
train = tf.train.AdamOptimizer(LR).minimize(error)
# train = tf.train.RMSPropOptimizer(0.0005, 0.99, 0.0, 1e-7).minimize(error)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# Start games
observation = env.reset()
buffer = list() # replay buffer

# sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
with tf.Session() as sess:
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state('./tmp/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, './tmp/model.ckpt')
        print("Model restored.")

    num_step = 0
    epsilon = EPSILON

    try:
        QValue_mean = np.genfromtxt('QValue.txt')
    except:
        QValue_mean = np.zeros(1)
    
    try:
        reward_running = np.genfromtxt('reward.txt')
    except:
        reward_running = np.zeros(1)
    reward_running_ = reward_running[-1]


    for episode in range(EPISODE_NUM):
        observation = env.reset()
        reward_sum = 0

        while True:
            image = preprocess(observation)

            image_sequence.append(image)

            # default action
            action = 0

            if len(image_sequence) <= IMAGE_SEQUENCE_SIZE:
                observation_, reward, done, info = env.step(action)

            else:
                image_sequence.pop(0)
                current_state = np.stack([image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])

                # epsilon-greedy action
                epsilon *= EPS_DECAY
                if np.random.rand(1) < epsilon:
                    action = env.action_space.sample()
                else:
                    QValue = sess.run(Q, feed_dict={image_input: [current_state]})
                    action = np.argmax(QValue)

                observation_, reward, done, info = env.step(action) 

                # add data to relay buffer
                image = preprocess(observation_)
                next_state = np.stack([image_sequence[1], image_sequence[2], image_sequence[3], image])
                
                action_state = encode_action(action)
                buffer.append((current_state, action_state, reward, next_state, done)) 

                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)

                # training
                if num_step > BURN_IN:
                    minibatch = random.sample(buffer, BATCH_SIZE)

                    state_batch = [data[0] for data in minibatch]
                    action_batch = [data[1] for data in minibatch]
                    reward_batch = [data[2] for data in minibatch]
                    nextState_batch = [data[3] for data in minibatch]
                    terminal_batch = [data[4] for data in minibatch]

                    y_batch = []
                    Qtarget_batch = sess.run(Target, feed_dict={image_input: nextState_batch}) # use target network
                    QValue_batch = sess.run(Q, feed_dict={image_input: nextState_batch}) # use Q-network
                    QValue_mean_ = np.mean(QValue_batch.max(1))
                    QValue_mean = np.append(QValue_mean, QValue_mean_)
                    # np.savetxt('QValue.txt', QValue_mean)

                    for i in range(BATCH_SIZE):
                        terminal = minibatch[i][4]
                        if terminal:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + GAMMA * np.max(Qtarget_batch[i]))

                    sess.run(train, feed_dict={y_target: y_batch, action_vec: action_batch, image_input: state_batch})
                    update_target(TAU)


                # if num_step % 100 == 0:
                #     saver.save(sess, './tmp/model.ckpt')

                reward_sum += reward
            
            num_step += 1
            observation = observation_

            if done:
                episode += 1
                print('Episode', episode, 'Rewards:', reward_sum)
                print('Average Q-Value:', QValue_mean[-1])
                np.savetxt('QValue.txt', QValue_mean)
                reward_running_ = REWARD_DECAY * reward_running_ + (1 - REWARD_DECAY) * reward_sum
                print('Running reward:', reward_running_)
                reward_running = np.append(reward_running, reward_running_)
                np.savetxt('reward.txt', reward_running)
                saver.save(sess, './tmp/model.ckpt')
                break

    saver.save(sess, './tmp/model.ckpt')


