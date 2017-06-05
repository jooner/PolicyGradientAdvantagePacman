import tensorflow as tf
import numpy as np

# Global Variables for Pacman model
STATE_VECTOR_SIZE = 4
WORLD_WIDTH = 20
WORLD_HEIGHT = 7
WORLD_STATES = WORLD_WIDTH * WORLD_HEIGHT * STATE_VECTOR_SIZE
POSS_ACTIONS = 5
PG_STEP_SIZE = 0.01
VG_STEP_SIZE = 0.1
HIDDEN_1_SIZE = 50
HIDDEN_2_SIZE = 50

def gridToArray(grid):
    a, b = grid
    return int(b * WORLD_WIDTH + a)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    with tf.variable_scope("policy"):
        # set up basic variables for PG calculation
        # params = tf.get_variable("policy_parameters",[WORLD_STATES, POSS_ACTIONS])
        state = tf.placeholder("float",[None, WORLD_STATES])
        actions = tf.placeholder("float",[None,POSS_ACTIONS])
        advantages = tf.placeholder("float",[None,1])

        w1 = tf.get_variable("w1",[WORLD_STATES, HIDDEN_1_SIZE])
        b1 = tf.get_variable("b1",[HIDDEN_1_SIZE])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[HIDDEN_1_SIZE, HIDDEN_2_SIZE])
        b2 = tf.get_variable("b2",[HIDDEN_2_SIZE]) 
        h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)
        w3 = tf.get_variable("w3",[HIDDEN_2_SIZE, POSS_ACTIONS])
        b3 = tf.get_variable("b3",[POSS_ACTIONS])
        h3 = tf.matmul(h2,w3) + b3

        probabilities = tf.nn.softmax(h3)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(PG_STEP_SIZE).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None, WORLD_STATES])
        newvals = tf.placeholder("float",[None, 1])

        w1 = tf.get_variable("w1",[WORLD_STATES, HIDDEN_1_SIZE])
        b1 = tf.get_variable("b1",[HIDDEN_1_SIZE])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[HIDDEN_1_SIZE, HIDDEN_2_SIZE])
        b2 = tf.get_variable("b2",[HIDDEN_2_SIZE])
        h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)
        w3 = tf.get_variable("w3",[HIDDEN_2_SIZE, POSS_ACTIONS])
        b3 = tf.get_variable("b3",[POSS_ACTIONS])
        h3 = tf.matmul(h2,w3) + b3

        diffs = h3 - newvals
        loss = tf.nn.l2_loss(diffs)
        # where the heavy-lifting happens
        optimizer = tf.train.AdamOptimizer(VG_STEP_SIZE).minimize(loss)
        return h3, state, newvals, optimizer, loss

def translateIdxToAction(idx):
    if idx == 0:
        return 'North'
    elif idx == 1:
        return 'South'
    elif idx == 2:
        return 'East'
    elif idx == 3:
        return 'West'
    return 'Stop'

def translateActionToIdx(action):
    if action == 'North':
        return 0
    elif action == 'South':
        return 1
    elif action == 'East':
        return 2
    elif action == 'West':
        return 3
    return 4
