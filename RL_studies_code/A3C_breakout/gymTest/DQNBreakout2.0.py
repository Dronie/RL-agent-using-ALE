import gym, collections, random, math, sys
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import *
#from keras.optimizers import *

from tensorflow.keras.optimizers import *

#np.set_printoptions(threshold=sys.maxsize)

# Hyperparameters
max_eps = 1
min_eps = 0.01
epsilon = max_eps
Lambda = 0.001
steps = 0
gamma = 0.99
Q = 0

# Create a breakout environment
env = gym.make("BreakoutDeterministic-v4")

# initialize a deque to prepare frames for stacking
stacked_frames = collections.deque([np.zeros((105,80), dtype=np.int) for i in range(4)], maxlen=4)
# initialize experience replay with max capacity of 250,000 elements 
experience_replay_memory = collections.deque(maxlen=250000)

# function to convert state to greyscale 
def to_greyscale(img):
    return np.mean(img, axis=2).astype(np.uint8) 

# function to downsample state by a factor of 2 (each img goes from (210, 160) to (105, 80))
def downsample(img):
    return img[::2, ::2]

# combines all preprocessing functions above
def preprocess(img):
    return to_greyscale(downsample(img))

# initializes frame stack deque to a full stack of the first state
def init_img_stack(img):
    frame = preprocess(img)
    for i in range(4):
        stacked_frames.append(frame)

# adds a state to the frame stack deque
def add_to_stack(img):
    frame = preprocess(img)
    stacked_frames.append(frame)
    return np.asarray(stacked_frames).reshape(105, 80, 4)

# define the Neural Network driving the Deep Q Network
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(105, 80, 4)),
    tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n)
])


# load weights of previously trained model
model.load_weights("DQNBreakout.h5")

# print a summary of the model, including the amount of traniable parameters
model.summary()

#define the optimizer of the model
optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)

#compile the model with the previously defined optimizer and a Mean Squared Error loss function
model.compile(optimizer, loss='mse')

# function that returns a prediction on 'state', where 'state' is a set of preprocessed, stacked, and committed to experience replay states 
def predict(state):
    return model.predict(state)

# adds 'sample' to experience_replay_memory and adjusts epsilon based on the epsilon decay equation
def observe(sample):
    global steps
    global epsilon
    experience_replay_memory.append(sample)

    steps += 1
    epsilon = min_eps + (max_eps - min_eps) * math.exp(-Lambda * steps)

# epsilon-greedy function, returns the action to take given 'state'
def act(state):
    if random.random() < epsilon:
        return random.randint(0, env.action_space.n)
    else:
        global Q
        Q = predict(state)
        return np.argmax(Q)

# returns a sample from experience_replay_memory of size 'batch_size'
def sample(batch_size):
    if batch_size > len(np.asarray(experience_replay_memory)):
        return np.asarray(experience_replay_memory)
    else:
        return np.asarray(random.sample(experience_replay_memory, batch_size))

# replay function, ultimately performs a single gradient descent step
def replay():
    # get 64 (s,a,r,s') samples
    batch = sample(64)
    
    # used as a placeholder for s' if the current state is terminal
    no_state = np.zeros(33600).reshape(105, 80, 4)

    # set states to 's' from batch
    states = np.array([ observations[0] for observations in batch ])
    
    # set states to 's'' from batch
    states_ = np.array([ (no_state if done else observations[3]) for observations in batch ])
    
    # make Q(s,a) predictions on states and states_
    predictions = predict(states)
    predictions_ = predict(states_)
    
    # initialize input and label data sets
    x = np.zeros((64, 105, 80, 4))
    y = np.zeros((64, env.action_space.n))

    # for every frame, use predictions to make a target on which to perform gradient descent
    for i in range(len(batch)):
        o = batch[i]
        s = o[0]; a = o[1]; r = o[2]

        target = predictions[i]
        if done:
            target[a-1] = r
        else:
            target[a-1] = r + gamma * np.argmax(predictions_[i])
        
        x[i] = s
        y[i] = target
    
    model.fit(x, y, batch_size=64, epochs=1, verbose=0)

episodes = 10000

# begin training
for i in range(episodes):
    # At the start of each episode:
    
    # save model
    model.save("DQNBreakout.h5")
    
    # print current episode
    print("Current episode:", i + 1, "/", episodes)
    
    # make sure the 'done' variable is set to false
    global done
    done = False
    
    # reset the environment
    frame = env.reset()
    
    # render the environment
    env.render()
    
    # Initialize the frame stack with the first frame of the game
    init_img_stack(frame)
    
    # set s to the current state of the frame stack
    s = add_to_stack(frame)
    
    # reset total reward to 0
    R = 0
    
    while True:
        # For each state:
        
        # set the current action (returned by act()) to a
        a = act(np.asarray([s]))
        
        # perform action a storing the next state, reward and whether the state is terminal (info is essentially ignored)
        s_, r, done, info = env.step(a-1)
        
        # render the new frame
        env.render()
        
        # ass s_ to the frame stack and store it as s_
        s_ = add_to_stack(s_)

        # add the most recent (s, a, r, s') tuple to memory and increment one step of epsilon greedy decay
        observe( (s, a, r, s_) )
        
        # use a sample from memory to predict targets and perform a single gradient descent step
        replay()
        
        # set the current state as the next state
        s = s_
        
        # Increment the total reward
        R += r
        
        # if a terminal state is reached, start a new episode
        if done:
            break
    
    print("Previous Reward:", R)

#model.save("DQNPong.h5")