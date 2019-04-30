import gym, collections, random, math, sys
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

#np.set_printoptions(threshold=sys.maxsize)

# Hyperparameters
max_eps = 1
min_eps = 0.01
epsilon = max_eps
Lambda = 0.001
steps = 0
gamma = 0.99

# Create a breakout environment
env = gym.make("BreakoutDeterministic-v4")

# initialize a deque to prepare frames for stacking
stacked_frames = collections.deque([np.zeros((105,80), dtype=np.int) for i in range(4)], maxlen=4)
# initialize experience replay with max capacity of 1 million elements 
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

def normalize(img):
    return img / 255.0
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

# adds a stack to experience replay memory
def add_stack_to_replay():
    s = conv_stack_to_numpy_array()
    experience_replay_memory.append(s)

# convert stack to numpy array (useful for manipulation by the network)
def conv_stack_to_numpy_array():
    return np.asarray(stacked_frames).reshape(105, 80, 4)


# define the DQN, input is a 
model = Sequential()

model.add(Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(105, 80, 4)))
model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(env.action_space.n))
#model.load_weights("DQNBreakout.h5")
model.summary()

optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
model.compile(optimizer, loss='mse')

# function that returns a prediction on 'state', where 'state' is a set of preprocessed, stacked, and committed to experience replay, states 
def predict(state):
    return model.predict(state)

# adds 'sample' to experience_replay_memory and adjusts epsilon based on the amount of steps taken so far
def observe(sample):
    global steps
    global epsilon
    experience_replay_memory.append(sample)

    steps += 1
    epsilon = min_eps + (max_eps - min_eps) * math.exp(-Lambda * steps)

# epsilon-greedy (input (1, 105, 80, 4))
def act(state):
    if random.random() < epsilon:
        return random.randint(0, env.action_space.n)
    else:
        return np.argmax(predict(state))

# returns a sample from experience_replay_memory of size 'batch_size'
def sample(batch_size):
    if batch_size > len(np.asarray(experience_replay_memory)):
        return np.asarray(experience_replay_memory)
    else:
        return np.asarray(random.sample(experience_replay_memory, batch_size))

def replay():
    batch = sample(64)

    no_state = np.zeros(33600).reshape(105, 80, 4)

    states = np.array([ observations[0] for observations in batch ])
    states_ = np.array([ (no_state if done else observations[3]) for observations in batch ])

    predictions = predict(states)
    predictions_ = predict(states_)

    x = np.zeros((64, 105, 80, 4))
    y = np.zeros((64, 4))

    for i in range(len(batch)):
        o = batch[i]
        s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

        target = predictions[i]
        if done:
            target[a-1] = r
        else:
            target[a-1] = r + gamma * np.argmax(predictions_[i])
        
        x[i] = s
        y[i] = target
    
    model.fit(x, y, batch_size=64, nb_epoch=1, verbose=0)

rewards = []
episodes = 10000
for i in range(episodes):
    done = False
    
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    #env.render()
    init_img_stack(frame)
    s = add_to_stack(frame)
    R = 0
    #add_stack_to_replay()
    while True:    
        a = act(np.asarray([s]))

        s_, r, done, info = env.step(a-1)
        #env.render()
        print("Current episode:", i, "/", episodes)
        s_ = add_to_stack(s_)

        observe( (s, a, r, s_) )
        replay()

        s = s_
        R += r
        print("Current total reward:", R)
        if done:
            break
    rewards.append(R)
plt.plot(range(episodes), rewards)

model.save("DQNBreakout.h5")