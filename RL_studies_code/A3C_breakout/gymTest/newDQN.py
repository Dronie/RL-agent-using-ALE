# Importing the libs
import numpy as np
import gym, random, collections

from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *

# creating the environment
env = gym.make('BreakoutDeterministic-v4')

print "The size of the frame is:", env.observation_space
print "The size of the action space is:", env.action_space.n

# one hot encode the action space
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

def one_hot_to_scalar(action):
    return np.argmax(action)

# defining the preprocessing functions
def preprocess_frame(frame):
    grey = np.mean(frame, axis=2).astype(np.uint8)
    
    downsampled_frame = grey[::2, ::2]
    
    normalized_frame = downsampled_frame / 255.0
    
    return normalized_frame

preprocessed_frame_shape = preprocess_frame(env.reset()).shape
print "The size of the preprocessed frame is:", preprocessed_frame_shape

# stacking the frames
stack_size = 4

stacked_frames = collections.deque([np.zeros(preprocessed_frame_shape, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames = collections.deque([np.zeros(preprocessed_frame_shape, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        
        for i in range(stack_size):
            stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis=2)
    
    else:
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis=2)
    
    return stacked_state, stacked_frames

# Model Hyperparameters
state_size = [4, 105, 80]
action_size = env.action_space.n,
learning_rate = 0.000025

# Training Hyperparameters
total_episodes = 50
max_steps = 50000
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

# Q learning hyperparameters
gamma = 0.9

# Memory hyperparameters
pretrain_length = batch_size
memory_size = 1000000

# Preprocessing Hyperparameters
stack_size = 4

view_only_training = False
episode_render = False

class DQN:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Define the model
        self.state_input = Input(shape=(self.state_size))
        self.action_input = Input(shape=(self.action_size))
        
        self.conv_1 = convolutional.Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(self.state_input)
        self.conv_2 = convolutional.Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(self.conv_1)
        
        self.flattened_conv = core.Flatten()(self.conv_2)
        
        self.final = Dense(256, activation='relu')(self.flattened_conv)
        
        self.out = Dense(4,)(self.final)
        
        self.filtered = multiply([self.out, self.action_input])

        self.model = Model(input=[self.state_input, self.action_input], output=self.filtered)
        
        self.optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(self.optimizer, loss='mse')

DQNetwork = DQN(state_size, action_size, learning_rate)

class Memory():
    def __init__(self, max_size):
        self.samples = collections.deque(maxlen = max_size)
    
    def add(self, state):
        self.samples.append(state)
    
    def sample(self, batch_size):
        n = min(batch_size, len(self.samples))
        return random.sample(self.samples, n)

memory = Memory(memory_size)
for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    action = one_hot_to_scalar(action)
    next_state, reward, done, info = env.step(action)
    
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    if done:
        next_state = np.zeros(state.shape)
        
        memory.add((state, action, reward, next_state, done))
        
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    else:
        memory.add((state, action, reward, next_state, done))
        
        state = next_state
        
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        choice = random.randint(1,len(possible_actions)) - 1
        action = possible_actions[choice]
    
    else:
        Qs = DQNetwork.predict([state, action])
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    
    return action, explore_probability



