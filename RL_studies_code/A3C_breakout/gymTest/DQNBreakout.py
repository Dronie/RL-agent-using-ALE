import numpy as np
import gym, random, collections, math

from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *

class Preprocessor:
    def __init__(self):
        self.stack_size = 4
        self.stacked_frames = collections.deque([np.zeros((105,80), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
    
    def to_greyscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]
    
    def normalize(self, img):
        return img / 255.0

    def preprocess(self, img):
        return self.normalize(self.to_greyscale(self.downsample(img)))
    
    def one_hot_encode(self, actions):
        return np.array(np.identity(actions,dtype='int').tolist())
    
    def stack(self, state):
        frame = self.preprocess(state)
        
        if is_new_episode:
            self.stacked_frames = collections.deque([np.zeros((105,80), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
            
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            
            stacked_state = np.stack(self.stacked_frames, axis=2)
        else:
            self.stacked_frames.append(frame)
            stacked_state = np.stack(self.stacked_frames, axis=2)
        
        return stacked_state, self.stacked_frames
    
class DQN:
    def __init__(self):
        self.model = self.create_model()
        self.process_model()
        self.gamma = 0.99
        
    #input expected as an array of stacked states (shape of (x, 105, 80, 4))
    def create_model(self):
        ATARI_STATE_SPACE = (105, 80, 4)
        ACTION_SPACE = 4,
        
        state_input = Input(shape=(ATARI_STATE_SPACE))
        action_input = Input(shape=(ACTION_SPACE))
        
        conv_1 = convolutional.Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(state_input)
        conv_2 = convolutional.Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        
        flattened_conv = core.Flatten()(conv_2)
        
        final = Dense(256, activation='relu')(flattened_conv)
        
        out = Dense(4,)(final)
        
        filtered = multiply([out, action_input])
        
        return Model(input=[state_input, action_input], output=filtered)

    def process_model(self):
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')
    
    def fit(self, initial_states, actions, rewards, next_states):
        next_Q_values = self.model.predict([next_states, np.ones(actions.shape)])
        #next_Q_values[is_terminal] = 0
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        self.model.fit([initial_states, actions], actions * Q_values[:, None], nb_epoch=1, batch_size=len(initial_states), verbose=0)

    def predict(self, s):
        return self.model.predict(s)
    
    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()
        

class Environment:
    def __init__(self, environment):
        self.environment = environment
        self.env = gym.make(self.environment)

    def return_action_space(self):
        return self.env.action_space.n        
    
    def run(self, agent, render):
        state = self.env.reset()
        reward = 0
        
        done = False
        while not done:
            
            if render:
                self.env.render()
        
        #action = agent.act(current_state)
        
        state, reward, done, info = self.env.step(env.action_space.sample())
        
class Memory:
    samples = []
    
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
    
    def add(self, sample):
        self.samples.append(sample)
        self.check_memory()
    
    def remove(self):
        self.samples.pop(0)
    
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
    def check_memory(self):
        while len(self.samples) > self.max_capacity:
            self.remove()

class Agent:
    def __init__(self, actionCnt, stateCnt):
        self.actionCnt = actionCnt
        self.stateCnt = stateCnt
        self.max_eps = 0.1
        self.min_eps = 0.01
        self.decay_rate = 0.01
        self.steps = 0

        self.dqn = DQN()
        self.memory = Memory(1000000)
    
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.dqn.predictOne(s))
    
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-self.decay_rate * self.steps)

    def replay(self):    
        batch = self.memory.sample(64)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.dqn.predict(states)
        p_ = self.dqn.predict(states_)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + dqn.gamma * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)



def random_play():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()

    is_done = False
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        # Render
        env.render()
            
if __name__ == '__main__':
    is_new_episode = True
    prep = Preprocessor()
    env = Environment('BreakoutDeterministic-v4')
    env = gym.make('BreakoutDeterministic-v4')
    