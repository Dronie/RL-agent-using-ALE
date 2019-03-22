import random, numpy, math, gym

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCount))
        model.add(Dense(output_dim=actionCount, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCount)).flatten()
    
class Memory: # stored as ( s, a, r, s_ )
        samples = []
        
        def __init__(self, capacity):
            self.capacity = capacity
            
        def add(self, sample):
            self.samples.append(sample)
            
            if len(self.samples) > self.capacity:
                self.samples.pop(0)
                
        def sample(self, n):
            n = min(n, len(self.samples))
            return random.sample(self.samples, n)

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001 # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount
        
        self.brain = Brain(stateCount, actionCount)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon: # epsilon-greedy
            return random.randint(0, self.actionCount-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))
        
    def observe(self, sample):
        self.memory.add(sample)
        
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
    
    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLength = len(batch)
        
        no_state = numpy.zeros(self.stateCount)
        
        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[0] is None else o[3]) for o in batch ])
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        
        x = numpy.zeros((batchLength, self.stateCount))
        y = numpy.zeros((batchLength, self.actionCount))
        
        for i in range(batchLength):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])
            
            x[i] = s
            y[i] = t
            
        self.brain.train(x, y)
            
class Environment: # abstraction for OpenAI Gym
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
    
    def run(self, agent):
        s = self.env.reset()
        R = 0
        
        while True:
            self.env.render()
            
            a = agent.act(s)
            
            s_, r, done, info = self.env.step(a)
            
            if done: # terminal state
                s_ = None # marks the end of an episode
                
            agent.observe( (s, a, r, s_) )
            agent.replay()
            
            s = s_
            R += r
            
            if done:
                break
            
        print("Total Reward:", R)
        
#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCount  = env.env.observation_space.shape[0]
actionCount = env.env.action_space.n

agent = Agent(stateCount, actionCount)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")




        