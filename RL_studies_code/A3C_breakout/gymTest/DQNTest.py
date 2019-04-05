import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self): # create the NN
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):# perform supervised training step
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s): # predict Q function values in state s
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = [] # the core of the class

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample): # add a sample to memeory
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0) # if the size of the samples list is bigger than capacity, pop the first element in the list

    def sample(self, n): # returns a random set of samples (of size n)
        n = min(n, len(self.samples)) # n = smallest out of n and the len(samples)
        return random.sample(self.samples, n) # return n random samples from samples[]

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99 

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt) # 
        self.memory = Memory(MEMORY_CAPACITY) 
        
    def act(self, s): # decide what action to take in state s
        if random.random() < self.epsilon: # epsilon-greedy
            return random.randint(0, self.actionCnt-1) # take random action
        else:
            return numpy.argmax(self.brain.predictOne(s)) # take greedy action

    def observe(self, sample):  # adds sample (s, a, r, s_) to memory
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self): # replays memories and improves
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
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

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem): # abstraction for the gym env
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent): # runs one episode
        s = self.env.reset() # reset the environment starting a new episode
        R = 0  # reset the total reward for the new episode

        while True: # loop forever
            #self.env.render() # render a visualisation of the current problem 

            a = agent.act(s) # decides what action to take (performs eps-greedy, if greedy action taken)

            s_, r, done, info = self.env.step(a)# environment performs this action and returns the next state and a reward

            if done: # terminal state (when done is true, it means the episode has terminated)
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0] # returns the shape of the 
actionCnt = env.env.action_space.n # returns the amount of actions possible in the environment

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")
