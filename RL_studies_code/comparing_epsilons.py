import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, upper_bound): # Constructor
        self.trueMean = m # True Mean Reward
        self.estimatedMean = upper_bound # Estimate of Bandit's Mean Reward
        self.N = 0 # Denotes the amount of times this bandit's arms has been pulled

    def pull(self): # Simulates pulling the bandit's arm
        return np.random.randn() + self.trueMean # spit out a random number from a gaussian distribution + true mean as the reward

    def update(self, x): # x is the latest sample recieved from pulling the bandit's arm
        self.N += 1 
        self.estimatedMean = (1 - 1.0/self.N)*self.estimatedMean + 1.0/self.N*x # Mean function

def ucb(mean, n, nj): # Upper Confidence Bound function
  if nj == 0:
    return float('inf')
  return mean + np.sqrt(2*np.log(n) / nj)

def run_experiment(m1, m2, m3, eps, N, m): # takes 3 different true means (one for each bandit respectively), eps for epsilon greedy, n = no of time to pull bandit's arm
    
    bandits = [Bandit(m1, 10), Bandit(m2, 10), Bandit(m3, 10)]
    
    data = np.empty(N) # empty array of size N to store results

    if (m == "e"):
        for i in range(N):
            # epsilon-greedy implementation
            p = np.random.random() # random number used to pick which bandit's arm will be pulled next
            if p < eps:
                j = np.random.choice(3) # choose a new bandit at random
            else:
                j = np.argmax([b.estimatedMean for b in bandits]) # choose the bandit with the best current estimated mean
            x = bandits[j].pull() # pull bandit j (j being the bandit chosen in the preceding algorithm)
            bandits[j].update(x) # update bandit with reward obtained from pulling the arm
            data[i] = x # for the plot
    
    elif(m == "o"):
        for i in range(N):
            j = np.argmax([b.estimatedMean for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)
            data[i] = x
            
    elif(m=="u"):
            for i in range(N):
                j = np.argmax([ucb(b.estimatedMean, i+1, b.N) for b in bandits])
                x = bandits[j].pull()
                bandits[j].update(x)
                data[i] = x
    else:
        print("ERROR: incompatible solution method!")
        return
    
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average, label='Cumulative Avg')
    plt.plot(np.ones(N)*m1, label='m1')
    plt.plot(np.ones(N)*m2, label='m2')
    plt.plot(np.ones(N)*m3, label='m3')
    plt.xscale('log')
    plt.xlabel("Number of Pulls")
    plt.ylabel("Cumulative Average Reward")
    plt.show()

    for b in bandits:
        print("Final error for bandit", bandits.index(b) + 1, (b.trueMean - b.estimatedMean))

    return cumulative_average

if __name__ == '__main__':
    print("Experiment 1 (10% Exploration Rate)")
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000, "o") # run first experiment with 10% eps
    print("Experiment 2 (5% Exploration Rate)")
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000, "e") # run second experiment with 0.5% eps
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000, "u") # run second experiment with 0.5% eps

    # log scale plot
    plt.plot(c_1, label='Opt')
    plt.plot(c_05, label='Eps')
    plt.plot(c_01, label='UCB')
    plt.legend()
    plt.xscale('log')
    print("Log Plot")
    plt.show()

    # linear plot
    plt.plot(c_1, label='EpsGre')
    plt.plot(c_05, label='OptIni')
    plt.plot(c_01, label='UCB1')
    plt.legend()
    print("Linear Plot")
    plt.show()
