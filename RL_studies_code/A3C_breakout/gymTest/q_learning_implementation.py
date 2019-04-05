import numpy as np

class Q_learning_example:
    def __init__(self, dim, gamma):
        self.dim = dim
        self.gamma = gamma
        self.env = np.zeros((self.dim, self.dim))
        self.q = np.zeros((self.dim, self.dim))
        self.end_state = np.random.randint(self.dim)
        
    def init_env(self, connections): # connections should be the adjacency matrix of your graph
        for i in range(0,len(self.env)):
            for j in range(0, len(self.env[0])):
                if connections[i][j] == 1:
                    self.env[i][j] = 0.0
                else:
                    self.env[i][j] = -1.0
                if connections[i][j] == 1 and j == self.end_state:
                    self.env[i][j] = 100.0
        self.env[self.end_state][self.end_state] = 100.0
    
    def get_next_state(self, current_state):
        next_state = np.random.randint(self.dim)
        while self.env[current_state][next_state] == -1:
            next_state = np.random.randint(self.dim)
        return next_state
    
    def run(self):
        current_state = np.random.randint(self.dim)
        while current_state != self.end_state:
            next_state = self.get_next_state(current_state)
            max_q = np.max(self.q[next_state])
            self.q[current_state][next_state] = self.env[current_state][next_state] + (self.gamma * max_q)
            current_state = next_state
    
    def find_best_path(self, initial_state):
        state = initial_state
        temp = []
        temp.append(state)
        while state != self.end_state:
            state = np.argmax(self.q[state])
            temp.append(state)
        print(temp)
            
if __name__ == '__main__':
    connections = [[0,0,0,0,1,0],
                   [0,0,0,1,0,1],
                   [0,0,0,1,0,0],
                   [0,1,1,0,1,0],
                   [1,0,0,1,0,1],
                   [0,1,0,0,1,0]]
    
    DIMENSIONS = len(connections)
    GAMMA = 0.8
    EPISODES = 100
    
    q = Q_learning_example(DIMENSIONS, GAMMA)
    q.init_env(connections)
    
    for i in range(0, EPISODES):
        print "       ---- EPISODE",i+1,"----"
        print "- End State:", q.end_state
        q.run()
        print(q.q)