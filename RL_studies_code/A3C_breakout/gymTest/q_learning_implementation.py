import numpy as np

class Q_learning_example:
    def __init__(self, dim, gamma):
		# dimensions of the adjacency matrix
        self.dim = dim
		# discount factor
        self.gamma = gamma
		
		# initialize the environment matrix
        self.env = np.zeros((self.dim, self.dim))
		# initialize the q matrix
        self.q = np.zeros((self.dim, self.dim))
        
		# set the goal state
		self.end_state = 10
    
	# function that initializes the Reward matrix
    def init_env(self, connections): # connections should be the adjacency matrix of your graph
		for i in range(0,len(self.env)):
            for j in range(0, len(self.env[0])):
				# For each element in the reward matrix:
				
				# if [i][j] in adjacency matrix is a connection
                if connections[i][j] == 1:
					# set correspoding reward to 0
                    self.env[i][j] = 0.0
                else:
					# otherwise set to -1
                    self.env[i][j] = -1.0
				
				# If [i][j] is a connection and is the goal state
                if connections[i][j] == 1 and j == self.end_state:
					# set corresponding reward to 100
                    self.env[i][j] = 100.0
		# set loop in goal state that gives reward 100 (relatively redundant)
        self.env[self.end_state][self.end_state] = 100.0
    
	# function that returns a random next possible state
    def get_next_state(self, current_state):
		#set next state to a random state within the dimensions of the adjacency matrix
        next_state = np.random.randint(self.dim)
        
		while self.env[current_state][next_state] == -1:
            # ensure that the next state is a state connected to the current state
			next_state = np.random.randint(self.dim)
        return next_state
    
	# function that runs the Q-Learning algorithm
    def run(self):
		# start in a random state
        current_state = np.random.randint(self.dim)
        while current_state != self.end_state:
			# while the current state is not the goal state:
			
			# get a random possible next state
            next_state = self.get_next_state(current_state)
			# set max_q to the action in the next state with the highest value
            max_q = np.max(self.q[next_state])
			# set the corresponding value in the q matrix to the action value of the next state
            self.q[current_state][next_state] = self.env[current_state][next_state] + (self.gamma * max_q)
            # move into the next state
			current_state = next_state
    
	
	# function used to fund the best path from one state to another, given that training has been completed
    def find_best_path(self, initial_state):
        state = initial_state
        temp = []
        temp.append(state)
        while state != self.end_state:
            state = np.argmax(self.q[state])
            temp.append(state)
        print(temp)

if __name__ == '__main__':
	
	# adjacency matrix of a graph with 10 nodes 
    connections = [[0,1,1,0,0,0,0,0,0,0,0],
                   [1,0,1,0,0,0,0,0,0,0,0],
                   [1,1,0,1,1,1,1,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0,1],
                   [0,0,1,0,0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,1,1,0,0],
                   [0,0,0,0,0,0,1,0,0,1,0],
                   [0,0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0,0],
                   [0,0,0,1,0,0,0,0,0,0,0]]
    
    DIMENSIONS = len(connections)
    GAMMA = 0.8
    EPISODES = 10
    
	# instantiate the 'Q_learning_example' class as 'q'
    q = Q_learning_example(DIMENSIONS, GAMMA)
    # initialize the reward and Q matricies for 'connections'
	q.init_env(connections)
    
	# run the training for 'EPISODES' amount of episodes
    for i in range(0, EPISODES):
		# print information for debugging (written in python 2, add brackets for python 3)
        print "       ---- EPISODE",i+1,"----"
        print "- End State:", q.end_state
		q.run()
		# print the current state of the q matrix
        print(q.q)