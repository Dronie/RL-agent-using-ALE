import numpy as np

class Environment:
    def __init__(self):
        self.board = np.zeros(9).reshape(3,3) # Define a 3x3 matrix to denote a Tic-Tac-Toe board
        self.game_over = True # define a game over state
    
    def conv(self, place):
        if place == 1:
            return "X" # if an element in the board matrix is a '1' return an 'X'
        elif place == 2:
            return "O" # if an element in the board matrix is a '2' return an '0'
        else:
            return " " # return a space 
    
    def draw_board(self):
        if not self.game_over: # code for drawing the board
            print("-------------")
            print "|", self.conv(self.board[0,0]), "|",self.conv(self.board[0,1]), "|", self.conv(self.board[0,2]), "|"
            print("-------------")
            print "|", self.conv(self.board[1,0]), "|",self.conv(self.board[1,1]), "|", self.conv(self.board[1,2]), "|"
            print("-------------")
            print "|", self.conv(self.board[2,0]), "|",self.conv(self.board[2,1]), "|", self.conv(self.board[2,2]), "|"
            print("-------------")
        else:
            return 0

class Agent:
    def __init__(self):
        self.state_history = np.array() # initialize state history as numpy array
        
    
    def take_action(self): # method for taking an action in the environment
        
    
    def update_state_history(self): # method for updateing the state history 
        
    
    def update(): # method for updating the values of state changes
        
    
        
class Player:
    def __init__(self):
        
    def make_move(self):
        
    
    
def play_game(p1, p2, env, draw=False):
    current_player = None
    while not env.game_over(): # loop until game is over
        # alternate between players
        # p1 always starts first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        # draw the board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()
        
        # current player makes a move
        current_player.take_action(env)
        
        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
    if draw:
        env.draw_board()
    
    # do the value function update
    p1.update(env)
    p2.update(env)
    

if __name__ == '__main__':
    env = Environment()