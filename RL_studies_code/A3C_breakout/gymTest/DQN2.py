# Import the gym module
import gym
import numpy as np
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
  
def to_grayscale(img): # convert 'img' to greyscale
    return np.mean(img, axis=2).astype(np.uint8) 

def downsample(img): # downsample 'img' by exactly 2x
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

def transform_reward(reward):
    return np.sign(reward)
