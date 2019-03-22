import gym
from keras import layers
from keras.models import Model
from keras import Input


env = gym.make('Breakout-v0')
for i_episode in range(20):
    observation = env.reset()
        
    for t in range(100):
        env.render()
        print(env.observation_space)
        action = env.action_space.sample()
        #print(env.action_space)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


#def Return(reward, discount):
    