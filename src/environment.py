import gymnasium as gym
import numpy as np

class CartPoleEnv():
    def __init__(self, render_mode=None):
        self.env = gym.make('CartPole-v1')


    def reset(self):
        observation, info = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_state = observation
        step_reward = reward
        done = terminated or truncated
        step_info = info
        return next_state, step_reward, done, step_info

    def get_action_space(self):
        return self.env.action_space.n
    
    def get_state_space(self):
        return self.env.observation_space.shape[0]