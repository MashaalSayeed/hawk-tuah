import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_spread_v2  # Multi-agent environment

# Fire Location (Random)
FIRE_X, FIRE_Y = np.random.randint(50, 100), np.random.randint(50, 100)

# Custom environment for drones searching fire
class FireSearchEnv(gym.Env):
    def __init__(self, num_drones=5):
        super(FireSearchEnv, self).__init__()
        self.num_drones = num_drones
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(self.num_drones, 4))

        # Initialize drone positions
        self.drones = np.random.randint(0, 50, size=(self.num_drones, 2))
    
    def step(self, actions):
        rewards = []
        done = False
        obs = []

        for i in range(self.num_drones):
            dx, dy = actions[i]
            self.drones[i][0] += dx * 2
            self.drones[i][1] += dy * 2
            
            # Compute extra observation features
            distance = np.linalg.norm(self.drones[i] - np.array([FIRE_X, FIRE_Y]))
            heat_intensity = max(0, 100 - distance)  # Simulate heat signal

            # Reward: Closer to fire
            reward = -distance / 100
            if distance < 5:
                reward += 100
                done = True

            rewards.append(reward)
            obs.append([self.drones[i][0], self.drones[i][1], distance, heat_intensity])

        return np.array(obs), np.array(rewards), done, {}

    def reset(self):
        self.drones = np.random.randint(0, 50, size=(self.num_drones, 2))
        obs = [[x, y, np.linalg.norm([x - FIRE_X, y - FIRE_Y]), max(0, 100 - np.linalg.norm([x - FIRE_X, y - FIRE_Y]))] 
               for x, y in self.drones]
        return np.array(obs)

# Initialize Environment
env = FireSearchEnv(num_drones=5)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Testing trained model
obs = env.reset()
for _ in range(100):
    actions, _ = model.predict(obs)
    obs, rewards, done, _ = env.step(actions)
    if done:
        print("Fire Found!")
        break

# Visualize the trained drones
plt.figure(figsize=(6, 6))
plt.scatter(FIRE_X, FIRE_Y, color="red", s=200, label="🔥 Fire")
plt.scatter(env.drones[:, 0], env.drones[:, 1], color="blue", label="Drones")
plt.legend()
plt.show()
