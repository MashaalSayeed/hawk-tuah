from pettingzoo.utils.env import ParallelEnv
from pettingzoo.test import parallel_api_test
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

"""
Multi-Agent Reinforcement Learning Environment for Firefighting
- The environment consists of multiple drones (scouts and suppressor) that need to extinguish a fire.
- The drones have to navigate to the fire location and extinguish it.
- The environment is partially observable, and the drones have to cooperate to extinguish the fire effectively.

The environment is inspired by the Firefighting Multi-Agent Reinforcement Learning (MARL) problem.

==========================================
    The Firefighting MARL Environment
==========================================

Scout: The scout drone is responsible for exploring the environment and finding the fire location.
Suppressor: The suppressor drone is responsible for extinguishing the fire once the fire location is found.
The environment consists of the following components:

Fire Grid:
- The fire grid is a 2D grid representing the intensity of the fire at each location.
- The fire intensity ranges from 0 to 1, where 0 represents no fire and 1 represents maximum fire intensity.
- The fire grid is partially observable to the drones, and they have to explore the environment to find the fire location.

Scout Drone:
- The scout drone has a sensing radius of 10 units.
- The scout drone has a speed of 1 unit per step.
- Observation Space: (x, y) position of the scout drone + battery level (normalized to [0, 1]) + fire intensity grid within the sensing radius.
- Action Space: Discrete(9) representing the actions: {8 Directions + Stay}.

Suppressor Drone:
- The suppressor drone has a speed of 1 unit per step.
- Observation Space: (x, y) position of the suppressor drone + battery level (normalized to [0, 1]) + fire intensity grid at the current location.
- Action Space: Discrete(10) representing the actions: {8 Directions + Stay + Extinguish}.
- The suppressor drone can extinguish the fire at its current location by taking the Extinguish action.

==================================
    Comparison Metrics
==================================

- Fire Extinguished %:      	Percentage of total fire cells extinguished by the end of an episode.
- Response Time: 	            Time taken to detect and start suppressing a fire after it starts.
- Average Extinguishing Time: 	Average time taken to extinguish a hotspot from detection.
- Total Time/Episode Length: 	Total time taken to extinguish all fires (or reach terminal condition).
- Energy/Battery Usage: 	    Total energy consumed across agents (battery-based cost).
- Number of Collisions: 	    For systems where agents can collide, track safety/efficiency.
- Idle Time: 	                Time agents spent doing nothing (e.g., no detection, no suppression).

==================================
"""

from drone import Drone, ScoutDrone, SuppressorDrone, FireGrid
from metrics import MetricsLogger



class FireFightingEnv(ParallelEnv):
    metadata = {
        "name": "FireFightingEnv"
    }

    def __init__(self, grid_size=(100, 100), num_scouts=2, num_suppresors=3, render_mode=None):
        self.num_scouts = num_scouts
        self.num_suppresors = num_suppresors
        self.render_mode = render_mode

        self.fire_grid = FireGrid(grid_size)
        self.scouts = [ScoutDrone(i, grid_size) for i in range(num_scouts)]
        self.suppressors = [SuppressorDrone(i, grid_size) for i in range(num_suppresors)]
        self.metrics_logger = MetricsLogger()

        self.agent_map = {drone.id: drone for drone in self.scouts + self.suppressors}
        self.agents = list(self.agent_map.keys())
        self.possible_agents = self.agents.copy()

        # Define the observation and action spaces for each agent
        # Scouts: (x, y) position + battery level + fire intensity grid
        # Suppressors: (x, y) position + battery level + fire intensity at current location
        self.observation_spaces = {
            drone.id: drone.observation_space() for drone in self.scouts + self.suppressors
        }

        # Scouts: Move Up, Move Down, Move Left, Move Right, Stay
        # Suppressors: Move Up, Move Down, Move Left, Move Right, Extinguish, Stay
        self.action_spaces = {
            scout.id: spaces.Discrete(9) for scout in self.scouts
        }
        
        self.action_spaces.update({
            suppressor.id: spaces.Discrete(10) for suppressor in self.suppressors
        })

        if self.render_mode != None:
            self.fig, self.ax = None, None

    def _get_obs(self, agent=None):
        obs = {}
        for agent_id in self.agents:
            drone = self.agent_map[agent_id]
            x, y = drone.position
            if isinstance(drone, ScoutDrone):
                fire_intensity = drone.sense_fire(self.fire_grid)
            else:  # SuppressorDrone
                fire_intensity = self.fire_grid.grid[x, y]

            # Normalize position
            x = x / self.fire_grid.grid.shape[0]
            y = y / self.fire_grid.grid.shape[1]

            obs[agent_id] = np.concatenate((
                np.array([x, y, drone.battery_level], dtype=np.float32),
                fire_intensity.flatten().astype(np.float32)
            ))

        return obs

    def get_global_state(self):
        fire_state = self.fire_grid.grid.flatten()
        drone_states = []
        for drone in self.scouts + self.suppressors:
            drone_type = 0 if isinstance(drone, ScoutDrone) else 1
            x, y = drone.position

            x_norm = x / self.fire_grid.grid.shape[0]
            y_norm = y / self.fire_grid.grid.shape[1]
            drone_states.extend([x_norm, y_norm, drone.battery_level, drone_type])
            
            if isinstance(drone, SuppressorDrone):
                drone_states.append(float(drone.fire_extinguishing))
        
        return np.concatenate([fire_state, np.array(drone_states)])
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fire_grid.reset()
        for drone in self.agent_map.values():
            drone.reset()

        self.metrics_logger.reset()
        self.metrics_logger.log_initial_battery(self.agent_map.values())

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, infos
    
    def set_reward(self):
        # Initialize base reward
        rewards = {agent: -0.1 for agent in self.agents}

        fire_locations = np.argwhere(self.fire_grid.grid > 0)  # locations of active fire
        for agent_id in self.agents:
            drone = self.agent_map[agent_id]

            # Penalty for dead battery
            if drone.battery_level <= 0:
                rewards[agent_id] -= 1.0

            # Bonus for suppressing fire
            if isinstance(drone, SuppressorDrone):
                if self.fire_grid.grid[drone.position[0], drone.position[1]] > 0:
                    rewards[agent_id] += 1.0
                elif drone.fire_extinguishing:
                    rewards[agent_id] += -0.5

            # Bonus for scouting near fire
            if isinstance(drone, ScoutDrone):
                drone_pos = np.array(drone.position)

                local_fire_intensity = drone.sense_fire(self.fire_grid)
                fires_found = np.argwhere(local_fire_intensity > 0)
                rewards[agent_id] += 0.05 * len(fires_found)

        return rewards

    def step(self, actions):
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        fire_cells = np.sum(self.fire_grid.grid > 0)

        # Step each agent
        rewards = self.set_reward()

        for agent_id, action in actions.items():
            drone = self.agent_map[agent_id]
            drone.step(action)

        suppressing = 0
        for suppressor in self.suppressors:
            if suppressor.fire_extinguishing or self.fire_grid.grid[suppressor.position[0], suppressor.position[1]] > 0:
                suppressor.suppress_fire(self.fire_grid)
                suppressing += 1

        # Check for collisions
        positions = {(drone.position[0], drone.position[1]) for drone in self.agent_map.values()}
        if len(positions) < len(self.agent_map):
            self.metrics_logger.log_collision()

        extinguished_cells = fire_cells - np.sum(self.fire_grid.grid > 0)

        self.metrics_logger.episode_length += 1
        self.metrics_logger.log_idle_time(self.agent_map.values())
        self.metrics_logger.log_extinguished_fire(extinguished_cells)
        self.metrics_logger.log_extinguishing_time(suppressing)

        # Check termination conditions
        env_done = np.argwhere(self.fire_grid.grid > 0).size == 0
        for agent_id, drone in self.agent_map.items():
            terminations[agent_id] = drone.battery_level <= 0 or env_done
            truncations[agent_id] = False
            infos[agent_id] = {}

        if env_done or all(terminations.values()):
            self.metrics_logger.log_final_battery(self.agent_map.values())
            self.metrics_logger.total_fire_cells = self.fire_grid.total_fire_count

        # Update fire grid
        self.fire_grid.spread_fire()

        # Update rewards, terminations, truncations, and infos for active agents
        obs = self._get_obs()
        terminations = {agent: terminations[agent] for agent in self.agents}
        truncations = {agent: truncations[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        infos = {agent: infos[agent] for agent in self.agents}

        # Remove terminated agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        return obs, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if self.render_mode is None:
            raise ValueError("Render mode is not set. Please set render_mode to 'human'.")
        
        if self.render_mode == "human":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots()

            self.ax.clear()
        else:
            self.fig, self.ax = plt.subplots()

        self.ax.imshow(self.fire_grid.grid.T, cmap="hot", interpolation="nearest")

        scout_positions = [drone.position for drone in self.scouts]
        suppressor_positions = [drone.position for drone in self.suppressors]

        if scout_positions:
            scout_x, scout_y = zip(*scout_positions)
            self.ax.scatter(scout_x, scout_y, color="blue", label="Scout", s=20)
        if suppressor_positions:
            suppressor_x, suppressor_y = zip(*suppressor_positions)
            self.ax.scatter(suppressor_x, suppressor_y, color="green", label="Suppressor", s=20)

        self.ax.legend()
        self.ax.set_xlim(0, self.fire_grid.grid.shape[0])
        self.ax.set_ylim(0, self.fire_grid.grid.shape[1])
        self.ax.set_title("Firefighting Environment")


        if self.render_mode == "human":
            plt.pause(0.01)
            plt.draw()
        else:
            plt.show(block=False)
            plt.close(self.fig)

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
    

if __name__ == "__main__":
    env = FireFightingEnv()
    parallel_api_test(env, num_cycles=1000)

    obs = env.reset()
    num_steps = 50
    mean_rewards = []

    for step in range(num_steps):
        print(f"--- Step {step + 1} ---")
        
        actions = {}
        # Generate random actions for scouts and suppressors
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Random action for scout

        # Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        mean_rewards.append(np.mean(list(rewards.values())))
        print(f"Mean Reward: {np.mean(list(rewards.values()))}")
        
        # env.render()
        if all(terminations.values()):
            print("Episode finished!")
            break

    print("Mean Reward:", np.mean(mean_rewards))
    env.close()